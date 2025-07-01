import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import cv2
import os
import sys
import math
import random
from torchvision import transforms

from PIL import Image
from scipy.spatial.transform import Rotation
sys.path.append('/data/I2PNet/')
import struct
import pickle as pkl
import h5py
import pandas as pd


# import open3d

if __name__ == '__main__':
    import calib as calib
    import utils as utils
else:
    import src.calib as calib
    import src.utils as utils
from tqdm import tqdm


def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == (4,), "Not a valid quaternion"
    if not np.isclose(np.linalg.norm(q), 1.):
        q = q / np.linalg.norm(q)
    mat = np.zeros((3, 3), np.float32)
    mat[0, 0] = 1 - 2 * q[2] ** 2 - 2 * q[3] ** 2
    mat[0, 1] = 2 * q[1] * q[2] - 2 * q[3] * q[0]
    mat[0, 2] = 2 * q[1] * q[3] + 2 * q[2] * q[0]
    mat[1, 0] = 2 * q[1] * q[2] + 2 * q[3] * q[0]
    mat[1, 1] = 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2
    mat[1, 2] = 2 * q[2] * q[3] - 2 * q[1] * q[0]
    mat[2, 0] = 2 * q[1] * q[3] - 2 * q[2] * q[0]
    mat[2, 1] = 2 * q[2] * q[3] + 2 * q[1] * q[0]
    mat[2, 2] = 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2

    return mat


def load_map(seq, root_path, start, end):
    map_path = os.path.join(root_path, 'kitti_processed_CMRNet',
                            'sequences', '%02d' % seq, 'map',
                            f'map-{seq:02d}_0.1_{start}-{end}.pcd')

    return open3d.io.read_point_cloud(map_path)


def make_dataset(root_path, mode, h5=False, larger=False):
    if mode == 'train':
        seq_list = [3, 5, 6, 7, 8, 9]
    elif mode == 'val':
        seq_list = [0]#[0, 10]
    elif mode == 'test':
        seq_list = [0]#[0, 10]
    else:
        raise Exception('Invalid mode.')

    maps = []
    dataset = []
    for seq in seq_list:
        # scan_path = os.path.join(root_path, 'data_odometry_velodyne', 'dataset', '%02d' % seq, 'velodyne')
        # [7,N]

        # sample_num = round(len(os.listdir(scan_path)))
        with open(os.path.join('/data/I2PNet/data_preprocess/CMRNet_script/data', f'kitti-{seq:02d}.csv')) as f:
        #with open(os.path.join(root_path, 'kitti_processed_CMRNet_small', 'poses', f'kitti-{seq:02d}.csv')) as f:
            lines = f.readlines()[1:]
            poses = [line.strip('\n').split(',')[1:] for line in lines]  # timestamp,x,y,z,qx,qy,qz,qw
        scans = len(poses)
        if not h5:
            print(f"[INFO] Load Map {seq:02d}")
            if seq == 8:
                maps.append([load_map(seq, root_path, 0, 3000), load_map(seq, root_path, 3000, scans)])
            else:
                maps.append([load_map(seq, root_path, 0, scans)])
            map_seq_idx = len(maps) - 1
            print(f"[INFO] Load Map {seq:02d} Finished")
        else:
            map_seq_idx = -1
        for i in range(0, scans):
            pc_path = os.path.join(root_path, 'kitti_processed_CMRNet_small', "sequences", '%02d' % seq,
                                   f"local_maps_{0.3 if larger else 0.1:.1f}")
            # pc_path = os.path.join(root_path, 'data_odometry_velodyne', 'dataset', '%02d' % seq, 'velodyne')
            map_idx = 0
            if seq == 8 and i >= 3000:
                map_idx = 1
            img_path1 = os.path.join(root_path, 'kitti_processed_DeepI2P', 'data_odometry_color_npy', 'sequences',
                                     '%02d' % seq, 'image_2')
            # img_path2 = os.path.join(root_path, 'kitti_processed_DeepI2P', 'data_odometry_color_npy', 'sequences',
            #                         '%02d' % seq, 'image_3')
            calib_path = os.path.join(root_path, 'kitti_processed_DeepI2P','data_odometry_calib', 'dataset', 'sequences', '%02d' % seq,
                                      'calib.txt')
            # poses = os.path.join(root_path, 'kitti_processed_DeepI2P', 'poses', '%02d' % seq, '%06d.npz' % i)
            dataset.append(
                (pc_path, img_path1, calib_path, seq, (map_seq_idx, map_idx), np.array(poses[i], np.float32), i))

    return dataset, maps


def angles2rotation_matrix(angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def transform_pc_np(P, pc_np):
    """
    :param pc_np: 3xN
    :param P: 4x4
    :return:
    """
    if pc_np.shape[0] != 3:
        pc_np = pc_np.T

    pc_homo_np = np.concatenate((pc_np,
                                 np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
                                axis=0)
    P_pc_homo_np = np.dot(P, pc_homo_np)
    return P_pc_homo_np[0:3, :]


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(
        view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def read_velodyne_bin(path, seq_i):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*4
    '''
    pc_list = []
    with open(os.path.join(path, '%06d.bin' % seq_i), 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2], point[3]])
    return np.asarray(pc_list, dtype=np.float32)


def read_calib(calib_file_path):
    with open(calib_file_path, 'r') as f:
        for line in f.readlines():
            key = line[0:2]
            mat = np.fromstring(line[4:], sep=' ').reshape((3, 4)).astype(np.float32)

            if 'Tr' == key:
                Tr = mat
            elif 'P2' == key:
                K = mat[0:3, 0:3]
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                intrinsic = np.asarray([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]])

                tz = mat[2, 3]
                tx = (mat[0, 3] - cx * tz) / fx
                ty = (mat[1, 3] - cy * tz) / fy
                P = np.identity(4)
                P[0:3, 3] = np.asarray([tx, ty, tz])

    return Tr, intrinsic, P


def camera_matrix_cropping(K, dx, dy):
    """no shift camera matrix"""
    K_crop = np.copy(K)
    K_crop[0, 2] -= dx
    K_crop[1, 2] -= dy
    return K_crop


class Kitti_Odometry_Dataset(Dataset):
    def __init__(self, params, use_raw = False):
        self.acc = False

        self.downsample = False  # default False

        self.filter = False  # default True

        self.normal = False

        self.use_raw = use_raw

        self.h5 = True

        self.large_sparse = False

        # self.vis_filter = False

        self.sample_point = 8192  # default 8192

        self.root_path = params['root_path']
        self.mode = params['mode']
        self.dataset, self.maps = make_dataset(self.root_path, self.mode, self.h5, self.large_sparse)

        self.d_rot = params['d_rot']
        self.d_trans = params['d_trans']

        # self.img_scale = 0.5
        self.img_H = 384
        self.img_W = 1280

        # self.img_H = 640
        # self.img_W = 1280

        #self.img_H = 160 
        #self.img_W = 640 

        self.fixed_decalib = params['fixed_decalib']

        max_r = 10.
        max_t = 2.

        if self.mode == "test":  # validation
            self.test_RT=[]
            test_RT_file = "/dataset/kitti_processed_CMRNet/"+f"test_RT_seq00_{max_r:.2f}_{max_t:.2f}.csv"
            df_test_RT = pd.read_csv(test_RT_file, sep=',')
            for index, row in df_test_RT.iterrows():
                self.test_RT.append(list(row))
            # with open("/data/I2PNet/data_preprocess/CMRNet_script/"
            #           f"test_set_trans/test_RT_seq09_{max_r:.2f}_{max_t:.2f}_0.pkl", 'rb') as f:
            #     self.test_RT = pkl.load(f)
        if self.mode == "val":  # test
            # self.test_RT=[]
            # test_RT_file = "/dataset/kitti_processed_CMRNet/"+f"test_RT_seq00_{max_r:.2f}_{max_t:.2f}.csv"
            # df_test_RT = pd.read_csv(test_RT_file, sep=',')
            # for index, row in df_test_RT.iterrows():
            #     self.test_RT.append(list(row))
            cmr_seed = params["cmr_seed"]  # 0-9
            cmr_seed = ((cmr_seed % 2) * 5) * 10 ** (cmr_seed // 2)
            with open("/data/I2PNet/data_preprocess/CMRNet_script/"
                      f"test_set_trans/test_RT_seq00_{max_r:.2f}_{max_t:.2f}_{cmr_seed:d}.pkl", 'rb') as f:
                test_RT_00 = pkl.load(f)
            # with open("/data/I2PNet/data_preprocess/CMRNet_script/"
            #           f"test_set_trans/test_RT_seq10_{max_r:.2f}_{max_t:.2f}_{cmr_seed:d}.pkl", 'rb') as f:
            #     test_RT_10 = pkl.load(f)
            # test_RT_00.extend(test_RT_10)
            #print(len(test_RT_00))
            self.test_RT = test_RT_00

        # self.rx = 10. * np.pi / 180.
        # self.ry = 10. * np.pi / 180.
        # self.rz = 10. * np.pi / 180.
        self.rx = max_r * np.pi / 180.
        self.ry = max_r * np.pi / 180.
        self.rz = max_r * np.pi / 180.

        # self.tx = 2.
        # self.ty = 2.
        # self.tz = 2.
        self.tx = max_t
        self.ty = max_t
        self.tz = max_t
        if self.mode == "val" or self.mode == "test":
            assert len(self.test_RT) == len(self.dataset), "Something wrong with test RTs"

    def __len__(self):
        return len(self.dataset)

    def jitter_point_cloud(self, pc_np, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
            CxN array, original point clouds
            Return:
            CxN array, jittered point clouds
        """
        C, N = pc_np.shape
        assert (clip > 0)
        jittered_pc = np.clip(sigma * np.random.randn(C, N), -1 * clip, clip).astype(pc_np.dtype)
        jittered_pc += pc_np
        return jittered_pc

    def sample_projected_pts(self, pcl_xyz, extrinsic, npoint=8192):
        """
        point process ours
        """
        if self.filter:
            is_ground = np.logical_or(pcl_xyz[:, 2] < -15, pcl_xyz[:, 2] < -15)
            not_ground = np.logical_not(is_ground)

            near_mask_x = np.logical_and(pcl_xyz[:, 0] < 1200, pcl_xyz[:, 0] > 2)
            near_mask_z = np.logical_and(pcl_xyz[:, 1] < 1200, pcl_xyz[:, 1] > -1200)

            near_mask = np.logical_and(near_mask_x, near_mask_z)
            near_mask = np.logical_and(not_ground, near_mask)
            indices_1 = np.where(near_mask)[0]

            pcl_xyz = pcl_xyz[indices_1, :]

        # S,3
        pcl_xyz = pcl_xyz[np.random.choice(pcl_xyz.shape[0], npoint, replace=False), :]

        pcl_xyz = np.hstack((pcl_xyz[:, 0:3], np.ones((pcl_xyz.shape[0], 1)))).T
        pcl_xyz_tmp = extrinsic @ pcl_xyz

        pcl_xyz_tmp = pcl_xyz_tmp.T
        pcl_xyz_tmp = pcl_xyz_tmp.reshape(1, -1, 3).astype(np.float32)

        # pcl_xyz_tmp = pcl_xyz.T
        # pcl_xyz = pcl_xyz_tmp.reshape(1, -1, 3)

        return pcl_xyz_tmp

    def generate_transformation(self, rx, ry, rz, tx, ty, tz):
        rotation_mat = Rotation.from_euler('xzy', [rx, rz, ry]).as_matrix().reshape(3, 3)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = [tx, ty, tz]
        return P_random

    def generate_random_transform(self,
                                  P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
                                  P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude):
        """

        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-P_tx_amplitude, P_tx_amplitude),
             random.uniform(-P_ty_amplitude, P_ty_amplitude),
             random.uniform(-P_tz_amplitude, min(P_tz_amplitude, 1.))]
        angles = [random.uniform(-P_Rx_amplitude, P_Rx_amplitude),
                  random.uniform(-P_Ry_amplitude, P_Ry_amplitude),
                  random.uniform(-P_Rz_amplitude, P_Rz_amplitude)]

        # rotation_mat = angles2rotation_matrix(angles)

        return self.generate_transformation(*angles, *t)

    def augment_img(self, img_np):
        """
        img data augmentation
        :param img: HxWx3, np.ndarray
        :return:
        """
        # color perturbation
        brightness = [0.8, 1.2]
        contrast = [0.8, 1.2]
        saturation = [0.8, 1.2]
        hue = [-0.1, 0.1]
        jitter = transforms.ColorJitter()
        jitter.get_params(brightness, contrast, saturation, hue)
        img_PIL = Image.fromarray(img_np)
        img_color_aug_np = np.array(jitter(img_PIL))
        return img_color_aug_np

    def __getitem__(self, index):
        # getting dataset paths
        # pose [x,y,z,qw,qx,qy,qz]
        pc_folder, img_folder, calib_path, seq, map_idx, poses, seq_i = self.dataset[index]
        # seq_pose_folder = os.path.join(self.root_path, 'kitti_processed_DeepI2P', 'poses', '%02d' % seq)

        # processing with calib params
        Tr, cam_intrinsic, P2 = read_calib(calib_path)
        Tr = np.vstack((Tr, [0, 0, 0, 1]))
        Pc = np.dot(P2, Tr)

        # accumulating successive frames of lidar images
        # pc_np = self.get_accumulated_pc(pc_folder, seq_pose_folder, seq_i, seq_sample_num, Tr)
        if not self.h5:
            global_map = self.maps[map_idx[0]][map_idx[1]]

            # generate the local map runtime
            # forward pose
            pose_R = quat2mat(poses[[6, 3, 4, 5]])  # qw,qx,qy,qz
            pose_t = poses[:3].reshape(3, 1)
            # inverse pose [R^T,-R^T@t]
            pose_R_inv = pose_R.T
            pose_t_inv = -pose_R_inv @ pose_t

            local_pc = np.asarray(global_map.points).T  # (3,N)
            # voxelized = voxelized.to(args.device)
            local_intensity = np.asarray(global_map.colors)[:, 0:1].T  # (1,N)

            local_pc = (pose_R_inv @ local_pc + pose_t_inv)

            indexes = local_pc[1] > -25.
            indexes = indexes & (local_pc[1] < 25.)
            indexes = indexes & (local_pc[0] > -10.)
            indexes = indexes & (local_pc[0] < 100.)

            pc_np = local_pc[:, indexes]
            intensity_np = local_intensity[:, indexes]
            # shuffle the point cloud data, this is necessary!
            # shuffle_idx = np.random.permutation(local_pc.shape[1])
            # pc_np = local_pc[:, shuffle_idx]
            # intensity_np = local_intensity[:, shuffle_idx]
        else:
            # [-5,15] [-10,10]
            with h5py.File(os.path.join(pc_folder, "%06d.h5" % seq_i), 'r') as hf:
                pc_np = np.asarray(hf['PC'], np.float32)[:3]
                intensity_np = np.asarray(hf['intensity'], np.float32)
        # submap generation
        # mask_x = np.logical_and(pc_np[0] > -5, pc_np[0] < 15)
        # mask_y = np.logical_and(pc_np[1] < 10, pc_np[1] > -10)
        # mask = np.logical_and(mask_x, mask_y)
        # pc_np = pc_np[:, mask]
        # intensity_np = intensity_np[:, mask]
        # print(local_pc.shape)     
        # print("----------------------")
        # print(np.max(pc_np[0]))
        # print(np.min(pc_np[0]))
        # print(np.max(pc_np[1]))
        # print(np.min(pc_np[1]))
        # print(np.max(pc_np[2]))
        # print(np.min(pc_np[2]))

        if self.use_raw:
            origin_pc_np = pc_np.copy().T.astype(np.float32) # (N,3)
            # ori_pc_np = pc_np
            # origin_pc_np = pc_np.copy().T.astype(np.float32)  # (N,3)
            # origin_pc_np[:, 0] = ori_pc_np[1, :]
            # origin_pc_np[:, 1] = -ori_pc_np[0, :]

        # The pose err between the coarse pose estimation and the accurate pose
        if self.mode == "train":
            Pr = self.generate_random_transform(self.tx, self.ty, self.tz,
                                                self.rx, self.ry, self.rz)
        else:
            """
            initial_RT = self.test_RT[seq_i]
            rz = initial_RT[6]
            ry = initial_RT[5]
            rx = initial_RT[4]
            tx = initial_RT[1]
            ty = initial_RT[2]
            tz = initial_RT[3]
            """
            rx, ry, rz, tx, ty, tz = self.test_RT[seq_i]
            Pr = self.generate_transformation(rx, ry, rz, tx, ty, tz)
        Pr_inv = np.linalg.inv(Pr)

        # print("----------------------")
        # print(np.max(pc_np[0]))
        # print(np.min(pc_np[0]))
        # print(np.max(pc_np[1]))
        # print(np.min(pc_np[1]))
        # print(np.max(pc_np[2]))
        # print(np.min(pc_np[2]))

        # The world coordination (PC)(I) @ the Pr (pose err) is the ground truth pose
        calib_extrinsic = Pr[:3, :]

        # calib_extrinsic = np.linalg.pinv(Pr)[:3, :]
        decalib_quat_real, decalib_quat_dual = utils.extrinsic_to_dual_quat(calib_extrinsic)
        decalib_quat_dual = calib_extrinsic[:, 3]

        decalib_quat_real = torch.from_numpy(decalib_quat_real).type(torch.FloatTensor)
        decalib_quat_dual = torch.from_numpy(decalib_quat_dual).type(torch.FloatTensor)

        # Generate the err PC
        init_extrinsic = np.dot(Pr_inv, Pc)[:3, :]
        #print(Pc)
        # init_extrinsic = Pr[:3,:]
        img_path = os.path.join(img_folder, '%06d.npy' % seq_i)
        rgb_img = np.load(img_path)
        rgb_ini = rgb_img

        # crop the first few rows
        crop_original_top_rows = 50
        rgb_img = rgb_img[crop_original_top_rows:, :, :]
        intrinsic = camera_matrix_cropping(cam_intrinsic, dx=0, dy=crop_original_top_rows)

        h, w, _ = rgb_img.shape
        # modify camera intrinsic because of resizing
        rgb_img = cv2.resize(rgb_img, (self.img_W, self.img_H),
                             interpolation=cv2.INTER_LINEAR)
        intrinsic[0, 0] = self.img_W / w * intrinsic[0, 0]  # width x
        intrinsic[0, 2] = self.img_W / w * intrinsic[0, 2]  # width x
        intrinsic[1, 1] = self.img_H / h * intrinsic[1, 1]  # height y
        intrinsic[1, 2] = self.img_H / h * intrinsic[1, 2]  # height y
        h, w, _ = rgb_img.shape

        rgb_ini = rgb_img
        if self.mode == 'train':
            # data augmentation
            # add N(0,0.0001(m^2)) abs not over 0.05(m) noise
            pc_np = self.jitter_point_cloud(pc_np)
            # sn_np = self.jitter_point_cloud(sn_np)
            rgb_img = self.augment_img(rgb_img)


        N = pc_np.shape[1]
        if N >= self.sample_point:
            choice_idx = np.random.choice(N, self.sample_point, replace=False)
        else:
            fix_idx = np.arange(N)
            while fix_idx.shape[0] + N < self.sample_point:
                fix_idx = np.concatenate([fix_idx, np.arange(N)], axis=0)
            random_idx = np.random.choice(N, self.sample_point - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate([fix_idx, random_idx], axis=0)
        select_idx = choice_idx

        pc_np = pc_np[:, select_idx]
        # sn_np = sn_np[:, select_idx]
        intensity_np = intensity_np[:, select_idx]
        if self.use_raw:
            origin_pc_np = origin_pc_np[select_idx, :]

        pc_np = np.concatenate([pc_np, np.ones((1, pc_np.shape[1]))], axis=0)
        pc_cam = (Pc[:3, :] @ pc_np).T.astype(np.float32)
        pc_np = init_extrinsic @ pc_np

        
        # lidar_img = self.sample_projected_pts(pc_np.T, init_extrinsic, npoint=self.sample_point)

        lidar_img = pc_np.T.astype(np.float32)

        #lidar_feats = np.concatenate([np.zeros_like(pc_np).astype(np.float32), intensity_np.astype(np.float32)],
        #                              axis=0).T
        lidar_feats = lidar_img

        # random crop into input size
        # if self.mode == 'train':
        #     img_crop_dx = random.randint(0, w - self.img_W)
        #     img_crop_dy = random.randint(0, h - self.img_H)
        # else:
        #     img_crop_dx = int((w - self.img_W) / 2)
        #     img_crop_dy = int((h - self.img_H) / 2)
        # crop image
        # rgb_img = rgb_img[img_crop_dy:img_crop_dy + self.img_H,
        #           img_crop_dx:img_crop_dx + self.img_W, :]
        # intrinsic = camera_matrix_cropping(intrinsic, dx=img_crop_dx, dy=img_crop_dy)

        # h, w, _ = rgb_img.shape
        # rgb_img = cv2.resize(rgb_img, (self.resize_w, self.resize_h))

        # img normalization
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        rgb_img = torch.from_numpy(rgb_img).type(torch.FloatTensor)
        if self.normal:
            rgb_img[:, :, 0] = (rgb_img[:, :, 0] / 255 - imagenet_mean[0]) / imagenet_std[0]
            rgb_img[:, :, 1] = (rgb_img[:, :, 1] / 255 - imagenet_mean[1]) / imagenet_std[1]
            rgb_img[:, :, 2] = (rgb_img[:, :, 2] / 255 - imagenet_mean[2]) / imagenet_std[2]
        rgb_img = rgb_img.permute(2, 0, 1)

        # resize_img = np.array([self.resize_w / w, self.resize_h / h])
        # # intrinsic = cam_intrinsic.copy()
        # intrinsic[0, 0] = resize_img[0] * intrinsic[0, 0]  # width x
        # intrinsic[0, 2] = resize_img[0] * intrinsic[0, 2]  # width x
        # intrinsic[1, 1] = resize_img[1] * intrinsic[1, 1]  # height y
        # intrinsic[1, 2] = resize_img[1] * intrinsic[1, 2]  # height y

        if not self.use_raw:
            origin_pc_np = np.zeros_like(lidar_feats)

        sample = {}
        sample['rgb'] = rgb_img
        sample['resize_rgb'] = rgb_ini
        sample['decalib_real_gt'] = decalib_quat_real
        sample['decalib_dual_gt'] = decalib_quat_dual
        sample['init_extrinsic'] = init_extrinsic
        sample['init_intrinsic'] = intrinsic
        sample['raw_intrinsic'] = cam_intrinsic
        sample['lidar'] = lidar_img
        # sample['raw_lidar'] = pc_raw
        sample['resize_img'] = np.array([self.img_H, self.img_W])
        sample['index'] = index
        # sample['tji'] = float(np.linalg.norm(t_ji, 2))
        sample["path_info"] = "%02d %06d %06d" % (seq, seq_i, seq_i)
        sample["lidar_feats"] = lidar_feats
        sample["raw_point_xyz"] = origin_pc_np
        sample["pc_cam"] = pc_cam

        return sample


if __name__ == '__main__':
    dataset_params = {
        'root_path': '/dataset',
        'mode': 'train',
        'd_rot': 10,  # not used
        'd_trans': 1.0,  # not used
        'fixed_decalib': False,
        'resize_w': 1216,
        'resize_h': 352,
    }
    save_path = "/data/debug/kd/"
    import open3d as o3d
    import matplotlib.pyplot as plt
    kitti = Kitti_Odometry_Dataset(params=dataset_params, use_raw = True)
    print(len(kitti))
    for i, data in enumerate(kitti, 0):
        sample = data
        #sample["raw_point_xyz"]
        # lidar = sample["raw_point_xyz"]
        # print(lidar.shape)
        # o3d_pc = o3d.geometry.PointCloud()
        # o3d_pc.points = o3d.utility.Vector3dVector(lidar)
        # o3d.io.write_point_cloud(save_path+"pc_raw_"+"test"+".pcd", o3d_pc)
        lidar_raw = sample['pc_cam']
        #print(lidar.shape)
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(lidar_raw)
        o3d.io.write_point_cloud(save_path+"pc_cam_"+"test"+".pcd", o3d_pc)
        img = sample['resize_rgb']  # .astype(unit8)
        plt.figure()
        plt.imshow(img)
        plt.savefig(save_path + "image_proc" + str(i) + ".jpg")

        calibs = sample['init_intrinsic']
        # calibs[0, 0] = 1 / 2 * calibs[0, 0]  # width x
        # calibs[0, 2] = 1 / 2 * calibs[0, 2]  # width x
        # calibs[1, 1] = 1 / 2 * calibs[1, 1]  # height y
        # calibs[1, 2] = 1 / 2 * calibs[1, 2]  # height y
        print(calibs)
        #print(lidar.shape)
        lidar = np.zeros_like(lidar_raw)
        lidar = lidar_raw
        # lidar[:, 0] = -lidar_raw[:, 1]
        # lidar[:, 1] = -lidar_raw[:, 2]
        # lidar[:, 2] = lidar_raw[:, 0]
        pix_pc = np.transpose(np.dot(calibs, np.transpose(lidar[:,:3])))
        #print(pix_pc)
        #print(pix_pc)
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pix_pc)
        o3d.io.write_point_cloud(save_path+"pc_pix0_"+"test"+".pcd", o3d_pc)
        pix_pc[:, :2] = np.divide(pix_pc[:, :2],  (pix_pc[:, 2])[:,None])
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pix_pc)
        o3d.io.write_point_cloud(save_path+"pc_pix_"+"test"+".pcd", o3d_pc)
        z_ = pix_pc[:, 2]
        xy = pix_pc[:, :2]
        is_in_picture = (xy[:, 0] >= 0) & (xy[:, 0] <= (1216  - 1)) & (xy[:, 1] >= 0) & (
                xy[:, 1] <= (352 - 1)) & (z_ > 0)
        z_ = z_[is_in_picture]
        #print("y max:", np.max(xy[:,1]))
        #print("z:",  z_)
        xy = xy[is_in_picture, :]

        #plt.savefig(file_img_seq+str(ts)+"_img_left"+".jpg")
        pc_draw = (z_-np.min(z_)/(np.max(z_)-np.min(z_)))
        plt.scatter(xy[:,0], xy[:,1], c=pc_draw, cmap='jet', alpha=0.7, s=1)
        plt.savefig(save_path + "pc_proj" + str(i) +".jpg")

        plt.close()

        break
