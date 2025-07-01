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
#from pointnet2.pointnet2_utils import furthest_point_sample
from PIL import Image
# sys.path.append('/data/regnet_batch_4_gyf_5/')
import struct
from torch_scatter import scatter_mean
# import open3d

if __name__ == '__main__':
    import calib as calib
    import utils as utils
else:
    import src.calib as calib
    import src.utils as utils
from tqdm import tqdm


# def downsample_pc(pointcloud, voxel_grid_downsample_size):
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))
#
#     down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
#     down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
#     pointcloud = down_pcd_points
#
#     return pointcloud


def make_dataset(root_path, mode, acc=False,raw=False):
    if mode == 'train':
        seq_list = list(range(9))
    elif mode == 'val':
        seq_list = [9, 10]
    elif mode == 'test':
        seq_list = [7, 8]
    else:
        raise Exception('Invalid mode.')

    skip_start_end = 40 if acc else 0
    dataset = []
    for seq in seq_list:
        # scan_path = os.path.join(root_path, 'data_odometry_velodyne', 'dataset', '%02d' % seq, 'velodyne')
        if raw:
            scan_path = os.path.join(root_path, 'data_odometry_velodyne_deepi2p_new', 'data_odometry_velodyne_NWU',
                                 'sequences', '%02d' % seq, 'snr0.6')  # [7,N]
        else:
            scan_path = os.path.join(root_path, 'data_odometry_velodyne_deepi2p_new', 'data_odometry_velodyne_NWU',
                                 'sequences', '%02d' % seq, 'voxel0.1_snr0.6')  # [7,N]
        scan_path_ori = os.path.join(root_path, 'data_odometry_velodyne', 'dataset',
                                 '%02d' % seq, 'velodyne')  # [7,N]
        if not os.path.exists(scan_path):
            scan_path = os.path.join(root_path, 'kitti_processed_DeepI2P', 'data_odometry_velodyne_NWU',
                                     'sequences', '%02d' % seq, 'voxel0.1-SNr0.6')  # [7,N]
        sample_num = round(len(os.listdir(scan_path)))
        for i in range(skip_start_end, sample_num - skip_start_end):
            # pc_path = os.path.join(root_path, 'data_odometry_velodyne', 'dataset', '%02d' % seq, 'velodyne')
            pc_path = scan_path_ori
            pc_snr_path = scan_path
            img_path1 = os.path.join(root_path, 'kitti_processed_DeepI2P', 'data_odometry_color_npy', 'sequences',
                                     '%02d' % seq, 'image_2')
            # img_path2 = os.path.join(root_path, 'kitti_processed_DeepI2P', 'data_odometry_color_npy', 'sequences',
            #                         '%02d' % seq, 'image_3')
            calib_path = os.path.join(root_path, 'kitti_processed_DeepI2P','data_odometry_calib', 'dataset', 'sequences', '%02d' % seq,
                                      'calib.txt')
            poses = os.path.join(root_path, 'kitti_processed_DeepI2P', 'poses', '%02d' % seq, '%06d.npz' % i)
            dataset.append((pc_path,pc_snr_path, img_path1, calib_path, seq, sample_num, poses, i))

    return dataset


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


def random_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3] 24*1024*3
        npoint: number of samples 512
    Return:
        centroids: sampled pointcloud index, [B, npoint] 24*512
    """
    B, N, C = xyz.shape
    randmatrix = torch.randint(0, N, (B, npoint), dtype=torch.long)
    return randmatrix


# def farthest_point_sample(xyz, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [B, N, 3] 24*1024*3
#         npoint: number of samples 512
#     Return:
#         centroids: sampled pointcloud index, [B, npoint] 24*512
#     """
#     B, N, C = xyz.shape
#     centroids = torch.zeros(B, npoint, dtype=torch.long)
#     # dynamic programming store the distance
#     distance = torch.ones(B, N,dtype=torch.float32) * 1e10
#     # random init start point
#     farthest = torch.randint(0, N, (B,), dtype=torch.long)
#     batch_indices = torch.arange(B, dtype=torch.long)
#     for i in range(npoint):
#         centroids[:, i] = farthest
#         centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
#         temp_sum = (xyz - centroid) ** 2
#         temp_sum = torch.from_numpy(temp_sum).float()
#         dist = torch.sum(temp_sum, -1) # the distance towards centroid (last picked point)
#         mask = dist < distance # distance is the minimum distance towards set A
#         distance[mask] = dist[mask]
#         farthest = torch.max(distance, -1)[1]
#     return centroids

def random_point_sample_nopeat(xyz, npoint):
    # common implement
    B, N, C = xyz.shape
    randmatrix = torch.stack([torch.randperm(N)[:npoint] for _ in range(B)])
    return randmatrix


def sample_n_points(pcl, npoint=8192):
    """
    Input:
        npoint:8192
        xyz: input points position data, [B, N, 3]
    Return:
        new_xyz: sampled points position data, [B, npoint, 3]
    """
    B, N, C = pcl.shape
    assert N >= npoint, "N is too small: %d < %d" % (N, npoint)
    fps_idx = random_point_sample_nopeat(pcl, npoint)
    # fps_idx = furthest_point_sample(torch.from_numpy(pcl).cuda(),npoint)
    new_xyz = index_points(pcl, fps_idx)
    return new_xyz


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
    def __init__(self, params):
        self.acc = False

        self.downsample = False  # default False

        self.filter = False  # default True

        self.normal = False

        self.raw = True

        self.voxel = False

        self.pc_range = None
        #self.pc_range = [[-35, -20, -5], [35, 20, 2.5]]
        self.crop = False

        print("pc_range:", self.pc_range)

        self.yaug = False

        self.voxel_size = 0.05
        
        self.sample_point = 150000  # default 8192
        print("sample_point:", self.sample_point)

        self.using_cam_coord = False
        print("using cam coord:", self.using_cam_coord)

        self.root_path = params['root_path']
        self.mode = params['mode']
        self.dataset = make_dataset(self.root_path, self.mode, self.acc,self.raw)

        self.d_rot = params['d_rot']
        self.d_trans = params['d_trans']

        self.img_scale = 0.5
        self.img_H = 160  # 320 * 0.5
        self.img_W = 512  # 1224 * 0.5
        # self.resize_w = params['resize_w']
        # self.resize_h = params['resize_h']
        self.fixed_decalib = params['fixed_decalib']
        self.accumulation_frame_num = 3
        self.accumulation_frame_skip = 6

        # self.delta_ij_max = 5
        # self.translation_max = 5.0

        self.delta_ij_max = 40
        self.translation_max = 10.0

        self.rx = 0.
        self.ry = 2 * math.pi
        self.rz = 0.

        self.tx = 10.
        #self.ty = 0.5
        if self.mode == 'train' and self.yaug:
            self.ty = 0.5
        else:
            self.ty = 0
        print("y rand:", self.ty)
        self.tz = 10.

        # self.rot = 2 * math.pi
        # self.rot = math.pi

    def __len__(self):
        return len(self.dataset)

    def get_sequence_j(self, seq_sample_num, seq_i, seq_pose_folder, delta_ij_max, translation_max):
        # get the max and min of possible j
        seq_j_min = max(seq_i - delta_ij_max, 0)
        seq_j_max = min(seq_i + delta_ij_max, seq_sample_num - 1)

        # pose of i
        Pi = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_i))['arr_0'].astype(np.float32)  # 4x4

        while True:
            seq_j = random.randint(seq_j_min, seq_j_max)
            # get the pose, if the pose is too large, ignore and re-sample
            Pj = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_j))['arr_0'].astype(np.float32)  # 4x4
            Pji = np.dot(np.linalg.inv(Pj), Pi)  # 4x4
            t_ji = Pji[0:3, 3]  # 3
            t_ji_norm = np.linalg.norm(t_ji)  # scalar

            if t_ji_norm < translation_max:
                break

        return seq_j, Pji, t_ji

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

    def get_projected_pts(self, pcl, intrinsic, extrinsic, img_shape):
        # pcl = self.load_lidar(index)
        pcl_uv, pcl_z = utils.get_2D_lidar_projection(
            pcl, intrinsic, extrinsic)
        # print("pcl_uv before mask", pcl_uv)
        mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & \
               (pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)

        return pcl_uv[mask], pcl_z[mask]

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

    def generate_random_transform(self,
                                  P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
                                  P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude):
        """

        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-P_tx_amplitude, P_tx_amplitude),
             random.uniform(-P_ty_amplitude, P_ty_amplitude),
             random.uniform(-P_tz_amplitude, P_tz_amplitude)]
        angles = [random.uniform(-P_Rx_amplitude, P_Rx_amplitude),
                  random.uniform(-P_Ry_amplitude, P_Ry_amplitude),
                  random.uniform(-P_Rz_amplitude, P_Rz_amplitude)]

        rotation_mat = angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random

    def search_for_accumulation(self, pc_folder, seq_pose_folder,
                                seq_i, seq_sample_num, Pc, P_oi,
                                stride):
        Pc_inv = np.linalg.inv(Pc)
        P_io = np.linalg.inv(P_oi)

        pc_np_list, intensity_np_list, sn_np_list = [], [], []

        counter = 0
        while len(pc_np_list) < self.accumulation_frame_num:
            counter += 1
            seq_j = seq_i + stride * counter
            if seq_j < 0 or seq_j >= seq_sample_num:
                break

            # npy_data = read_velodyne_bin(pc_folder, seq_j)[:, 0:3].T.astype(np.float32)
            npy_data = np.load(os.path.join(pc_folder, '%06d.npy' % seq_j))

            pc_np = npy_data[0:3, :]  # 3xN

            P_oj = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_j))['arr_0'].astype(np.float32)  # 4x4
            P_ij = np.dot(P_io, P_oj)

            P_transform = np.dot(Pc_inv, np.dot(P_ij, Pc))
            pc_np = transform_pc_np(P_transform, pc_np)
            P_transform_rot = np.copy(P_transform)
            P_transform_rot[0:3, 3] = 0

            pc_np_list.append(pc_np)

        return pc_np_list

    def get_accumulated_pc(self, pc_folder, seq_pose_folder, seq_i, seq_sample_num, Pc):
        # npy_data = read_velodyne_bin(pc_folder, seq_i)[:, 0:3].T.astype(np.float32)

        npy_data = np.load(os.path.join(pc_folder, '%06d.npy' % seq_i)).astype(np.float32)
        # shuffle the point cloud data, this is necessary!
        npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
        pc_np = npy_data[0:3, :]  # 3xN

        # single frame is enough
        pc_np_list = [pc_np]

        if self.acc:
            # pose of i
            P_oi = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_i))['arr_0'].astype(np.float32)  # 4x4

            # search for previous
            prev_pc_np_list = self.search_for_accumulation(pc_folder,
                                                           seq_pose_folder,
                                                           seq_i,
                                                           seq_sample_num,
                                                           Pc,
                                                           P_oi,
                                                           -self.accumulation_frame_skip)
            # search for next
            next_pc_np_list = self.search_for_accumulation(pc_folder,
                                                           seq_pose_folder,
                                                           seq_i,
                                                           seq_sample_num,
                                                           Pc,
                                                           P_oi,
                                                           self.accumulation_frame_skip)

            pc_np_list = pc_np_list + prev_pc_np_list + next_pc_np_list

        pc_np = np.concatenate(pc_np_list, axis=1)

        return pc_np
    
    def augment_img_crop(self, img_np, intrinsics):
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
        jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2)
        #jitter.get_params(brightness, contrast, saturation, hue)
        img_PIL = Image.fromarray(img_np)
        img_color_aug_np = np.array(jitter(img_PIL))

        # random crop
        crop = utils.RandomScaleCrop()
        img_crop, intrinsics = crop(img_color_aug_np, intrinsics)
        return img_crop, intrinsics

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
        pc_folder,pc_snr_folder, img_folder, calib_path, seq, seq_sample_num, poses, seq_i = self.dataset[index]
        seq_pose_folder = os.path.join(self.root_path, 'kitti_processed_DeepI2P', 'poses', '%02d' % seq)

        # processing with calib params
        Tr, cam_intrinsic, P2 = read_calib(calib_path)
        Tr = np.vstack((Tr, [0, 0, 0, 1]))
        Pc = np.dot(P2, Tr)

        # accumulating successive frames of lidar images
        # pc_np = self.get_accumulated_pc(pc_folder, seq_pose_folder, seq_i, seq_sample_num, Tr)
        if self.raw:
            bin_data = np.fromfile(os.path.join(pc_folder,"%06d.bin"%seq_i),np.float32).reshape(-1,4).T
            # shuffle the point cloud data, this is necessary!
            shuffle_idx = np.random.permutation(bin_data.shape[1])
            bin_data = bin_data[:,shuffle_idx]
            sn_np = np.load(os.path.join(pc_snr_folder, '%06d.npy' % seq_i)).astype(np.float32)[-3:,shuffle_idx]
            pc_np = bin_data[0:3, :]  # 3xN
            intensity_np = bin_data[3:4, :]  # 1xN
        else:
            npy_data = np.load(os.path.join(pc_snr_folder, '%06d.npy' % seq_i)).astype(np.float32)
            # shuffle the point cloud data, this is necessary!
            npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
            pc_np = npy_data[0:3, :]  # 3xN
            intensity_np = npy_data[3:4, :]  # 1xN
            sn_np = npy_data[4:7, :]  # 3xN
        
        ### testing the pc range
        x = pc_np[0, :]
        y = pc_np[1, :]
        z = pc_np[2, :]
        #print("x min nax percent 5 95: ", np.min(x)," ", np.max(x), ",", np.percentile(x, (5, 95)))
        #print("y min nax percent 5 95: ", np.min(y)," ", np.max(y), ",", np.percentile(y, (5, 95)))
        #print("z min nax percent 5 95: ", np.min(z)," ", np.max(z), ",", np.percentile(z, (5, 95)))
        x_perc = np.percentile(x, (5, 95))
        y_perc = np.percentile(y, (5, 95))
        z_perc = np.percentile(z, (5, 95))

        # # degree filter
        # x = pc_np[0, :]
        # y = pc_np[1, :]
        # z = pc_np[2, :]
        # dist = np.sqrt(x*x+y*y)
        # tan2 = 0.03492076949
        # tan24 = -0.4452286853
        # ratio = z/dist
        # angle_mask = np.logical_and(ratio < tan2, ratio > tan24)
        # print(np.mean(angle_mask))
        # pc_np = pc_np[:, angle_mask]
        # intensity_np = intensity_np[:, angle_mask]
        if self.pc_range is not None:
            # range filter
            # print(pc_np.shape)
            x = pc_np[0, :]
            y = pc_np[1, :]
            z = pc_np[2, :]
            range_mask = np.logical_and(abs(x)<self.pc_range[1][0], abs(y)<self.pc_range[1][1])
            range_mask = range_mask & np.logical_and(z<self.pc_range[1][2], z>self.pc_range[0][2])
            pc_np = pc_np[:, range_mask]
            intensity_np = intensity_np[:, range_mask]
            sn_np = sn_np[:, range_mask]

        """   
        if self.pc_range:
            # range filter
            # print(pc_np.shape)
            x = pc_np[0, :]
            y = pc_np[1, :]
            range_mask = np.logical_and(abs(x)<35, abs(y)<35)
            pc_np = pc_np[:, range_mask]
            intensity_np = intensity_np[:, range_mask]
            sn_np = sn_np[:, range_mask]
            origin_pc_np = origin_pc_np[range_mask, :]
            # print(pc_np.shape)
        """
        # if self.downsample and pc_np.shape[1] > 2 * self.sample_point:
        #     # point cloud too huge, voxel grid downsample first
        #     pc_np = downsample_pc(pc_np, voxel_grid_downsample_size=0.3)
        #     pc_np = pc_np.astype(np.float32)

        # choosing frames of images under limits of index and translations diffs between lidar image.
        # seq_j, Pji, t_ji = self.get_sequence_j(seq_sample_num, seq_i, seq_pose_folder,
        #                                        self.delta_ij_max, self.translation_max)
        #print(pc_np.shape)
        Pr = self.generate_random_transform(self.tx, self.ty, self.tz,
                                            self.rx, self.ry, self.rz)
        Pr_inv = np.linalg.inv(Pr)
        calib_extrinsic = Pr_inv[:3, :]

        # calib_extrinsic = np.linalg.pinv(Pr)[:3, :]
        decalib_quat_real, decalib_quat_dual = utils.extrinsic_to_dual_quat(calib_extrinsic)
        decalib_quat_dual = calib_extrinsic[:, 3]

        decalib_quat_real = torch.from_numpy(decalib_quat_real).type(torch.FloatTensor)
        decalib_quat_dual = torch.from_numpy(decalib_quat_dual).type(torch.FloatTensor)

        init_extrinsic = np.dot(Pr, Pc)[:3, :]
        # init_extrinsic = Pr[:3,:]
        img_path = os.path.join(img_folder, '%06d.npy' % seq_i)
        rgb_img = np.load(img_path)
        
        # data augmentation
        if self.mode == 'train':
            # add N(0,0.0001(m^2)) abs not over 0.05(m) noise
            pc_np = self.jitter_point_cloud(pc_np)
            sn_np = self.jitter_point_cloud(sn_np)
            

        # N = pc_np.shape[1]
        # if N >= self.sample_point:
        #     choice_idx = np.random.choice(N, self.sample_point, replace=False)
        # else:
        #     fix_idx = np.arange(N)
        #     while fix_idx.shape[0] + N < self.sample_point:
        #         fix_idx = np.concatenate([fix_idx, np.arange(N)], axis=0)
        #     random_idx = np.random.choice(N, self.sample_point - fix_idx.shape[0], replace=False)
        #     choice_idx = np.concatenate([fix_idx, random_idx], axis=0)
        # select_idx = choice_idx

        # pc_np = np.concatenate([pc_np,])

        # pc_np = pc_np[:, select_idx]
        # sn_np = sn_np[:, select_idx]
        # intensity_np = intensity_np[:, select_idx]
        if self.using_cam_coord:
            pc_cam = Pc[:3] @ np.concatenate([pc_np, np.ones((1, pc_np.shape[1]))], axis=0)
            origin_pc_np = pc_cam.T.astype(np.float32)
        else:
            origin_pc_np = pc_np.copy().T.astype(np.float32) # (N,3)
        lidar_img_raw = origin_pc_np
        pc_np = np.concatenate([pc_np, np.ones((1, pc_np.shape[1]))], axis=0)
        
        pc_np = init_extrinsic @ pc_np

        sn_np = init_extrinsic[:, :3] @ sn_np  # normal not transition

        
        lidar_img = pc_np.T.astype(np.float32)
        # lidar_img = self.sample_projected_pts(pc_np.T, init_extrinsic, npoint=self.sample_point)

        # padding
        if self.voxel:
            if self.pc_range is not None:
                # range filter
                max_coord = np.array(self.pc_range[1])
                min_coord = np.array(self.pc_range[0])
            else:
                max_coord = np.max(lidar_img_raw, axis=0)
                min_coord = np.min(lidar_img_raw, axis=0)
            #print("min:", min_coord)
            spatial_size = np.ceil((max_coord - min_coord) / self.voxel_size).astype(np.int32)
            # print(spatial_size)
            voxel_coord = np.floor_divide((lidar_img_raw - min_coord[None]), self.voxel_size).astype(np.int32)
            #print("raw_voxel_center:",voxel_coord[0] * self.voxel_size + min_coord)
            
            #print(voxel_coord)
            voxel_coord, inv = np.unique(voxel_coord, axis=0, return_inverse=True)
            inv = torch.from_numpy(inv)
            lidar_img = scatter_mean(torch.from_numpy(lidar_img), inv, dim=0)
            origin_pc_np = scatter_mean(torch.from_numpy(lidar_img_raw), inv, dim=0)

        #### wyx_changed
        #lidar_feats = np.concatenate([sn_np.astype(np.float32), intensity_np.astype(np.float32)], axis=0).T
        #lidar_feats = np.concatenate([np.zeros_like(sn_np), np.zeros_like(intensity_np)], axis=0).T
        lidar_feats = intensity_np.astype(np.float32).T
        #print("lidar first shape:", lidar_feats.shape)
        
        #lidar_feats = None
        if self.voxel:
            N = voxel_coord.shape[0]
            if N >= self.sample_point:
                choice_idx = np.random.choice(N, self.sample_point, replace=False)
                select_idx = choice_idx
                # print("voxel_coord:", voxel_coord.shape)
                origin_pc_np = origin_pc_np[select_idx, :]
                # inv = inv[select_idx, :]
                lidar_img = lidar_img[select_idx, :]
                lidar_feats = lidar_feats[select_idx, :]
        else:
            N = lidar_img.shape[0]
            #print(N)
            lidar_img = np.concatenate([lidar_img,np.zeros((self.sample_point-N,3),dtype=np.float32)],axis=0)

            #lidar_feats = np.concatenate([lidar_feats, np.zeros((self.sample_point - N, 4), dtype=np.float32)],
            #                           axis=0)
            lidar_feats = np.concatenate([lidar_feats, np.zeros((self.sample_point - N, 1), dtype=np.float32)],
                                    axis=0)
            #print("lidar second shape:", lidar_feats.shape)

            #print("pc shape:", pc_np.shape)
            #print("lidar_shape:", lidar_feats.shape)
            origin_pc_np = np.concatenate(
                [origin_pc_np, np.zeros((self.sample_point - N, 3), dtype=np.float32)],
                axis=0)

        # crop the first few rows
        crop_original_top_rows = 50
        rgb_img = rgb_img[crop_original_top_rows:, :, :]
        intrinsic = camera_matrix_cropping(cam_intrinsic, dx=0, dy=crop_original_top_rows)

        h, w, _ = rgb_img.shape
        # modify camera intrinsic because of resizing
        rgb_img = cv2.resize(rgb_img,
                             (int(round(w * self.img_scale)),
                              int(round((h * self.img_scale)))),
                             interpolation=cv2.INTER_LINEAR)
        intrinsic[0, 0] = self.img_scale * intrinsic[0, 0]  # width x
        intrinsic[0, 2] = self.img_scale * intrinsic[0, 2]  # width x
        intrinsic[1, 1] = self.img_scale * intrinsic[1, 1]  # height y
        intrinsic[1, 2] = self.img_scale * intrinsic[1, 2]  # height y
        h, w, _ = rgb_img.shape
        # random crop into input size
        
        if self.mode == 'train':
            img_crop_dx = random.randint(0, w - self.img_W)
            img_crop_dy = random.randint(0, h - self.img_H)
        else:
            img_crop_dx = int((w - self.img_W) / 2)
            img_crop_dy = int((h - self.img_H) / 2)

        # crop image
        rgb_img = rgb_img[img_crop_dy:img_crop_dy + self.img_H,
                  img_crop_dx:img_crop_dx + self.img_W, :]
        intrinsic = camera_matrix_cropping(intrinsic, dx=img_crop_dx, dy=img_crop_dy)
        rgb_ini = rgb_img

        # h, w, _ = rgb_img.shape
        # rgb_img = cv2.resize(rgb_img, (self.resize_w, self.resize_h))
        if self.mode == 'train':
            if self.crop:
                rgb_img, intrinsic = self.augment_img_crop(rgb_img, intrinsic)
            else:
                rgb_img = self.augment_img(rgb_img)
        rgb_proc = rgb_img
        #print(rgb_img.shape)
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

        sample = {}
        sample['rgb'] = rgb_img
        #sample['raw_rgb'] = rgb_ini
        #sample['proc_rgb'] = rgb_proc
        sample['decalib_real_gt'] = decalib_quat_real
        sample['decalib_dual_gt'] = decalib_quat_dual
        sample['init_extrinsic'] = init_extrinsic
        sample['init_intrinsic'] = intrinsic
        sample['raw_intrinsic'] = cam_intrinsic
        sample['lidar'] = lidar_img
        #sample['raw_lidar'] = pc_cam
        sample['resize_img'] = np.array([self.img_scale, self.img_scale])
        sample['index'] = index
        # sample['tji'] = float(np.linalg.norm(t_ji, 2))
        sample["path_info"] = "%02d %06d %06d" % (seq, seq_i, seq_i)
        sample["lidar_feats"] = lidar_feats
        sample["raw_point_xyz"] = origin_pc_np
        sample["pc_stat"] = (x_perc, y_perc, z_perc)

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
    kitti = Kitti_Odometry_Dataset(params=dataset_params)
    print("data num:")
    print(len(kitti))
    import open3d as o3d
    import matplotlib.pyplot as plt
    save_path =  '/data/debug2/img_aug/'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    x_avg = np.array([0., 0.])
    y_avg = np.array([0., 0.])
    z_avg = np.array([0., 0.])
    x_all = np.array([0., 0.])
    y_all = np.array([0., 0.])
    z_all = np.array([0., 0.])
    count = 0
    for i, data in enumerate(kitti, 0):
        sample = data
        x_perc, y_perc, z_perc = sample['pc_stat']
        x_avg += x_perc
        y_avg += y_perc
        z_avg += z_perc
        x_all += x_perc
        y_all += y_perc
        z_all += z_perc
        count += 1
        if i % 100 == 0:
            if i == 0:
                x_avg = 0
                y_avg = 0
                z_avg = 0
            else:
                print("------iter", i, "---------")
                print("x percent 5 95: ", x_avg/100)
                print("y percent 5 95: ", y_avg/100)
                print("z percent 5 95: ", z_avg/100)
                x_avg = 0
                y_avg = 0
                z_avg = 0
            # print(sample['lidar'].shape)
            # lidar = sample['raw_lidar']
            # o3d_pc = o3d.geometry.PointCloud()
            # o3d_pc.points = o3d.utility.Vector3dVector(lidar)
            # o3d.io.write_point_cloud(save_path+"pc_"+str(i)+".pcd", o3d_pc)

            # lidar = sample['raw_point_xyz']
            # o3d_pc = o3d.geometry.PointCloud()
            # o3d_pc.points = o3d.utility.Vector3dVector(lidar)
            # o3d.io.write_point_cloud(save_path+"pc_raw_"+str(i)+".pcd", o3d_pc)
            
            # img = sample['raw_rgb']#.astype(unit8)
            # plt.figure()
            # plt.imshow(img)
            # plt.savefig(save_path+"image"+str(i)+".jpg")

            # img = sample['proc_rgb']#.astype(unit8)
            # plt.figure()
            # plt.imshow(img)
            # plt.savefig(save_path+"image_proc"+str(i)+".jpg")
        # else:
        #     #print(i)
        #     continue
    print("x percent 5 95: ", x_all/count)
    print("y percent 5 95: ", y_all/count)
    print("z percent 5 95: ", z_all/count)


