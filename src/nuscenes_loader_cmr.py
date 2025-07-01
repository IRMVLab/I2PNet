import torch.utils.data as data
import random
import os
import os.path
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
import pickle
from pyquaternion import Quaternion
import pickle as pkl
import sys
sys.path.append('/data/I2PNet/')
import src.utils as utils
import math
from scipy.spatial.transform import Rotation
from nuscenes.utils.data_classes import LidarPointCloud
import h5py
import pandas as pd


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


def camera_matrix_scaling(K: np.ndarray, s: float):
    K_scale = s * K
    K_scale[2, 2] = 1
    return K_scale


def random_sample_pc(pc_np, k):
    if pc_np.shape[1] >= k:
        choice_idx = np.random.choice(pc_np.shape[1], k, replace=False)
    else:
        fix_idx = np.asarray(range(pc_np.shape[1]))
        while pc_np.shape[1] + fix_idx.shape[0] < k:
            fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
        random_idx = np.random.choice(pc_np.shape[1], k - fix_idx.shape[0], replace=False)
        choice_idx = np.concatenate((fix_idx, random_idx), axis=0)

    return pc_np[:, choice_idx].T


def camera_matrix_cropping(K: np.ndarray, dx: float, dy: float):
    K_crop = np.copy(K)
    K_crop[0, 2] -= dx
    K_crop[1, 2] -= dy
    return K_crop


def downsample_pc(pointcloud, voxel_grid_downsample_size):
    import open3d
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))

    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
    pointcloud = down_pcd_points

    return pointcloud


def get_P_from_Rt(R, t):
    P = np.identity(4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t
    return P


def transform_pc_np(P, pc_np):
    """

    :param pc_np: 3xN
    :param P: 4x4
    :return:
    """
    pc_homo_np = np.concatenate((pc_np,
                                 np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
                                axis=0)
    P_pc_homo_np = np.dot(P, pc_homo_np)
    return P_pc_homo_np[0:3, :]


class nuScenesLoader(data.Dataset):
    def __init__(self, params, use_raw = False):
        super(nuScenesLoader, self).__init__()
        self.root = params['root_path']
        self.mode = params['mode']

        self.skip_night = True
        self.random_car = True
        skip = 1

        self.use_raw = use_raw
        self.norm = True

        self.crop_original_top_rows = 100
        # self.img_scale_H = 0.2
        # #self.img_scale_H = 0.32
        # self.img_scale_W = 0.4
        # self.img_H = 160 # (900-100)*0.2
        # self.img_W = 640 # (1600)*0.4 original 320

        self.img_scale_H = 0.8
        #self.img_scale_H = 0.32
        self.img_scale_W = 0.8
        self.img_H = 640 # (900-100)*0.2
        self.img_W = 1280 # (1600)*0.4 original 320

        # self.img_scale_H = 0.48
        # #self.img_scale_H = 0.32
        # self.img_scale_W = 0.8
        # self.img_H = 384
        # self.img_W = 1280
        #self.img_W = 512
        # change the Width similar to KITTI in testing

        max_r = 10.
        max_t = 2.

        if self.mode == "test":  # validation
            self.test_RT=[]
            test_RT_file = "/dataset/nus_processed_CMRNet/"+f"test_RT_{max_r:.2f}_{max_t:.2f}.csv"
            df_test_RT = pd.read_csv(test_RT_file, sep=',')
            for index, row in df_test_RT.iterrows():
                self.test_RT.append(list(row))
            #print("len test RT", len(self.test_RT))
            len_test = len(self.test_RT)
            # with open(self.root+"/nus_processed_CMRNet/"
            #           f"test_RT_{max_r:.2f}_{max_t:.2f}.csv", 'rb') as f:
            #     self.test_RT = pkl.load(f)
        if self.mode == "val":  # test
            self.test_RT=[]
            test_RT_file = "/dataset/nus_processed_CMRNet/"+f"test_RT_{max_r:.2f}_{max_t:.2f}.csv"
            df_test_RT = pd.read_csv(test_RT_file, sep=',')
            for index, row in df_test_RT.iterrows():
                self.test_RT.append(list(row))
            # with open("/dataset"+"/nus_processed_CMRNet/"
            #           f"test_RT_{max_r:.2f}_{max_t:.2f}.csv", 'rb') as f:
            #     self.test_RT = pkl.load(f)

        self.rx = max_r * np.pi / 180.
        self.ry = max_r * np.pi / 180.
        self.rz = max_r * np.pi / 180.

        self.tx = max_t
        self.ty = max_t
        self.tz = max_t

        self.sample_point = 8192

        info = "randominfo" if self.random_car else "info"

        # train_info_path = os.path.join("nuScenes_datasplit", f'train_dataset_{info}_proj_day.list')
        # val_info_path = os.path.join("nuScenes_datasplit", f'val_dataset_{info}_proj_day.list')
        # test_info_path = os.path.join("nuScenes_datasplit", f'test_dataset_{info}_proj_day.list')

        train_info_path = os.path.join('/dataset/nus_processed_CMRNet', f'train_dataset_map_small_short.list')
        val_info_path = os.path.join('/dataset/nus_processed_CMRNet', f'val_dataset_map_small_short.list')
        test_info_path = os.path.join('/dataset/nus_processed_CMRNet', f'test_dataset_map_small_short.list')

        if self.mode == "train": # 1834*8 = 14,672
            self.root = os.path.join(self.root, 'trainval')

            self.skip = skip
            train_dataset = []
            with open(train_info_path, 'rb') as f:
                # a list [((lidar["filename"], init_camera['filename']), K, Tr,night_tag)]
                train_dataset.extend(pkl.load(f))

            with open(val_info_path, 'rb') as f:
                train_dataset.extend(pkl.load(f))

            self.dataset = train_dataset
        elif self.mode == "val": # 126*8 = 1008
            self.root = os.path.join(self.root, 'trainval')
            self.skip = skip

            with open(val_info_path, 'rb') as f:
                val_dataset = pkl.load(f)

            self.dataset = val_dataset

        elif self.mode == "test":  # 25733
            self.root = os.path.join(self.root, 'test')
            self.skip = skip

            with open(test_info_path, 'rb') as f:
                test_dataset = pkl.load(f)
            self.dataset = test_dataset[:len_test]
            #print("test:", len(self.dataset))

        else:
            raise NotImplementedError

        self.length = int(np.ceil(len(self.dataset) / self.skip))

    def augment_img(self, img_np):
        """

        :param img: HxWx3, np.ndarray
        :return:
        """
        # color perturbation
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(np.uint8(img_np))))

        return img_color_aug_np

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
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random.astype(np.float32)

    def generate_transformation(self, rx, ry, rz, tx, ty, tz):
        rotation_mat = Rotation.from_euler('xzy', [rx, rz, ry]).as_matrix().reshape(3, 3)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = [tx, ty, tz]
        return P_random

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        lc_path, K, velo2cam2, _ = self.dataset[index * self.skip]
        
        cam_intrinsic = K.copy()

        lp, cp = lc_path
        # print(lp)
        # print(cp)
        #print(velo2cam2)
        # load pc
        with h5py.File(lp, 'r') as hf:
                pc = hf['PC'][:]
                reflectance = hf['intensity'][:]
        #pc = LidarPointCloud.from_file(os.path.join(self.root, lp))
        #print(pc.shape)
        #print(reflectance.shape)
        random_idx = np.random.permutation(pc.shape[1])
        pc_np = pc[0:3, random_idx]
        intensity_np = reflectance[:, random_idx]

        # remove point lying on the ego car
        # x_inside = np.logical_and(pc_np[0, :] < 0.8, pc_np[0, :] > -0.8)
        # y_inside = np.logical_and(pc_np[1, :] < 2.7, pc_np[1, :] > -2.7)
        # inside_mask = np.logical_and(x_inside, y_inside)
        # outside_mask = np.logical_not(inside_mask)

        # pc_np = pc_np[:, outside_mask]
        # intensity_np = intensity_np[:, outside_mask]

        # load image

        img = np.array(Image.open(os.path.join(self.root, cp)), np.uint8)  # RGB

        img = img[self.crop_original_top_rows:, :, :]
        K = camera_matrix_cropping(K, dx=0, dy=self.crop_original_top_rows)
        img = cv2.resize(img,
                         (int(round(img.shape[1] * self.img_scale_W)),
                          int(round((img.shape[0] * self.img_scale_H)))),
                         interpolation=cv2.INTER_LINEAR)
        K[0, 0] = self.img_scale_W * K[0, 0]  # width x
        K[0, 2] = self.img_scale_W * K[0, 2]  # width x
        K[1, 1] = self.img_scale_H * K[1, 1]  # height y
        K[1, 2] = self.img_scale_H * K[1, 2]  # height y

        if self.mode == "train":
            img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
        # crop image
        img = img[img_crop_dy:img_crop_dy + self.img_H,
              img_crop_dx:img_crop_dx + self.img_W, :]
        intrinsic = camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)

        if self.mode == 'train':
            img = self.augment_img(img)
            pc_np = self.jitter_point_cloud(pc_np)
            
        if self.mode == "train":
            Pr = self.generate_random_transform(self.tx, self.ty, self.tz,
                                                self.rx, self.ry, self.rz)
        else:
            initial_RT = self.test_RT[index * self.skip]
            rz = initial_RT[6]
            ry = initial_RT[5]
            rx = initial_RT[4]
            tx = initial_RT[1]
            ty = initial_RT[2]
            tz = initial_RT[3]
        
            Pr = self.generate_transformation(rx, ry, rz, tx, ty, tz)
        Pr_inv = np.linalg.inv(Pr)
        calib_extrinsic = Pr[:3, :]
        
        init_extrinsic = np.dot(Pr_inv, velo2cam2)[:3, :]

        decalib_quat_real, decalib_quat_dual = utils.extrinsic_to_dual_quat(calib_extrinsic)
        decalib_quat_dual = calib_extrinsic[:, 3]

        decalib_quat_real = torch.from_numpy(decalib_quat_real).type(torch.FloatTensor)
        decalib_quat_dual = torch.from_numpy(decalib_quat_dual).type(torch.FloatTensor)

        #cv2.imwrite('/data/debug/nus_cmr/raw.png', img)
        # print("----------------------")
        # print(np.max(pc_np[0]))
        # print(np.min(pc_np[0]))
        # print(np.max(pc_np[1]))
        # print(np.min(pc_np[1]))
        # print(np.max(pc_np[2]))
        # print(np.min(pc_np[2]))
        rgb_ini = img
        ### for debug
        if False:
            camera_K = K
           
            pc_test = np.concatenate([pc_np, np.ones((1, pc_np.shape[1]))], axis=0)
            local_map = velo2cam2 @ pc_test

            #local_map[2,:] = -local_map[2,:]
            #local_map = pc_np
            
            import matplotlib.pyplot as plt
            ## project pc
            pix_pc = np.transpose(np.dot(camera_K,local_map[0:3, :]))
            
            #print(pix_pc)
            pix_pc[:, :2] = np.divide(pix_pc[:, :2],  (pix_pc[:, 2])[:,None])
            z_ = pix_pc[:, 2]
            xy = pix_pc[:, :2]
            is_in_picture = (xy[:, 0] >= 0) & (xy[:, 0] <= (img.shape[1]  - 1)) & (xy[:, 1] >= 0) & (
                    xy[:, 1] <= (img.shape[0] - 1)) & (z_ > 0)
            z_ = z_[is_in_picture]
            #print("y max:", np.max(xy[:,1]))
            #print("z:",  z_)
            xy = xy[is_in_picture, :]

            plt.figure()
            plt.imshow(img)
            #plt.savefig(file_img_seq+str(ts)+"_img_left"+".jpg")
            try:
                pc_draw = (z_-np.min(z_)/(np.max(z_)-np.min(z_)))
            except:
                print("proj failed")
            else:
                print("proj success")
                cv2.imwrite('/data/debug/nus_cmr/raw.png', img)
                plt.scatter(xy[:,0], xy[:,1], c=pc_draw, cmap='jet', alpha=0.7, s=1)
                plt.savefig('/data/debug/nus_cmr/proj.png')
                #exit()

        if self.use_raw:
            #origin_pc_np = pc_np.copy().T.astype(np.float32)  # (N,3)
            ori_pc_np = pc_np
            origin_pc_np = pc_np.copy().T.astype(np.float32)  # (N,3)
            origin_pc_np[:, 0] = ori_pc_np[1, :]
            origin_pc_np[:, 1] = -ori_pc_np[0, :]
        else:
            origin_pc_np = pc_np.copy().T.astype(np.float32)  # (N,3)

        pc_np = np.concatenate([pc_np, np.ones((1, pc_np.shape[1]))], axis=0)
        pc_cam = (velo2cam2[:3, :] @ pc_np).numpy().T.astype(np.float32)
        pc_np = init_extrinsic @ pc_np
        #pc_cam = local_map
        N = pc_np.shape[1]
        #print(N)
        if N >= self.sample_point:
            choice_idx = np.random.choice(N, self.sample_point, replace=False)
            select_idx = choice_idx

            pc_np = pc_np[:, select_idx]
            intensity_np = intensity_np[:, select_idx]
            origin_pc_np = origin_pc_np[select_idx, :]
            pc_cam = pc_cam[select_idx, :]
            N = self.sample_point

        lidar_img = pc_np.T.astype(np.float32)

        # lidar_feats = np.concatenate([np.zeros_like(pc_np), intensity_np.astype(np.float32)], axis=0).T
        lidar_feats = lidar_img

        lidar_img = np.concatenate([lidar_img, np.zeros((self.sample_point - N, 3), dtype=np.float32)], axis=0 )

        # lidar_feats = np.concatenate([lidar_feats, np.zeros((self.sample_point - N, 4), dtype=np.float32)],
        #                              axis=0)
        lidar_feats = np.concatenate([lidar_feats, np.zeros((self.sample_point - N, 3), dtype=np.float32)],
                                     axis=0)

        origin_pc_np = np.concatenate(
            [origin_pc_np, np.zeros((self.sample_point - N, 3), dtype=np.float32)],
            axis=0)

        # img normalization
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        rgb_img = torch.from_numpy(img).type(torch.FloatTensor)
        if self.norm:
            rgb_img[:, :, 0] = (rgb_img[:, :, 0] / 255 - imagenet_mean[0]) / imagenet_std[0]
            rgb_img[:, :, 1] = (rgb_img[:, :, 1] / 255 - imagenet_mean[1]) / imagenet_std[1]
            rgb_img[:, :, 2] = (rgb_img[:, :, 2] / 255 - imagenet_mean[2]) / imagenet_std[2]
        rgb_img = rgb_img.permute(2, 0, 1)

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
        sample['resize_img'] = np.array([self.img_scale_H,self.img_scale_W])
        sample['index'] = index
        # sample['tji'] = float(np.linalg.norm(t_ji, 2))
        sample["path_info"] = "%d" % (index * self.skip)
        sample["lidar_feats"] = lidar_feats
        sample["raw_point_xyz"] = origin_pc_np
        sample["pc_cam"] = pc_cam

        return sample




if __name__ == '__main__':
    dataset_params = {
        'root_path': '/dataset/nuScenes',
        'mode': 'train',
        'd_rot': 10,  # not used
        'd_trans': 1.0,  # not used
        'fixed_decalib': False,
        'resize_w': 1216,
        'resize_h': 352,
    }
    import open3d as o3d
    import matplotlib.pyplot as plt
    save_path = "/data/debug/nus_cmr/"
    kitti = nuScenesLoader(params=dataset_params, use_raw = True)
    print(len(kitti))
    for i, data in enumerate(kitti, 0):
      
        sample = data
        #sample["raw_point_xyz"]
        lidar = sample['raw_point_xyz']
        print(lidar.shape)
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(lidar)
        o3d.io.write_point_cloud(save_path+"pc_raw_"+"test"+".pcd", o3d_pc)
        lidar_raw = sample['pc_cam']
        print(lidar.shape)
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(lidar)
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
        # lidar[:, 1] = lidar_raw[:, 0]
        # lidar[:, 0] = -lidar_raw[:, 1]
        # lidar[:, 2] = lidar_raw[:, 2]
        
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
        is_in_picture = (xy[:, 0] >= 0) & (xy[:, 0] <= (img.shape[1]  - 1)) & (xy[:, 1] >= 0) & (
                xy[:, 1] <= (img.shape[0] - 1)) & (z_ > 0)
        z_ = z_[is_in_picture]
        #print("y max:", np.max(xy[:,1]))
        #print("z:",  z_)
        xy = xy[is_in_picture, :]

        #plt.savefig(file_img_seq+str(ts)+"_img_left"+".jpg")
        pc_draw = (z_-np.min(z_)/(np.max(z_)-np.min(z_)))
        plt.scatter(xy[:,0], xy[:,1], c=pc_draw, cmap='jet', alpha=0.7, s=1)
        plt.savefig(save_path + "pc_proj" + str(i) +".jpg")

        plt.close()

        #break
        