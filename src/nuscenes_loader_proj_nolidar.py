import torch.utils.data as data
import random
import os
import os.path
import sys
sys.path.append("..")
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
import pickle
from pyquaternion import Quaternion
import pickle as pkl
import src.utils as utils
import math

from nuscenes.utils.data_classes import LidarPointCloud


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
    def __init__(self, params):
        super(nuScenesLoader, self).__init__()
        self.root = params['root_path']
        self.mode = params['mode']

        self.skip_night = True
        self.random_car = True
        skip = 1

        self.crop_original_top_rows = 100
        self.pc_range = None
        #self.pc_range = [[-35, -20, -5], [35, 20, 2.5]]
        print("pc range:", self.pc_range)

        self.using_cam_coord = False
        print("using cam coord:", self.using_cam_coord)
        self.img_scale_H = 0.2
        self.img_scale_W = 0.32
        #self.img_scale_W = 0.4
        self.img_H = 160 # (900-100)*0.2
        #self.img_W = 640 # (1600)*0.4 original 320
        self.img_W = 512
        print("SIZE changed")
        # change the Width similar to KITTI in testing

        self.tx = 10.
        self.ty = 0.
        self.tz = 10.

        self.rx = 0.
        self.ry = 2. * math.pi
        self.rz = 0.

        self.sample_point = 150000

        info = "randominfo" if self.random_car else "info"

        train_info_path = os.path.join("./nuScenes_datasplit", f'train_dataset_{info}_proj_day.list')
        val_info_path = os.path.join("./nuScenes_datasplit", f'val_dataset_{info}_proj_day.list')
        test_info_path = os.path.join("./nuScenes_datasplit", f'test_dataset_{info}_proj_day.list')

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

            self.dataset = test_dataset

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

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        lc_path, K, Tr, night_tag = self.dataset[index * self.skip]

        cam_intrinsic = K.copy()

        lp, cp = lc_path

        # load pc

        pc = LidarPointCloud.from_file(os.path.join(self.root, lp))
        random_idx = np.random.permutation(pc.points.shape[1])
        pc_np = pc.points[0:3, random_idx]
        intensity_np = pc.points[3:4, random_idx]
        #print(pc_np.shape)
        # remove point lying on the ego car
        x_inside = np.logical_and(pc_np[0, :] < 0.8, pc_np[0, :] > -0.8)
        y_inside = np.logical_and(pc_np[1, :] < 2.7, pc_np[1, :] > -2.7)
        inside_mask = np.logical_and(x_inside, y_inside)
        outside_mask = np.logical_not(inside_mask)

        pc_np = pc_np[:, outside_mask]
        intensity_np = intensity_np[:, outside_mask]
        #print(" ",pc_np.shape)
        #print(" ", np.max(pc_np[2,:]), np.min(pc_np[2,:]))

        ### testing the pc range
        y = -pc_np[0, :]
        x = pc_np[1, :]
        z = pc_np[2, :]
        #print("x min nax percent 5 95: ", np.min(x)," ", np.max(x), ",", np.percentile(x, (5, 95)))
        #print("y min nax percent 5 95: ", np.min(y)," ", np.max(y), ",", np.percentile(y, (5, 95)))
        #print("z min nax percent 5 95: ", np.min(z)," ", np.max(z), ",", np.percentile(z, (5, 95)))
        x_perc = np.percentile(x, (5, 95))
        y_perc = np.percentile(y, (5, 95))
        z_perc = np.percentile(z, (5, 95))

        if self.pc_range is not None:
            # range filter
            # print(pc_np.shape)
            y = -pc_np[0, :]
            x = pc_np[1, :]
            z = pc_np[2, :]
            range_mask = np.logical_and(abs(x)<self.pc_range[1][0], abs(y)<self.pc_range[1][1])
            range_mask = range_mask & np.logical_and(z<self.pc_range[1][2], z>self.pc_range[0][2])
            pc_np = pc_np[:, range_mask]
            intensity_np = intensity_np[:, range_mask]
    

        # degree filter
        x = pc_np[0, :]
        y = pc_np[1, :]
        z = pc_np[2, :]
        dist = np.sqrt(x*x+y*y)
        tan2 = 0.03492076949
        #tan24 = -0.4452286853
        tan24 = -0.4620648698
        ratio = z/dist
        angle_mask = np.logical_and(ratio < tan2, ratio > tan24)
        #print(np.mean(angle_mask))
        pc_np = pc_np[:, angle_mask]
        intensity_np = intensity_np[:, angle_mask]
        
        y = -pc_np[0, :]
        x = pc_np[1, :]
        z = pc_np[2, :]

        # load image

        img = np.array(Image.open(os.path.join(self.root, cp)), np.uint8)  # RGB
        rgb_ini = img
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
        rgb_proc = img
        if self.mode == 'train':
            img = self.augment_img(img)
            pc_np = self.jitter_point_cloud(pc_np)

        # if self.pc_range:
        #     # range filter
        #     # print(pc_np.shape)
        #     x = pc_np[0, :]
        #     z = pc_np[2, :]
        #     range_mask = np.logical_and(abs(x)<35, abs(z)<35)
        #     pc_np = pc_np[:, range_mask]
        #     intensity_np = intensity_np[:, range_mask]

        Pr = self.generate_random_transform(self.tx, self.ty, self.tz,
                                            self.rx, self.ry, self.rz)
        Pr_inv = np.linalg.inv(Pr)
        calib_extrinsic = Pr_inv[:3, :]

        init_extrinsic = np.dot(Pr, Tr)[:3]

        decalib_quat_real, decalib_quat_dual = utils.extrinsic_to_dual_quat(calib_extrinsic)
        decalib_quat_dual = calib_extrinsic[:, 3]
        
        decalib_quat_real = torch.from_numpy(decalib_quat_real).type(torch.FloatTensor)
        decalib_quat_dual = torch.from_numpy(decalib_quat_dual).type(torch.FloatTensor)

        if self.using_cam_coord:
            pc_cam = Tr[:3] @ np.concatenate([pc_np, np.ones((1, pc_np.shape[1]))], axis=0)
            pc_cam[1, :] = pc_cam[1, :] + 0.2
            origin_pc_np = pc_cam.T.astype(np.float32)
        else:
            ori_pc_np = pc_np
            origin_pc_np = pc_np.copy().T.astype(np.float32)  # (N,3)
            origin_pc_np[:, 0] = ori_pc_np[1, :]
            origin_pc_np[:, 1] = -ori_pc_np[0, :]

        pc_np = np.concatenate([pc_np, np.ones((1, pc_np.shape[1]))], axis=0)
        #pc_cam = Tr[:3] @ pc_np
        #pc_cam[1, :] = pc_cam[1, :] + 0.2
        #pc_cam = pc_cam.T.astype(np.float32)
        pc_np = init_extrinsic @ pc_np

        # height correct
        # pc_np[1, :] = pc_np[1, :] + 0.2
        # print("clip:",pc_np.shape)
        # print("clip and align: ", -np.min(pc_np[1,:]), -np.max(pc_np[1,:]))
        # # degree filter
        # x = pc_np[0, :]
        # y = pc_np[1, :]
        # z = pc_np[2, :]
        # dist = np.sqrt(x*x+z*z)
        # tan2 = 0.03492076949
        # tan24 = -0.4452286853
        # ratio = y/dist
        # angle_mask = np.logical_and(ratio > -tan2, ratio < -tan24)
        # pc_np = pc_np[:, angle_mask]
        # intensity_np = intensity_np[:, angle_mask]
        # origin_pc_np = origin_pc_np[angle_mask, :]
        # pc_cam = pc_cam[angle_mask, :]
        # pc_cam[1, :] = pc_cam[1, :] - 0.19

        lidar_img = pc_np.T.astype(np.float32)

        #lidar_feats = np.concatenate([np.zeros_like(pc_np), intensity_np.astype(np.float32)], axis=0).T
        lidar_feats = intensity_np.astype(np.float32).T
        N = lidar_img.shape[0]
        #print(N)
        lidar_img = np.concatenate([lidar_img, np.zeros((self.sample_point - N, 3), dtype=np.float32)], axis=0 )

        #lidar_feats = np.concatenate([lidar_feats, np.zeros((self.sample_point - N, 3), dtype=np.float32)],
       #                              axis=0)
        lidar_feats = np.concatenate([lidar_feats, np.zeros((self.sample_point - N, 1), dtype=np.float32)],
                                    axis=0)
        origin_pc_np = np.concatenate(
            [origin_pc_np, np.zeros((self.sample_point - N, 3), dtype=np.float32)],
            axis=0)

        # img normalization
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        rgb_img = torch.from_numpy(img).type(torch.FloatTensor)
        # rgb_img[:, :, 0] = (rgb_img[:, :, 0] / 255 - imagenet_mean[0]) / imagenet_std[0]
        # rgb_img[:, :, 1] = (rgb_img[:, :, 1] / 255 - imagenet_mean[1]) / imagenet_std[1]
        # rgb_img[:, :, 2] = (rgb_img[:, :, 2] / 255 - imagenet_mean[2]) / imagenet_std[2]
        rgb_img = rgb_img.permute(2, 0, 1)

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
        # sample['raw_lidar'] = pc_raw
        sample['resize_img'] = np.array([self.img_scale_H,self.img_scale_W])
        sample['index'] = index
        # sample['tji'] = float(np.linalg.norm(t_ji, 2))
        sample["path_info"] = "%d" % (index * self.skip)
        sample["lidar_feats"] = lidar_feats
        sample["raw_point_xyz"] = origin_pc_np
        sample["pc_stat"] = (x_perc, y_perc, z_perc)

        return sample

if __name__ == '__main__':
    dataset_params = {
        'root_path': '/dataset/nuScenes',
        'mode': 'train',
        'd_rot': 10,
        'd_trans': 1.0,
        'fixed_decalib': False,
        'resize_w': 1216,
        'resize_h': 352,
    }

    nus = nuScenesLoader(params=dataset_params)
    print("data num:")
    print(len(nus))
    exit(0)
    import open3d as o3d
    import matplotlib.pyplot as plt
    save_path =  '/data/debug2/nuscenes_clip/'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    x_avg = np.array([0., 0.])
    y_avg = np.array([0., 0.])
    z_avg = np.array([0., 0.])
    x_all = np.array([0., 0.])
    y_all = np.array([0., 0.])
    z_all = np.array([0., 0.])
    count = 0
    for i, data in enumerate(nus, 0):
        #print("hi")
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
        # if i % 503 == 0:
        #     sample = data
        #     print(sample['lidar'].shape)
        #     lidar = sample['raw_lidar']
        #     o3d_pc = o3d.geometry.PointCloud()
        #     o3d_pc.points = o3d.utility.Vector3dVector(lidar)
        #     o3d.io.write_point_cloud(save_path+"pc_"+str(i)+".pcd", o3d_pc)
            
        #     lidar = sample['raw_point_xyz']
        #     o3d_pc = o3d.geometry.PointCloud()
        #     o3d_pc.points = o3d.utility.Vector3dVector(lidar)
        #     o3d.io.write_point_cloud(save_path+"pc_raw_"+str(i)+".pcd", o3d_pc)

        #     img = sample['raw_rgb']#.astype(unit8)
        #     plt.figure()
        #     plt.imshow(img)
        #     plt.savefig(save_path+"image"+str(i)+".jpg")
            
        #     img = sample['proc_rgb']#.astype(unit8)
        #     plt.figure()
        #     plt.imshow(img)
        #     plt.savefig(save_path+"image_proc"+str(i)+".jpg")
        # else:
        #     #print(i)
        #     continue
    print("x percent 5 95: ", x_all/count)
    print("y percent 5 95: ", y_all/count)
    print("z percent 5 95: ", z_all/count)