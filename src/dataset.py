import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import cv2
import sys
#sys.path.append('/data/regnet_batch_4_gyf_5/')
if __name__ == '__main__':
    import calib as calib
    import utils as utils
    import dataset_params as dp
else:
    import src.calib as calib
    import src.utils as utils
    import src.dataset_params as dp

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


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3] 24*1024*3
        npoint: number of samples 512
    Return:
        centroids: sampled pointcloud index, [B, npoint] 24*512
    """
    B, N, C = xyz.shape
    # print("##################new######")
    # print("xyz.shape", end=': ')
    # print(xyz.shape)  # torch.Size([24, 1024, 3])
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    # print("centroids.shape", end=': ')
    # print(centroids.shape)  # torch.Size([24, 512])
    distance = torch.ones(B, N) * 1e10
    # print("distance.shape", end=': ')  # torch.Size([24, 1024])
    # print(distance.shape)
    farthest = torch.randint(0, N, (B,), dtype=torch.long)
    # print("farthest.shape", end=': ')  # torch.Size([24])
    # print(farthest.shape)
    # print(farthest)
    batch_indices = torch.arange(B, dtype=torch.long)
    # print("batch_indices.shape", end=': ')
    # print(batch_indices.shape)  # torch.Size([24])
    # print(batch_indices)
    for i in range(npoint):
        centroids[:, i] = farthest
        # temp = xyz[batch_indices, farthest, :]
        # print("temp", end=': ')
        # print(temp.shape)  # torch.Size([24, 3])
        # print(temp)
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
        # print("centroid", end=': ')
        # print(centroid.shape)  # torch.Size([24, 1, 3])
        # print(centroid)
        temp_sum = (xyz - centroid) ** 2
        temp_sum = torch.from_numpy(temp_sum)
        dist = torch.sum(temp_sum, -1)
        # print("dist", end=': ')
        # print(dist.shape)  # torch.Size([24, 1024])
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        # print("farthest", end=': ')
        # print(farthest.shape)  # torch.Size([24])
        # print(farthest)

    # print('centroids.shape', end=': ')
    # print(centroids.shape)  # centroids.shape: torch.Size([24, 512])
    # print("##################")
    return centroids


def random_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3] 24*1024*3
        npoint: number of samples 512
    Return:
        centroids: sampled pointcloud index, [B, npoint] 24*512
    """
    B, N, C = xyz.shape
    # print("##################new######")
    # print("xyz.shape", xyz.shape)
    # torch.Size([24, 1024, 3])
    randmatrix = torch.randint(0, N, (B, npoint), dtype=torch.long)
    # print('centroids.shape', end=': ')
    # print(centroids.shape)  # centroids.shape: torch.Size([24, 512])
    # print("##################")
    return randmatrix

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
    # fps_idx = random_point_sample(pcl, npoint)  # [B, npoint, C]
    # print("fps_idx.shape", fps_idx.shape)
    fps_idx = random_point_sample_nopeat(pcl,npoint)
    new_xyz = index_points(pcl, fps_idx)
    return new_xyz


def flit_points(pcl_uv, pcl_z):
    mask = (pcl_uv[:, 0] > 0)
    return pcl_uv[mask], pcl_z[mask]


class Kitti_Dataset(Dataset):

    def __init__(self, params):

        base_path = params['base_path']
        date = params['date']
        drives = params['drives']

        self.d_rot = params['d_rot']
        self.d_trans = params['d_trans']
        self.resize_h = params['resize_h']
        self.resize_w = params['resize_w']
        self.fixed_decalib = params['fixed_decalib']

        self.img_path = []
        self.lidar_path = []
        for drive in drives:
            cur_img_path = Path(
                base_path)/date/(date + '_drive_{:04d}_sync'.format(drive))/'image_02'/'data'
            cur_lidar_path = Path(
                base_path)/date/(date + '_drive_{:04d}_sync'.format(drive))/'velodyne_points'/'data'
            for file_name in sorted(cur_img_path.glob('*.png')):
                self.img_path.append(str(file_name))
            for file_name in sorted(cur_lidar_path.glob('*.bin')):
                self.lidar_path.append(str(file_name))

        CAM02_PARAMS, VELO_PARAMS = calib.get_calib(date)
        self.cam_intrinsic = utils.get_intrinsic(
            CAM02_PARAMS['fx'], CAM02_PARAMS['fy'], CAM02_PARAMS['cx'], CAM02_PARAMS['cy'])
        self.velo_extrinsic = utils.get_extrinsic(
            VELO_PARAMS['rot'], VELO_PARAMS['trans'])

    def load_image(self, index):
        return cv2.imread(self.img_path[index])[:, :, ::-1] # BGR to RGB

    def load_lidar(self, index):
        return np.fromfile(self.lidar_path[index], dtype=np.float32).reshape(-1, 4)

    def get_projected_pts(self, index, extrinsic, img_shape):
        pcl = self.load_lidar(index)
        pcl_uv, pcl_z = utils.get_2D_lidar_projection(
            pcl, self.cam_intrinsic, extrinsic)
        # print("pcl_uv before mask", pcl_uv)
        mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & \
               (pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)

        return pcl_uv[mask], pcl_z[mask]

    def sample_projected_pts(self, index, extrinsic, img_shape, resize_img, resize_w, resize_h, intrinsic, npoint=8192):
        pcl = self.load_lidar(index)
        
        pcl_xyz = pcl[:, 0:3]        
        
        is_ground = np.logical_or(pcl_xyz[:, 2] < -15, pcl_xyz[:, 2] < -15)
        not_ground = np.logical_not(is_ground)

        near_mask_x = np.logical_and(pcl_xyz[:, 0] < 1200, pcl_xyz[:, 0] > 2)
        near_mask_z = np.logical_and(pcl_xyz[:, 1] < 1200, pcl_xyz[:, 1] > -1200)

        near_mask = np.logical_and(near_mask_x, near_mask_z)
        near_mask = np.logical_and(not_ground, near_mask)
        indices_1 = np.where(near_mask)[0]

        pcl_xyz = pcl_xyz[indices_1, :]
        # pcl_jiequ = pcl_xyz
        # np.savetxt('jiequ.txt', pcl_jiequ)
        pcl_xyz = np.hstack((pcl_xyz[:, 0:3], np.ones((pcl_xyz.shape[0], 1)))).T
        pcl_xyz_tmp = extrinsic @ pcl_xyz

        pcl_xyz_tmp = pcl_xyz_tmp.T
        pcl_xyz_tmp = pcl_xyz_tmp.reshape(1, -1, 3)

        # pcl_xyz_tmp = pcl_xyz.T
        # pcl_xyz = pcl_xyz_tmp.reshape(1, -1, 3)
        pcl_xyz_sample = sample_n_points(pcl_xyz_tmp, npoint=npoint)
        return pcl_xyz_sample

    def get_depth_image(self, index, extrinsic, img_shape):
        pcl_uv, pcl_z = self.get_projected_pts(index, extrinsic, img_shape)
        flag1 = pcl_uv[:, 0].astype(np.uint32)
        flag2 = pcl_uv[:, 1].astype(np.uint32)
        near = np.zeros((img_shape[1], img_shape[0])).astype(np.uint32)
        dis = np.zeros((img_shape[1], img_shape[0])).astype(np.uint32)
        dis = dis - 1 # -1 means invalid
        for i in range(0, pcl_uv.shape[0]):
            # round operation TODO
            if ((pcl_uv[i, 0] - flag1[i]) >= 0.5) & (pcl_uv[i, 0] < (img_shape[1] - 1)):
                x1 = flag1[i] + 1
            else:
                x1 = flag1[i]
            if ((pcl_uv[i, 1] - flag2[i]) >= 0.5) & (pcl_uv[i, 1] < (img_shape[0] - 1)):
                y1 = flag2[i] + 1
            else:
                y1 = flag2[i]
            distance = pcl_z[i]
            if dis[x1, y1] == -1:
                dis[x1, y1] = distance
                near[x1, y1] = i
                pcl_uv[i, 0] = x1
                pcl_uv[i, 1] = y1
            else:
                if distance <= dis[x1, y1]: # perform occlusion
                    dis[x1, y1] = distance
                    t = near[x1, y1]
                    pcl_uv[t, 0] = -1
                    near[x1, y1] = i
                    pcl_uv[i, 0] = x1
                    pcl_uv[i, 1] = y1
                else:
                    pcl_uv[i, 0] = -1

        pcl_uv, pcl_z = flit_points(pcl_uv, pcl_z)
        pcl_uv = pcl_uv.astype(np.uint32)
        pcl_z = pcl_z.reshape(-1, 1)
        depth_img = np.zeros((img_shape[0], img_shape[1], 1))
        depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
        return depth_img

    def __len__(self):
        return len(self.img_path)

    def get_decalibration(self):
        def get_rand():
            # [-1,1]
            return np.random.rand()*2-1

        if self.fixed_decalib:
            d_roll = utils.degree_to_rad(self.d_rot)
            d_pitch = utils.degree_to_rad(self.d_rot)
            d_yaw = utils.degree_to_rad(self.d_rot)
            d_x = self.d_trans
            d_y = self.d_trans
            d_z = self.d_trans
        else:
            d_roll = get_rand()*utils.degree_to_rad(self.d_rot)
            d_pitch = get_rand()*utils.degree_to_rad(self.d_rot)
            d_yaw = get_rand()*utils.degree_to_rad(self.d_rot)
            d_x = get_rand()*self.d_trans
            d_y = get_rand()*self.d_trans
            d_z = get_rand()*self.d_trans
            
        decalib_val = dict(
            d_rot_angle=[d_roll, d_pitch, d_yaw],
            d_trans=[d_x, d_y, d_z],
        )
        decalib_rot = utils.euler_to_rotmat(d_roll, d_pitch, d_yaw)
        decalib_trans = np.asarray([d_x, d_y, d_z]).reshape(3, 1)
        decalib_extrinsic = utils.get_extrinsic(decalib_rot, decalib_trans)

        return decalib_extrinsic, decalib_val

    def __getitem__(self, index):
        rgb_img = self.load_image(index)
        rgb_img = np.ascontiguousarray(rgb_img)
        # cv2.imwrite('rgb.png', rgb_img)
        # print('rgb done!!!')

        cloud_path = self.lidar_path[index]

        resize_rgb = rgb_img

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        # velo_extrinsic: GT matrix
        # decalib_extrinsic: error matrix
        # H_initial: decalib_extrinsic*velo_extrinsic
        # target: calib_extrinsic = inverse(decalib_extrinsic)
        decalib_extrinsic, _ = self.get_decalibration()
        calib_extrinsic = np.linalg.pinv(np.vstack((decalib_extrinsic, [0, 0, 0, 1])))[:3, :]
        decalib_quat_real, decalib_quat_dual = utils.extrinsic_to_dual_quat(calib_extrinsic)

        decalib_quat_dual = calib_extrinsic[:, 3] # dual quat=>tvec
        # print(self.velo_extrinsic.shape)
        init_extrinsic = utils.mult_extrinsic(decalib_extrinsic, self.velo_extrinsic)

        # I = Hgt * P
        # I, H_initial * P = H_error * H_gt * P
        # I = fai * H_initial * P
        # fai = inverse (H_error)
        # 
        # H_initial=fai^-1 @ H_gt
        # output = fai
        # output @ H_initial vs Hgt
        
        
        # H_gt = x^-1 * H_initial
        # x=fai^-1 decalib_extrinsic
        # formula: Hgt= fai * H_initial

        # TODO: depth_img is not used
        # depth_img = self.get_depth_image(index, init_extrinsic, rgb_img.shape)
        # depth_img = utils.mean_normalize_pts(depth_img).astype('float32')
        #
        # # adjust intrinsic
        # h, w, _ = depth_img.shape  # 375,1242

        h,w,_ = rgb_img.shape

        resize_img = np.array([self.resize_w/w, self.resize_h/h])
        intrinsic = self.cam_intrinsic.copy()
        intrinsic[0, 0] = resize_img[0]*intrinsic[0, 0]  # width x
        intrinsic[0, 2] = resize_img[0]*intrinsic[0, 2]  # width x
        intrinsic[1, 1] = resize_img[1]*intrinsic[1, 1]  # height y
        intrinsic[1, 2] = resize_img[1]*intrinsic[1, 2]  # height y

        # real_lidar_img = self.load_lidar(index)[:, 0:3]
        # np.savetxt('cloud_original.txt', real_lidar_img)
        # print('lidar done!!!')

        lidar_img = self.sample_projected_pts(index, init_extrinsic, rgb_img.shape, resize_img, self.resize_w,
                                              self.resize_h, intrinsic, npoint=8192)
        # print("lidar_img.shape", lidar_img.shape)
        lidar_img = lidar_img.reshape(-1, 3)

        rgb_img = cv2.resize(rgb_img, (self.resize_w, self.resize_h))
        # resize_rgb = rgb_img
        # depth_img = cv2.resize(depth_img, (self.resize_w, self.resize_h))
        # depth_img = depth_img[:, :, np.newaxis]

        decalib_quat_real = torch.from_numpy(decalib_quat_real).type(torch.FloatTensor)
        decalib_quat_dual = torch.from_numpy(decalib_quat_dual).type(torch.FloatTensor)
        rgb_img = torch.from_numpy(rgb_img).type(torch.FloatTensor)
        rgb_img[:, :, 0] = (rgb_img[:, :, 0]/255 - imagenet_mean[0]) / imagenet_std[0]
        rgb_img[:, :, 1] = (rgb_img[:, :, 1]/255 - imagenet_mean[1]) / imagenet_std[1]
        rgb_img[:, :, 2] = (rgb_img[:, :, 2]/255 - imagenet_mean[2]) / imagenet_std[2]
        rgb_img = rgb_img.permute(2, 0, 1)
        # depth_img = torch.from_numpy(depth_img).permute(2, 0, 1).type(torch.FloatTensor)

        sample = {}
        sample['resize_img'] = resize_img                     # unused in model forwarding
        sample['rgb'] = rgb_img
        sample['resize_rgb'] = resize_rgb                     # unused 
        # print('rgb:', rgb_img)
        # print('rgb.shape:', sample['rgb'].shape)
        sample['decalib_real_gt'] = decalib_quat_real         # 四元数
        sample['decalib_dual_gt'] = decalib_quat_dual         # 平移向量
        sample['init_extrinsic'] = init_extrinsic             # 传感器间变换
        sample['real_extrinsic'] = self.velo_extrinsic        # not used
        sample['init_intrinsic'] = intrinsic                  # 相机内参, 由于resize需做变换
        sample['index'] = index                               # 文件编号
        # sample['depth'] = depth_img
        sample['lidar'] = lidar_img                           # pc data

        sample["path_info"] = cloud_path

        # sample['real_lidar'] = real_lidar_img
        # print('real_lidar.shape:', sample['real_lidar'].shape)
        # print("hello")
        # print('lidar.shape:', sample['lidar'].shape)
        return sample

if __name__ == "__main__":
    dataset_params = {
    'base_path': dp.TRAIN_SET_2011_09_26['base_path'],
    'date': dp.TRAIN_SET_2011_09_26['date'],
    'drives': dp.TRAIN_SET_2011_09_26['drives'],
    'd_rot': 10,
    'd_trans': 1.0,
    'fixed_decalib': False,
    'resize_w': 1216,
    'resize_h': 352,
    }

    kitti = Kitti_Dataset(params=dataset_params)
    print(len(kitti))

    for i, data in enumerate(kitti):
        sample = kitti