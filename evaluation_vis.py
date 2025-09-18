import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import numpy as np
from datetime import datetime
import pickle as pkl
from importlib import import_module
from src.calib2 import CALIB
from nuscenes.utils.data_classes import LidarPointCloud
# arg parser
import src.visualize as vis
from src import utils
import struct
import open3d
def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z
def load_vel_hits(filename):

    f_bin = open(filename, "rb")

    hits = []

    while True:

        x_str = f_bin.read(2)
        if x_str == b'': # eof
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert(x, y, z)

        # Load in homogenous
        hits += [[x, y, z, i]]

    f_bin.close()
    hits = np.asarray(hits)
    # hits[:,2] = -hits[:,2]

    #print("height median:", np.median(hits[:,2]))

    return hits.transpose()


parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', required=True, help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dataset', type=str, default="kitti", choices=["kitti", "kd", "kitti_rgg", "kd_cmr",
                                                                     "nus", "nus_ori","realloc"
                             ],
                    help="choose which dataset to train [default: kitti]")
parser.add_argument('--vis_num', type=int, default=10, help="which num to vis")
parser.add_argument('--rot_test', type=float, default=10., help="when dataset is kitti, choose the fixed decalib")
parser.add_argument('--out', action="store_true")
parser.add_argument('--vis_target', type=str, default=None)
parser.add_argument('--coarse', action="store_true")
parser.add_argument('--visibility', action="store_true")
FLAGS = parser.parse_args()

LOGDIR = FLAGS.log_dir
DATASET = FLAGS.dataset
VISNUM = FLAGS.vis_num
ROT = FLAGS.rot_test
OUT = FLAGS.out
TAG = FLAGS.vis_target
COARSE = FLAGS.coarse
VISB = FLAGS.visibility

try:
    rgg_calib = CALIB()
except:
    print("try rgg but failed")

if DATASET == "kitti":
    from src.dataset import Kitti_Dataset as testdataset
    from src.dataset_params import KITTI_ONLINE_CALIB as cfg
if DATASET == "kitti_rgg":
    from src.dataset import Kitti_Dataset as testdataset
    from src.dataset_params import KITTI_ONLINE_CALIB as cfg
elif DATASET == "kd" or DATASET == "kd_cmr":
    from src.kitti_odometry_corr_snr import Kitti_Odometry_Dataset as testdataset
    from src.kitti_odometry import read_calib
    from src.dataset_params import KITTI_ODOMETRY as cfg
elif "nus" in DATASET:
    if "corr" in DATASET:
        dataset_file = "nuscenes_loader_processed"
    else:
        dataset_file = "nuscenes_loader"
    DA = import_module("src.{0}".format(dataset_file))
    testdataset = DA.nuScenesLoader
    from src.dataset_params import NUSCENES as cfg
elif "realloc" in DATASET:
    dataset_file = "real_dataset"
    DA = import_module("src.{0}".format(dataset_file))
    testdataset = DA.Real_Dataset
    from src.dataset_params import REAL_DATA as cfg

class Evaluator(object):
    def __init__(self):

        save_path = Path(LOGDIR) / "info_test"
        # logs
        save_path.mkdir(parents=True, exist_ok=True)

        save_path_tensorboard = save_path / "tensorboard"
        save_path_tensorboard.mkdir(parents=True, exist_ok=True)

        now_time = datetime.now()
        ts_info = now_time.strftime('%Y_%m_%d_') + '_'.join(now_time.strftime("%X").split(':'))

        pred_path = f"prediction_{int(ROT)}.txt" if DATASET == "kitti" else "prediction.txt"
        with open(str(save_path / pred_path), "r") as f:
            self.lines = f.readlines()

        writer_info = f"vis_{int(ROT)}_" + ts_info if DATASET == "kitti" else "vis_" + ts_info
        # self.writer = SummaryWriter(log_dir=str(save_path_tensorboard), filename_suffix=writer_info)
        self.save_dir = os.path.join(save_path, "vis", writer_info)
        os.makedirs(os.path.join(save_path, "vis"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "vis", writer_info), exist_ok=True)
        # validation data
        params = cfg.dataset_params_test
        if DATASET == "kitti" or DATASET == "kitti_rgg":
            params["d_rot"] = 10
            params["d_trans"] = 0.1 * 10
        dataset_test = testdataset(params)
        self.params = params
        self.dataset = dataset_test

        self.save_path = save_path

        if OUT:
            with open(str(save_path / "outlier.pkl"), 'rb') as f:
                self.outlier = pkl.load(f)
        count = -1
        sections, last = self.calculate_sections(self.lines)

        tag = TAG
        if TAG is None:
            tag = last

        self.num, self.start, self.pose_t = sections[tag]

        if "cmr" in DATASET:
            seq = 0
            with open(os.path.join("/dataset", 'kitti_processed_CMRNet', 'poses', f'kitti-{seq:02d}.csv')) as f:
                lines = f.readlines()[1:]
                self.poses = [line.strip('\n').split(',')[1:] for line in lines]  # timestamp,x,y,z,qx,qy,qz,qw
            map_path = os.path.join("/dataset", 'kitti_processed_CMRNet',
                                    'sequences', '00', 'map',
                                    f'map-{seq:02d}_0.1_0-{len(self.poses)}.pcd')

            print("Load Map...")
            self.map = np.asarray(open3d.io.read_point_cloud(map_path).points).T  # [3,N]
            print("Load Done...")

        if "ori" in DATASET:
            with open(os.path.join("nuScenes_datasplit", f'test_dataset_randominfo_proj_day.list'), 'rb') as nusf:
                self.nus_dataset = pkl.load(nusf)
            self.nusroot = os.path.join('/dataset/nuScenes', 'test')

    def decode_meta(self, meta_info):
        if "kitti" == DATASET:
            scan_id, drive_code = meta_info.strip('\n').split(' ')
            base_path = self.params['base_path']
            date = self.params["date"]
            imp = os.path.join(base_path, date,
                               date + '_drive_{:s}_sync'.format(drive_code), 'image_02', 'data', scan_id + ".png")
            pcp = os.path.join(base_path, date,
                               date + '_drive_{:s}_sync'.format(drive_code), 'velodyne_points', 'data',
                               scan_id + ".bin")

            pcl = np.fromfile(pcp, dtype=np.float32).reshape(-1, 4)[:, :3]
            img = cv2.imread(imp)
            K = self.dataset.cam_intrinsic.copy()
        elif "rgg" in DATASET:

            testset, ind = meta_info.strip('\n').split(' ')
            ind = int(ind)
            with open(f"rgg_datas/rgg_data_{testset}.pkl", 'rb') as f:
                test = pkl.load(f)
            lidar_path = test["lidar"]
            img_path = test["img"]
            img = cv2.imread(img_path[ind])
            pcl = np.fromfile(lidar_path[ind], dtype=np.float32).reshape(-1, 4)[:, :3]
            if testset in ["T1", "T2a", "T2b"]:
                K = rgg_calib.K_0926
            else:
                K = rgg_calib.K_1003
        elif "cmr" in DATASET:
            seq, seq_i, seq_j = meta_info.strip('\n').split(' ')
            pose = np.array(self.poses[int(seq_i)], np.float32)
            R = quat2mat(pose[[6, 3, 4, 5]]).T
            local_pc = R @ self.map + (-R @ pose[:3, None])  # 3,N

            indexes = local_pc[1] > -25.
            indexes = indexes & (local_pc[1] < 25.)
            indexes = indexes & (local_pc[0] > -10.)
            indexes = indexes & (local_pc[0] < 100.)

            pcl = local_pc[:, indexes].T  # without visibility filter

            # pcp = os.path.join(self.params['root_path'], 'data_odometry_velodyne_deepi2p_new',
            #                    'data_odometry_velodyne_NWU',
            #                    'sequences', seq, 'voxel0.1', seq_i + '.npy')

            # pcp = os.path.join(self.params['root_path'], 'data_odometry_velodyne', "dataset",
            #                    'sequences', seq, 'velodyne', seq_i + '.bin')
            imp = os.path.join(self.params['root_path'], 'kitti_processed_DeepI2P', 'data_odometry_color_npy',
                               'sequences',
                               seq, 'image_2', seq_j + ".npy")
            # pcl = np.fromfile(pcp, dtype=np.float32).reshape(-1, 4)[:, :3]
            img = cv2.cvtColor(np.load(imp), cv2.COLOR_RGB2BGR)
            _, K, _ = read_calib(
                os.path.join(self.params['root_path'], 'data_odometry_calib', 'dataset', 'sequences', seq,
                             'calib.txt'))

        elif "kd" in DATASET:
            seq, seq_i, seq_j = meta_info.strip('\n').split(' ')
            # pcp = os.path.join(self.params['root_path'], 'data_odometry_velodyne_deepi2p_new',
            #                    'data_odometry_velodyne_NWU',
            #                    'sequences', seq, 'voxel0.1', seq_i + '.npy')
            pcp = os.path.join(self.params['root_path'], 'data_odometry_velodyne', "dataset",
                               'sequences', seq, 'velodyne', seq_i + '.bin')
            imp = os.path.join(self.params['root_path'], 'kitti_processed_DeepI2P', 'data_odometry_color_npy',
                               'sequences',
                               seq, 'image_2', seq_j + ".npy")
            pcl = np.fromfile(pcp, dtype=np.float32).reshape(-1, 4)[:, :3]
            img = cv2.cvtColor(np.load(imp), cv2.COLOR_RGB2BGR)
            _, K, _ = read_calib(
                os.path.join(self.params['root_path'], 'data_odometry_calib', 'dataset', 'sequences', seq,
                             'calib.txt'))
        elif "nus" in DATASET:
            if "ori" in DATASET:
                ind = int(''.join(meta_info.strip('\n').split(' ')))
                lc_path, K, Tr, night_tag = self.nus_dataset[ind]
                lp, cp = lc_path
                K = np.array(K, np.float32)
                pc = np.asarray(LidarPointCloud.from_file(os.path.join(self.nusroot, lp)).points)
                x_inside = np.logical_and(pc[0, :] < 0.8, pc[0, :] > -0.8)
                y_inside = np.logical_and(pc[1, :] < 2.7, pc[1, :] > -2.7)
                inside_mask = np.logical_and(x_inside, y_inside)
                outside_mask = np.logical_not(inside_mask)
                pcl = pc[:3, outside_mask].T.astype(np.float32)
                img = cv2.imread(os.path.join(self.nusroot, cp))

                img = cv2.resize(img, (800, 450))
                K[0, 0] = K[0, 0] * 0.5
                K[0, 2] = K[0, 2] * 0.5
                K[1, 1] = K[1, 1] * 0.5
                K[1, 2] = K[1, 2] * 0.5

            else:
                dir = os.path.join(self.params['root_path'], "nuScenes2", "test")
                ind = int(''.join(meta_info.strip('\n').split(' ')))

                npy_data = np.load(os.path.join(dir, "PC", "%06d.npy" % ind))
                pcl = npy_data[:3, :].T.astype(np.float32)
                img = cv2.cvtColor(np.load(os.path.join(dir, "img", "%06d.npy" % ind)), cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, dsize=(int(round(img.shape[1] * 5)),
                                             int(round(img.shape[0] * 5))))
                K = np.load(os.path.join(dir, "K", "%06d.npy" % ind)).astype(np.float32)
                K[0, 0] = K[0, 0] * 5
                K[0, 2] = K[0, 2] * 5
                K[1, 1] = K[1, 1] * 5
                K[1, 2] = K[1, 2] * 5
        elif "realloc" in DATASET:
            seq, ts, _ = meta_info.strip('\n').split(' ')

            pcl = np.fromfile(os.path.join(self.params["root_path"], "sequences",
                                           seq, "velodyne",ts+".bin"),np.float32).reshape(-1,4)[:,:3]
            img = cv2.imread(os.path.join(self.params["root_path"], "sequences",
                                           seq, "images",ts+".jpg"))
            K = np.array([[1265.1835, 0, 650.6398],
                          [0, 1265.3955, 536.6536],
                          [0, 0, 1.]], np.float32)
        elif "m2dgr" in DATASET:
            print(meta_info.strip('\n').split(' '))
            seq, ts, _ = meta_info.strip('\n').split(' ')
            pc_path = os.path.join(self.params['root_path'], seq, "velodyne", ts + ".bin")
            pcl =np.fromfile(pc_path, np.float32).reshape(-1, 4)[:, :3]
            img_path = os.path.join(self.params['root_path'], seq, "images", ts + ".jpg")
            img = cv2.imread(img_path)

            D = np.array([0.148000794688248, -0.217835187249065, 0, 0])
            K = np.array([617.971050917033, 0, 0,
                  0, 616.445131524790, 0,
                  327.710279392468, 253.976983707814, 1], np.float32).reshape(3, 3).T
            
            img = cv2.undistort(img, K, D)
            #img = cv2.cvtColor(cv2.undistort(img, K, D), cv2.COLOR_BGR2RGB)
            H, W , _= img.shape 
            offset_y = 6
            offset_x = 23
            img = img[offset_y:473,
                    offset_x:631, :]
            K[0, 2] -= offset_x
            K[1, 2] -= offset_y
            h, w , _= img.shape
            img = cv2.resize(img,
                             (int(round(W)),
                              int(round(H))),
                             interpolation=cv2.INTER_LINEAR)
            K[0, 0] = W / w * K[0, 0]  # width x
            K[0, 2] = W / w * K[0, 2]  # width x
            K[1, 1] = H / h * K[1, 1]  # height y
            K[1, 2] = H / h * K[1, 2]  # height y

        elif "seasons4" in DATASET:
            print(meta_info.strip('\n').split(' '))
            seq, ts, _ = meta_info.strip('\n').split(' ')
            pc_path = os.path.join(self.params['root_path'], seq, "stereo_pc", ts + ".bin")
            pcl =np.fromfile(pc_path, np.float64).reshape(-1, 4)[:, :3].astype(np.float32)
            #img_path = os.path.join(self.params['root_path'], seq, "images", ts + ".jpg")
            img_path = os.path.join(self.params['root_path'],seq, "distorted_images", "cam0", ts + ".png")
            img = cv2.imread(img_path)

            K = np.array([501.4757919305817, 0, 0,
                    0, 501.4757919305817, 0,
                    421.7953735163109, 167.65799492501083, 1], np.float32).reshape(3, 3).T
            
            H, W , _= img.shape 
            #offset_y = 6
            #offset_y = 320
            img = img[:320, :, :]
            
        elif "nclt" in DATASET:
            print(meta_info.strip('\n').split(' '))
            seq, ts, _ = meta_info.strip('\n').split(' ')
            pc_path = os.path.join(self.params['root_path'], seq, "velodyne_sync", ts + ".bin")
            pcl = load_vel_hits(pc_path).astype(np.float32).T

            #img_path = os.path.join(self.params['root_path'], seq, "images", ts + ".jpg")
            img_path = os.path.join(self.params['root_path'],seq, "undistorted_img/Cam5", ts + ".jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            K = self.dataset.K.copy()
    
            
            H, W , _= img.shape 
            #offset_y = 6
            #offset_y = 320
            #img = img[:320, :, :]

        else:
            pcl, img, K = None, None, None
        return pcl, img, K

    def get_projected_pts(self, pcl, intrinsic, extrinsic, img_shape):
        pcl_uv, pcl_z = utils.get_2D_lidar_projection(pcl, intrinsic, extrinsic)
        #print(pcl_uv)
        mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & \
               (pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)

        return pcl_uv[mask], pcl_z[mask], mask

    def vis(self):
        # step = 100
        if abs(self.start) == len(self.lines) and "section" not in self.lines[self.start]:
            return
        # num = abs(self.start + 1) // 4
        num = self.num
    
        vis_target = np.random.permutation(num)[:VISNUM]
        # vis_target = [2108,982,1898]

        if OUT:
            out_nums = np.array([len(outs) for outs in self.outlier])
            vis_target = np.argsort(out_nums)[-VISNUM:]
        self.start += 1
        init_start = self.start
        for vis_t in tqdm(vis_target, total=len(vis_target)):
            self.start = init_start + self.pose_t * vis_t
            #print("start")
            #breakpoint()
            pcl, img, intrinsic = self.decode_meta(self.lines[self.start])

            init_extrinsic = np.array(self.lines[self.start + 1].strip('\n').split(' '), np.float32).reshape(3, 4)
            if self.pose_t == 5:
                pred_extrinsic_coarse = np.array(self.lines[self.start + 2].
                                                 strip('\n').split(' '), np.float32).reshape(3, 4)
            pred_extrinsic = np.array(self.lines[self.start + self.pose_t - 2].strip('\n').split(' '),
                                      np.float32).reshape(3, 4)
            gt_extrinsic = np.array(self.lines[self.start + self.pose_t - 1].strip('\n').split(' '),
                                    np.float32).reshape(3, 4)

            # import matplotlib.pyplot as plt
            # pc_np = pcl[0:3, :]
            # pc_np = np.concatenate([pc_np, np.ones((1, pc_np.shape[1]))], axis=0)
            # cam_pc = (gt_extrinsic @ pc_np)[:3, :].astype(np.float32)
            # lidar = cam_pc.T
            # pix_pc = np.transpose(np.dot(intrinsic, np.transpose(lidar)))
            # #print(pix_pc)
            # #print(pix_pc)
            # pix_pc[:, :2] = np.divide(pix_pc[:, :2],  (pix_pc[:, 2])[:,None])
            # z_ = pix_pc[:, 2]
            # xy = pix_pc[:, :2]
            # is_in_picture = (xy[:, 0] >= 0) & (xy[:, 0] <= (640  - 1)) & (xy[:, 1] >= 0) & (
            #         xy[:, 1] <= (400 - 1)) & (z_ > 0)
            # z_ = z_[is_in_picture]
            # #print("y max:", np.max(xy[:,1]))
            # #print("z:",  z_)
            # xy = xy[is_in_picture, :]

            # #plt.savefig(file_img_seq+str(ts)+"_img_left"+".jpg")
            # pc_draw = (z_-np.min(z_)/(np.max(z_)-np.min(z_)))
            # plt.figure()
            # plt.imshow(img)
            # plt.scatter(xy[:,0], xy[:,1], c=pc_draw, cmap='jet', alpha=0.7, s=1)
            # plt.savefig("pc_proj"+".jpg")

            # plt.close()

            if "cmr" not in DATASET:
                pcl_uv, pcl_z, _ = self.get_projected_pts(pcl, intrinsic, init_extrinsic, img.shape)
                init_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img, 255)

                pcl_uv, pcl_z, _ = self.get_projected_pts(pcl, intrinsic, pred_extrinsic, img.shape)

                pj_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img, 255)

                if self.pose_t == 5:
                    pcl_uv, pcl_z, _ = self.get_projected_pts(pcl, intrinsic, pred_extrinsic_coarse, img.shape)

                    pj_projected_img_coarse = vis.get_projected_img(pcl_uv, pcl_z, img, 255)
                #print(gt_extrinsic)
                pcl_uv, pcl_z, mask = self.get_projected_pts(pcl, intrinsic, gt_extrinsic, img.shape)
             
                gt_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img, 255)
            else:
                def project_cmr(Pr, pc_np, img_project):

                    from data_preprocess.CMRNet_script.depth_map_script.py_visibility import \
                        depth_image, pixel_depth, visibility2
                    img_project = img_project.copy()
                    device = torch.device("cuda:0")
                    h, w, _ = img_project.shape
                    pc_np_cam = intrinsic @ (Pr[:3, :3] @ pc_np.T + Pr[:3, 3][:, None])

                    pc_np_z = pc_np_cam[2:, :]
                    pc_np_uv = pc_np_cam[:2, :] / (pc_np_z + 1e-10)

                    pc_np_uv = pc_np_uv.astype(np.int_)
                    pc_fore_mask = pc_np_z[0] > 0
                    pc_fore_insidey = np.logical_and(pc_np_uv[1] >= 0, pc_np_uv[1] < h)
                    pc_fore_insidex = np.logical_and(pc_np_uv[0] >= 0, pc_np_uv[0] < w)
                    pc_fore_inside = np.logical_and(pc_fore_insidey, pc_fore_insidex)
                    pc_fore_mask = np.logical_and(pc_fore_mask, pc_fore_inside)
                    pc_np_uv = pc_np_uv[:, pc_fore_mask]
                    pc_np_z = pc_np_z[:, pc_fore_mask]

                    lidar_uv = torch.from_numpy(pc_np_uv.T).to(device).int()
                    lidar_depth = torch.from_numpy(pc_np_z[0]).to(device).float()
                    cam_intrinsic = torch.from_numpy(intrinsic).to(device).float()

                    depth_map = depth_image(lidar_uv, lidar_depth, (h, w))

                    if VISB:
                        new_depth_map = visibility2(depth_map, cam_intrinsic, (h, w))
                    else:
                        new_depth_map = depth_map
                    img_project = pixel_depth(img_project, new_depth_map)

                    return cv2.cvtColor(img_project, cv2.COLOR_BGR2RGB)

                init_projected_img = project_cmr(init_extrinsic, pcl, img)
                pj_projected_img = project_cmr(pred_extrinsic, pcl, img)
                gt_projected_img = project_cmr(gt_extrinsic, pcl, img)

            # cv2.imwrite(os.path.join(self.save_path,"00.jpg"),cv2.resize(gt_projected_img,(1600,800)))
            if OUT:
                ex = utils.mult_extrinsic(gt_extrinsic, utils.inv_extrinsic(init_extrinsic))
                pcl_uv, pcl_z = self.get_projected_pts(self.outlier[vis_t], intrinsic, ex, img.shape)
                if len(pcl_z) > 0:
                    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                    dist_norm = utils.max_normalize_pts(pcl_z) * 90

                    for i in range(pcl_uv.shape[0]):
                        cv2.circle(hsv_img, (int(pcl_uv[i, 0]), int(pcl_uv[i, 1])), radius=10, color=(
                            int(dist_norm[i]), 255, 255), thickness=-1)

                    outlier_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
                else:
                    outlier_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            n_iter = abs(self.start)

            os.makedirs(os.path.join(self.save_dir, "%06d" % vis_t), exist_ok=True)

            # create_output(pc_np[mask],color,os.path.join(self.save_dir,"%06d"%vis_t,"pc.ply"))
            save_img = lambda im, path: cv2.imwrite(os.path.join(self.save_dir, "%06d" % vis_t, path),
                                                    cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

            save_img(init_projected_img, "init_projected_img.png")
            save_img(pj_projected_img, "pred_projected_img.png")
            save_img(gt_projected_img, "gt_projected_img.png")
            if self.pose_t == 5:
                save_img(pj_projected_img_coarse, "pred_projected_img_coarse.png")

            # self.writer.add_image("init_projected_img", init_projected_img, n_iter
            #                       , dataformats="HWC")
            # self.writer.add_image("pj_projected_img", pj_projected_img, n_iter
            #                       , dataformats="HWC")
            # self.writer.add_image("gt_projected_img", gt_projected_img, n_iter
            #                       , dataformats="HWC")
            # if OUT:
            #     self.writer.add_image("outlier_projected_img", outlier_img, n_iter
            #                       , dataformats="HWC")
            # self.writer.add_images("comparison_img", concat_img, n_iter
            #                        , dataformats="NHWC")

            # self.start += 4*step

    # def decode_path(self, path: str):
    #     if DATASET == "kitti":
    #         # base_path/date/date_drive_{drive_code}_sync/velodyne_points/data/{scan_id}.bin
    #         paths = path.split(os.sep)
    #         scan_id = paths[-1].split('.')[0]
    #         drive_code = paths[-4].split('_')[-2]
    #         info = (scan_id, drive_code)
    #     elif "kd" in DATASET:
    #         seq, seq_i, seq_j = path.split(' ')
    #         info = (seq, seq_i, seq_j)
    #     elif DATASET == "nus":
    #         info = (int(path))
    #     else:
    #         info = None
    #     return info

    def calculate_sections(self, lines):
        count = -1
        section = {}
        count2 = 0
        last = None
        while count + len(lines) >= 0:
            # if abs(count) == len(self.lines):
            #     break
            if "section" in lines[count]:
                if count2 % 4 == 0:  # no coarse:
                    section[lines[count].strip("[section sign] prediction on ")[:19]] = (count2 // 4, count, 4)
                elif count2 % 5 == 0:  # coarse
                    section[lines[count].strip("[section sign] prediction on ")[:19]] = (count2 // 5, count, 5)
                else:
                    continue

                if COARSE and count2 % 5 == 0:  # coarse
                    section[lines[count].strip("[section sign] prediction on ")[:19]] = (count2 // 5, count, 5)
                count2 = 0
                if last is None:
                    last = lines[count].strip("[section sign] prediction on ")[:19]

            else:
                count2 += 1
            count -= 1
        # print(section)
        return section, last


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


if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.vis()
