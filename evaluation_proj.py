import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from importlib import import_module
from sklearn.metrics import roc_curve, accuracy_score
import cv2
import yaml
from datetime import datetime
import torch.nn.functional as F
import time
import numpy as np
import pickle

from avgmeter import AverageMeter
# import src.modellearn as mod
from src.deterministic import set_seed, seed_worker
import src.utils as utils
from compute_loss import Get_loss
from metric import getExtrinsic, RteRreEval, calibration_error_batch, eval_mrr, eval_msee, cal_rete_once, mult_extrinsic_batch, inv_extrinsic, quaternion_distance
from scipy.spatial.transform import Rotation
import math
# from src.config_proj import I2PNetConfig as modelcfg

# arg parser
import src.visualize as vis

parser = argparse.ArgumentParser()
# TODO: support use the network in the train log
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--abs_checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--checkpoint_path', default="model_rotation_best.pt", help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', required=True, help='Dump dir to save model checkpoint [default: log]')
parser.add_argument("--network", default="modellearn_proj_center", type=str, help="the network to train [default: modellearn]")
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers [default: 8]')
parser.add_argument('--dataset', type=str, default="kd_corr_snr_proj", choices=["kitti", "nus_corr_nolidar", "kd_corr_nolidar",
                             ],
                    help="choose which dataset to train [default: kitti]")
parser.add_argument('--rot_test', type=float, default=10., help="when dataset is kitti, choose the fixed decalib")
parser.add_argument('--delete', action="store_true", help="clear the previous results")
parser.add_argument('--save_model', action="store_true")
parser.add_argument('--coarse', action="store_true")
parser.add_argument('--cmr_seed', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
parser.add_argument('--modelcfg', type=str,default="config_proj")
FLAGS = parser.parse_args()

WORKERS = FLAGS.num_workers
LOGDIR = FLAGS.log_dir
CKPT = FLAGS.checkpoint_path
ABSCKPT = FLAGS.abs_checkpoint_path
BATCH_SIZE = FLAGS.batch_size
NETWORK = FLAGS.network
DATASET = FLAGS.dataset
ROT_TEST = FLAGS.rot_test
DELETE = FLAGS.delete
COARSE = FLAGS.coarse
CMRSEED = FLAGS.cmr_seed
MODELCFG = FLAGS.modelcfg

modelcfg = import_module("src.{0}".format(MODELCFG)).I2PNetConfig

if "kd" in DATASET:
    dataset_file = "kitti_odometry_corr_lidarnone_proj"                    
    
    DA = import_module("src.{0}".format(dataset_file))
    testdataset = DA.Kitti_Odometry_Dataset
    from src.dataset_params import KITTI_ODOMETRY as cfg
elif "nus" in DATASET:
    dataset_file = "nuscenes_loader_proj_nolidar"

    DA = import_module("src.{0}".format(dataset_file))
    testdataset = DA.nuScenesLoader
    from src.dataset_params import NUSCENES as cfg

    
# Setup
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
set_seed(0, True)  # deterministic

# mod = import_module("{0}.{1}".format(LOGDIR,"network"))
# if os.path.exists(os.path.join(LOGDIR,"config.py"))
mod = import_module("{0}.{1}".format("src", NETWORK))
RegNet_v2 = mod.RegNet_v2


def get_2D_lidar_projection(pcl, K, img_size, velo_extrinsic):
    pcl_xyz = np.hstack((pcl[:, :3], np.ones((pcl.shape[0], 1)))).T
    pcl_xyz = velo_extrinsic @ pcl_xyz  # [3,4]@[4,N]
    pcl_xyz = pcl_xyz.T
    pcl_norm_xyz = pcl_xyz / pcl_xyz[:, 2:]
    pcl_uv = (K @ (pcl_norm_xyz.T))[:2, :].T
    pcl_z = pcl_xyz[:, 2]
    inlier = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_size[1]) & (pcl_uv[:, 1] > 0) & \
             (pcl_uv[:, 1] < img_size[0]) & (pcl_z > 0)
    return inlier.astype(np.int32)


class Evaluator(object):
    def __init__(self):
        RUN_ID = 5

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        # print(device)

        if ABSCKPT is not None:
            ckpt_path = ABSCKPT
        else:
            ckpt_path = str(Path(LOGDIR) / 'checkpoints_new' /
                            'run_{:05d}'.format(RUN_ID) / CKPT)

        save_path = Path(LOGDIR) / "info_test"
        # logs
        save_path.mkdir(parents=True, exist_ok=True)

        save_path_tensorboard = save_path / "tensorboard"
        save_path_tensorboard.mkdir(parents=True, exist_ok=True)

        time_now = datetime.now()
        ts_info_txt = time_now.strftime('%Y-%m-%d %X')
        ts_info = time_now.strftime('%Y_%m_%d_') + '_'.join(time_now.strftime('%X').split(':'))

        # Model
        self.model = RegNet_v2(cfg=modelcfg)
        self.model.to(self.device)

        self.metric_path = os.path.join(str(save_path), "metrics_" + ts_info + ".npz")

        ckpt = torch.load(ckpt_path)

        save_path_model = save_path / "models"
        save_path_model.mkdir(parents=True, exist_ok=True)

        self.model.load_state_dict(ckpt["model_state_dict"])

        if FLAGS.save_model:
            torch.save(ckpt, str((save_path_model / ("model_best_" + ts_info + ".pt")).resolve()))

        if DATASET == "kitti":
            self.f_write = open(str(save_path / f"log_test_{int(ROT_TEST)}.txt"), "a+" if not DELETE else "w+")
            self.f_result = open(str(save_path / f"prediction_{int(ROT_TEST)}.txt"), "a+" if not DELETE else "w+")
        else:
            self.f_write = open(str(save_path / f"log_test.txt"), "a+" if not DELETE else "w+")
            self.f_result = open(str(save_path / f"prediction.txt"), "a+" if not DELETE else "w+")
        model_info = "rotation_best_model" if "rotation" in ckpt_path else "transition_best_model"
        try:
            model_info += f"rmae:{ckpt['cur_rotation_error']:.3f}/{ckpt['best_rotation_error']:.3f} " \
                          f"tmae:{ckpt['cur_transition_error']:3f}/{ckpt['best_transition_error']:.3f}"
        except:
            model_info += f"rmae:{ckpt['best_rotation_error']:.3f} " \
                          f"tmae:{ckpt['best_translation_error']:.3f}"
        if "kitti_rgg" in DATASET:
            model_info = f"msee_best_model msee:{ckpt['best_msee']}"
            self.f_write.write(f"[section sign] test on {ts_info_txt} {model_info}\n")
        elif "cmr" in DATASET:
            self.f_write.write(f"[section sign] test on {ts_info_txt} test_seed {CMRSEED:d} {model_info}\n")
        else:
            self.f_write.write(f"[section sign] test on {ts_info_txt} rot_test {ROT_TEST:.3f} {model_info}\n")

        self.f_write.flush()
        self.f_result.write(f"[section sign] prediction on {ts_info_txt} rot_test {ROT_TEST:.3f} {model_info}\n")
        self.f_result.flush()

        file = open(os.path.join(str(save_path), "config.yaml"), mode="w", encoding="utf-8")
        yaml.dump(vars(FLAGS), file)
        file.close()

        writer_info = f"test_{int(ROT_TEST)}" + ts_info if DATASET == "kitti" else "test_" + ts_info
        self.writer = SummaryWriter(log_dir=str(save_path_tensorboard),
                                    filename_suffix=writer_info)
        # for deterministic
        g = torch.Generator()
        g.manual_seed(0)

        # validation data
        if "kitti_rgg" in DATASET:
            if "t1" in DATASET:
                params = cfg.dataset_params_T1
            elif "t2a" in DATASET:
                params = cfg.dataset_params_T2a
            elif "t2b" in DATASET:
                params = cfg.dataset_params_T2b
            elif "t3" in DATASET:
                params = cfg.dataset_params_T3
            else:
                raise NotImplementedError
        else:
            params = cfg.dataset_params_test
            if "cmr" in DATASET:
                params["cmr_seed"] = CMRSEED

        dataset_test = testdataset(params)
        self.dataset = dataset_test
        self.test_loader = DataLoader(dataset_test,
                                      batch_size=BATCH_SIZE,
                                      num_workers=WORKERS,
                                      pin_memory=True,
                                      worker_init_fn=seed_worker,
                                      generator=g,
                                      shuffle=False,
                                      drop_last=False)

    def __del__(self):

        self.f_write.close()
        self.f_result.close()

    def validate(self):
        self.model.eval()
        # self.model.set_bn()
        mean_roll_error = AverageMeter()
        mean_pitch_error = AverageMeter()
        mean_yaw_error = AverageMeter()
        mean_x_error = AverageMeter()
        mean_y_error = AverageMeter()
        mean_z_error = AverageMeter()
        batch_time = AverageMeter()
        mean_see = AverageMeter()
        mean_rr = AverageMeter()

        running_rre = AverageMeter()
        running_rte = AverageMeter()

        evaluator = RteRreEval()
        err_rotq = []
        err_transq = []
        count = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            for valid_count, data_valid in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                torch.cuda.synchronize()
                t1 = time.time()

                resize_img = data_valid['resize_img'].to(self.device)
                rgb_img = data_valid['rgb'].to(self.device)
                lidar_img = data_valid['lidar'].to(self.device)  # load lidar
                H_initial = data_valid['init_extrinsic'].to(self.device)
                intrinsic = data_valid['init_intrinsic'].to(self.device)
                calib = None
                # if modelcfg.efgh:
                #     calib = data_valid['calib'].to(self.device)

                lidar_feats = None

                lidar_feats = data_valid["lidar_feats"].to(self.device).float()

                lidar_img_raw = data_valid["raw_point_xyz"].to(self.device).float()

                gt_project = None

                out3, out4, pm3, pm4, sx, sq = self.model(rgb_img, lidar_img, lidar_img_raw, H_initial,
                                                          intrinsic, resize_img, gt_project,
                                                          calib, lidar_feats,cfg=modelcfg)

                torch.cuda.synchronize()
                batch_time.update(time.time() - t1)

                rre, rte = cal_rete_once(out3, data_valid)
                pred_extrinsic, gt_extrinsic, pred_raw, gt_raw = getExtrinsic(out3, data_valid, out_raw=True)
                if COARSE:
                    pred_extrinsic_coarse, _ = getExtrinsic(out4, data_valid)
                init_extrinsic = H_initial.detach().cpu().numpy()

                if "kitti_rgg" in DATASET:
                    gt_se3 = data_valid['decalib_se3'].cpu().numpy()
                    msee = eval_msee(out3, gt_se3)
                    mrr = eval_mrr(msee, gt_se3)

                cur_roll_error, cur_pitch_error, cur_yaw_error, \
                cur_x_error, cur_y_error, cur_z_error = calibration_error_batch(pred_raw, gt_raw)

                r_diff, t_diff = evaluator.addBatch(pred_raw, gt_raw)
                if "coarse" in DATASET:
                    P_diff = mult_extrinsic_batch(inv_extrinsic(pred_extrinsic), gt_extrinsic)
                    r_diff_q = P_diff[:, :3, :3]
                    R_diff = Rotation.from_matrix(r_diff_q)
                    angles_diff = np.sum(np.abs(R_diff.as_euler('xzy', degrees=True)), -1)
                    
                    rotq = R_diff.as_quat()
                    rot_norm = np.linalg.norm(rotq, axis=1)
                    rotq = rotq/rot_norm[:, None]
                    # for i in range(0,rot_norm.shape[0]):
                    #     rotq[i] = rotq[i]/rot_norm[i]
                    rotq = np.roll(rotq, 1)
                    #print(rotq)
                    total_rot_error = quaternion_distance(rotq, np.array([[1., 0., 0., 0.]]))
                    #print(angles_diff)
                    total_rot_error = total_rot_error * 180. / math.pi
                    #print(total_rot_error)
                    #print(total_rot_error)
                
                
                # if valid_count % VIS_RATE == 0:
                #     self.vis(data_valid,pred_extrinsic,gt_extrinsic,valid_count)
                running_rre.update(float(rre), 1)
                running_rte.update(float(rte), 1)

                if (valid_count + 1) % 20 == 0:
                    self.f_write.write("step {:05d}| rre: {:05f} | rte: {:05f}\n".format(
                        (valid_count + 1) // 20, running_rre.avg, running_rte.avg
                    ))
                    running_rre.reset()
                    running_rte.reset()

                for i in range(len(cur_x_error)):
                    # data_path qw,qx,qy,qz tx,ty,tz
                    info = self.decode_path(data_valid["path_info"][i])
                    meta_data = ' '.join(info) + '\n'

                    self.f_result.write(meta_data)
                    ex2str = lambda x, i: ' '.join(['%.9f' % v for v in x[i].reshape(-1)]) + '\n'
                    if not COARSE:
                        self.f_result.write(
                            ex2str(init_extrinsic, i) + ex2str(pred_extrinsic, i) + ex2str(gt_extrinsic, i))
                    else:
                        self.f_result.write(
                            ex2str(init_extrinsic, i) + ex2str(pred_extrinsic_coarse, i) + ex2str(
                                pred_extrinsic, i)
                            + ex2str(gt_extrinsic, i))
                    self.f_write.flush()

                    # self.writer.add_scalar("AUC",auc,count)

                    self.writer.add_scalar("MRE", (cur_roll_error[i] + cur_yaw_error[i] + cur_pitch_error[i]) / 3,
                                           count)
                    self.writer.add_scalar("MTE", (cur_x_error[i] + cur_y_error[i] + cur_z_error[i]) / 3, count)
                    self.writer.add_scalar("RRE", r_diff[i], count)
                    self.writer.add_scalar("RTE", t_diff[i], count)
                    if "kitti_rgg" in DATASET:
                        self.writer.add_scalar("SEE", msee[i], count)
                        self.writer.add_scalar("RR", mrr[i], count)

                    mean_roll_error.update(cur_roll_error[i])
                    mean_pitch_error.update(cur_pitch_error[i])
                    mean_yaw_error.update(cur_yaw_error[i])
                    mean_x_error.update(cur_x_error[i])
                    mean_y_error.update(cur_y_error[i])
                    mean_z_error.update(cur_z_error[i])
                    if "coarse" in DATASET:
                        err_rotq.append(total_rot_error[i])
                        err_transq.append(t_diff[i])
                    if "kitti_rgg" in DATASET:
                        mean_see.update(float(msee[i]))
                        mean_rr.update(float(mrr[i]))

                    count += 1
        # self.f_write.write(modelcfg.debug_timing.summary())
        if "kitti_rgg" in DATASET:
            self.f_write.write('TESTSET: {}\n'.format(DATASET.split('_')[-1]))
        self.f_write.write('rot_test_set= {:3f}\n'.format(ROT_TEST))

        self.f_write.write('mean_FPS= {:3f}\n'.format(1.0 / batch_time.avg))

        self.f_write.write('mean_time= {:3f} ms\n'.format(batch_time.avg * 1e3))

        self.f_write.write('mean_roll_error= {:3f}\n'.format(mean_roll_error.avg))

        self.f_write.write('mean_pitch_error= {:3f}\n'.format(mean_pitch_error.avg))

        self.f_write.write('mean_yaw_error= {:3f}\n'.format(mean_yaw_error.avg))

        self.f_write.write('mean_x_error= {:3f}\n'.format(mean_x_error.avg))

        self.f_write.write('mean_y_error= {:3f}\n'.format(mean_y_error.avg))

        self.f_write.write('mean_z_error= {:3f}\n'.format(mean_z_error.avg))

        cur_mean_rotation_error = (mean_roll_error.avg + mean_pitch_error.avg + mean_yaw_error.avg) / 3
        cur_mean_translation_error = (mean_x_error.avg + mean_y_error.avg + mean_z_error.avg) / 3

        self.f_write.write('mean_rotation_error= {:3f}\n'.format(cur_mean_rotation_error))

        self.f_write.write('mean_translation_error= {:3f}\n'.format(cur_mean_translation_error))

        if "coarse" in DATASET:
            err_rotq = np.array(err_rotq)
            err_transq = np.array(err_transq)
            self.f_write.write('RE_for_CMR %.2f +- %.2f\n' % (np.mean(err_rotq), np.std(err_rotq)))
            self.f_write.write('TE_for_CMR %.2f +- %.2f\n' % (np.mean(err_transq), np.std(err_transq)))
            self.f_write.write('median_rot_error= {:3f}\n'.format(np.median(err_rotq)))
            self.f_write.write('median_trans_error= {:3f}\n'.format(np.median(err_transq)))



        

        if "kitti_rgg" in DATASET:
            self.f_write.write('MSEE= {:8f}\n'.format(mean_see.avg))
            self.f_write.write('MRR= {:8f}%\n'.format(mean_rr.avg * 100))

        rte_mean, rte_std, rre_mean, rre_std = evaluator.evalSeq()

        self.f_write.write('RTE %.2f +- %.2f, RRE %.2f +- %.2f\n' % (rte_mean, rte_std, rre_mean, rre_std))

        self.f_write.flush()

        evaluator.save_metric(self.metric_path)

    def decode_path(self, path: str):
        if DATASET == "kitti":
            # base_path/date/date_drive_{drive_code}_sync/velodyne_points/data/{scan_id}.bin
            paths = path.split(os.sep)
            scan_id = paths[-1].split('.')[0]
            drive_code = paths[-4].split('_')[-2]
            info = (scan_id, drive_code)
        elif "kitti_rgg" in DATASET:
            testset, ind = path.split(' ')
            info = (testset, ind)
        elif "kd" in DATASET:
            seq, seq_i, seq_j = path.split(' ')
            info = (seq, seq_i, seq_j)
        elif "nus" in DATASET:
            info = (path)
        elif "realloc" in DATASET:
            seq,pc_ts, camera_ts = path.split(' ')
            info = (seq, pc_ts, camera_ts)
        else:
            info = None
        return info


if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.validate()
