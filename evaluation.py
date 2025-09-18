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
from metric import getExtrinsic, RteRreEval, calibration_error_batch, eval_mrr, eval_msee, cal_rete_once
#from src.config import I2PNetConfig as modelcfg
try:
    from src.deepi2p_modules.multimodal_classifier_my_snr import MMClassifer
except:
    print("no deepi2p")
# arg parser
import src.visualize as vis

parser = argparse.ArgumentParser()
# TODO: support use the network in the train log
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--abs_checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--checkpoint_path', default="model_rotation_best.pt", help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', required=True, help='Dump dir to save model checkpoint [default: log]')
parser.add_argument("--network", default="modellearn", type=str, help="the network to train [default: modellearn]")
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers [default: 8]')
parser.add_argument('--dataset', type=str, default="kitti", choices=["kitti", "kd", "kd_corr", "kd_small",
                                                                     "kd_efgh", "nus", "kd_sdeep",
                                                                     "kd_cmr_snr",
                                                                     "kd_efgh_snr","kd_corr_nolidar",
                                                                     "kd_corr_snr_proj",
                                                                     "nus_corr_snr_ori", "nus_corr_nolidar", "nus_cmr_snr",
                                                                     "nus_corr", "nus_corr_snr", "nus_corr_snr_ex",
                                                                     "kd_corr_snr",
                                                                     "kitti_rgg_t1", "kitti_rgg_t2a", "kitti_rgg_t2b",
                                                                     "kitti_rgg_t3", "kitti_rgg_snr_t1",
                                                                     "kitti_rgg_snr_t2a", "kitti_rgg_snr_t2b",
                                                                     "kitti_rgg_snr_t3", 
                                                                    ],
                    help="choose which dataset to train [default: kitti]")
parser.add_argument('--rot_test', type=float, default=10., help="when dataset is kitti, choose the fixed decalib")
parser.add_argument('--delete', action="store_true", help="clear the previous results")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--outlier_record', action="store_true")
parser.add_argument('--use_deepi2p', action="store_true")
parser.add_argument('--threshold', action="store_true")
parser.add_argument('--validation', action="store_true")
parser.add_argument('--coarse', action="store_true")
parser.add_argument('--save_model', action="store_true")
parser.add_argument('--cmr_seed', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
parser.add_argument('--spawn', action="store_true")
parser.add_argument('--evaluation_seed', type=int, default=0)
parser.add_argument('--modelcfg', default="config",type=str)
FLAGS = parser.parse_args()

MODELCFG = FLAGS.modelcfg
WORKERS = FLAGS.num_workers
LOGDIR = FLAGS.log_dir
CKPT = FLAGS.checkpoint_path
ABSCKPT = FLAGS.abs_checkpoint_path
BATCH_SIZE = FLAGS.batch_size
NETWORK = FLAGS.network
DATASET = FLAGS.dataset
ROT_TEST = FLAGS.rot_test
DELETE = FLAGS.delete
DEBUG = FLAGS.debug
OUT = FLAGS.outlier_record
DEEP = FLAGS.use_deepi2p
THRESH = FLAGS.threshold
VALI = FLAGS.validation
COARSE = FLAGS.coarse
MODELSAVE = FLAGS.save_model
CMRSEED = FLAGS.cmr_seed
SEED_E = FLAGS.evaluation_seed


modelcfg = import_module("src.{0}".format(MODELCFG)).I2PNetConfig
print(MODELCFG)
print(modelcfg.featmode)

if DATASET == "kitti":
    from src.dataset import Kitti_Dataset as testdataset
    from src.dataset_params import KITTI_ONLINE_CALIB as cfg

    dataset_file = "dataset"
elif "kitti_rgg" in DATASET:
    if "snr" in DATASET:
        from src.dataset_rggnet_snr import Kitti_Dataset as testdataset
        from src.dataset_params import KITTI_RGG_CALIB as cfg

        dataset_file = "dataset_rggnet_snr"
    else:
        from src.dataset_rggnet import Kitti_Dataset as testdataset
        from src.dataset_params import KITTI_RGG_CALIB as cfg

        dataset_file = "dataset_rggnet"
elif 'kd' in DATASET:
    if 'corr_snr' in DATASET:
        dataset_file = 'kitti_odometry_corr_snr'
    elif DATASET == 'kd_corr_nolidar':
        print("no lidar feat use")
        dataset_file = "kitti_odometry_corr_lidarnone"
    elif 'corr' in DATASET:
        dataset_file = 'kitti_odometry_corr'
    elif 'cmr' in DATASET:
        dataset_file = 'kitti_odometry_cmr'
    elif 'pr' in DATASET:
        dataset_file = 'kitti_odometry_efgh_pr_snr'
    elif 'efgh_snr' in DATASET:
        dataset_file = 'kitti_odometry_efgh_snr'
    elif 'efgh' in DATASET:
        dataset_file = 'kitti_odometry_efgh'
    elif 'small' in DATASET:
        dataset_file = 'kitti_odometry_small'
    elif 'sdeep' in DATASET:
        dataset_file = 'kitti_odometry_small1'
    else:
        dataset_file = "kitti_odometry"
    DA = import_module("src.{0}".format(dataset_file))
    testdataset = DA.Kitti_Odometry_Dataset
    from src.dataset_params import KITTI_ODOMETRY as cfg
elif "nus" in DATASET:
    if DATASET =="nus_corr_nolidar":
        dataset_file = "nuscenes_loader_nolidar"
    elif 'cmr' in DATASET:
        dataset_file = 'nuscenes_loader_cmr'
    elif "ex" in DATASET:
        dataset_file = "nuscenes_loader_snr"
    elif "ori" in DATASET:
        dataset_file = "nuscenes_loader_trans"
    elif "corr_snr" in DATASET:
        #dataset_file = "nuscenes_loader_processed_snr"
        dataset_file = "nuscenes_loader_snr"
    elif "corr" in DATASET:
        dataset_file = "nuscenes_loader_processed"
    else:
        dataset_file = "nuscenes_loader"
    DA = import_module("src.{0}".format(dataset_file))
    testdataset = DA.nuScenesLoader
    from src.dataset_params import NUSCENES as cfg

# Setup
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
set_seed(SEED_E, True)  # deterministic

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

        save_path_model = save_path / "models"
        save_path_model.mkdir(parents=True, exist_ok=True)

        time_now = datetime.now()
        ts_info_txt = time_now.strftime('%Y-%m-%d %X')
        ts_info = time_now.strftime('%Y_%m_%d_') + '_'.join(time_now.strftime('%X').split(':'))

        # Model
        self.model = RegNet_v2(cfg=modelcfg, eval_info=True)
        self.model.to(self.device)

        self.metric_path = os.path.join(str(save_path), "metrics_" + ts_info + ".npz")

        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt["model_state_dict"])

        if MODELSAVE:
            torch.save(ckpt, str((save_path_model / ("model_best_" + ts_info + ".pt")).resolve()))

        if OUT:
            self.out_path = str(save_path / "outlier.pkl")

        if not DEBUG:
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
                if THRESH:
                    self.rre_th = 10.
                    self.rte_th = 5.
                    self.f_write.write(f"[section sign] test on {ts_info_txt} rot_test {ROT_TEST:.3f} {model_info} "
                                       f"threshold: rre {self.rre_th} rte {self.rte_th}\n")
                else:
                    self.f_write.write(
                        f"[section sign] test on {ts_info_txt} rot_test {ROT_TEST:.3f} {model_info} seed{SEED_E:d}\n")

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
        g.manual_seed(SEED_E)

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
            if VALI:
                params = cfg.dataset_params_valid3
            else:
                params = cfg.dataset_params_test
            if DATASET == "kitti":
                params["d_rot"] = ROT_TEST
                params["d_trans"] = 0.1 * ROT_TEST
            elif "cmr" in DATASET:
                params["cmr_seed"] = CMRSEED

        dataset_test = testdataset(params, use_raw=modelcfg.raw_feat_point)
        self.dataset = dataset_test
        self.test_loader = DataLoader(dataset_test,
                                      batch_size=BATCH_SIZE,
                                      num_workers=WORKERS,
                                      pin_memory=True,
                                      worker_init_fn=seed_worker,
                                      generator=g,
                                      shuffle=False,
                                      drop_last=False)

        if DEEP:
            self.deepi2p = MMClassifer()
            self.deepi2p.to(self.device)
            self.deepi2p.load_model("../DeepI2P/runs/1.32_continue/best.pt")

    def __del__(self):
        if not DEBUG:
            self.f_write.close()
            self.f_result.close()

    def validate(self):
        # skip_value = 5176
        # VIS_RATE = 40
        self.model.eval()
        mean_roll_error = AverageMeter()
        mean_pitch_error = AverageMeter()
        mean_yaw_error = AverageMeter()
        mean_x_error = AverageMeter()
        mean_y_error = AverageMeter()
        mean_z_error = AverageMeter()
        auc_total = AverageMeter()
        pre_total = AverageMeter()
        fn_total = AverageMeter()
        batch_time = AverageMeter()
        mean_see = AverageMeter()
        mean_rr = AverageMeter()

        running_rre = AverageMeter()
        running_rte = AverageMeter()

        evaluator = RteRreEval() if not THRESH else RteRreEval(THRESH, self.rre_th, self.rte_th)

        count = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if OUT:
            outliers = []

        with torch.no_grad():
            for valid_count, data_valid in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                if valid_count > 10 and DEBUG:
                    break
                torch.cuda.synchronize()
                t1 = time.time()

                resize_img = data_valid['resize_img'].to(self.device)
                rgb_img = data_valid['rgb'].to(self.device)
                lidar_img = data_valid['lidar'].to(self.device)  # load lidar
                H_initial = data_valid['init_extrinsic'].to(self.device)
                intrinsic = data_valid['init_intrinsic'].to(self.device)
                calib = None
                if modelcfg.efgh:
                    calib = data_valid['calib'].to(self.device)  # 3x4

                if modelcfg.raw_feat_point:
                    lidar_img_raw = data_valid['raw_point_xyz'].to(self.device)
                else:
                    lidar_img_raw = None

                lidar_feats = None
                if "snr" in DATASET or DATASET=="kd_corr_nolidar":
                    lidar_feats = data_valid["lidar_feats"].to(self.device).float()

                gt_project = None
                if modelcfg.ground_truth_projection_mask_eval and not DEEP:
                    decalib_quat_real = data_valid['decalib_real_gt'].to(self.device)
                    decalib_quat_dual = data_valid['decalib_dual_gt'].to(self.device)
                    gt_project = utils.get_projection_gt(lidar_img, intrinsic, rgb_img.shape[2:], decalib_quat_real,
                                                         decalib_quat_dual)
                    gt_project = F.one_hot(gt_project, num_classes=2)
                if DEEP:
                    decalib_quat_real = data_valid['decalib_real_gt'].to(self.device)
                    decalib_quat_dual = data_valid['decalib_dual_gt'].to(self.device)
                    deep_data = data_valid["deepi2p_data"].to(self.device).float()
                    deep_idx = data_valid["deepi2p_index"].to(self.device).long()
                    gt_project = utils.get_projection_gt(lidar_img, intrinsic, rgb_img.shape[2:], decalib_quat_real,
                                                         decalib_quat_dual)
                    deepi2p_pred = self.deepi2p.forward(deep_data, rgb_img, deep_idx)
                    gt_1 = torch.eq(gt_project, 1)
                    pred_1 = torch.eq(deepi2p_pred, 1)
                    gt_0 = torch.eq(gt_project, 0)
                    pred_0 = torch.eq(deepi2p_pred, 0)
                    false_neg = torch.logical_and(gt_0, pred_1)
                    #
                    # deepi2p_pred[false_neg] = 0
                    #
                    deepi2p_pred_2 = F.one_hot(deepi2p_pred, 2)

                    out3, out4, sx, sq, _, p3, l3_prediction_mask, _, _ = self.model(rgb_img, lidar_img,
                                                                                     H_initial, intrinsic, resize_img,
                                                                                     deepi2p_pred_2, calib, lidar_feats, cfg=modelcfg, lidar_img_raw=lidar_img_raw)

                else:
                    out3, out4, sx, sq, _, p3, l3_prediction_mask, _, _ = self.model(rgb_img, lidar_img,
                                                                                     H_initial, intrinsic, resize_img,
                                                                                     gt_project, calib, lidar_feats, cfg=modelcfg, lidar_img_raw=lidar_img_raw)

                torch.cuda.synchronize()
                batch_time.update(time.time() - t1)

                pred_extrinsic, gt_extrinsic, pred_ex_raw, gt_ex_raw = getExtrinsic(out3, data_valid, out_raw=True)
                #print(gt_extrinsic)
                if COARSE:
                    pred_extrinsic_coarse, _ = getExtrinsic(out4, data_valid)

                # get l3_w and l3_p and total_p and decalib_gt
                if modelcfg.use_projection_mask and modelcfg.layer_mask[1] and not DEEP:
                    mcW_l3 = l3_prediction_mask.argmax(-1).cpu().detach().numpy()  # [B,N3]
                    p3 = p3.detach().cpu().numpy()  # [B,N3,3]
                    K = intrinsic[0].cpu().detach().numpy()
                    # pcl = lidar_img.detach().cpu().numpy() # [B,N,3]
                    gt_decalib_quat_real = data_valid['decalib_real_gt'].numpy()
                    gt_decalib_quat_dual = data_valid['decalib_dual_gt'].numpy().reshape(-1, 3, 1)

                init_extrinsic = H_initial.detach().cpu().numpy()

                # if DATASET == "kitti_rgg":
                #     gt_se3 = data_valid['decalib_se3'].cpu().numpy()
                #     msee = eval_msee(out3, gt_se3)
                #     mrr = eval_mrr(msee, gt_se3)
                if "kitti_rgg" in DATASET:
                    gt_se3 = data_valid['decalib_se3'].cpu().numpy()
                    msee = eval_msee(out3, gt_se3)
                    mrr = eval_mrr(msee, gt_se3)

                cur_roll_error, cur_pitch_error, cur_yaw_error, \
                cur_x_error, cur_y_error, cur_z_error = calibration_error_batch(pred_ex_raw, gt_ex_raw)

                r_diff, t_diff = evaluator.addBatch(pred_ex_raw, gt_ex_raw)

                # if valid_count % VIS_RATE == 0:
                #     self.vis(data_valid,pred_extrinsic,gt_extrinsic,valid_count)

                if DEEP:
                    B, N = gt_project.shape
                    right = torch.eq(gt_project, deepi2p_pred)
                    acc = (right.sum(-1) / N).cpu().numpy()
                    gt_1 = torch.eq(gt_project, 1)
                    pred_1 = torch.eq(deepi2p_pred, 1)
                    gt_0 = torch.eq(gt_project, 0)
                    pred_0 = torch.eq(deepi2p_pred, 0)
                    pred_true_1 = torch.logical_and(gt_1, pred_1)
                    pred_false_1 = torch.logical_and(gt_0, pred_1)
                    precision = ((pred_true_1.sum(-1)) / (gt_1.sum(-1))).cpu().numpy()  # recall
                    # actually false positive
                    false_neg = ((pred_false_1.sum(-1)) / (gt_0.sum(-1))).cpu().numpy()
                rre, rte = cal_rete_once(out3, data_valid)
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
                    if not DEBUG:
                        self.f_result.write(meta_data)

                        if "efgh" in DATASET:
                            calib_np = data_valid["calib"].numpy()  # 3,4
                            ex2str = lambda x, i: ' '.join(
                                ['%.9f' % v for v in (utils.mult_extrinsic(calib_np[i], x[i]))
                                    .reshape(-1)]) + '\n'
                        else:
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

                    if modelcfg.use_projection_mask and modelcfg.layer_mask[1] and not DEEP:
                        # eval w auc
                        R = utils.quat_to_rotmat(*gt_decalib_quat_real[i])
                        ex = utils.get_extrinsic(R, gt_decalib_quat_dual[i])
                        h, w = rgb_img.shape[-2:]
                        label2 = get_2D_lidar_projection(p3[i], K, [h, w], ex)
                        auc = accuracy_score(label2, mcW_l3[i])

                        if OUT:
                            outlier = np.abs(label2 - mcW_l3[i]) == 1
                            outliers.append(p3[i][outlier])

                    # if "kitti_rgg" in DATASET:
                    #     gt_se3 = data_valid['decalib_se3'].cpu().numpy()
                    #     msee = eval_msee(out3, gt_se3)
                    #     mrr = eval_mrr(msee, gt_se3)


                    # self.writer.add_scalar("AUC",auc,count)
                    if not DEBUG:
                        self.writer.add_scalar("MRE", (cur_roll_error[i] + cur_yaw_error[i] + cur_pitch_error[i]) / 3,
                                               count)
                        self.writer.add_scalar("MTE", (cur_x_error[i] + cur_y_error[i] + cur_z_error[i]) / 3, count)
                        self.writer.add_scalar("RRE", r_diff[i], count)
                        self.writer.add_scalar("RTE", t_diff[i], count)
                        if "kitti_rgg" in DATASET:
                            self.writer.add_scalar("SEE", msee[i], count)
                            self.writer.add_scalar("RR", mrr[i], count)
                        if modelcfg.use_projection_mask and modelcfg.layer_mask[1] and not DEEP:
                            self.writer.add_scalar("ACC", auc, count)
                        if DEEP:
                            self.writer.add_scalar("ACC", acc[i], count)
                            self.writer.add_scalar("PRE", precision[i], count)
                            self.writer.add_scalar("FN", false_neg[i], count)
                    else:
                        print("==================")
                        print("MRE", (cur_roll_error[i] + cur_yaw_error[i] + cur_pitch_error[i]) / 3)
                        print("MTE", (cur_x_error[i] + cur_y_error[i] + cur_z_error[i]) / 3)
                        print("RRE", r_diff[i])
                        print("RTE", t_diff[i])
                        if "kitti_rgg" in DATASET:
                            print("SEE", msee[i])
                            print("RR", mrr[i])
                        if modelcfg.use_projection_mask and modelcfg.layer_mask[1]:
                            print("ACC", auc)
                        print("==================")

                    mean_roll_error.update(cur_roll_error[i])
                    mean_pitch_error.update(cur_pitch_error[i])
                    mean_yaw_error.update(cur_yaw_error[i])
                    mean_x_error.update(cur_x_error[i])
                    mean_y_error.update(cur_y_error[i])
                    mean_z_error.update(cur_z_error[i])
                    if "kitti_rgg" in DATASET:
                        mean_see.update(float(msee[i]))
                        mean_rr.update(float(mrr[i]))

                    if modelcfg.use_projection_mask and modelcfg.layer_mask[1] and not DEEP:
                        auc_total.update(auc)
                    if DEEP:
                        auc_total.update(acc[i])
                        pre_total.update(precision[i])
                        fn_total.update(false_neg[i])

                    count += 1
        if not DEBUG:
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

            self.f_write.write('MSEE= {:8f}\n'.format(mean_see.avg))
            self.f_write.write('MRR= {:8f}%\n'.format(mean_rr.avg * 100))
        # metrics = {
        #     "rot_test":ROT_TEST,
        #     "mean_roll_error":mean_roll_error.avg,
        #     "mean_pitch_error":mean_pitch_error.avg,
        #     "mean_yaw_error":mean_yaw_error.avg,
        #     "mean_x_error":mean_x_error.avg,
        #     "mean_y_error":mean_y_error.avg,
        #     "mean_z_error":mean_z_error.avg,
        #     "mean_rotate_error":cur_mean_rotation_error,
        #     "mean_trans_error":cur_mean_translation_error,
        # }

        rte_mean, rte_std, rre_mean, rre_std = evaluator.evalSeq()
        # metrics["rte"] = {"mean":rte_mean,"std":rte_std}
        # metrics["rre"] = {"mean":rre_mean,"std":rre_std}
        if not DEBUG:
            self.f_write.write('RTE %.2f +- %.2f, RRE %.2f +- %.2f\n' % (rte_mean, rte_std, rre_mean, rre_std))

            if THRESH:
                self.f_write.write('Rigistration Recall %.3f%%\n' % (evaluator.get_recall() * 100))

            if modelcfg.use_projection_mask and modelcfg.layer_mask[1] and not DEEP:
                self.f_write.write('mean_l3_mask_auc= {:3f}\n'.format(auc_total.avg))
            if DEEP:
                self.f_write.write('mean_acc= {:3f}\n'.format(auc_total.avg))
                self.f_write.write('mean_pre= {:3f}\n'.format(pre_total.avg))
                self.f_write.write('mean_fn= {:3f}\n'.format(fn_total.avg))
            self.f_write.flush()
        else:
            print('RTE %.2f +- %.2f, RRE %.2f +- %.2f\n' % (rte_mean, rte_std, rre_mean, rre_std))
            if modelcfg.use_projection_mask and modelcfg.layer_mask[1] and not DEEP:
                print('mean_l3_mask_auc= {:3f}\n'.format(auc_total.avg))

        if modelcfg.use_projection_mask and modelcfg.layer_mask[1]:
            if OUT:
                with open(self.out_path, 'wb') as f:
                    pickle.dump(outliers, f)
        evaluator.save_metric(self.metric_path)

        # self.writer.add_custom_scalars(metrics)

    def vis(self, data_valid, pred_extrinsic, gt_extrinsic, n_iter):
        """visualize the first image in the batch"""
        init_extrinsic = data_valid['init_extrinsic'][0].detach().cpu().numpy()
        img = data_valid['resize_rgb'][0].detach().cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        pcl = data_valid['raw_lidar'][0].detach().cpu().numpy()
        intrinsic = data_valid['raw_intrinsic'][0].detach().cpu().numpy()

        pred_extrinsic = pred_extrinsic[0]  # [3,4]
        gt_extrinsic = gt_extrinsic[0]

        pcl_uv, pcl_z = self.dataset.get_projected_pts(pcl, intrinsic, init_extrinsic, img.shape)
        init_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img)

        pcl_uv, pcl_z = self.dataset.get_projected_pts(pcl, intrinsic, pred_extrinsic, img.shape)

        pj_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img)

        pcl_uv, pcl_z = self.dataset.get_projected_pts(pcl, intrinsic, gt_extrinsic, img.shape)
        gt_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img)

        init_projected_img = torch.from_numpy(init_projected_img)
        pj_projected_img = torch.from_numpy(pj_projected_img)
        gt_projected_img = torch.from_numpy(gt_projected_img)

        concat_img = torch.stack([init_projected_img, pj_projected_img, gt_projected_img])

        self.writer.add_image("init_projected_img", init_projected_img, n_iter
                              , dataformats="HWC")
        self.writer.add_image("pj_projected_img", pj_projected_img, n_iter
                              , dataformats="HWC")
        self.writer.add_image("gt_projected_img", gt_projected_img, n_iter
                              , dataformats="HWC")
        self.writer.add_images("comparison_img", concat_img, n_iter
                               , dataformats="NHWC")

    def decode_path(self, path: str):
        #print("path:", path)
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
        else:
            info = None
        return info


if __name__ == '__main__':
    if FLAGS.spawn:
        torch.multiprocessing.set_start_method('spawn')
    evaluator = Evaluator()
    evaluator.validate()
