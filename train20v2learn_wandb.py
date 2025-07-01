import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
from importlib import import_module
import shutil
import numpy as np

from avgmeter import AverageMeter
# import src.modellearn as mod
from src.deterministic import set_seed, seed_worker
import src.utils as utils
from compute_loss import Get_loss, GetProjectionLoss, GetPointwiseReProjectionLoss
import torch.nn.functional as F
#from src.config import I2PNetConfig as modelcfg
from metric import getExtrinsic, RteRreEval, calibration_error_batch, eval_acc, eval_msee, eval_mrr
from monitor.base import UniWriter

# arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--test_num', default=887, type=int, help='The number of used data to validate [default: -1] '
                                                              'negative value means all')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 4]')
parser.add_argument('--save_rate', type=int, default=20, help='save report rate [default: 20]')
parser.add_argument('--validate_rate', type=int, default=-1, help='validation rate (after n reports) [default: 80]')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='training initial learning rate [default: 0.001]')
parser.add_argument("--network", default="modellearn", type=str,
                    help="the network to train [default: modellearn]")
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers [default: 8]')
parser.add_argument('--dataset', type=str, default="kd_corr_snr",
                    choices=["kitti",  "kd_cmr_snr",
                             "nus_cmr_snr", 
                             ],
                    help="choose which dataset to train [default: kitti]")
parser.add_argument('--train_target', default='all', choices=['all', 'class', 'regist'], type=str,
                    help="choose what to train")
parser.add_argument('--continue_train_ckpt', type=str, default=None, help="The directory of the ckpt of classification")
parser.add_argument('--debug', action='store_true')
parser.add_argument('--clip', type=float, default=-1.)
parser.add_argument('--spawn', action='store_true')
parser.add_argument('--modelcfg', default="config",type=str)
FLAGS = parser.parse_args()

MODELCFG = FLAGS.modelcfg
EPOCHS = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
WORKERS = FLAGS.num_workers
LOGDIR = FLAGS.log_dir
LEARNING_RATE = FLAGS.lr
CKPT = FLAGS.checkpoint_path
TEST_NUM = FLAGS.test_num
SAVE_RATE = FLAGS.save_rate
VALIDATION_RATE = FLAGS.validate_rate
NETWORK = FLAGS.network
DATASET = FLAGS.dataset
TRAINMODE = FLAGS.train_target
CCKPT = FLAGS.continue_train_ckpt
DEBUG = FLAGS.debug
CLIP = FLAGS.clip


modelcfg = import_module("src.{0}".format(MODELCFG)).I2PNetConfig

if DATASET == "kitti":
    from src.dataset import Kitti_Dataset as traindataset
    from src.dataset_params import KITTI_ONLINE_CALIB as cfg

    dataset_file = "dataset"
elif 'kd' in DATASET:
    dataset_file = 'kitti_odometry_cmr'
    DA = import_module("src.{0}".format(dataset_file))
    traindataset = DA.Kitti_Odometry_Dataset
    # from src.kitti_odometry_corr import Kitti_Odometry_Dataset as traindataset
    from src.dataset_params import KITTI_ODOMETRY as cfg
elif "nus" in DATASET:
    dataset_file = 'nuscenes_loader_cmr'
    DA = import_module("src.{0}".format(dataset_file))
    traindataset = DA.nuScenesLoader
    from src.dataset_params import NUSCENES as cfg

# Setup
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
set_seed(0)  # deterministic

mod = import_module("src.{0}".format(NETWORK))
RegNet_v2 = mod.RegNet_v2


class Trainer(object):
    def __init__(self):
        if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR, exist_ok=True)
        # Config
        RUN_ID = 5

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        # print(device)

        # logs

        SAVE_PATH = str(Path(LOGDIR) / 'checkpoints_new' /
                        'run_{:05d}'.format(RUN_ID))
        LOG_PATH = str(Path(LOGDIR) / 'tensorboard_new' /
                       'run_{:05d}'.format(RUN_ID))

        self.f_write = open(str(Path(LOGDIR) / "log.txt"), "a+")
        self.f_log_rotate = open(str(Path(LOGDIR) / "log_rotate.txt"), "a+")
        self.f_log_trans = open(str(Path(LOGDIR) / "log_trans.txt"), "a+")

        Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
        Path(LOG_PATH).mkdir(parents=True, exist_ok=True)

        self.save_path = SAVE_PATH
        self.log_path = LOG_PATH

        if CKPT is not None:
            if os.path.exists(os.path.join(LOGDIR, "config.yaml")):
                origin_config = open(os.path.join(LOGDIR, "config.yaml"), mode="r", encoding="utf-8")
                origin_cfg = yaml.load(origin_config, yaml.SafeLoader)
                origin_config.close()
                now_config = vars(FLAGS)
                for key, v in origin_cfg.items():
                    if key not in ["checkpoint_path", "batch_size", "gpu", "num_workers"]:
                        assert v == now_config[key], f"config error in {key} origin {v} now {now_config[key]}"
            file = open(os.path.join(LOGDIR, "config.yaml"), mode="w", encoding="utf-8")
            yaml.dump(vars(FLAGS), file)
            file.close()
            # still copy the necessary files
            shutil.copy("src/{0}.py".format(dataset_file), "{0}/dataset.py".format(LOGDIR))
            shutil.copy("src/{0}.py".format(NETWORK), "{0}/network.py".format(LOGDIR))
            shutil.copy("src/{0}.py".format(MODELCFG), "{0}/config.py".format(LOGDIR))
        else:
            shutil.copy("src/{0}.py".format(dataset_file), "{0}/dataset.py".format(LOGDIR))
            shutil.copy("src/{0}.py".format(NETWORK), "{0}/network.py".format(LOGDIR))
            shutil.copy("src/{0}.py".format(MODELCFG), "{0}/config.py".format(LOGDIR))
            file = open(os.path.join(LOGDIR, "config.yaml"), mode="w", encoding="utf-8")
            yaml.dump(vars(FLAGS), file)
            file.close()

        # Hyperparameters
        # LEARNING_RATE = 3e-4

        # training data

        # for deterministic
        g = torch.Generator()
        g.manual_seed(0)

        dataset = traindataset(cfg.dataset_params, use_raw=modelcfg.raw_feat_point)
        self.train_loader = DataLoader(dataset,
                                       batch_size=BATCH_SIZE,
                                       num_workers=WORKERS,
                                       pin_memory=True,
                                       worker_init_fn=seed_worker,
                                       generator=g,
                                       shuffle=True,
                                       drop_last=True)
        # self.dataset = dataset

        # validation data
   
        dataset_valid3 = traindataset(cfg.dataset_params_valid3, use_raw=modelcfg.raw_feat_point)
        self.test_loader3 = DataLoader(dataset_valid3,
                                    batch_size=BATCH_SIZE,
                                    num_workers=WORKERS,
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    shuffle=False,
                                    drop_last=False)

        # Model
        self.model = RegNet_v2(cfg=modelcfg)
        self.criterion = nn.MSELoss()

        if CCKPT is not None:
            if "ex" in DATASET:
                CCKPT_PATH = str(Path(CCKPT) / 'checkpoints_new' /
                                 'run_{:05d}'.format(RUN_ID) / "model_msee_best.pt")
                cckpt = torch.load(CCKPT_PATH)
                self.model.load_state_dict(cckpt["model_state_dict"])
            else:
                CCKPT_PATH = str(Path(CCKPT) / 'checkpoints_new' /
                                 'run_{:05d}'.format(RUN_ID) / "model_rotation_best.pt")
                cckpt = torch.load(CCKPT_PATH)
                self.model.load_state_dict(cckpt["model_state_dict"])
                # CCKPT_PATH = str(Path(CCKPT) / 'checkpoints_new' /
                #                  'run_{:05d}'.format(RUN_ID) / "ckpt.pt")
                # cckpt = torch.load(CCKPT_PATH)
                # self.model.load_state_dict(cckpt["model_state_dict"])

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=LEARNING_RATE,
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          weight_decay=0.0001)

        # delay_rate = (0.99) ** (1 / SAVE_RATE / VALIDATION_RATE)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99)

        self.model.to(self.device)

        self.epoch = 0
        self.n_iter = 0
        self.best_rotation_error = float("inf")
        self.best_transition_error = float("inf")

        self.af_best_rotation_error = float("inf")
        self.af_best_transition_error = float("inf")
        self.ni_best_rotation_error = float("inf")
        self.ni_best_transition_error = float("inf")

        self.best_acc = 0.

        self.af_best_acc = 0.
        self.ni_best_acc = 0.
        if "kitti_rgg" in DATASET:
            self.best_msee = float("inf")

        if CKPT is not None:
            ckpt = torch.load(CKPT)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.epoch = ckpt["epoch"] + 1
            self.n_iter = ckpt["n_iter"]
            self.best_rotation_error = ckpt["best_rotation_error"]
            self.best_transition_error = ckpt["best_transition_error"]
            self.best_acc = ckpt["best_acc"]
            if "kitti_rgg" in DATASET:
                self.best_msee = ckpt["best_msee"]

        # self.writer = SummaryWriter(log_dir=self.log_path)
        self.writer = UniWriter(log_dir=self.log_path, project="i2pnet",
                                wandb_name=LOGDIR) if not DEBUG else \
            SummaryWriter(log_dir=self.log_path)
        if not DEBUG:
            self.writer.config(FLAGS)
            self.writer.define_step(["loss", "metric"])

        # add hyperparameter to tensorboard
        # self.writer.add_custom_scalars(vars(FLAGS))
        # add model graph
        # self.writer.add_graph(self.model)

        self.training_params = {
            'dataset_params': cfg.dataset_params,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
        }

    def __del__(self):
        self.f_write.close()
        self.f_log_trans.close()
        self.f_log_rotate.close()

    def save_ckpt(self, n_iter, path="ckpt.pt"):
        model_save = {
            'epoch': self.epoch,
            "n_iter": n_iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_params': self.training_params,
            "best_rotation_error": self.best_rotation_error,
            "best_transition_error": self.best_transition_error,
            "best_acc": self.best_acc
        }
        if "kitti_rgg" in DATASET:
            model_save["best_msee"] = self.best_msee
        torch.save(model_save, os.path.join(self.save_path, path))

    def save_model(self, rotate, trans, path):
        model_save = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'training_params': self.training_params,
            "best_rotation_error": self.best_rotation_error,
            "best_transition_error": self.best_transition_error,
            "best_acc": self.best_acc,
            "cur_rotation_error": rotate,
            "cur_transition_error": trans,
        }
        if "kitti_rgg" in DATASET:
            model_save["best_msee"] = self.best_msee
        torch.save(model_save, os.path.join(self.save_path, path))

    def save_model_af(self, rotate, trans, path):
        model_save = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'training_params': self.training_params,
            "best_rotation_error": self.af_best_rotation_error,
            "best_transition_error": self.af_best_transition_error,
            "best_acc": self.af_best_acc,
            "cur_rotation_error": rotate,
            "cur_transition_error": trans,
        }
        torch.save(model_save, os.path.join(self.save_path, path))

    def save_model_ni(self, rotate, trans, path):
        model_save = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'training_params': self.training_params,
            "best_rotation_error": self.ni_best_rotation_error,
            "best_transition_error": self.ni_best_transition_error,
            "best_acc": self.ni_best_acc,
            "cur_rotation_error": rotate,
            "cur_transition_error": trans,
        }
        torch.save(model_save, os.path.join(self.save_path, path))

    def validate(self, n_iter):
        self.model.eval()
        mean_roll_error = AverageMeter()
        mean_pitch_error = AverageMeter()
        mean_yaw_error = AverageMeter()
        mean_x_error = AverageMeter()
        mean_y_error = AverageMeter()
        mean_z_error = AverageMeter()
        acc = AverageMeter()
        mean_msee = AverageMeter()
        mean_mrr = AverageMeter()

        evaluator = None
        if "kd" in DATASET or "nus" in DATASET:
            evaluator = RteRreEval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.no_grad():
            for count, data_valid in tqdm(enumerate(self.test_loader3), total=len(self.test_loader3)):
                if TEST_NUM >= 0 and count == TEST_NUM:
                    break
                resize_img = data_valid['resize_img'].to(self.device)
                rgb_img = data_valid['rgb'].to(self.device)
                lidar_img = data_valid['lidar'].to(self.device)  # load lidar
                H_initial = data_valid['init_extrinsic'].to(self.device)
                intrinsic = data_valid['init_intrinsic'].to(self.device)
                decalib_quat_real = data_valid['decalib_real_gt'].to(self.device)
                decalib_quat_dual = data_valid['decalib_dual_gt'].to(self.device)
                lidar_feats = None
                if "corr_snr" in DATASET or "snr" in DATASET or DATASET=="kd_corr_nolidar":
                    lidar_feats = data_valid["lidar_feats"].float().to(self.device)
                gt_project = None
                if modelcfg.ground_truth_projection_mask_eval:
                    gt_project = utils.get_projection_gt(lidar_img, intrinsic, rgb_img.shape[2:], decalib_quat_real,
                                                         decalib_quat_dual)
                    gt_project = F.one_hot(gt_project, num_classes=2)

                calib = None
                if modelcfg.efgh:
                    calib = data_valid['calib'].to(self.device)
                if modelcfg.raw_feat_point:
                    lidar_img_raw = data_valid['raw_point_xyz'].to(self.device)
                else:
                    lidar_img_raw = None
                out3, out4, pm3, pm4, sx, sq = self.model(rgb_img, lidar_img,
                                                          H_initial, intrinsic, resize_img,
                                                          gt_project, calib, lidar_feats, cfg=modelcfg, lidar_img_raw=lidar_img_raw)

                pred_extrinsic, gt_extrinsic, pred_ex_raw, gt_ex_raw = getExtrinsic(out3, data_valid, out_raw=True)

                if "kitti_rgg" in DATASET:
                    gt_se3 = data_valid['decalib_se3'].cpu().numpy()
                    msee = eval_msee(out3, gt_se3)
                    mrr = eval_mrr(msee, gt_se3)

                acclist = np.zeros((out3.shape[0],))
                if modelcfg.use_projection_mask:
                    if pm3 is not None and modelcfg.layer_mask[1]:
                        acclist = eval_acc(pm3, intrinsic, rgb_img.shape[2:], decalib_quat_real, decalib_quat_dual,
                                           modelcfg.mask_sigmoid)
                    elif pm4 is not None and modelcfg.layer_mask[0]:
                        acclist = eval_acc(pm4, intrinsic, rgb_img.shape[2:], decalib_quat_real, decalib_quat_dual,
                                           modelcfg.mask_sigmoid)
                elif pm3 is not None and modelcfg.one_head_mask:
                    acclist = eval_acc(pm3, intrinsic, rgb_img.shape[2:], decalib_quat_real, decalib_quat_dual)

                cur_roll_error, cur_pitch_error, cur_yaw_error, \
                cur_x_error, cur_y_error, cur_z_error = calibration_error_batch(pred_ex_raw, gt_ex_raw)

                if "kd" in DATASET or "nus" in DATASET:
                    evaluator.addBatch(pred_ex_raw, gt_ex_raw)

                for i in range(len(cur_x_error)):
                    mean_roll_error.update(cur_roll_error[i])
                    mean_pitch_error.update(cur_pitch_error[i])
                    mean_yaw_error.update(cur_yaw_error[i])
                    mean_x_error.update(cur_x_error[i])
                    mean_y_error.update(cur_y_error[i])
                    mean_z_error.update(cur_z_error[i])
                    acc.update(acclist[i])
                    
                    if "kitti_rgg" in DATASET:
                        mean_msee.update(float(msee[i]))
                        mean_mrr.update(float(mrr[i]))

        if TEST_NUM == 0:
            return
        cur_mean_rotation_error = (mean_roll_error.avg + mean_pitch_error.avg + mean_yaw_error.avg) / 3
        cur_mean_transition_error = (mean_x_error.avg + mean_y_error.avg + mean_z_error.avg) / 3
        cur_acc = acc.avg

        self.f_write.write("cur_mean_rotation_error: {}\n".format(cur_mean_rotation_error))
        self.f_write.write("cur_mean_transition_error: {}\n".format(cur_mean_transition_error))
        self.f_write.write("cur_mean_l3_acc: {}\n".format(acc.avg))

        self.writer.add_scalar("metric/MRE", cur_mean_rotation_error, n_iter)
        self.writer.add_scalar("metric/MTE", cur_mean_transition_error, n_iter)
        self.writer.add_scalar("metric/l3_acc", acc.avg, n_iter)

        rotate_report = f"Epoch {self.epoch}| Iter {n_iter}|cur_mean_rotation_error {cur_mean_rotation_error:3f}"
        trans_report = f"Epoch {self.epoch}| Iter {n_iter}|cur_mean_transition_error {cur_mean_transition_error:3f}"

        if "kitti_rgg" in DATASET:
            self.f_write.write("cur_MSEE: {} | cur_MRR: {}%\n".format(mean_msee.avg, mean_mrr.avg * 100))
            self.writer.add_scalar("metric/MSEE", mean_msee.avg, n_iter)
            self.writer.add_scalar("metric/MRR", mean_mrr.avg * 100, n_iter)
            rotate_report += f"|cur_MSEE {mean_msee.avg:.4f}|cur_MRR {mean_mrr.avg * 100}%"
        if "kd" in DATASET or "nus" in DATASET:
            rte_mean, rte_std, rre_mean, rre_std = evaluator.evalSeq()
            self.f_write.write('RTE %.2f +- %.2f, RRE %.2f +- %.2f\n' % (rte_mean, rte_std, rre_mean, rre_std))
            self.writer.add_scalar("metric/RTE_mean", rte_mean, n_iter)
            self.writer.add_scalar("metric/RTE_std", rte_std, n_iter)
            self.writer.add_scalar("metric/RRE_mean", rre_mean, n_iter)
            self.writer.add_scalar("metric/RRE_std", rre_std, n_iter)

            rotate_report += f"|cur_RRE {rre_mean:.2f}+-{rre_std:.2f}"
            trans_report += f"|cur_RTE {rte_mean:.2f}+-{rte_std:.2f}"

        self.f_log_rotate.write(rotate_report + '\n')
        self.f_log_trans.write(trans_report + '\n')

        self.f_write.flush()
        self.f_log_rotate.flush()
        self.f_log_trans.flush()

        if TRAINMODE in ['all', 'regist']:
            if "kitti_rgg" in DATASET:
                if mean_msee.avg < self.best_msee:
                    self.best_msee = mean_msee.avg
                    self.save_model(cur_mean_rotation_error, cur_mean_transition_error, 'model_msee_best.pt')
            else:
                if cur_mean_rotation_error < self.best_rotation_error:
                    self.best_rotation_error = cur_mean_rotation_error
                    self.save_model(cur_mean_rotation_error, cur_mean_transition_error, 'model_rotation_best.pt')
                if cur_mean_transition_error < self.best_transition_error:
                    self.best_transition_error = cur_mean_transition_error
                    self.save_model(cur_mean_rotation_error, cur_mean_transition_error, 'model_transition_best.pt')
                if cur_acc > self.best_acc:
                    self.best_acc = cur_acc

        elif TRAINMODE == "class":
            if cur_acc > self.best_acc:
                self.best_acc = cur_acc
                self.save_model(0, 0, 'model_acc_best.pt')
        else:
            raise NotImplementedError
        
    def validate_old_town_a(self, n_iter):
        self.model.eval()
        mean_roll_error = AverageMeter()
        mean_pitch_error = AverageMeter()
        mean_yaw_error = AverageMeter()
        mean_x_error = AverageMeter()
        mean_y_error = AverageMeter()
        mean_z_error = AverageMeter()
        acc = AverageMeter()
        mean_msee = AverageMeter()
        mean_mrr = AverageMeter()

        evaluator = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.no_grad():
            for count, data_valid in tqdm(enumerate(self.test_loader3_a), total=len(self.test_loader3_a)):
                if TEST_NUM >= 0 and count == TEST_NUM:
                    break
                resize_img = data_valid['resize_img'].to(self.device)
                rgb_img = data_valid['rgb'].to(self.device)
                lidar_img = data_valid['lidar'].to(self.device)  # load lidar
                H_initial = data_valid['init_extrinsic'].to(self.device)
                intrinsic = data_valid['init_intrinsic'].to(self.device)
                decalib_quat_real = data_valid['decalib_real_gt'].to(self.device)
                decalib_quat_dual = data_valid['decalib_dual_gt'].to(self.device)
                lidar_feats = None
                if "corr_snr" in DATASET or "snr" in DATASET or DATASET=="kd_corr_nolidar":
                    lidar_feats = data_valid["lidar_feats"].float().to(self.device)
                gt_project = None
                if modelcfg.ground_truth_projection_mask_eval:
                    gt_project = utils.get_projection_gt(lidar_img, intrinsic, rgb_img.shape[2:], decalib_quat_real,
                                                         decalib_quat_dual)
                    gt_project = F.one_hot(gt_project, num_classes=2)

                calib = None
                if modelcfg.efgh:
                    calib = data_valid['calib'].to(self.device)

                if modelcfg.raw_feat_point:
                    lidar_img_raw = data_valid['raw_point_xyz'].to(self.device)
                else:
                    lidar_img_raw = None

                out3, out4, pm3, pm4, sx, sq = self.model(rgb_img, lidar_img,
                                                          H_initial, intrinsic, resize_img,
                                                          gt_project, calib, lidar_feats, cfg=modelcfg, lidar_img_raw=lidar_img_raw)

                pred_extrinsic, gt_extrinsic, pred_ex_raw, gt_ex_raw = getExtrinsic(out3, data_valid, out_raw=True)

                acclist = np.zeros((out3.shape[0],))
                if modelcfg.use_projection_mask:
                    if pm3 is not None and modelcfg.layer_mask[1]:
                        acclist = eval_acc(pm3, intrinsic, rgb_img.shape[2:], decalib_quat_real, decalib_quat_dual,
                                           modelcfg.mask_sigmoid)
                    elif pm4 is not None and modelcfg.layer_mask[0]:
                        acclist = eval_acc(pm4, intrinsic, rgb_img.shape[2:], decalib_quat_real, decalib_quat_dual,
                                           modelcfg.mask_sigmoid)
                elif pm3 is not None and modelcfg.one_head_mask:
                    acclist = eval_acc(pm3, intrinsic, rgb_img.shape[2:], decalib_quat_real, decalib_quat_dual)

                cur_roll_error, cur_pitch_error, cur_yaw_error, \
                cur_x_error, cur_y_error, cur_z_error = calibration_error_batch(pred_ex_raw, gt_ex_raw)

                for i in range(len(cur_x_error)):
                    mean_roll_error.update(cur_roll_error[i])
                    mean_pitch_error.update(cur_pitch_error[i])
                    mean_yaw_error.update(cur_yaw_error[i])
                    mean_x_error.update(cur_x_error[i])
                    mean_y_error.update(cur_y_error[i])
                    mean_z_error.update(cur_z_error[i])
                    acc.update(acclist[i])

        if TEST_NUM == 0:
            return
        cur_mean_rotation_error = (mean_roll_error.avg + mean_pitch_error.avg + mean_yaw_error.avg) / 3
        cur_mean_transition_error = (mean_x_error.avg + mean_y_error.avg + mean_z_error.avg) / 3
        cur_acc = acc.avg

        self.f_write.write("af_cur_mean_rotation_error: {}\n".format(cur_mean_rotation_error))
        self.f_write.write("af_cur_mean_transition_error: {}\n".format(cur_mean_transition_error))
        self.f_write.write("af_cur_mean_l3_acc: {}\n".format(acc.avg))

        self.writer.add_scalar("metric/af_MRE", cur_mean_rotation_error, n_iter)
        self.writer.add_scalar("metric/af_MTE", cur_mean_transition_error, n_iter)
        self.writer.add_scalar("metric/af_l3_acc", acc.avg, n_iter)

        rotate_report = f"Epoch {self.epoch}| Iter {n_iter}|af_cur_mean_rotation_error {cur_mean_rotation_error:3f}"
        trans_report = f"Epoch {self.epoch}| Iter {n_iter}|af_cur_mean_transition_error {cur_mean_transition_error:3f}"

        self.f_log_rotate.write(rotate_report + '\n')
        self.f_log_trans.write(trans_report + '\n')

        self.f_write.flush()
        self.f_log_rotate.flush()
        self.f_log_trans.flush()

        if TRAINMODE in ['all', 'regist']:
            if cur_mean_rotation_error < self.af_best_rotation_error:
                self.af_best_rotation_error = cur_mean_rotation_error
                self.save_model_af(cur_mean_rotation_error, cur_mean_transition_error, 'af_model_rotation_best.pt')
            if cur_mean_transition_error < self.af_best_transition_error:
                self.af_best_transition_error = cur_mean_transition_error
                self.save_model_af(cur_mean_rotation_error, cur_mean_transition_error, 'af_model_transition_best.pt')
            if cur_acc > self.af_best_acc:
                self.af_best_acc = cur_acc

        elif TRAINMODE == "class":
            if cur_acc > self.af_best_acc:
                self.af_best_acc = cur_acc
                self.save_model_af(0, 0, 'af_model_acc_best.pt')
        else:
            raise NotImplementedError
        
    def validate_old_town_n(self, n_iter):
        self.model.eval()
        mean_roll_error = AverageMeter()
        mean_pitch_error = AverageMeter()
        mean_yaw_error = AverageMeter()
        mean_x_error = AverageMeter()
        mean_y_error = AverageMeter()
        mean_z_error = AverageMeter()
        acc = AverageMeter()
        mean_msee = AverageMeter()
        mean_mrr = AverageMeter()

        evaluator = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.no_grad():
            for count, data_valid in tqdm(enumerate(self.test_loader3_n), total=len(self.test_loader3_n)):
                if TEST_NUM >= 0 and count == TEST_NUM:
                    break
                resize_img = data_valid['resize_img'].to(self.device)
                rgb_img = data_valid['rgb'].to(self.device)
                lidar_img = data_valid['lidar'].to(self.device)  # load lidar
                H_initial = data_valid['init_extrinsic'].to(self.device)
                intrinsic = data_valid['init_intrinsic'].to(self.device)
                decalib_quat_real = data_valid['decalib_real_gt'].to(self.device)
                decalib_quat_dual = data_valid['decalib_dual_gt'].to(self.device)
                lidar_feats = None
                if "corr_snr" in DATASET or "snr" in DATASET or DATASET=="kd_corr_nolidar":
                    lidar_feats = data_valid["lidar_feats"].float().to(self.device)
                gt_project = None
                if modelcfg.ground_truth_projection_mask_eval:
                    gt_project = utils.get_projection_gt(lidar_img, intrinsic, rgb_img.shape[2:], decalib_quat_real,
                                                         decalib_quat_dual)
                    gt_project = F.one_hot(gt_project, num_classes=2)

                calib = None
                if modelcfg.efgh:
                    calib = data_valid['calib'].to(self.device)

                if modelcfg.raw_feat_point:
                    lidar_img_raw = data_valid['raw_point_xyz'].to(self.device)
                else:
                    lidar_img_raw = None

                out3, out4, pm3, pm4, sx, sq = self.model(rgb_img, lidar_img,
                                                          H_initial, intrinsic, resize_img,
                                                          gt_project, calib, lidar_feats, cfg=modelcfg, lidar_img_raw=lidar_img_raw)

                pred_extrinsic, gt_extrinsic, pred_ex_raw, gt_ex_raw = getExtrinsic(out3, data_valid, out_raw=True)

                acclist = np.zeros((out3.shape[0],))
                if modelcfg.use_projection_mask:
                    if pm3 is not None and modelcfg.layer_mask[1]:
                        acclist = eval_acc(pm3, intrinsic, rgb_img.shape[2:], decalib_quat_real, decalib_quat_dual,
                                           modelcfg.mask_sigmoid)
                    elif pm4 is not None and modelcfg.layer_mask[0]:
                        acclist = eval_acc(pm4, intrinsic, rgb_img.shape[2:], decalib_quat_real, decalib_quat_dual,
                                           modelcfg.mask_sigmoid)
                elif pm3 is not None and modelcfg.one_head_mask:
                    acclist = eval_acc(pm3, intrinsic, rgb_img.shape[2:], decalib_quat_real, decalib_quat_dual)

                cur_roll_error, cur_pitch_error, cur_yaw_error, \
                cur_x_error, cur_y_error, cur_z_error = calibration_error_batch(pred_ex_raw, gt_ex_raw)

                for i in range(len(cur_x_error)):
                    mean_roll_error.update(cur_roll_error[i])
                    mean_pitch_error.update(cur_pitch_error[i])
                    mean_yaw_error.update(cur_yaw_error[i])
                    mean_x_error.update(cur_x_error[i])
                    mean_y_error.update(cur_y_error[i])
                    mean_z_error.update(cur_z_error[i])
                    acc.update(acclist[i])

        if TEST_NUM == 0:
            return
        cur_mean_rotation_error = (mean_roll_error.avg + mean_pitch_error.avg + mean_yaw_error.avg) / 3
        cur_mean_transition_error = (mean_x_error.avg + mean_y_error.avg + mean_z_error.avg) / 3
        cur_acc = acc.avg

        self.f_write.write("ni_cur_mean_rotation_error: {}\n".format(cur_mean_rotation_error))
        self.f_write.write("ni_cur_mean_transition_error: {}\n".format(cur_mean_transition_error))
        self.f_write.write("ni_cur_mean_l3_acc: {}\n".format(acc.avg))

        self.writer.add_scalar("metric/ni_MRE", cur_mean_rotation_error, n_iter)
        self.writer.add_scalar("metric/ni_MTE", cur_mean_transition_error, n_iter)
        self.writer.add_scalar("metric/ni_l3_acc", acc.avg, n_iter)

        rotate_report = f"Epoch {self.epoch}| Iter {n_iter}|ni_cur_mean_rotation_error {cur_mean_rotation_error:3f}"
        trans_report = f"Epoch {self.epoch}| Iter {n_iter}|ni_cur_mean_transition_error {cur_mean_transition_error:3f}"

        self.f_log_rotate.write(rotate_report + '\n')
        self.f_log_trans.write(trans_report + '\n')

        self.f_write.flush()
        self.f_log_rotate.flush()
        self.f_log_trans.flush()

        if TRAINMODE in ['all', 'regist']:
            if cur_mean_rotation_error < self.ni_best_rotation_error:
                self.ni_best_rotation_error = cur_mean_rotation_error
                self.save_model_ni(cur_mean_rotation_error, cur_mean_transition_error, 'ni_model_rotation_best.pt')
            if cur_mean_transition_error < self.ni_best_transition_error:
                self.ni_best_transition_error = cur_mean_transition_error
                self.save_model_ni(cur_mean_rotation_error, cur_mean_transition_error, 'ni_model_transition_best.pt')
            if cur_acc > self.ni_best_acc:
                self.ni_best_acc = cur_acc

        elif TRAINMODE == "class":
            if cur_acc > self.ni_best_acc:
                self.ni_best_acc = cur_acc
                self.save_model_ni(0, 0, 'ni_model_acc_best.pt')
        else:
            raise NotImplementedError

    def train(self):
        # LOG_RATE = 1
        # QUAT_FACTOR = 1
        valid_flag = False
        n_iter = self.n_iter
        start_epoch = self.epoch
        for epoch in range(start_epoch, EPOCHS):
            self.epoch = epoch
            running_loss = AverageMeter()
            running_real_loss = AverageMeter()
            running_dual_loss = AverageMeter()
            running_p_loss = AverageMeter()

            # running_layer1_loss_real = 0.0
            # running_layer1_loss_dual = 0.0
            for i, data in enumerate(tqdm(self.train_loader, total=len(self.train_loader))):
                self.model.train()
                resize_img = data['resize_img'].to(self.device)
                rgb_img = data['rgb'].to(self.device)
                # depth_img = data['depth'].cuda()
                decalib_quat_real = data['decalib_real_gt'].to(self.device)
                decalib_quat_dual = data['decalib_dual_gt'].to(self.device)
                lidar_img = data['lidar'].to(self.device)  # load lidar
                H_initial = data['init_extrinsic'].to(self.device)
                intrinsic = data['init_intrinsic'].to(self.device)
                lidar_feats = None
                if "corr_snr" in DATASET or "snr" in DATASET or DATASET=="kd_corr_nolidar":
                    lidar_feats = data["lidar_feats"].float().to(self.device)

                gt_project = None
                if modelcfg.ground_truth_projection_mask:
                    gt_project = utils.get_projection_gt(lidar_img, intrinsic, rgb_img.shape[2:], decalib_quat_real,
                                                         decalib_quat_dual, cfg=modelcfg)
                    gt_project = F.one_hot(gt_project, num_classes=2)

                calib = None
                assert not (modelcfg.efgh and (modelcfg.use_projection_mask or modelcfg.ground_truth_projection_mask
                                               or modelcfg.ground_truth_projection_mask_eval)), \
                    "TODO: efgh and projection_mask exist meanwhile"
                if modelcfg.efgh:
                    calib = data['calib'].to(self.device)

                if modelcfg.raw_feat_point:
                    lidar_img_raw = data['raw_point_xyz'].to(self.device)
                else:
                    lidar_img_raw = None
                # Forward pass
                out3, out4, pm3, pm4, sx, sq = self.model(
                    rgb_img, lidar_img, H_initial, intrinsic, resize_img,
                    gt_project, calib, lidar_feats, cfg=modelcfg, lidar_img_raw=lidar_img_raw)

                # Zero optimizer
                self.optimizer.zero_grad()

                if TRAINMODE in ['regist', 'all']:
                    loss, real_loss, dual_loss = Get_loss(out3, out4, decalib_quat_real, decalib_quat_dual, sx, sq)
                if modelcfg.use_projection_mask:
                    if pm3 is not None or pm4 is not None:
                        l3_loss = GetProjectionLoss(pm3, intrinsic, (rgb_img.shape[-2], rgb_img.shape[-1]),
                                                    decalib_quat_real, decalib_quat_dual)
                        l4_loss = GetProjectionLoss(pm4, intrinsic, (rgb_img.shape[-2], rgb_img.shape[-1]),
                                                    decalib_quat_real, decalib_quat_dual)

                        if l3_loss is None:
                            p_loss = l4_loss
                        elif l4_loss is None:
                            p_loss = l3_loss
                        else:
                            p_loss = 1.6 * l4_loss + 0.8 * l3_loss

                        if TRAINMODE == "all":
                            loss = loss + 1.5 * p_loss
                        elif TRAINMODE == "regist":
                            pass
                        elif TRAINMODE == "class":
                            loss = p_loss
                        running_p_loss.update(p_loss.item())
                elif modelcfg.one_head_mask:
                    if pm3 is not None:
                        p_loss = GetProjectionLoss(pm3, intrinsic, (rgb_img.shape[-2], rgb_img.shape[-1]),
                                                   decalib_quat_real, decalib_quat_dual)
                        if TRAINMODE == "all":
                            loss = loss + 1.5 * p_loss
                        elif TRAINMODE == "regist":
                            pass
                        elif TRAINMODE == "class":
                            loss = p_loss

                        running_p_loss.update(p_loss.item())
                    else:
                        running_p_loss.update(0)
                else:
                    running_p_loss.update(0)

                # if modelcfg.pointwise_reproject_loss:
                #     point_loss = GetPointwiseReProjectionLoss(lidar_img,intrinsic,(rgb_img.shape[-2],rgb_img.shape[-1]),out3,out4,
                #                                               decalib_quat_real,decalib_quat_dual)
                #     loss += point_loss
                #     running_p_loss.update(point_loss.item())
                # else:
                #     running_p_loss.update(0)

                # print('epoch:', epoch, 'loss:', loss, 'counter:', counter, 'sx:', sx, 'sq:', sq)

                running_loss.update(loss.item())
                if TRAINMODE in ['regist', 'all']:
                    running_real_loss.update(real_loss.item())
                    running_dual_loss.update(dual_loss.item())
                else:
                    running_real_loss.update(0)
                    running_dual_loss.update(0)
                # Backward pass
                loss.backward()

                # GRAD_CLIP
                if CLIP > 0.:
                    nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
                # Optimize
                self.optimizer.step()

                if running_loss.count == SAVE_RATE:
                    n_iter = n_iter + 1
                    self.f_write.write(
                        'Epoch: {:5d} | Batch: {:5d} | Loss: {:03f}| realLoss: {:03f} | dualLoss: {:03f} | sx: {:03f} |'
                        ' sq: {:03f} | p_loss: {:03f} | lr: {:05f}\n'
                            .format(self.epoch + 1, n_iter, running_loss.avg, running_real_loss.avg,
                                    running_dual_loss.avg, sx.item(), sq.item(), running_p_loss.avg,
                                    self.scheduler.get_last_lr()[-1]))
                    self.f_write.flush()

                    self.writer.add_scalar("loss/Loss", running_loss.avg, n_iter)
                    self.writer.add_scalar("loss/realLoss", running_real_loss.avg, n_iter)
                    self.writer.add_scalar("loss/dualLoss", running_dual_loss.avg, n_iter)
                    self.writer.add_scalar("loss/pLoss", running_p_loss.avg, n_iter)
                    self.writer.add_scalar("loss/sx", sx.item(), n_iter)
                    self.writer.add_scalar("loss/sq", sq.item(), n_iter)

                    # f_write.close()

                    running_loss.reset()
                    running_real_loss.reset()
                    running_dual_loss.reset()
                    running_p_loss.reset()

                    if VALIDATION_RATE > 0 and n_iter % VALIDATION_RATE == 0 and n_iter > 0:
                        valid_flag = True

                    if valid_flag:
                        valid_flag = False
                        # delay per valid (1600step)
                        
                        self.validate(n_iter // VALIDATION_RATE)

            self.scheduler.step()
            if VALIDATION_RATE <= 0:
                self.validate(epoch + 1)
            # every epoch save ckpt once
            self.save_ckpt(n_iter)


if __name__ == '__main__':
    if FLAGS.spawn:
        torch.multiprocessing.set_start_method('spawn')
    trainer = Trainer()
    trainer.train()
