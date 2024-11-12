# !/usr/bin/env Python
# coding=utf-8
import torch
import torch.nn as nn
from torchvision import models
from pointnet_util import PointNetSetAbstraction, index_points
# from pointconv_util import PointNetSaModule
import numpy as np
import torch.nn.functional as F
import os

from src.config import I2PNetConfig as cfg
from src.modules.basicConv import Conv2d, Conv1d, createCNNs
from src.modules.pointnet2_module import SetUpconvModule
from src.modules.HRegnet import SimSVDCrossVolume, SVDCrossVolume
import src.modules.warp_utils as warp_utils
import src.utils as utils

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # make error more user-friendly


# use_bn = False

class RegNet_v2(nn.Module):
    def __init__(self, bn_decay=None, eval_info=False):
        super(RegNet_v2, self).__init__()
        self.eval_info = eval_info

        lidar_layer_points = [cfg.lidar_in_points // s
                              for s in np.cumprod(cfg.lidar_downsample_rate)]

        lidar_mlps = cfg.lidar_encoder_mlps
        self.LiDAR_lv1 = PointNetSetAbstraction(
            npoint=lidar_layer_points[0], radius=0.5, nsample=cfg.lidar_group_samples[0],
            in_channel=3 + 3, mlp=lidar_mlps[0], group_all=False)
        self.LiDAR_lv2 = PointNetSetAbstraction(
            npoint=lidar_layer_points[1], radius=0.5, nsample=cfg.lidar_group_samples[1],
            in_channel=lidar_mlps[0][-1] + 3, mlp=lidar_mlps[1], group_all=False)

        self.LiDAR_lv3 = PointNetSetAbstraction(
            npoint=lidar_layer_points[2], radius=1.0, nsample=cfg.lidar_group_samples[2],
            in_channel=lidar_mlps[1][-1] + 3, mlp=lidar_mlps[2], group_all=False)

        self.LiDAR_lv4 = PointNetSetAbstraction(
            npoint=lidar_layer_points[3], radius=2.0, nsample=cfg.lidar_group_samples[3],
            in_channel=lidar_mlps[2][-1] + 3, mlp=lidar_mlps[3], group_all=False)

        self.layer_idx = PointNetSetAbstraction(
            npoint=lidar_layer_points[3], radius=2.0, nsample=cfg.lidar_group_samples[4],
            in_channel=cfg.cost_volume_mlps[-1][-1] + 3, mlp=lidar_mlps[4], group_all=False)

        self.RGB_net1 = createCNNs(3, [16, 16, 16, 16, 32],
                                   [2, 1, 1, 1, 2])

        self.RGB_net2 = createCNNs(32, [32, 32, 32, 32, 64],
                                   [1, 1, 1, 1, 2])

        self.RGB_net3 = createCNNs(64, [64, 64, 64, 64, 128],
                                   [1, 1, 1, 1, 2])

        self.RGB_net4 = createCNNs(128, [128, 128, 128, 128, 256],
                                   [1, 1, 1, 1, 2])

        ##########################################################
        self.coarse_regist = SimSVDCrossVolume(256, 256, 16, 3, [128, 128, 64])
        self.fine_regist = SVDCrossVolume(128, 128, 32, [128, 128, 64])

        ##########################################################
        # loss learnable parameters
        self.sq = torch.nn.Parameter(torch.tensor([cfg.sq_init]), requires_grad=True)
        self.sx = torch.nn.Parameter(torch.tensor([cfg.sx_init]), requires_grad=True)

    def forward(self, rgb_img, lidar_img, H_initial, intrinsic, resize_img,
                gt_project=None, calib=None,use_gt=False):
        # resize_img = resize_img[0]
        device = rgb_img.device
        intrinsic = intrinsic.float()  # camera matrix is the same

        # H_initial = H_initial[0].reshape(3, 4)  # H_initial is not used

        B, _, h, w = rgb_img.shape  # [B,3,352,1216]
        _, N, _ = lidar_img.shape

        # image branch

        RF1 = self.RGB_net1(rgb_img)

        RF2 = self.RGB_net2(RF1)

        RF3 = self.RGB_net3(RF2)  # /16

        RF4 = self.RGB_net4(RF3)  # /32

        # img discrete index

        lidar_img = lidar_img.permute(0, 2, 1)  # [B,C,N]
        # lidar feature initial as zeros
        lidar_norm = torch.zeros(B, N, 3, device=device)
        lidar_norm = lidar_norm.permute(0, 2, 1)

        # lidar layers
        P1, LF1, group_xyz_1, fps_idx_1 = self.LiDAR_lv1(lidar_img.float(), lidar_norm.float())
        P2, LF2, group_xyz_2, fps_idx_2 = self.LiDAR_lv2(P1, LF1)
        P3, LF3, group_xyz_3, fps_idx_3 = self.LiDAR_lv3(P2, LF2)  # LF3.shape[B,C,N]=[B,64,256]
        P4, LF4, group_xyz_4, fps_idx_4 = self.LiDAR_lv4(P3, LF3)  # LF4.shape[B,C,N]=[B,128,64]


            # gt_project [B,N,2]

        if gt_project is not None:
            gt_project_l1 = index_points(gt_project, fps_idx_1)
            gt_project_l2 = index_points(gt_project_l1, fps_idx_2)
            gt_project_l3 = index_points(gt_project_l2, fps_idx_3)
            gt_project_l4 = index_points(gt_project_l3, fps_idx_4)
        else:
            gt_project_l3 = None
            gt_project_l4 = None

        if use_gt:
            pred_project_l3 = gt_project_l3
            pred_project_l4 = gt_project_l4
        else:
            pred_project_l3 = None
            pred_project_l4 = None

        # project the uv to normalized camera plane
        _, C, H, W = RF4.shape
        RF4_index = set_id_grid(RF4.permute(0, 2, 3, 1))  # [B,418,3]
        intrinsic_4 = change_intrinsic(intrinsic, RF4, rgb_img)  # K3
        # cuda not support matrix inverse
        intrinsic_4_inv = torch.inverse(intrinsic_4)
        RF4_index = torch.bmm(intrinsic_4_inv, RF4_index.permute(0, 2, 1))
        RF4_index = RF4_index.view(B, 3, H, W)

        # project to the normalization camera plane

        lidar_uv, lidar_z, LF4 = warp_utils.projection_initial(P4, None, None, None, LF4)

        lidar_uv = lidar_uv.reshape(B, -1, 3)

        # RF3 = RF3.reshape(B, C, H * W).permute(0, 2, 1)  # B,N,C

        # r: [B,3,3]
        # t: [B,3]
        R4, t4, weight_4 = self.coarse_regist(lidar_uv, LF4.permute(0, 2, 1),
                                              RF4, RF4_index,
                                              lidar_z, pred_project_l4)  # [B,256,64]=[B,N,C]

        new_P3 = torch.bmm(R4, P3) + t4.unsqueeze(-1)  # B,3,N

        # project the uv to normalized camera plane
        _, C, H, W = RF3.shape
        RF3_index = set_id_grid(RF3.permute(0, 2, 3, 1))  # [B,418,3]
        intrinsic_3 = change_intrinsic(intrinsic, RF3, rgb_img)  # K3
        # cuda not support matrix inverse
        intrinsic_3_inv = torch.inverse(intrinsic_3)
        RF3_index = torch.bmm(intrinsic_3_inv, RF3_index.permute(0, 2, 1))
        RF3_index = RF3_index.view(B, 3, H, W)

        lidar_uv, lidar_z, LF3 = warp_utils.projection_initial(new_P3, None, None, None, LF3)

        R3, t3, weight_3 = self.fine_regist(lidar_uv, LF3.permute(0, 2, 1),
                                            RF3, RF3_index,
                                            lidar_z, pred_project_l3)  # [B,256,64]=[B,N,C]

        new_R3 = torch.bmm(R3, R4)
        new_t3 = torch.bmm(R3, t4.unsqueeze(-1)).squeeze(-1) + t3

        return new_R3, new_t3, R4, t4, (weight_3, gt_project_l3), (weight_4, gt_project_l4), self.sq, self.sx


def set_id_grid(rf):
    B = rf.shape[0]
    device = rf.device
    # input: B*h*w*3
    b, h, w, _ = rf.shape
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(rf)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(rf)  # [1, H, W]
    one_range = torch.ones(1, h, w, device=device)
    pixel_coords = torch.stack((j_range, i_range, one_range), dim=1)
    pixel_coords = pixel_coords.permute(0, 2, 3, 1)
    pixel_coords = pixel_coords.reshape(1, -1, 3)
    a = pixel_coords

    return a.repeat(B, 1, 1)  # B,M,3


def change_intrinsic(intrinsic, RF, rgb_img):
    intrinsic = intrinsic.clone()  # present the change of origin intrinsic
    intrinsic[:, 0, 0] = RF.shape[3] / rgb_img.shape[3] * intrinsic[:, 0, 0]
    intrinsic[:, 0, 2] = RF.shape[3] / rgb_img.shape[3] * intrinsic[:, 0, 2]
    intrinsic[:, 1, 1] = RF.shape[2] / rgb_img.shape[2] * intrinsic[:, 1, 1]
    intrinsic[:, 1, 2] = RF.shape[2] / rgb_img.shape[2] * intrinsic[:, 1, 2]
    return intrinsic


def remove_layer(model, n):
    modules = list(model.children())[:-n]
    model = nn.Sequential(*modules)
    return model


def get_num_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
