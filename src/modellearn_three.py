# !/usr/bin/env Python
# coding=utf-8
import torch
import torch.nn as nn
from torchvision import models
from pointnet_util import PointNetSetAbstraction
# from pointconv_util import PointNetSaModule
from fusion_net import fusion_module_C
import numpy as np
import torch.nn.functional as F
import os

from src.modules.basicConv import Conv2d, Conv1d,createCNNs
from src.modules.pointnet2_module import SetUpconvModule
from src.modules.MainModules import CostVolume, FlowPredictor
import src.modules.warp_utils as warp_utils

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # make error more user-friendly





def change_intrinsic(intrinsic, RF, rgb_img):
    intrinsic[0][0] = RF.shape[3] / rgb_img.shape[3] * intrinsic[0][0]
    intrinsic[0][2] = RF.shape[3] / rgb_img.shape[3] * intrinsic[0][2]
    intrinsic[1][1] = RF.shape[2] / rgb_img.shape[2] * intrinsic[1][1]
    intrinsic[1][2] = RF.shape[2] / rgb_img.shape[2] * intrinsic[1][2]
    return intrinsic


def set_id_grid(rf):
    B = rf.shape[0]
    device = rf.device
    # input: B*h*w*3
    b, h, w, _ = rf.shape
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(rf)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(rf)  # [1, H, W]
    one_range = torch.ones(1, h, w,device=device)
    pixel_coords = torch.stack((j_range, i_range, one_range), dim=1)
    pixel_coords = pixel_coords.permute(0, 2, 3, 1)
    pixel_coords = pixel_coords.reshape(1, -1, 3)
    a = pixel_coords
    for i in range(B - 1):
        a = torch.cat((a, pixel_coords), dim=0)
    return a


def conj_quat(q):
    output = q
    helper = torch.Tensor([1, -1, -1, -1]).cuda()
    output = output.mul(helper)
    output = output.reshape(-1)
    return output


def remove_layer(model, n):
    modules = list(model.children())[:-n]
    model = nn.Sequential(*modules)
    return model


def get_num_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


one_dp = False
new_dim = False

class RegNet_v2(nn.Module):
    def __init__(self, bn_decay=None,eval_info=False):
        super(RegNet_v2, self).__init__()
        self.eval_info = eval_info
        if not new_dim:
            self.LiDAR_lv1 = PointNetSetAbstraction(
                npoint=2048, radius=0.5, nsample=32, in_channel=3 + 3, mlp=[8, 16, 32], group_all=False)
        else:
            self.LiDAR_lv1 = PointNetSetAbstraction(
            npoint=2048, radius=0.5, nsample=32, in_channel=3 + 3, mlp=[16, 16, 32], group_all=False)
        self.LiDAR_lv2 = PointNetSetAbstraction(
            npoint=1024, radius=0.5, nsample=16, in_channel=32 + 3, mlp=[32, 32, 64], group_all=False)

        self.LiDAR_lv3 = PointNetSetAbstraction(
            npoint=256, radius=1.0, nsample=16, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)

        if not new_dim:
            self.LiDAR_lv4 = PointNetSetAbstraction(
                npoint=64, radius=2.0, nsample=16, in_channel=128 + 3, mlp=[128, 128, 128], group_all=False)
        else:
            self.LiDAR_lv4 = PointNetSetAbstraction(
            npoint=64, radius=2.0, nsample=16, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)

        self.layer_idx = PointNetSetAbstraction(
            npoint=64, radius=2.0, nsample=16, in_channel=64 + 3, mlp=[128, 64, 64], group_all=False)

        # self.fusion3 = fusion_module_C(256, 256, 256)
        # self.fusion2 = fusion_module_C(32, 32, 32)
        # self.fusion1 = fusion_module_C(16, 16, 16)
        # self.RGB_net1 = createCNNs(3, [16,16,16,16,32])
        # self.RGB_net1.apply(self.init_weights)
        # self.RGB_net2 = createCNNs(32, [32,32,32,32,64])
        # self.RGB_net2.apply(self.init_weights)
        # self.RGB_net3 = createCNNs(64, [64,64,64,64,128])
        self.RGB_net1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(16, 16, 3, 1, 1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=1, padding=1),

            nn.Conv2d(16, 16, 3, 1, 1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=1, padding=1),

            nn.Conv2d(16, 16, 3, 1, 1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=1, padding=1),

            nn.Conv2d(16, 32, 3, 1, 1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # self.RGB_net1.apply(self.init_weights)
        self.RGB_net2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=1, padding=1),

            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=1, padding=1),

            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=1, padding=1),

            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # self.RGB_net2.apply(self.init_weights)
        self.RGB_net3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=1, padding=1),

            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=1, padding=1),

            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=1, padding=1),

            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=1, padding=1),

            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # self.RGB_net3.apply(self.init_weights)
        ##########################################################
        self.conv1_l3 = Conv1d(256, 4, use_activation=False)  # quat head
        self.conv1_l2 = Conv1d(256, 4, use_activation=False)
        # self.conv1_l1 = Conv1d(256, 4, use_activation=False)
        # self.conv1_l0 = Conv1d(256, 4, use_activation=False)
        self.conv2_l3 = Conv1d(256, 3, use_activation=False)  # trans head
        self.conv2_l2 = Conv1d(256, 3, use_activation=False)
        # self.conv2_l1 = Conv1d(256, 3, use_activation=False)
        # self.conv2_l0 = Conv1d(256, 3, use_activation=False)
        self.conv3_l3 = Conv1d(64, 256, use_activation=False)  # in FC
        self.conv3_l2 = Conv1d(64, 256, use_activation=False)
        # self.conv3_l1 = Conv1d(64, 256, use_activation=False)
        # self.conv3_l0 = Conv1d(64, 256, use_activation=False)
        ##########################################################
        self.sx = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.sq = torch.nn.Parameter(torch.tensor([-2.5]), requires_grad=True)
        ##########################################################
        self.cost_volume1 = CostVolume(radius=10.0, nsample=4, nsample_q=32,
                                       rgb_in_channels=128, lidar_in_channels=128,
                                       mlp1=[128, 64, 64], mlp2=[128, 64], is_training=self.training, bn_decay=bn_decay,
                                       bn=True, pooling='max', knn=True, corr_func=CostVolume.CorrFunc.ELEMENTWISE_PRODUCT)
        # self.cost_volume2 = CostVolume(radius=10.0, nsample=4, nsample_q=6,
        #                               rgb_in_channels=64, lidar_in_channels=64,
        #                               mlp1=[128, 64, 64], mlp2=[128, 64], is_training=self.training, bn_decay=bn_decay,
        #                               bn=True, pooling='max', knn=True, corr_func='concat')
        # self.cost_volume3 = CostVolume(radius=10.0, nsample=4, nsample_q=6,
        #                               rgb_in_channels=32, lidar_in_channels=32,
        #                               mlp1=[128, 64, 64], mlp2=[128, 64], is_training=self.training, bn_decay=bn_decay,
        #                               bn=True, pooling='max', knn=True, corr_func='concat')
        ##########################################################
        if not new_dim:
            self.flow_predictor0 = FlowPredictor(in_channels=128 + 64, mlp=[128, 64], is_training=self.training,
                                                 bn_decay=bn_decay)
        else:
            self.flow_predictor0 = FlowPredictor(in_channels=256 + 64, mlp=[128, 64], is_training=self.training,
                                             bn_decay=bn_decay)
        self.flow_predictor0_predict = FlowPredictor(in_channels=128 + 64 + 64, mlp=[128, 64],
                                                     is_training=self.training,
                                                     bn_decay=bn_decay)
        self.flow_predictor0_w = FlowPredictor(in_channels=128 + 64 + 64, mlp=[128, 64], is_training=self.training,
                                               bn_decay=bn_decay)
        # self.flow_predictor1_predict = FlowPredictor(in_channels=64 + 64 + 64, mlp=[128, 64], is_training=self.training,
        #                                             bn_decay=bn_decay)  # .to(device)
        # self.flow_predictor1_w = FlowPredictor(in_channels=64 + 64 + 64, mlp=[128, 64], is_training=self.training,
        #                                       bn_decay=bn_decay)  # .to(device)
        # self.flow_predictor2_predict = FlowPredictor(in_channels=32 + 64 + 64, mlp=[128, 64], is_training=self.training,
        #                                             bn_decay=bn_decay)  # .to(device)
        # self.flow_predictor2_w = FlowPredictor(in_channels=32 + 64 + 64, mlp=[128, 64], is_training=self.training,
        #                                       bn_decay=bn_decay)  # .to(device)

        ##########################################################
        self.set_upconv0_w_upsample = SetUpconvModule(nsample=8, radius=2.4,
                                                      in_channels=[128, 64],
                                                      mlp=[128, 64], mlp2=[64], is_training=self.training,
                                                      bn_decay=bn_decay, knn=True)  # .to(device)
        self.set_upconv0_upsample = SetUpconvModule(nsample=8, radius=2.4,
                                                    in_channels=[128, 64],
                                                    mlp=[128, 64], mlp2=[64], is_training=self.training,
                                                    bn_decay=bn_decay, knn=True)  # .to(device)
        # self.set_upconv1_w_upsample = SetUpconvModule(nsample=8, radius=2.4,
        #                                              in_channels=[64, 64],
        #                                              mlp=[128, 64], mlp2=[64], is_training=self.training,
        #                                              bn_decay=bn_decay, knn=True)  # .to(device)
        # self.set_upconv1_upsample = SetUpconvModule(nsample=8, radius=2.4,
        #                                            in_channels=[64, 64],
        #                                            mlp=[128, 64], mlp2=[64], is_training=self.training,
        #                                            bn_decay=bn_decay, knn=True)  # .to(device)
        # self.set_upconv2_w_upsample = SetUpconvModule(nsample=8, radius=2.4,
        #                                              in_channels=[32, 64],
        #                                              mlp=[128, 64], mlp2=[64], is_training=self.training,
        #                                              bn_decay=bn_decay, knn=True)  # .to(device)
        # self.set_upconv2_upsample = SetUpconvModule(nsample=8, radius=2.4,
        #                                            in_channels=[32, 64],
        #                                            mlp=[128, 64], mlp2=[64], is_training=self.training,
        #                                            bn_decay=bn_decay, knn=True)  # .to(device)

    def init_weights(self, layer):
        """not used"""
        if type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        elif type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, rgb_img, lidar_img, H_initial, intrinsic, resize_img):
        # resize_img = resize_img[0]
        device = rgb_img.device

        intrinsic = intrinsic[0].float()  # camera matrix is the same

        H_initial = H_initial[0].reshape(3, 4)  # H_initial is not used

        B, _, h, w = rgb_img.shape  # [B,3,352,1216]
        _, N, _ = lidar_img.shape

        # image branch

        RF1 = self.RGB_net1(rgb_img)

        RF2 = self.RGB_net2(RF1)

        RF3 = self.RGB_net3(RF2)  # [B,256,11,38]

        # RF1 = F.normalize(RF1, p=2, dim=1, eps=1e-12, out=None)
        # RF2 = F.normalize(RF2, p=2, dim=1, eps=1e-12, out=None)
        # RF3 = F.normalize(RF3, p=2, dim=1, eps=1e-12, out=None)

        # RF1_index = set_id_grid(RF1.permute(0, 2, 3, 1))
        # RF2_index = set_id_grid(RF2.permute(0, 2, 3, 1))
        # img discrete index
        RF3_index = set_id_grid(RF3.permute(0, 2, 3, 1))  # [B,418,3]

        lidar_img = lidar_img.permute(0, 2, 1)  # [B,C,N]
        # lidar feature initial as zeros
        lidar_norm = torch.zeros(B, N, 3, device=device)
        lidar_norm = lidar_norm.permute(0, 2, 1)

        # lidar layers
        P1, LF1, group_xyz_1, fps_idx_1 = self.LiDAR_lv1(lidar_img.float(), lidar_norm.float())
        P2, LF2, group_xyz_2, fps_idx_2 = self.LiDAR_lv2(P1, LF1)
        P3, LF3, group_xyz_3, fps_idx_3 = self.LiDAR_lv3(P2, LF2)  # LF3.shape[B,C,N]=[B,64,256]
        P4, LF4, group_xyz_4, fps_idx_4 = self.LiDAR_lv4(P3, LF3)  # LF4.shape[B,C,N]=[B,128,64]

        # LF1 = F.normalize(LF1, p=2, dim=1, eps=1e-12, out=None)
        # LF2 = F.normalize(LF2, p=2, dim=1, eps=1e-12, out=None)
        # LF3 = F.normalize(LF3, p=2, dim=1, eps=1e-12, out=None)
        # LF4 = F.normalize(LF4, p=2, dim=1, eps=1e-12, out=None)

        # project the uv to normalized camera plane
        intrinsic_3 = change_intrinsic(intrinsic, RF3, rgb_img)  # K3
        # cuda not support matrix inverse
        intrinsic_3_inv = torch.from_numpy(np.linalg.inv(intrinsic_3.cpu()))
        RF3_index = intrinsic_3_inv.cpu() @ RF3_index.permute(0, 2, 1).cpu()
        RF3_index = RF3_index.permute(0, 2, 1).to(device)  # [B,418,3]
        # P3_npoint = P3.shape[2]

        # project to the normalization camera plane
        lidar_uv, lidar_z, LF3 = warp_utils.projection_initial(P3, H_initial, intrinsic_3, RF3[0][0].shape, LF3)

        lidar_uv = lidar_uv.reshape(B, -1, 3)
        # sampled_points = lidar_uv.shape[1]
        #
        # radius = 4
        # nsample = 32

        _, C, H, W = RF3.shape
        RF3 = RF3.reshape(B, C, H * W).permute(0, 2, 1) # B,N,C

        # B,N,C embedding cost volume
        concat_4 = self.cost_volume1(lidar_uv, LF3.permute(0, 2, 1), RF3_index, RF3, lidar_z)  # [B,256,64]=[B,N,C]


        # resample the cost volume to l4 B,C,N
        P4, l4_points_f1_cost_volume, _, _ = self.layer_idx(P3, concat_4.permute(0, 2, 1), sample_idx=fps_idx_4)

        l4_points_predict = l4_points_f1_cost_volume.permute(0, 2, 1)  # [B,64,64]=[B,N,C]

        # mask F+E [B,N,C]
        l4_cost_volume_w = self.flow_predictor0(LF4.permute(0, 2, 1), None, l4_points_predict)
        W_l4_feat1 = F.softmax(l4_cost_volume_w, dim=1)

        # mask_4 = P4.permute(0, 2, 1)[:, :, 2] > 0
        # mask_4 = torch.unsqueeze(mask_4, dim=2).detach()

        # W_l4_feat1 = mask_4 * W_l4_feat1
        # W_l4_feat1 = F.softmax(W_l4_feat1, dim=1)

        l4_points_f1_new = torch.sum(l4_points_predict * W_l4_feat1, dim=1, keepdim=True)  # [B,1,C]

        _, _, C = l4_points_f1_new.shape
        result_4 = l4_points_f1_new.reshape(B, 1, C)

        # pose prediction
        l4_points_f1_new_big = self.conv3_l3(result_4)
        # l4_points_f1_new_big = F.dropout(self.conv3_l3(result_4),p=0.5,training=self.training)
        l4_points_f1_new_q = F.dropout(l4_points_f1_new_big, p=0.5, training=self.training)
        l4_points_f1_new_t = F.dropout(l4_points_f1_new_big, p=0.5, training=self.training)

        result_4_real = self.conv1_l3(l4_points_f1_new_q)
        # result_4_real = self.conv1_l3(l4_points_f1_new_big)

        result_4_real = result_4_real / (
                torch.sqrt(torch.sum(result_4_real * result_4_real, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        result_4_dual = self.conv2_l3(l4_points_f1_new_t)
        # result_4_dual = self.conv2_l3(l4_points_f1_new_big)
        result_4_real = torch.squeeze(result_4_real, dim=1)
        result_4_dual = torch.squeeze(result_4_dual, dim=1)
        result_4 = torch.cat((result_4_real, result_4_dual), dim=1)

        result_4 = result_4[:, 0:7]

        decalib_quat_real4 = result_4[:, :4]
        decalib_quat_dual4 = result_4[:, 4:]

        # layer 3  ################################################
        # vec to quat
        H3_trans = torch.cat([torch.zeros((B, 1),device=device),
                              result_4[:, 4:7].reshape(B, 3)],-1).reshape(B, 4)
        lidar_uv, lidar_z, LF3 = warp_utils.warp_quat(P3, decalib_quat_real4,
                                                      H3_trans, intrinsic_3, RF3[0][0].shape, LF3)
        lidar_uv = lidar_uv.reshape(B, -1, 3)
        # sampled_points = lidar_uv.shape[1]

        concat_3 = self.cost_volume1(lidar_uv, LF3.permute(0, 2, 1), RF3_index, RF3, lidar_z)

        # mask upsampling
        l3_cost_volume_w_upsample = self.set_upconv0_w_upsample(
            P3.permute(0, 2, 1), P4.permute(0, 2, 1), LF3.permute(0, 2, 1), l4_cost_volume_w)  # [B,256,64]

        # cost volume upsampling TODO: concat_4 [B,256,C] not be upsampled
        # l3_cost_volume_upsample = concat_4
        # l3_cost_volume_upsample = self.set_upconv0_upsample(
        #     P3.permute(0, 2, 1), P4.permute(0, 2, 1), LF3.permute(0, 2, 1), concat_4)  # [B,256,64]
        l3_cost_volume_upsample = self.set_upconv0_upsample(
            P3.permute(0, 2, 1), P4.permute(0, 2, 1), LF3.permute(0, 2, 1), l4_points_predict)  # [B,256,64]

        # predict refined embedding
        l3_cost_volume_predict = self.flow_predictor0_predict(
            LF3.permute(0, 2, 1), l3_cost_volume_upsample, concat_3)  # [B,256,64]
        # predict refined mask
        l3_cost_volume_w = self.flow_predictor0_w(
            LF3.permute(0, 2, 1), l3_cost_volume_w_upsample, l3_cost_volume_predict)  # [B,256,64]
        W_l3_cost_volume = F.softmax(l3_cost_volume_w, dim=1)

        # origin mask
        # mask_3 = lidar_z > 0
        # mask_3 = mask_3.detach()
        # W_l3_cost_volume = mask_3 * W_l3_cost_volume
        # W_l3_cost_volume = F.softmax(W_l3_cost_volume, dim=1)

        l3_cost_volume_sum = torch.sum(l3_cost_volume_predict * W_l3_cost_volume, dim=1, keepdim=True)  # [B,1,64]

        _, _, C = l3_cost_volume_sum.shape
        result_3 = l3_cost_volume_sum.reshape(B, 1, C)
        l3_points_f1_new_big = self.conv3_l2(result_3)
        # l3_points_f1_new_big = F.dropout(self.conv3_l2(result_3), p=0.5, training=self.training)
        l3_points_f1_new_q = F.dropout(l3_points_f1_new_big, p=0.5, training=self.training)
        l3_points_f1_new_t = F.dropout(l3_points_f1_new_big, p=0.5, training=self.training)
        result_3_real = self.conv1_l2(l3_points_f1_new_q)
        # result_3_real = self.conv1_l2(l3_points_f1_new_big)
        result_3_real = result_3_real / (
                torch.sqrt(torch.sum(result_3_real * result_3_real, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        result_3_dual = self.conv2_l2(l3_points_f1_new_t)
        # result_3_dual = self.conv2_l2(l3_points_f1_new_big)
        result_3_real = torch.squeeze(result_3_real, dim=1)
        result_3_dual = torch.squeeze(result_3_dual, dim=1)
        result_3 = torch.cat((result_3_real, result_3_dual), dim=1)
        result_3 = result_3[:, 0:7]

        out_3_real = warp_utils.mul_q(result_3[:, :4].reshape(B, 1, 4),
                                      result_4[:, :4].reshape(B, 1, 4))

        result_4_copy_trans = torch.cat((torch.zeros((B, 1)).cuda(),
                                         result_4[:, 4:7].reshape(B, 3)), 1).reshape(B, 4)
        result_3_copy_trans = torch.cat((torch.zeros((B, 1)).cuda(),
                                         result_3[:, 4:7].reshape(B, 3)), 1).reshape(B, 4)

        result_4_copy_trans = torch.reshape(result_4_copy_trans, [B, 1, 4])
        result_3_copy_trans = torch.reshape(result_3_copy_trans, [B, 1, 4])

        # q = q3*q4
        out_3_dual = warp_utils.mul_q(result_3[:, :4], result_4_copy_trans)  # B,1,4
        out_3_dual = warp_utils.mul_q(out_3_dual, warp_utils.inv_q(result_3[:, :4])) + result_3_copy_trans
        out_3_real = torch.squeeze(out_3_real, dim=1)
        out_3_dual = torch.squeeze(out_3_dual, dim=1)
        out_3 = torch.cat((out_3_real, out_3_dual[:, 1:]), 1)

        out_3 = out_3[:, 0:7]

        # out_3_real = out_3[:, :4]
        # out_3_dual = out_3[:, 4:]

        if self.eval_info:
            return out_3.reshape(B, 7).float(), \
                   result_4.reshape(B, 7).float(), \
                   self.sx, self.sq, W_l3_cost_volume, P3.permute(0, 2, 1)

        return out_3.reshape(B, 7).float(), result_4.reshape(B, 7).float(), self.sx, self.sq

    def l2l1forward(self):
        pass
        # # layer 2  ################################################
        # H2_trans = torch.cat((torch.zeros((B, 1)).cuda(), out_3_copy[:, 4:7].reshape(B, 3)),
        #                      -1).reshape(B, 4)
        # intrinsic_2 = change_intrinsic(intrinsic, RF2, rgb_img)
        # intrinsic_2_inv = torch.from_numpy(np.linalg.inv(intrinsic_2.cpu()))
        # RF2_index = intrinsic_2_inv.cpu() @ RF2_index.permute(0, 2, 1).cpu()
        # RF2_index = RF2_index.permute(0, 2, 1).cuda()
        #
        # P2_npoint = P2.shape[2]
        # lidar_uv, lidar_z, LF2 = self.warp_quat(P2, out_3_real, H2_trans, intrinsic_2, RF2[0][0].shape, LF2)
        #
        # lidar_uv = lidar_uv.reshape(B, -1, 3)
        # sampled_points = lidar_uv.shape[1]
        #
        # radius = 3
        # nsample = 18
        # # idx = query_ball_point(radius, nsample, RF2_index, lidar_uv)
        # _, C, H, W = RF2.shape
        # RF2_copy = RF2.reshape(B, C, H * W).permute(0, 2, 1)
        # concat_2 = self.cost_volume2(lidar_uv, LF2.permute(0, 2, 1), RF2_index, RF2_copy,
        #                              lidar_z)  # [B,1024,64] l2_cost_volume
        #
        # l2_cost_volume_w_upsample = self.set_upconv1_w_upsample(
        #     P2.permute(0, 2, 1), P3.permute(0, 2, 1), LF2.permute(0, 2, 1), l3_cost_volume_w)  # [B,1024,64]
        # l2_cost_volume_upsample = self.set_upconv1_upsample(
        #     P2.permute(0, 2, 1), P3.permute(0, 2, 1), LF2.permute(0, 2, 1), concat_3)  # [B,1024,64]
        # l2_cost_volume_predict = self.flow_predictor1_predict(
        #     LF2.permute(0, 2, 1), l2_cost_volume_upsample, concat_2)  # [B,1024,64]
        # l2_cost_volume_w = self.flow_predictor1_w(
        #     LF2.permute(0, 2, 1), l2_cost_volume_w_upsample, l2_cost_volume_predict)  # [B,1024,64]
        # W_l2_cost_volume = F.softmax(l2_cost_volume_w, dim=1)
        #
        # # mask_2 = lidar_z > 0
        # # mask_2 = mask_2.detach()
        # # W_l2_cost_volume = mask_2 * W_l2_cost_volume
        # # W_l2_cost_volume = F.softmax(W_l2_cost_volume, dim=1)
        #
        # l2_cost_volume_sum = torch.sum(concat_2 * W_l2_cost_volume, dim=1, keepdim=True)  # [B,1,64]
        #
        # _, _, C = l2_cost_volume_sum.shape
        # result_2 = l2_cost_volume_sum.reshape(B, 1, C)
        # l2_points_f1_new_big = self.conv3_l1(result_2)
        # # l2_points_f1_new_q = F.dropout(l2_points_f1_new_big, p=0.5, training=self.training)
        # # l2_points_f1_new_t = F.dropout(l2_points_f1_new_big, p=0.5, training=self.training)
        # # result_2_real = self.conv1_l1(l2_points_f1_new_q)
        # result_2_real = self.conv1_l1(l2_points_f1_new_big)
        # result_2_real = result_2_real / (
        #         torch.sqrt(torch.sum(result_2_real * result_2_real, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        # # result_2_dual = self.conv2_l1(l2_points_f1_new_t)
        # result_2_dual = self.conv2_l1(l2_points_f1_new_big)
        # result_2_real = torch.squeeze(result_2_real, dim=1)
        # result_2_dual = torch.squeeze(result_2_dual, dim=1)
        # result_2 = torch.cat((result_2_real, result_2_dual), dim=1)
        #
        # out_3_copy = out_3[:, 0:7]
        # result_2_copy = result_2[:, 0:7]
        #
        # out_2_real = self.mul_q_point(result_2_copy[:, :4].reshape(B, 1, 4),
        #                               out_3_copy[:, :4].reshape(B, 1, 4), batch_size=B)
        # out_3_copy_trans = torch.cat((torch.zeros((B, 1)).cuda(),
        #                               out_3_copy[:, 4:7].reshape(B, 3)), 1).reshape(B, 4)
        # result_2_copy_trans = torch.cat((torch.zeros((B, 1)).cuda(),
        #                                  result_2_copy[:, 4:7].reshape(B, 3)), 1).reshape(B, 4)
        #
        # out_3_copy_trans = torch.reshape(out_3_copy_trans, [B, 1, 4])
        # result_2_copy_trans = torch.reshape(result_2_copy_trans, [B, 1, 4])
        #
        # out_2_dual = self.mul_q_point(result_2_copy[:, :4], out_3_copy_trans, batch_size=B)
        # out_2_dual = self.mul_point_q(out_2_dual, self.inv_q(result_2_copy[:, :4]),
        #                               batch_size=B) + result_2_copy_trans
        # out_2_real = torch.squeeze(out_2_real, dim=1)
        # out_2_dual = torch.squeeze(out_2_dual, dim=1)
        # out_2 = torch.cat((out_2_real, out_2_dual[:, 1:]), 1)
        #
        #
        # # layer 1  ################################################
        # H1_trans = out_2_dual
        # intrinsic_1 = change_intrinsic(intrinsic, RF1, rgb_img)
        # intrinsic_1_inv = torch.from_numpy(np.linalg.inv(intrinsic_1.cpu()))
        # RF1_index = intrinsic_1_inv.cpu() @ RF1_index.permute(0, 2, 1).cpu()
        # RF1_index = RF1_index.permute(0, 2, 1).cuda()
        #
        # lidar_uv, lidar_z, LF1 = self.warp_quat(P1, out_2_real, H1_trans, intrinsic_1, RF1[0][0].shape, LF1)
        #
        # lidar_uv = lidar_uv.reshape(B, -1, 3)
        # P1_npoint = P1.shape[2]
        # sampled_points = lidar_uv.shape[1]
        # # print("P1: ", sampled_points)
        # radius = 3
        # nsample = 18
        # _, C, H, W = RF1.shape
        # RF1_copy = RF1.reshape(B, C, H*W).permute(0, 2, 1)
        # concat_1 = self.cost_volume3(lidar_uv, LF1.permute(0, 2, 1), RF1_index, RF1_copy, lidar_z)  # [B,2048,64]
        #
        # l1_cost_volume_w_up_sample = self.set_upconv2_w_upsample(
        #     P1.permute(0, 2, 1), P2.permute(0, 2, 1), LF1.permute(0, 2, 1), l2_cost_volume_w)  # [B,2048,64]
        # l1_cost_volume_up_sample = self.set_upconv2_upsample(
        #     P1.permute(0, 2, 1), P2.permute(0, 2, 1), LF1.permute(0, 2, 1), l2_cost_volume_predict)  # [B,2048,64]
        # l1_cost_volume_predict = self.flow_predictor2_predict(
        #     LF1.permute(0, 2, 1), l1_cost_volume_up_sample, concat_1)  # [B,2048,64]
        # l1_cost_volume_w = self.flow_predictor2_w(
        #     LF1.permute(0, 2, 1), l1_cost_volume_w_up_sample, l1_cost_volume_predict)  # [B,2048,64]
        # W_l1_cost_volume = F.softmax(l1_cost_volume_w, dim=1)
        #
        # #mask_1 = lidar_z > 0
        # #mask_1 = mask_1.detach()
        # #W_l1_cost_volume = mask_1 * W_l1_cost_volume
        # #W_l1_cost_volume = F.softmax(W_l1_cost_volume, dim=1)
        #
        # l1_cost_volume_8 = torch.sum(concat_1 * W_l1_cost_volume, dim=1, keepdim=True)  # [B,1,64]
        #
        # _, _, C = l1_cost_volume_8.shape
        # result_1 = l1_cost_volume_8.reshape(B, 1, C)
        #
        # l1_points_f1_new_big = self.conv3_l0(result_1)
        # #l1_points_f1_new_q = F.dropout(l1_points_f1_new_big, p=0.5, training=self.training)
        # #l1_points_f1_new_t = F.dropout(l1_points_f1_new_big, p=0.5, training=self.training)
        #
        # #result_1_real = self.conv1_l0(l1_points_f1_new_q)
        # result_1_real = self.conv1_l0(l1_points_f1_new_big)
        # result_1_real = result_1_real / (
        #             torch.sqrt(torch.sum(result_1_real * result_1_real, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        #
        # #result_1_dual = self.conv2_l0(l1_points_f1_new_t)
        # result_1_dual = self.conv2_l0(l1_points_f1_new_big)
        # result_1_real = torch.squeeze(result_1_real, dim=1)
        # result_1_dual = torch.squeeze(result_1_dual, dim=1)
        #
        # result_1 = torch.cat((result_1_real, result_1_dual), dim=1)
        #
        # result_1_copy = result_1[:, 0:7]
        #
        # out_real = self.mul_q_point(result_1_copy[:, :4].reshape(B, 1, 4),
        #                             out_2_real.reshape(B, 1, 4), batch_size=B)
        # out_2_copy_trans = out_2_dual
        # result_1_copy_trans = torch.cat((torch.zeros((B, 1)).cuda(),
        #                                  result_1_copy[:, 4:7].reshape(B, 3)), 1).reshape(B, 4)
        #
        # out_2_copy_trans = torch.reshape(out_2_copy_trans, [B, 1, 4])
        # result_1_copy_trans = torch.reshape(result_1_copy_trans, [B, 1, 4])
        #
        # out_dual = self.mul_q_point(result_1_copy[:, :4], out_2_copy_trans, batch_size=B)
        # out_dual = self.mul_point_q(out_dual, self.inv_q(result_1_copy[:, :4]),
        #                             batch_size=B) + result_1_copy_trans
        #
        # out_real = torch.squeeze(out_real, dim=1)
        # out_dual = torch.squeeze(out_dual, dim=1)
        #
        # out = torch.cat((out_real, out_dual[:, 1:]), 1)

        # return out.reshape(B, 7).float(), out_2.reshape(B, 7).float(), \
        #       out_3_copy.reshape(B, 7).float(), result_4_copy.reshape(B, 7).float(), self.sx, self.sq