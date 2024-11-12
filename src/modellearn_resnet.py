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
from src.modules.MainModules import CostVolume, FlowPredictor, PoseHead, ProjectMask,DelayWeight,MaskCostVolume
import src.modules.warp_utils as warp_utils
import src.utils as utils
from src.resnet import ImageEncoder

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

        self.img_encoder = ImageEncoder() # last channel 512
        ##########################################################
        self.cost_volume1 = CostVolume(radius=10.0, nsample=cfg.cost_volume_nsamples[0],
                                       nsample_q=cfg.cost_volume_nsamples[1][0],
                                       rgb_in_channels=512,
                                       lidar_in_channels=lidar_mlps[-3][-1],  # LF3 dim
                                       mlp1=cfg.cost_volume_mlps[0], mlp2=cfg.cost_volume_mlps[1],
                                       is_training=self.training, bn_decay=bn_decay,
                                       bn=True, pooling='max', knn=True,
                                       corr_func=cfg.cost_volume_corr_func,
                                       backward_validation=cfg.backward_validation[0])
        self.cost_volume2 = CostVolume(radius=10.0, nsample=cfg.cost_volume_nsamples[0],
                                       nsample_q=cfg.cost_volume_nsamples[1][1],
                                       rgb_in_channels=512,
                                       lidar_in_channels=lidar_mlps[-3][-1],  # LF3 dim
                                       mlp1=cfg.cost_volume_mlps[0], mlp2=cfg.cost_volume_mlps[1],
                                       is_training=self.training, bn_decay=bn_decay,
                                       bn=True, pooling='max', knn=True,
                                       corr_func=cfg.cost_volume_corr_func,
                                       backward_validation=cfg.backward_validation[1])
        ##########################################################

        self.flow_predictor0 = FlowPredictor(in_channels=lidar_mlps[-2][-1] +  # l4 feature
                                                         lidar_mlps[-1][-1]  # resampled_costvolume_l4
                                             , mlp=cfg.flow_predictor_mlps[0], is_training=self.training,
                                             bn_decay=bn_decay)

        ##########################################################
        self.set_upconv0_w_upsample = SetUpconvModule(nsample=cfg.setupconv_nsamples[0], radius=2.4,
                                                      in_channels=[lidar_mlps[-3][-1],
                                                                   cfg.flow_predictor_mlps[0][-1]],
                                                      mlp=cfg.setupconv_mlps[0][0],
                                                      mlp2=cfg.setupconv_mlps[0][1], is_training=self.training,
                                                      bn_decay=bn_decay, knn=True)
        self.set_upconv0_upsample = SetUpconvModule(nsample=cfg.setupconv_nsamples[1], radius=2.4,
                                                    in_channels=[lidar_mlps[-3][-1],  # l3 feature
                                                                 lidar_mlps[-1][-1]],  # resampled l4 EM
                                                    mlp=cfg.setupconv_mlps[1][0],
                                                    mlp2=cfg.setupconv_mlps[1][1], is_training=self.training,
                                                    bn_decay=bn_decay, knn=True)
        ###############################################
        self.flow_predictor0_predict = FlowPredictor(in_channels=lidar_mlps[-3][-1] +  # LF3 dim
                                                                 cfg.setupconv_mlps[1][1][-1] +  # upsampled l4 EM dim
                                                                 cfg.cost_volume_mlps[-1][-1]  # l3 EM dim
                                                     , mlp=cfg.flow_predictor_mlps[1],
                                                     is_training=self.training,
                                                     bn_decay=bn_decay)
        self.flow_predictor0_w = FlowPredictor(in_channels=lidar_mlps[-3][-1] +  # LF3 dim
                                                           cfg.setupconv_mlps[0][-1][-1] +  # upsampled l4 mask dim
                                                           cfg.flow_predictor_mlps[1][-1],  # l3 refined EM dim
                                               mlp=cfg.flow_predictor_mlps[2], is_training=self.training,
                                               bn_decay=bn_decay)

        ##########################################################
        # shared MLPs
        # FC to hidden layer
        # self.conv3_l3 = Conv1d(lidar_mlps[-1][-1], cfg.head_hidden_dim, use_activation=False) # setconv to the l3 EM
        # # self.conv3_l2 = Conv1d(cfg.flow_predictor_mlps[1][-1], cfg.head_hidden_dim, use_activation=False)
        # # rotation quat head
        # self.conv1_l3 = Conv1d(cfg.head_hidden_dim, cfg.rotation_quat_head_dim, use_activation=False)
        # # self.conv1_l2 = Conv1d(cfg.head_hidden_dim, cfg.rotation_quat_head_dim, use_activation=False)
        # # trans vector head
        # self.conv2_l3 = Conv1d(cfg.head_hidden_dim, cfg.transition_vec_head_dim, use_activation=False)
        # # self.conv2_l2 = Conv1d(cfg.head_hidden_dim, cfg.transition_vec_head_dim, use_activation=False)
        self.l4_head = PoseHead(in_channels=[lidar_mlps[-1][-1], lidar_mlps[-2][-1]],
                                mlp1=cfg.pose_head_mlps[0][0],
                                mlp2=cfg.pose_head_mlps[0][1],
                                hidden=cfg.head_hidden_dim,
                                q_dim=cfg.rotation_quat_head_dim,
                                t_dim=cfg.transition_vec_head_dim,
                                dropout_rate=cfg.head_dropout_rate,
                                split_dp=cfg.split_dp,
                                corr_func=cfg.head_corr_func,
                                pos_embed=cfg.head_pos_embedding,
                                sigmoid=cfg.mask_sigmoid)
        self.l3_head = PoseHead(in_channels=[cfg.flow_predictor_mlps[1][-1], lidar_mlps[-3][-1]],
                                mlp1=cfg.pose_head_mlps[1][0],
                                mlp2=cfg.pose_head_mlps[1][1],
                                hidden=cfg.head_hidden_dim,
                                q_dim=cfg.rotation_quat_head_dim,
                                t_dim=cfg.transition_vec_head_dim,
                                dropout_rate=cfg.head_dropout_rate,
                                split_dp=cfg.split_dp,
                                corr_func=cfg.head_corr_func,
                                pos_embed=cfg.head_pos_embedding,
                                sigmoid=cfg.mask_sigmoid)

        if cfg.use_projection_mask:
            if cfg.layer_mask[0]:
                self.l4_projection_mask = ProjectMask(lidar_mlps[-1][-1]  # resampled l4 EM
                                                  + lidar_mlps[-2][-1],  # l4 feature
                                                  cfg.projection_mask_mlps[0],cfg.mask_sigmoid)
                self.l4_delay = DelayWeight(cfg.mask_delay_step,cfg.mask_delay)
            if cfg.layer_mask[1]:
                self.l3_projection_mask = ProjectMask(lidar_mlps[-3][-1] +  # l3 feature
                                                  cfg.flow_predictor_mlps[1][-1],  # l3 refined EM
                                                  cfg.projection_mask_mlps[1],cfg.mask_sigmoid)
                self.l3_delay = DelayWeight(cfg.mask_delay_step,cfg.mask_delay)
        elif cfg.one_head_mask:
            self.mask_costvolume = MaskCostVolume(rgb_in_channels=cfg.rgb_encoder_channels[-1][1][-1],
                                       lidar_in_channels=lidar_mlps[-3][-1],
                                                  mlp1=[128,128],img_size=418)
            self.proj_mask = ProjectMask(lidar_mlps[-3][-1],
                                         cfg.projection_mask_mlps[0])

        ##########################################################
        # loss learnable parameters
        self.sq = torch.nn.Parameter(torch.tensor([cfg.sq_init]), requires_grad=True)
        self.sx = torch.nn.Parameter(torch.tensor([cfg.sx_init]), requires_grad=True)

    # def init_weights(self, layer):
    #     """not used"""
    #     if type(layer) == nn.Conv2d:
    #         nn.init.xavier_uniform_(layer.weight)
    #         nn.init.constant_(layer.bias, 0.0)
    #     elif type(layer) == nn.Linear:
    #         nn.init.xavier_uniform_(layer.weight)
    #         nn.init.constant_(layer.bias, 0.0)

    def forward(self, rgb_img, lidar_img, H_initial, intrinsic, resize_img, gt_project=None):
        # resize_img = resize_img[0]
        device = rgb_img.device
        K = intrinsic.float()
        intrinsic = intrinsic[0].float()  # camera matrix is the same

        # H_initial = H_initial[0].reshape(3, 4)  # H_initial is not used

        B, _, h, w = rgb_img.shape  # [B,3,352,1216]
        _, N, _ = lidar_img.shape

        # image branch

        # RF1 = self.RGB_net1(rgb_img)
        #
        # RF2 = self.RGB_net2(RF1)
        #
        # RF3 = self.RGB_net3(RF2)  # [B,256,11,38]
        RF2,RF3,_ = self.img_encoder(rgb_img)

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
        lidar_uv, lidar_z, LF3 = warp_utils.projection_initial(P3, None, None, None, LF3)

        lidar_uv = lidar_uv.reshape(B, -1, 3)
        # sampled_points = lidar_uv.shape[1]
        #
        # radius = 4
        # nsample = 32

        _, C, H, W = RF3.shape
        RF3 = RF3.reshape(B, C, H * W).permute(0, 2, 1)  # B,N,C

        # B,N,C embedding cost volume
        concat_4 = self.cost_volume1(lidar_uv, LF3.permute(0, 2, 1), RF3_index, RF3, lidar_z)  # [B,256,64]=[B,N,C]

        if cfg.one_head_mask:
            feature_cost = self.mask_costvolume(LF3.permute(0,2,1),RF3)
            p3_mask = self.proj_mask(None,feature_cost)
            p4_mask = index_points(p3_mask,fps_idx_4)

        # resample the cost volume to l4 B,C,N
        P4, l4_points_f1_cost_volume, _, _ = self.layer_idx(P3, concat_4.permute(0, 2, 1), sample_idx=fps_idx_4)

        l4_points_predict = l4_points_f1_cost_volume.permute(0, 2, 1)  # [B,64,64]=[B,N,C]

        # mask F+E [B,N,C]
        l4_cost_volume_w = self.flow_predictor0(LF4.permute(0, 2, 1), None, l4_points_predict)

        # mask_4 = P4.permute(0, 2, 1)[:, :, 2] > 0
        # mask_4 = torch.unsqueeze(mask_4, dim=2).detach()

        # W_l4_feat1 = mask_4 * W_l4_feat1
        # W_l4_feat1 = F.softmax(W_l4_feat1, dim=1)

        # TODO: l4 prediction
        l4_projection_mask = None
        if cfg.use_projection_mask and cfg.layer_mask[0]:
            l4_projection_mask = self.l4_projection_mask(LF4.permute(0, 2, 1), l4_points_predict)
        elif cfg.one_head_mask:
            l4_projection_mask = p4_mask
        if gt_project is not None:
            # gt_project [B,N,2]
            gt_project_l1 = index_points(gt_project, fps_idx_1)
            gt_project_l2 = index_points(gt_project_l1, fps_idx_2)
            gt_project_l3 = index_points(gt_project_l2, fps_idx_3)
            gt_project_l4 = index_points(gt_project_l3, fps_idx_4)
            if cfg.ground_truth_mask_layer[0]:
                l4_projection_mask_predict = l4_projection_mask
                if l4_projection_mask is not None:
                    l4_projection_mask = self.l4_delay(gt_project_l4,l4_projection_mask)
                else:
                    l4_projection_mask = gt_project_l4

        if cfg.ground_truth_mask_layer[0]:
            assert l4_projection_mask is not None
        result_4_real, result_4_dual, _ = self.l4_head(l4_points_predict, l4_cost_volume_w,
                                                       P4.permute(0, 2, 1), LF4.permute(0, 2, 1), l4_projection_mask)
        if gt_project is not None and cfg.ground_truth_mask_layer[0]:
            l4_projection_mask = l4_projection_mask_predict
        # result_4_real = torch.squeeze(result_4_real, dim=1)
        # result_4_dual = torch.squeeze(result_4_dual, dim=1)
        result_4 = torch.cat([result_4_real, result_4_dual], dim=1)

        decalib_quat_real4 = result_4_real
        decalib_quat_dual4 = result_4_dual

        # layer 3  ################################################
        # [0,t4]
        H3_trans = torch.cat([torch.zeros((B, 1), device=device),
                              decalib_quat_dual4], -1).reshape(B, 4)
        # p3_warp = q4*[0,p3]*q4'+[0,t4]
        lidar_uv, lidar_z, LF3 = warp_utils.warp_quat(P3, decalib_quat_real4,
                                                      H3_trans, None, None, LF3)
        lidar_uv = lidar_uv.reshape(B, -1, 3)
        # sampled_points = lidar_uv.shape[1]

        concat_3 = self.cost_volume2(lidar_uv, LF3.permute(0, 2, 1), RF3_index, RF3, lidar_z)

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

        # origin mask
        # mask_3 = lidar_z > 0
        # mask_3 = mask_3.detach()
        # W_l3_cost_volume = mask_3 * W_l3_cost_volume
        # W_l3_cost_volume = F.softmax(W_l3_cost_volume, dim=1)
        # TODO: l3 prediction
        l3_prediction_mask = None
        if cfg.use_projection_mask and cfg.layer_mask[1]:
            l3_prediction_mask = self.l3_projection_mask(LF3.permute(0, 2, 1), l3_cost_volume_predict)
        elif cfg.one_head_mask:
            l3_prediction_mask = p3_mask
        if gt_project is not None and cfg.ground_truth_mask_layer[1]:
            # gt_project [B,N,2]
            l3_prediction_mask_predict = l3_prediction_mask
            if l3_prediction_mask is not None:
                l3_prediction_mask = self.l3_delay(gt_project_l3,l3_prediction_mask)
            else:
                l3_prediction_mask = gt_project_l3
        # when train using gt but test without gt and pred
        if not cfg.layer_mask[1] and cfg.ground_truth_mask_layer[1] and gt_project is None:
            l3_prediction_mask = F.one_hot(utils.get_projection_gt(P3.permute(0,2,1),
                                                                   K,rgb_img.shape[2:],
                                                                   decalib_quat_real4,decalib_quat_dual4),2)


        result_3_real, result_3_dual, W_l3_cost_volume = self.l3_head(l3_cost_volume_predict, l3_cost_volume_w,
                                                                      P3.permute(0, 2, 1), LF3.permute(0, 2, 1),
                                                                      l3_prediction_mask)
        if gt_project is not None and cfg.ground_truth_mask_layer[1]:
            l3_prediction_mask = l3_prediction_mask_predict

        result_3 = torch.cat([result_3_real, result_3_dual], dim=1)
        # result_3 = result_3[:, 0:7]

        decalib_quat_real3 = result_3_real
        decalib_quat_dual3 = result_3_dual

        # q = q3*q4 <=> R = R3@R4
        out_3_real = warp_utils.mul_q(decalib_quat_real3.view(B, 1, 4),
                                      decalib_quat_real4.view(B, 1, 4))

        # [0,t]
        result_4_copy_trans = torch.cat((torch.zeros((B, 1), device=device),
                                         decalib_quat_dual4), 1).view(B, 1, 4)
        result_3_copy_trans = torch.cat((torch.zeros((B, 1), device=device),
                                         decalib_quat_dual3), 1).view(B, 1, 4)

        # q = q3*q4
        # q3*[0,t4]*q3'+t3 <=> t = R3@t4+t3
        out_3_dual = warp_utils.mul_q(decalib_quat_real3, result_4_copy_trans)  # B,1,4
        out_3_dual = warp_utils.mul_q(out_3_dual, warp_utils.inv_q(decalib_quat_real3)) + result_3_copy_trans

        out_3_real = torch.squeeze(out_3_real, dim=1)
        out_3_dual = torch.squeeze(out_3_dual, dim=1)
        out_3 = torch.cat((out_3_real, out_3_dual[:, 1:]), 1)  # B,7

        if self.eval_info:
            if gt_project is not None:
                return out_3.float(), result_4.float(), self.sx, self.sq, \
                       W_l3_cost_volume, P3.permute(0, 2, 1), gt_project_l3, gt_project_l4, P4.permute(0, 2, 1)
            else:
                return out_3.float(), result_4.float(), self.sx, self.sq, \
                       W_l3_cost_volume, P3.permute(0, 2, 1), l3_prediction_mask, l4_projection_mask, P4.permute(0, 2,
                                                                                                                 1)

        pm3 = None
        if l3_prediction_mask is not None:
            pm3 = [l3_prediction_mask, P3.permute(0, 2, 1)]
        if gt_project is not None and pm3 is not None:
            pm3.append(gt_project_l3)
        pm4 = None
        if l4_projection_mask is not None and not cfg.one_head_mask:
            pm4 = [l4_projection_mask, P4.permute(0, 2, 1)]
        if gt_project is not None and pm4 is not None:
            pm4.append(gt_project_l4)

        return out_3.float(), result_4.float(), pm3, pm4, self.sx, self.sq


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
    for i in range(B - 1):
        a = torch.cat((a, pixel_coords), dim=0)
    return a


def change_intrinsic(intrinsic, RF, rgb_img):
    intrinsic[0][0] = RF.shape[3] / rgb_img.shape[3] * intrinsic[0][0]
    intrinsic[0][2] = RF.shape[3] / rgb_img.shape[3] * intrinsic[0][2]
    intrinsic[1][1] = RF.shape[2] / rgb_img.shape[2] * intrinsic[1][1]
    intrinsic[1][2] = RF.shape[2] / rgb_img.shape[2] * intrinsic[1][2]
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
