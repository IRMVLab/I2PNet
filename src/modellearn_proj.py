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

from src.config_proj import I2PNetConfig as cfg_default
from src.modules.basicConv import createCNNs

from src.projectPN.PPBackbone import CostVolume, ProjSetUpconvModule, ProjectPointNet,PoseHead,FlowPredictor
from src.projectPN.utils import check_valid, project, get_idx_cuda, project_seq
import src.modules.warp_utils as warp_utils
import src.utils as utils

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # make error more user-friendly


# use_bn = False

class RegNet_v2(nn.Module):
    def __init__(self, bn_decay=None, eval_info=False,cfg=cfg_default):
        super(RegNet_v2, self).__init__()
        self.eval_info = eval_info

        self.lidar_Hs = [int(np.ceil(cfg.init_H / s))
                         for s in np.cumprod(cfg.stride_Hs)]
        self.lidar_Ws = [int(np.ceil(cfg.init_W / s))
                         for s in np.cumprod(cfg.stride_Ws)]

        self.LiDAR_lv1 = ProjectPointNet(H=cfg.init_H, W=cfg.init_W,
                                         out_h=self.lidar_Hs[0], out_w=self.lidar_Ws[0],
                                         stride_H=cfg.stride_Hs[0], stride_W=cfg.stride_Ws[0],
                                         kernel_size=cfg.kernel_sizes[0], nsample=cfg.lidar_group_samples[0],
                                         distance=cfg.down_conv_dis[0], in_channel=cfg.lidar_feature_size + 3,
                                         mlp=cfg.lidar_encoder_mlps[0],use_trans=cfg.use_trans,
                                         use_bn_p=cfg.use_bn_p,
                                         use_bn_input=cfg.use_bn_input
                                         )

        self.LiDAR_lv2 = ProjectPointNet(H=self.lidar_Hs[0], W=self.lidar_Ws[0],
                                         out_h=self.lidar_Hs[1], out_w=self.lidar_Ws[1],
                                         stride_H=cfg.stride_Hs[1], stride_W=cfg.stride_Ws[1],
                                         kernel_size=cfg.kernel_sizes[1], nsample=cfg.lidar_group_samples[1],
                                         distance=cfg.down_conv_dis[1], in_channel=cfg.lidar_encoder_mlps[0][-1] + 3,
                                         mlp=cfg.lidar_encoder_mlps[1],use_trans=cfg.use_trans,
                                         use_bn_p=cfg.use_bn_p,
                                         use_bn_input=cfg.use_bn_input
                                         )

        self.LiDAR_lv3 = ProjectPointNet(H=self.lidar_Hs[1], W=self.lidar_Ws[1],
                                         out_h=self.lidar_Hs[2], out_w=self.lidar_Ws[2],
                                         stride_H=cfg.stride_Hs[2], stride_W=cfg.stride_Ws[2],
                                         kernel_size=cfg.kernel_sizes[2], nsample=cfg.lidar_group_samples[2],
                                         distance=cfg.down_conv_dis[2], in_channel=cfg.lidar_encoder_mlps[1][-1] + 3,
                                         mlp=cfg.lidar_encoder_mlps[2],use_trans=cfg.use_trans,
                                         use_bn_p=cfg.use_bn_p,
                                         use_bn_input=cfg.use_bn_input
                                         )

        self.LiDAR_lv4 = ProjectPointNet(H=self.lidar_Hs[2], W=self.lidar_Ws[2],
                                         out_h=self.lidar_Hs[3], out_w=self.lidar_Ws[3],
                                         stride_H=cfg.stride_Hs[3], stride_W=cfg.stride_Ws[3],
                                         kernel_size=cfg.kernel_sizes[3], nsample=cfg.lidar_group_samples[3],
                                         distance=cfg.down_conv_dis[3], in_channel=cfg.lidar_encoder_mlps[2][-1] + 3,
                                         mlp=cfg.lidar_encoder_mlps[3],use_trans=cfg.use_trans,
                                         use_bn_p=cfg.use_bn_p,
                                         use_bn_input=cfg.use_bn_input
                                         )

        self.layer_idx = ProjectPointNet(H=self.lidar_Hs[2], W=self.lidar_Ws[2],
                                         out_h=self.lidar_Hs[3], out_w=self.lidar_Ws[3],
                                         stride_H=cfg.stride_Hs[3], stride_W=cfg.stride_Ws[3],
                                         kernel_size=cfg.kernel_sizes[3], nsample=cfg.lidar_group_samples[4],
                                         distance=cfg.down_conv_dis[3], in_channel=cfg.cost_volume_mlps[-1][-1] + 3,
                                         mlp=cfg.lidar_encoder_mlps[4],use_trans=cfg.use_trans,
                                         use_bn_p=cfg.use_bn_p,
                                         use_bn_input=cfg.use_bn_input
                                         )

        self.RGB_net1 = createCNNs(cfg.rgb_encoder_channels[0][0], cfg.rgb_encoder_channels[0][1],
                                   cfg.rgb_encoder_channels[0][2])

        self.RGB_net2 = createCNNs(cfg.rgb_encoder_channels[1][0], cfg.rgb_encoder_channels[1][1],
                                   cfg.rgb_encoder_channels[1][2])

        self.RGB_net3 = createCNNs(cfg.rgb_encoder_channels[2][0], cfg.rgb_encoder_channels[2][1],
                                   cfg.rgb_encoder_channels[2][2])

        ##########################################################

        self.cost_volume1 = CostVolume(H=self.lidar_Hs[2], W=self.lidar_Ws[2],
                                       kernel_size=cfg.cost_volume_kernel_size[0],
                                       distance=cfg.cost_volume_dis[0],
                                       nsample=cfg.cost_volume_nsamples[0],
                                       nsample_q=cfg.cost_volume_nsamples[1][0],
                                       rgb_in_channels=cfg.rgb_encoder_channels[-1][1][-1],
                                       lidar_in_channels=cfg.lidar_encoder_mlps[-3][-1],  # LF3 dim
                                       mlp1=cfg.cost_volume_mlps[0], mlp2=cfg.cost_volume_mlps[1],
                                       backward_validation=cfg.backward_validation[0],
                                       use_trans=cfg.use_trans,
                                       use_bn_p=cfg.use_bn_p,
                                       use_bn_input=cfg.use_bn_input
                                       )
        self.cost_volume2 = CostVolume(H=self.lidar_Hs[2], W=self.lidar_Ws[2],
                                       kernel_size=cfg.cost_volume_kernel_size[1],
                                       distance=cfg.cost_volume_dis[1],
                                       nsample=cfg.cost_volume_nsamples[0],
                                       nsample_q=cfg.cost_volume_nsamples[1][1],
                                       rgb_in_channels=cfg.rgb_encoder_channels[-1][1][-1],
                                       lidar_in_channels=cfg.lidar_encoder_mlps[-3][-1],  # LF3 dim
                                       mlp1=cfg.cost_volume_mlps[0], mlp2=cfg.cost_volume_mlps[1],
                                       backward_validation=cfg.backward_validation[1],
                                       use_trans=cfg.use_trans,
                                       use_bn_p=cfg.use_bn_p,
                                       use_bn_input=cfg.use_bn_input
                                       )

        ##########################################################
        # mask predictor
        self.flow_predictor0 = FlowPredictor(in_channels=cfg.lidar_encoder_mlps[-2][-1] +  # l4 feature
                                                         cfg.lidar_encoder_mlps[-1][-1]  # resampled_costvolume_l4
                                             , mlp=cfg.flow_predictor_mlps[0], is_training=self.training,
                                             bn_decay=bn_decay,bn=cfg.use_bn_p,
                                                        use_bn_input=cfg.use_bn_input)

        ##########################################################
        self.set_upconv0_w_upsample = ProjSetUpconvModule(H=self.lidar_Hs[-1], W=self.lidar_Ws[-1],
                                                          out_h=self.lidar_Hs[-2], out_w=self.lidar_Ws[-2],
                                                          kernel_size=cfg.up_conv_kernel_size[0],
                                                          nsample=cfg.setupconv_nsamples[0],
                                                          stride_H=cfg.stride_Hs[-1], stride_W=cfg.stride_Ws[-1],
                                                          distance=cfg.up_conv_dis[0],
                                                          in_channels=[cfg.lidar_encoder_mlps[-3][-1],
                                                                       cfg.flow_predictor_mlps[0][-1]],
                                                          mlp=cfg.setupconv_mlps[0][0],
                                                          mlp2=cfg.setupconv_mlps[0][1],
                                                          use_trans=cfg.use_trans,
                                                          use_bn_p=cfg.use_bn_p,
                                                          use_bn_input=cfg.use_bn_input
                                                          )
        self.set_upconv0_upsample = ProjSetUpconvModule(H=self.lidar_Hs[-1], W=self.lidar_Ws[-1],
                                                        out_h=self.lidar_Hs[-2], out_w=self.lidar_Ws[-2],
                                                        kernel_size=cfg.up_conv_kernel_size[1],
                                                        nsample=cfg.setupconv_nsamples[1],
                                                        stride_H=cfg.stride_Hs[-1], stride_W=cfg.stride_Ws[-1],
                                                        distance=cfg.up_conv_dis[1],
                                                        in_channels=[cfg.lidar_encoder_mlps[-3][-1],  # l3 feature
                                                                     cfg.lidar_encoder_mlps[-1][-1]],  # resampled l4 EM
                                                        mlp=cfg.setupconv_mlps[1][0],
                                                        mlp2=cfg.setupconv_mlps[1][1],
                                                        use_trans=cfg.use_trans,
                                                        use_bn_p=cfg.use_bn_p,
                                                        use_bn_input=cfg.use_bn_input)
        ###############################################
        self.flow_predictor0_predict = FlowPredictor(in_channels=cfg.lidar_encoder_mlps[-3][-1] +  # LF3 dim
                                                                 cfg.setupconv_mlps[1][1][-1] +  # upsampled l4 EM dim
                                                                 cfg.cost_volume_mlps[-1][-1]  # l3 EM dim
                                                     , mlp=cfg.flow_predictor_mlps[1],
                                                     is_training=self.training,
                                                     bn_decay=bn_decay,bn=cfg.use_bn_p,use_bn_input=cfg.use_bn_input)
        self.flow_predictor0_w = FlowPredictor(in_channels=cfg.lidar_encoder_mlps[-3][-1] +  # LF3 dim
                                                           cfg.setupconv_mlps[0][-1][-1] +  # upsampled l4 mask dim
                                                           cfg.flow_predictor_mlps[1][-1],  # l3 refined EM dim
                                               mlp=cfg.flow_predictor_mlps[2], is_training=self.training,
                                               bn_decay=bn_decay,bn=cfg.use_bn_p,use_bn_input=cfg.use_bn_input)

        ##########################################################
        self.l4_head = PoseHead(in_channels=[cfg.lidar_encoder_mlps[-1][-1], cfg.lidar_encoder_mlps[-2][-1]],
                                mlp1=cfg.pose_head_mlps[0][0],
                                mlp2=cfg.pose_head_mlps[0][1],
                                hidden=cfg.head_hidden_dim,
                                q_dim=cfg.rotation_quat_head_dim,
                                t_dim=cfg.transition_vec_head_dim,
                                dropout_rate=cfg.head_dropout_rate,
                                split_dp=cfg.split_dp,
                                pos_embed=cfg.head_pos_embedding,
                                sigmoid=cfg.mask_sigmoid,
                                maxhead=cfg.max_head)
        self.l3_head = PoseHead(in_channels=[cfg.flow_predictor_mlps[1][-1], cfg.lidar_encoder_mlps[-3][-1]],
                                mlp1=cfg.pose_head_mlps[1][0],
                                mlp2=cfg.pose_head_mlps[1][1],
                                hidden=cfg.head_hidden_dim,
                                q_dim=cfg.rotation_quat_head_dim,
                                t_dim=cfg.transition_vec_head_dim,
                                dropout_rate=cfg.head_dropout_rate,
                                split_dp=cfg.split_dp,
                                pos_embed=cfg.head_pos_embedding,
                                sigmoid=cfg.mask_sigmoid,
                                maxhead=cfg.max_head)

        ##########################################################
        # loss learnable parameters
        self.sq = torch.nn.Parameter(torch.tensor([cfg.sq_init]), requires_grad=True)
        self.sx = torch.nn.Parameter(torch.tensor([cfg.sx_init]), requires_grad=True)



    def init_weights(self, layer):
        """not used"""
        if type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        elif type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, rgb_img, lidar_img, lidar_img_raw, H_initial, intrinsic, resize_img,
                gt_project=None, calib=None, lidar_feature=None,cfg=cfg_default):
        # lidar_img_raw is intended for the fast query

        device = rgb_img.device
        # B,3,3
        intrinsic = intrinsic.float()  # camera matrix is the same

        B, _, h, w = rgb_img.shape  # [B,3,352,1216]
        _, N, _ = lidar_img.shape
        if cfg.debug_time:
            cfg.debug_timing.reset()
        # image branch

        RF1 = self.RGB_net1(rgb_img)

        RF2 = self.RGB_net2(RF1)

        RF3 = self.RGB_net3(RF2)  # [B,256,11,38]
        if cfg.debug_time:
            cfg.debug_timing.time("rgb_ex")
        # img discrete index
        RF3_index = set_id_grid(RF3.permute(0, 2, 3, 1))  # [B,418,3]

        # lidar feature initial as zeros
        if lidar_feature is None:
            lidar_norm = torch.zeros(B, N, 3, device=device)
        else:
            lidar_norm = lidar_feature

        # proj
        lidar_img_raw, feats = project_seq(lidar_img_raw.float(), [lidar_norm.float(), lidar_img.float()], cfg.init_H,
                                           cfg.init_W, cfg.rank,cfg.fup,cfg.fdown)

        lidar_norm, lidar_img = feats

        if cfg.debug_time:
            cfg.debug_timing.time("projection")
        # [B,H,W,3] [B,H,W,C]
        # lidar layers
        #print("lidar_img shape:", lidar_img.shape)
        P1_raw, P1, LF1, _, sample_idx_1 = self.LiDAR_lv1(lidar_img_raw, lidar_img, lidar_norm,cfg=cfg)
        P2_raw, P2, LF2, _, sample_idx_2 = self.LiDAR_lv2(P1_raw, P1, LF1,cfg=cfg)
        P3_raw, P3, LF3, _, sample_idx_3 = self.LiDAR_lv3(P2_raw, P2, LF2,cfg=cfg)  # LF3.shape[B,C,N]=[B,64,256]
        P4_raw, P4, LF4, _, sample_idx_4 = self.LiDAR_lv4(P3_raw, P3, LF3,cfg=cfg)  # LF4.shape[B,C,N]=[B,128,64]
        if cfg.debug_time:
            cfg.debug_timing.time("point_ex")
        if cfg.debug and cfg.debug_count < cfg.debug_storage:
            def point_save(pc):
                pc = pc[0]
                pc_img = pc.clone()
                torch.any(torch.ne(pc, 0), dim=-1)
                pc = pc[torch.any(torch.ne(pc, 0), dim=-1),:]
                return (pc.view(-1,3).cpu().numpy(),pc_img.cpu().numpy())
            cfg.debug_dict["global_point_sample"].append([
                point_save(lidar_img_raw),
                point_save(P1_raw),
                point_save(P2_raw),
                point_save(P3_raw),
                point_save(P4_raw),
            ])

        # project the uv to normalized camera plane
        intrinsic_3 = change_intrinsic(intrinsic, RF3, rgb_img)  # K3
        # cuda not support matrix inverse
        # TODO: inverse bug
        # intrinsic_3_inv = torch.inverse(intrinsic_3)
        intrinsic_3_inv = torch.inverse(intrinsic_3.cpu()).to(P4.device)  # B,3,3
        RF3_index = torch.bmm(intrinsic_3_inv, RF3_index.permute(0, 2, 1))  # B,3,3 @ B,3,N
        RF3_index = RF3_index.permute(0, 2, 1)  # [B,418,3]

        # project to the normalization camera plane
        H3, W3 = self.lidar_Hs[2], self.lidar_Ws[2]
        P3_l4 = P3.reshape(B, H3 * W3, 3)  # B,H*W,3
        LF3_cv1 = LF3.reshape(B, H3 * W3, -1)

        lidar_z = P3_l4[:, :, 2:]
        lidar_uv = P3_l4 / (lidar_z + 1e-10)

        _, C, H, W = RF3.shape

        RF3 = RF3.reshape(B, C, H * W).permute(0, 2, 1)  # B,N,C

        RF3_cv1 = RF3

        l3_idx_n2 = get_idx_cuda(B, H3, W3, device)
        if cfg.debug_time:
            cfg.debug_timing.time("cv1_pre")
        # B,H3,W3,C embedding cost volume
        concat_4 = self.cost_volume1(P3_raw, lidar_uv, LF3_cv1, l3_idx_n2, RF3_index, RF3_cv1,
                                     lidar_z,cfg=cfg)  # [B,256,64]=[B,N,C]
        if cfg.debug_time:
            cfg.debug_timing.time("cv1")
        # resample the cost volume to l4 B,H,W,C
        _, _, l4_points_f1_cost_volume, _, _ = self.layer_idx(P3_raw, P3, concat_4, sample_idx=sample_idx_4,cfg=cfg)
        # E4_new
        l4_points_predict = l4_points_f1_cost_volume

        l4_valid_mask = check_valid(P4_raw).view(B, -1, 1)
        H4, W4 = self.lidar_Hs[-1], self.lidar_Ws[-1]
        # mask F+E [B,N,C]
        l4_cost_volume_w = self.flow_predictor0(LF4.view(B, H4 * W4, -1), None, l4_points_predict.view(B, H4 * W4, -1))
        # valid mask
        l4_cost_volume_w = l4_cost_volume_w * l4_valid_mask + -1e10 * (1 - l4_valid_mask)

        l4_projection_mask = None

        result_4_real, result_4_dual, _ = self.l4_head(l4_points_predict.view(B, H4 * W4, -1), l4_cost_volume_w,
                                                       P4.view(B, H4 * W4, 3), LF4.view(B, H4 * W4, -1),
                                                       l4_projection_mask)

        result_4 = torch.cat([result_4_real, result_4_dual], dim=1)

        decalib_quat_real4 = result_4_real
        decalib_quat_dual4 = result_4_dual
        if cfg.debug_time:
            cfg.debug_timing.time("l4_reg")
        # layer 3  ################################################
        # [0,t4]
        H3_trans = torch.cat([torch.zeros((B, 1), device=device),
                              decalib_quat_dual4], -1).reshape(B, 4)
        # p3_warp = q4*[0,p3]*q4'+[0,t4]

        l3_nowarp_valid_mask = check_valid(P3_l4)  # B,N,1

        P3_warped_l3 = warp_utils.warp_quat_xyz(P3_l4, decalib_quat_real4,
                                                H3_trans) * l3_nowarp_valid_mask

        # mask upsampling
        l3_cost_volume_w_upsample = self.set_upconv0_w_upsample(
            P3_raw, P4_raw, P3, P4, l3_idx_n2, LF3, l4_cost_volume_w.view(B, H4, W4, -1),cfg=cfg)  # [B,H,W,64]

        l3_cost_volume_upsample = self.set_upconv0_upsample(
            P3_raw, P4_raw, P3, P4, l3_idx_n2, LF3, l4_points_predict,cfg=cfg)  # [B,H,W,64]
        if cfg.debug_time:
            cfg.debug_timing.time("upsample")
        # reproject [B,H,W,3] [B,H,W,C]
        # P3_warped_proj, feats = project_seq(P3_warped_l3, [LF3_cv1, l3_cost_volume_upsample,
        #                                                    l3_cost_volume_w_upsample], H3, W3)
        # LF3_proj = LF3_cv1
        # LF3_proj, l3_cost_volume_upsample, l3_cost_volume_w_upsample = feats
        # P3_warped_n3 = P3_warped_proj.reshape(B, H3 * W3, 3)

        lidar_z = P3_warped_l3[:, :, 2:]
        lidar_uv = P3_warped_l3 / (lidar_z + 1e-10)

        # sampled_points = lidar_uv.shape[1]
        LF3_cv2 = LF3_cv1

        concat_3 = self.cost_volume2(P3_raw, lidar_uv, LF3_cv2, l3_idx_n2, RF3_index, RF3, lidar_z,cfg=cfg)
        if cfg.debug_time:
            cfg.debug_timing.time("cv2")
        # predict refined embedding
        l3_cost_volume_predict = self.flow_predictor0_predict(
            LF3_cv2, l3_cost_volume_upsample.view(B, H3 * W3, -1), concat_3.view(B, H3 * W3, -1))
        # predict refined mask
        l3_cost_volume_w = self.flow_predictor0_w(
            LF3_cv2, l3_cost_volume_w_upsample.view(B, H3 * W3, -1), l3_cost_volume_predict)

        valid_mask_l3 = check_valid(P3_raw).view(B, -1, 1)

        l3_cost_volume_w = l3_cost_volume_w * valid_mask_l3 + -1e10 * (1 - valid_mask_l3)
        l3_prediction_mask = None

        result_3_real, result_3_dual, W_l3_cost_volume = self.l3_head(l3_cost_volume_predict, l3_cost_volume_w,
                                                                      P3_warped_l3, LF3_cv2,
                                                                      l3_prediction_mask)
        if cfg.debug_time:
            cfg.debug_timing.time("l3_reg")
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

        if cfg.debug and cfg.debug_count < cfg.debug_storage:
            cfg.debug_count += 1
            if cfg.debug_count == cfg.debug_storage:
                os.makedirs(cfg.debug_dir, exist_ok=True)
                if cfg.debug_no_sample:
                    del cfg.debug_dict["global_point_sample"]
                    cfg.debug_dict["global_point_sample"] = None
                torch.save(cfg.debug_dict, os.path.join(cfg.debug_dir, cfg.debug_path))
                print("[INFO] debug file saved...")
        if self.eval_info:
            return out_3.float(), result_4.float(), self.sx, self.sq, \
                   W_l3_cost_volume, P3_l4, l3_prediction_mask, l4_projection_mask, \
                   P4.view(B, H4 * W4, 3)

        pm3 = None

        pm4 = None

        return out_3.float(), result_4.float(), pm3, pm4, self.sx, self.sq
    def set_bn(self):
        self.flow_predictor0.set_bn()
        self.flow_predictor0_w.set_bn()
        self.flow_predictor0_predict.set_bn()
        self.LiDAR_lv1.set_bn()
        self.LiDAR_lv2.set_bn()
        self.LiDAR_lv3.set_bn()
        self.LiDAR_lv4.set_bn()
        self.layer_idx.set_bn()
        self.set_upconv0_upsample.set_bn()
        self.set_upconv0_w_upsample.set_bn()
        self.cost_volume1.set_bn()
        self.cost_volume2.set_bn()


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
    a = pixel_coords.repeat(B, 1, 1)
    # for i in range(B - 1):
    #     a = torch.cat((a, pixel_coords), dim=0)
    return a


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
