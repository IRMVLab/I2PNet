# !/usr/bin/env Python
# coding=utf-8
import torch
import torch.nn as nn
import os

from src.config_sp import I2PNetConfig as cfg_default
from src.modules.basicConv import createCNNs

import src.modules.warp_utils as warp_utils
from src.modules.basicSPConv import ResidualBlock, BasicConvolutionBlock, UpSampleLayer, CostVolume, FlowPredictor, \
    PoseHead, get_downsample_info, nc2bnc
import spconv.pytorch as spconv


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # make error more user-friendly

class RegNet_v2(nn.Module):
    def __init__(self, bn_decay=None, eval_info=False, cfg=cfg_default):
        super(RegNet_v2, self).__init__()
        self.eval_info = eval_info

        lidarc = cfg.lidar_channels
        lidarstrides = cfg.lidar_strides

        self.stem1 = BasicConvolutionBlock(cfg.lidar_feature_size, lidarc[0],
                                           indice_key='stem', subm=True)

        self.stem2 = BasicConvolutionBlock(lidarc[0], lidarc[0],
                                           indice_key='stem', subm=True)

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(lidarc[0], lidarc[0], ks=lidarstrides[0], stride=lidarstrides[0],
                                  indice_key="stem_down"),
            ResidualBlock(lidarc[0], lidarc[1], ks=3, stride=1, indice_key='conv1'),
            ResidualBlock(lidarc[1], lidarc[1], ks=3, stride=1, indice_key='conv1')
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(lidarc[1], lidarc[1], ks=lidarstrides[1], stride=lidarstrides[1],
                                  indice_key="conv1_down"),
            ResidualBlock(lidarc[1], lidarc[2], ks=3, stride=1, indice_key='conv2'),
            ResidualBlock(lidarc[2], lidarc[2], ks=3, stride=1, indice_key='conv2')
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(lidarc[2], lidarc[2], ks=lidarstrides[2], stride=lidarstrides[2],
                                  indice_key="conv2_down"),
            ResidualBlock(lidarc[2], lidarc[3], ks=3, stride=1, indice_key='conv3'),
            ResidualBlock(lidarc[3], lidarc[3], ks=3, stride=1, indice_key='conv3')
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(lidarc[3], lidarc[3], ks=lidarstrides[3], stride=lidarstrides[3],
                                  indice_key="conv3_down"),
            ResidualBlock(lidarc[3], lidarc[4], ks=3, stride=1, indice_key='conv4'),
            ResidualBlock(lidarc[4], lidarc[4], ks=3, stride=1, indice_key='conv4')
        )

        self.layer_idx = nn.Sequential(
            BasicConvolutionBlock(cfg.cost_volume_mlps[-1][-1],
                                  cfg.cost_volume_mlps[-1][-1], ks=lidarstrides[3], stride=lidarstrides[3],
                                  indice_key="conv3_down"),
            ResidualBlock(cfg.cost_volume_mlps[-1][-1], lidarc[5], ks=3, stride=1, indice_key='conv4'),
            ResidualBlock(lidarc[5], lidarc[5], ks=3, stride=1, indice_key='conv4')
        )

        self.RGB_net1 = createCNNs(cfg.rgb_encoder_channels[0][0], cfg.rgb_encoder_channels[0][1],
                                   cfg.rgb_encoder_channels[0][2])

        self.RGB_net2 = createCNNs(cfg.rgb_encoder_channels[1][0], cfg.rgb_encoder_channels[1][1],
                                   cfg.rgb_encoder_channels[1][2])

        self.RGB_net3 = createCNNs(cfg.rgb_encoder_channels[2][0], cfg.rgb_encoder_channels[2][1],
                                   cfg.rgb_encoder_channels[2][2])

        ##########################################################

        self.cost_volume1 = CostVolume(nsample=cfg.cost_volume_nsamples[0],
                                       nsample_q=cfg.cost_volume_nsamples[1][0],
                                       rgb_in_channels=cfg.rgb_encoder_channels[-1][1][-1],
                                       lidar_in_channels=lidarc[3],  # LF3 dim
                                       backward_validation=cfg.backward_validation[0],
                                       mlp1=cfg.cost_volume_mlps[0], mlp2=cfg.cost_volume_mlps[1],
                                       )
        self.cost_volume2 = CostVolume(nsample=cfg.cost_volume_nsamples[0],
                                       nsample_q=cfg.cost_volume_nsamples[1][1],
                                       rgb_in_channels=cfg.rgb_encoder_channels[-1][1][-1],
                                       lidar_in_channels=lidarc[3],  # LF3 dim
                                       mlp1=cfg.cost_volume_mlps[0], mlp2=cfg.cost_volume_mlps[1],
                                       backward_validation=cfg.backward_validation[1])

        ##########################################################
        # mask predictor
        self.flow_predictor0 = FlowPredictor(in_channels=lidarc[4] +  # l4 feature
                                                         lidarc[5]
                                             , mlp=cfg.flow_predictor_mlps[0],
                                             is_training=self.training,
                                             bn_decay=bn_decay)

        ##########################################################
        self.set_upconv0_w_upsample = UpSampleLayer(
            inc1=cfg.flow_predictor_mlps[0][-1],  # flow_predictor0
            out_c1=cfg.up_channels[0][0],
            inc2=lidarc[3],  # l3 feature
            out_c2=cfg.up_channels[0][1], ks=lidarstrides[3],
            indice_key_down='conv3_down',
            indice_key='conv3'
        )

        self.set_upconv0_upsample = UpSampleLayer(
            inc1=lidarc[5],  # resampled l4 EM
            out_c1=cfg.up_channels[1][0],
            inc2=lidarc[3],  # l3 feature
            out_c2=cfg.up_channels[1][1], ks=lidarstrides[3],
            indice_key_down='conv3_down',
            indice_key='conv3'
        )
        ###############################################
        self.flow_predictor0_predict = FlowPredictor(in_channels=lidarc[3] +  # LF3 dim
                                                                 cfg.up_channels[0][1] +  # upsampled l4 EM dim
                                                                 cfg.cost_volume_mlps[-1][-1]  # l3 EM dim
                                                     , mlp=cfg.flow_predictor_mlps[1],
                                                     is_training=self.training,
                                                     bn_decay=bn_decay)
        self.flow_predictor0_w = FlowPredictor(in_channels=lidarc[3] +  # LF3 dim
                                                           cfg.up_channels[0][1] +  # upsampled l4 mask dim
                                                           cfg.flow_predictor_mlps[1][-1],  # l3 refined EM dim
                                               mlp=cfg.flow_predictor_mlps[2], is_training=self.training,
                                               bn_decay=bn_decay)

        ##########################################################
        self.l4_head = PoseHead(in_channels=lidarc[5],  # context gathered feature
                                hidden=cfg.head_hidden_dim,
                                q_dim=cfg.rotation_quat_head_dim,
                                t_dim=cfg.transition_vec_head_dim,
                                dropout_rate=cfg.head_dropout_rate,
                                split_dp=cfg.split_dp)
        self.l3_head = PoseHead(in_channels=cfg.flow_predictor_mlps[1][-1],  # flow predict0_predict

                                hidden=cfg.head_hidden_dim,
                                q_dim=cfg.rotation_quat_head_dim,
                                t_dim=cfg.transition_vec_head_dim,
                                dropout_rate=cfg.head_dropout_rate,
                                split_dp=cfg.split_dp)

        ##########################################################
        # loss learnable parameters
        self.sq = torch.nn.Parameter(torch.tensor([cfg.sq_init]), requires_grad=True)
        self.sx = torch.nn.Parameter(torch.tensor([cfg.sx_init]), requires_grad=True)

    def forward(self, batch_data, cfg=cfg_default):
        rgb_img = batch_data["rgb"]
        intrinsic = batch_data["init_intrinsic"]
        lidar_feature = batch_data["lidar_feats"]
        voxel_coord = batch_data["voxel_coord"].int()
        spatial_size = batch_data["spatial_size"]
        batch_size = batch_data["batch_size"]
        lidar_img_voxel = batch_data["lidar_img_voxel"]
        batch_info = batch_data["batch_info"].int()

        device = rgb_img.device
        # B,3,3
        intrinsic = intrinsic.float()  # camera matrix is the same

        B, _, h, w = rgb_img.shape  # [B,3,352,1216]

        # image branch

        RF1 = self.RGB_net1(rgb_img)

        RF2 = self.RGB_net2(RF1)

        RF3 = self.RGB_net3(RF2)  # [B,256,11,38]

        # img discrete index
        RF3_index = set_id_grid(RF3.permute(0, 2, 3, 1))  # [B,418,3]

        # lidar feature initial as zeros

        lidar_norm = lidar_feature

        x = spconv.SparseConvTensor(lidar_norm, voxel_coord, spatial_size,
                                    batch_size)

        # [B,H,W,3] [B,H,W,C]
        # lidar layers

        LF_stem = self.stem2(self.stem1(x))
        LF1 = self.stage1(LF_stem)

        P1, B1 = get_downsample_info(lidar_img_voxel, batch_info,
                                     LF1.indice_dict["stem_down"].indice_pairs)

        LF2 = self.stage2(LF1)

        P2, B2 = get_downsample_info(P1, B1, LF2.indice_dict["conv1_down"].indice_pairs)

        LF3 = self.stage3(LF2)

        P3, B3 = get_downsample_info(P2, B2, LF3.indice_dict["conv2_down"].indice_pairs)

        LF4 = self.stage4(LF3)

        P4, B4 = get_downsample_info(P3, B3, LF4.find_indice_pair("conv3_down").indice_pairs)

        # project the uv to normalized camera plane
        intrinsic_3 = change_intrinsic(intrinsic, RF3, rgb_img)  # K3
        # cuda not support matrix inverse
        # TODO: inverse bug
        # intrinsic_3_inv = torch.inverse(intrinsic_3)
        intrinsic_3_inv = torch.inverse(intrinsic_3.cpu()).to(P3.device)  # B,3,3
        RF3_index = torch.bmm(intrinsic_3_inv, RF3_index.permute(0, 2, 1))  # B,3,3 @ B,3,N
        RF3_index = RF3_index.permute(0, 2, 1)  # [B,418,3]

        lidar_z = P3[:, 2:]
        lidar_uv = P3 / (lidar_z + 1e-10)

        _, C, H, W = RF3.shape

        RF3 = RF3.reshape(B, C, H * W).permute(0, 2, 1)  # B,N,C

        RF3_cv1 = RF3

        # B,H3,W3,C embedding cost volume
        concat_4 = self.cost_volume1(lidar_uv, LF3.features, B3, batch_size,
                                     RF3_index, RF3_cv1,
                                     lidar_z)

        # resample the cost volume to l4 B,H,W,C
        l4_points_f1_cost_volume = self.layer_idx(LF3.replace_feature(concat_4))
        # E4_new
        l4_points_predict = l4_points_f1_cost_volume.features

        # mask F+E [B,N,C]
        l4_cost_volume_w = self.flow_predictor0(LF4.features, None, l4_points_predict)

        result_4_real, result_4_dual = self.l4_head(l4_points_predict, l4_cost_volume_w, B4)

        result_4 = torch.cat([result_4_real, result_4_dual], dim=1)

        decalib_quat_real4 = result_4_real
        decalib_quat_dual4 = result_4_dual

        H3_trans = torch.cat([torch.zeros((B, 1), device=device),
                              decalib_quat_dual4], -1).reshape(B, 4)
        # p3_warp = q4*[0,p3]*q4'+[0,t4]

        bnc, inv = nc2bnc([P3], B3, batch_size)
        P3_bnc = bnc[0]

        P3_warped_l3 = warp_utils.warp_quat_xyz(P3_bnc, decalib_quat_real4,
                                                H3_trans)

        b, n, c = P3_warped_l3.shape
        P3_warped_l3 = P3_warped_l3.reshape(b * n, c).gather(0, inv[:, None].repeat(1, c))

        # mask upsampling
        l3_cost_volume_w_upsample = self.set_upconv0_w_upsample(LF4.replace_feature(
            l4_cost_volume_w
        ), LF3.features)

        l3_cost_volume_upsample = self.set_upconv0_upsample(LF4.replace_feature(
            l4_points_predict
        ), LF3.features)

        lidar_z = P3_warped_l3[:, 2:]
        lidar_uv = P3_warped_l3 / (lidar_z + 1e-10)

        concat_3 = self.cost_volume2(lidar_uv, LF3.features, B3, batch_size,
                                     RF3_index, RF3_cv1,
                                     lidar_z)

        # predict refined embedding
        l3_cost_volume_predict = self.flow_predictor0_predict(LF3.features,
                                                              l3_cost_volume_upsample,
                                                              concat_3)
        # predict refined mask
        l3_cost_volume_w = self.flow_predictor0_w(LF3.features,
                                                  l3_cost_volume_w_upsample,
                                                  l3_cost_volume_predict)

        result_3_real, result_3_dual = self.l3_head(l3_cost_volume_predict, l3_cost_volume_w, B3)

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

        return out_3.float(), result_4.float(), None, None, self.sx, self.sq


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
