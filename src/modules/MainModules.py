import torch.nn as nn
import torch
from src.modules.basicConv import Conv2d, Conv1d
from src.modules.point_utils import grouping
import torch.nn.functional as F
from enum import Enum
import math


class FlowPredictor(nn.Module):

    def __init__(self, in_channels, mlp, is_training, bn_decay, bn=True):
        super(FlowPredictor, self).__init__()
        self.in_channels = in_channels
        self.mlp = mlp
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.mlp_conv = nn.ModuleList()

        for i, num_out_channel in enumerate(mlp):
            self.mlp_conv.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=bn))
            self.in_channels = num_out_channel

    def forward(self, points_f1, upsampled_feat, cost_volume):
        """
        Input:
            points_f1: (b,n,c1)
            upsampled_feat: (b,n,c2)
            cost_volume: (b,n,c3)
        Output:
            points_concat:(b,n,mlp[-1])
        """
        if upsampled_feat is not None:
            points_concat = torch.cat([points_f1, cost_volume, upsampled_feat], -1)  # b,n,c1+c2+c3
        else:
            points_concat = torch.cat([points_f1, cost_volume], -1)

        points_concat = torch.unsqueeze(points_concat, 2)  # B,n,1,c1+c2+c3

        for i, conv in enumerate(self.mlp_conv):
            points_concat = conv(points_concat)

        points_concat = torch.squeeze(points_concat, 2)

        return points_concat


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.num_heads = 4
        self.scale = 1. / math.sqrt(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, feature):
        # feature B,N,C
        B, N, C = feature.shape
        assert C % self.num_heads == 0, f"[ASSERT] {C} dim not suitable for {self.num_heads} heads."
        qkv = self.qkv(feature).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # B,N,3C
        q, k, v = qkv[0], qkv[1], qkv[2]  # B,H,N,C//H

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B,H,N,N
        attn = F.softmax(attn, -1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.qkv_proj = nn.Linear(dim, dim, bias=True)
        self.num_heads = 4
        self.scale = 1. / math.sqrt(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, F1, F2):
        B, N, C = F1.shape
        B, M, C = F2.shape
        assert C % self.num_heads == 0, f"[ASSERT] {C} dim not suitable for {self.num_heads} heads."

        # B,
        q = self.qkv_proj(F1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.qkv_proj(F2).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.qkv_proj(F2).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B,H,N,M
        attn = F.softmax(attn, -1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


# class CrossAttention2(nn.Module):
#     def __init__(self, N, M, r=6, tau=0.025):
#         """
#         Cross Attention Network for Few-shot Classification
#         Args:
#             N: Number of querying
#             M: Number of Key
#             r: reduction factor
#             tau: temperature rate
#         """
#         super(CrossAttention2, self).__init__()
#         self.tau = tau
#
#         # the correlation vector to key is the length of query
#         self.wk1 = nn.Conv1d(N, N // r, 1, bias=False)
#         self.wk2 = nn.Conv1d(N // r, N, 1, bias=False)
#         self.wq1 = nn.Conv1d(M, M // r, 1, bias=False)
#         self.wq2 = nn.Conv1d(M // r, M, 1, bias=False)
#
#     def forward(self, q, k):
#         # q [B,N,C] k [B,M,C] return v [B,N,C]
#         # (1) CAM
#         nq = F.normalize(q, p=2, dim=-1)
#         nk = F.normalize(k, p=2, dim=-1)
#         cross_dis = (nq.unsqueeze(2)) * (nk.unsqueeze(1))  # B,N,M,C
#         cross_dis = cross_dis.sum(-1)
#
#         cross_dis_q = cross_dis.permute(0, 2, 1)  # B,M,N
#         cross_dis_k = cross_dis_q
#
#         # w = W2(σ(W1(GAP(Rp)) use the average correlation to generate
#         # correlation aggregation weights MLP(mapping)
#         # B,M,1
#         wq = self.wq2(F.relu(self.wq1(torch.mean(cross_dis_q, dim=-1, keepdim=True))))
#         # B,N,1
#         wk = self.wk2(F.relu(self.wk1(torch.mean(cross_dis_k, dim=-1, keepdim=True))))
#         # aggregate to a scalar attention weight
#         Aq = F.softmax(torch.matmul(cross_dis_k, wq) / self.tau, -1)  # B,N
#         Ak = F.softmax(torch.matmul(cross_dis_q, wk) / self.tau, -1)  # B,M
#
#         return (Aq * q + q), (Ak * k + k)


class CostVolume(nn.Module):
    class CorrFunc(Enum):
        ELEMENTWISE_PRODUCT = 1
        CONCAT = 2
        COSINE_DISTANCE = 3

    def __init__(self, radius, nsample, nsample_q, rgb_in_channels, lidar_in_channels, mlp1, mlp2,
                 is_training, bn_decay, bn=True, pooling='max', knn=True, corr_func=CorrFunc.ELEMENTWISE_PRODUCT,
                 backward_validation=False, max_cost=False, backward_fc=False):
        super(CostVolume, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.nsample_q = nsample_q
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.corr_func = corr_func
        self.backward_validation = backward_validation
        self.max_cost = max_cost
        self.backward_fc = backward_fc

        if corr_func == CostVolume.CorrFunc.CONCAT:
            corr_channel = rgb_in_channels + lidar_in_channels
        elif corr_func == CostVolume.CorrFunc.ELEMENTWISE_PRODUCT:
            corr_channel = rgb_in_channels
        elif corr_func == CostVolume.CorrFunc.COSINE_DISTANCE:
            corr_channel = rgb_in_channels
        else:
            raise NotImplementedError

        if backward_validation:
            corr_channel += lidar_in_channels

        if backward_fc and backward_validation:
            self.inverse_fc = Conv2d(lidar_in_channels, lidar_in_channels, [1, 1], [1, 1], bn=True)

        self.in_channels = corr_channel + 6

        self.mlp1_convs = nn.ModuleList()
        if not self.max_cost:
            self.mlp2_convs = nn.ModuleList()
        self.mlp2_convs_2 = nn.ModuleList()

        for i, num_out_channel in enumerate(mlp1):
            self.mlp1_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        self.pi_encoding = Conv2d(6, mlp1[-1], [1, 1], stride=[1, 1], bn=True)

        if not self.max_cost:
            self.in_channels = 2 * mlp1[-1]
            for j, num_out_channel in enumerate(mlp2):
                self.mlp2_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
                self.in_channels = num_out_channel

        self.pc_encoding = Conv2d(10, mlp1[-1], [1, 1], stride=[1, 1], bn=True)

        self.in_channels = 2 * mlp1[-1] + lidar_in_channels
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs_2.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

    def forward(self, warped_xyz, warped_points, f2_xyz, f2_points, lidar_z):
        """
            Input:
                warped_xyz: (b,npoint,3)
                warped_points: (b,npoint,c)
                f2_xyz:  (b,ndataset,3)
                f2_points: (b,ndataset,c)
                lidar_z: (b, npoint, 1)

            Output:
                pc_feat1_new: batch_size, npoints, mlp2[-1]
        """

        # normalized 3d point searches image pixels
        # [B,N,K,3] [B,N,K,C] [B,N,K]
        if self.nsample_q > 0:
            qi_xyz_grouped, _, qi_points_grouped, idx, _ = grouping(f2_points, self.nsample_q, f2_xyz,
                                                                 warped_xyz)
        else:
            # B,N1,N2,C
            qi_xyz_grouped, qi_points_grouped = f2_xyz.unsqueeze(1).repeat(1, warped_xyz.shape[1], 1, 1), \
                                                f2_points.unsqueeze(1).repeat(1, warped_xyz.shape[1], 1, 1)

        # important
        warped_xyz = warped_xyz.mul(lidar_z)  # restore depth

        # lidar_z_repeat = torch.unsqueeze(lidar_z, dim=2).repeat(1, 1, self.nsample_q, 1)
        K = qi_xyz_grouped.shape[2]
        pi_xyz_expanded = warped_xyz[:, :, None, :].repeat(1, 1, K, 1)  # B,N,K,3
        pi_points_expanded = warped_points[:, :, None, :].repeat(1, 1, K, 1)  # B,N,K,C

        # pi_xyz_diff = qi_xyz_grouped - pi_xyz_expanded  # batch_size, npoints, nsample_q, 3

        # pi_euc_diff = torch.sqrt(torch.sum(torch.mul(pi_xyz_diff, pi_xyz_diff), dim=-1,
        #                                   keepdim=True) + 1e-20)  # batch_size, npoints, nsample_q, 1

        # position embedding
        pi_xyz_diff_concat = torch.cat([pi_xyz_expanded, qi_xyz_grouped], dim=3)  # B,N,K,6

        if self.corr_func == CostVolume.CorrFunc.ELEMENTWISE_PRODUCT:
            pi_points_expanded = (pi_points_expanded - torch.mean(pi_points_expanded, -1, keepdim=True)) / torch.clip(torch.std(
                pi_points_expanded, -1, keepdim=True),min=1e-12)
            qi_points_grouped = (qi_points_grouped - torch.mean(qi_points_grouped, -1, keepdim=True)) / torch.clip(torch.std(
                qi_points_grouped, -1, keepdim=True),min=1e-12)

            pi_feat_diff = pi_points_expanded * qi_points_grouped  # B,N,K,C

        elif self.corr_func == CostVolume.CorrFunc.CONCAT:

            pi_feat_diff = torch.cat([pi_points_expanded, qi_points_grouped], dim=-1)  # B,N,K,2c

        elif self.corr_func == CostVolume.CorrFunc.COSINE_DISTANCE:
            pi_points_expanded = F.normalize(pi_points_expanded, p=2, dim=-1, eps=1e-12)

            qi_points_grouped = F.normalize(qi_points_grouped, p=2, dim=-1, eps=1e-12)

            pi_feat_diff = pi_points_expanded * qi_points_grouped  # B,N,K,C
        else:
            raise NotImplementedError
        pi_feat1_new = torch.cat([pi_xyz_diff_concat, pi_feat_diff], dim=3)  # B,N,K, 6+c

        if self.backward_validation:
            # B,N1,N2,C
            repeat_image_feature = qi_points_grouped
            # B,N1,N2,C
            repeat_lidar_feature = pi_points_expanded
            # correlation
            repeat_correlation = repeat_image_feature * repeat_lidar_feature
            image_max_respond = torch.max(repeat_correlation, 1, keepdim=True)[0].repeat(1, warped_xyz.shape[1], 1, 1)

            if self.backward_fc:
                image_max_respond = self.inverse_fc(image_max_respond)

            pi_feat1_new = torch.cat([pi_feat1_new, image_max_respond], dim=-1)
        # mlp1 processes pi corresponding values
        for i, conv in enumerate(self.mlp1_convs):
            pi_feat1_new = conv(pi_feat1_new)  # B,N,K, mlp1[-1], to be weighted sum

        # position encoding for generating weights
        pi_xyz_encoding = self.pi_encoding(pi_xyz_diff_concat)  # B,N,K,mlp1[-1]

        if not self.max_cost:
            pi_concat = torch.cat([pi_xyz_encoding, pi_feat1_new], dim=3)  # B,N,K,2*mlp1[-1]

            # mlp2 processes the pi features to generate weights
            for j, conv in enumerate(self.mlp2_convs):
                pi_concat = conv(pi_concat)  # B,N,K,mlp2[-1]

            WQ = F.softmax(pi_concat, dim=2)

            pi_feat1_new = WQ * pi_feat1_new  # mlp1[-1]=mlp2[-1]
            pi_feat1_new = torch.sum(pi_feat1_new, dim=2, keepdim=False)  # B,N,mlp1[-1]
        else:
            pi_feat1_new = torch.max(pi_feat1_new, dim=2, keepdim=False)[0]

        # 3d find 3d grouped features to be weighted
        pc_xyz_grouped, _, pc_points_grouped, idx, _ = grouping(pi_feat1_new, self.nsample, warped_xyz, warped_xyz)

        pc_xyz_new = warped_xyz[:, :, None, :].repeat(1, 1, self.nsample, 1)  # B,N,K,3
        pc_points_new = warped_points[:, :, None, :].repeat(1, 1, self.nsample, 1)  # B,N,K,C

        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new  # B,N,K, 3

        pc_euc_diff = torch.sqrt(torch.sum(torch.mul(pc_xyz_diff, pc_xyz_diff), dim=3,
                                           keepdim=True) + 1e-20)  # B,N,K,1

        pc_xyz_diff_concat = torch.cat([pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff],
                                       dim=3)  # B,N,K,10

        pc_xyz_encoding = self.pc_encoding(pc_xyz_diff_concat)  # B,N,K, mlp1[-1]

        # position encoding + center pi features + neighbors pi features
        pc_concat = torch.cat([pc_xyz_encoding, pc_points_new, pc_points_grouped],
                              dim=-1)  # B,N,K, mlp[-1]+3+mlp[-1]

        # mlp3 for generating weights
        for j, conv in enumerate(self.mlp2_convs_2):
            pc_concat = conv(pc_concat)  # B,N,K, mlp2[-1]

        WP = F.softmax(pc_concat, dim=2)

        pc_feat1_new = WP * pc_points_grouped  # B,N,K, mlp2[-1]

        pc_feat1_new = torch.sum(pc_feat1_new, dim=2, keepdim=False)  # B,N, mlp2[-1]

        return pc_feat1_new


class CrossCostVolume(nn.Module):
    def __init__(self, nsample, rgb_in_channels, lidar_in_channels, mlp1, mlp2):
        super(CrossCostVolume, self).__init__()
        self.nsample = nsample
        self.mlp1 = mlp1
        self.mlp2 = mlp2

        corr_channel = rgb_in_channels + lidar_in_channels

        self.in_channels = corr_channel + 6

        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs_2 = nn.ModuleList()

        for i, num_out_channel in enumerate(mlp1):
            self.mlp1_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        self.pi_encoding = Conv2d(6, mlp1[-1], [1, 1], stride=[1, 1], bn=True)

        self.pc_encoding = Conv2d(10, mlp1[-1], [1, 1], stride=[1, 1], bn=True)

        self.in_channels = 2 * mlp1[-1] + lidar_in_channels
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs_2.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

    def forward(self, warped_xyz, warped_points, f2_xyz, f2_points, lidar_z):
        """
            Input:
                warped_xyz: (b,npoint,3)
                warped_points: (b,npoint,c)
                f2_xyz:  (b,ndataset,3)
                f2_points: (b,ndataset,c)
                lidar_z: (b, npoint, 1)

            Output:
                pc_feat1_new: batch_size, npoints, mlp2[-1]
        """

        N2 = f2_xyz.shape[1]
        # B,N1,N2,C

        qi_xyz_grouped, qi_points_grouped = f2_xyz.unsqueeze(1).repeat(1, warped_xyz.shape[1], 1, 1), \
                                            f2_points.unsqueeze(1).repeat(1, warped_xyz.shape[1], 1, 1)

        # important
        warped_xyz = warped_xyz.mul(lidar_z)  # restore depth

        # lidar_z_repeat = torch.unsqueeze(lidar_z, dim=2).repeat(1, 1, self.nsample_q, 1)

        pi_xyz_expanded = warped_xyz[:, :, None, :].repeat(1, 1, N2, 1)  # B,N,K,3
        pi_points_expanded = warped_points[:, :, None, :].repeat(1, 1, N2, 1)  # B,N,K,C

        # position embedding
        pi_xyz_diff_concat = torch.cat([pi_xyz_expanded, qi_xyz_grouped], dim=3)  # B,N,K,6

        pi_points_expanded = F.normalize(pi_points_expanded, p=2, dim=-1, eps=1e-12)

        qi_points_grouped = F.normalize(qi_points_grouped, p=2, dim=-1, eps=1e-12)

        # cosine similarity
        pi_feat_diff = torch.sum(pi_points_expanded * qi_points_grouped, -1, keepdim=True)  # B,N,K,1
        # row maximum
        pi_feature_diff_RM_rate = pi_feat_diff / (torch.max(pi_feat_diff, dim=2, keepdim=True)[0] + 1e-10)
        # column maximum
        pi_feature_diff_CM_rate = pi_feat_diff / (torch.max(pi_feat_diff, dim=1, keepdim=True)[0] + 1e-10)
        # joint possibility
        pi_feature_sim = pi_feature_diff_CM_rate * pi_feature_diff_RM_rate  # B,N,K,1

        # p+LF+RF concatenation
        pi_feat1_new = torch.cat([pi_xyz_diff_concat, pi_points_expanded, qi_points_grouped], dim=3)  # B,N,K, 6+2c

        # mlp1 processes pi corresponding values
        for i, conv in enumerate(self.mlp1_convs):
            pi_feat1_new = conv(pi_feat1_new)  # B,N,K, mlp1[-1], to be weighted sum

        pi_feat1_new = torch.max(pi_feat1_new * pi_feature_sim, dim=2, keepdim=False)[0]

        # 3d find 3d grouped features to be weighted
        pc_xyz_grouped, _, pc_points_grouped, idx = grouping(pi_feat1_new, self.nsample, warped_xyz, warped_xyz)

        pc_xyz_new = warped_xyz[:, :, None, :].repeat(1, 1, self.nsample, 1)  # B,N,K,3
        pc_points_new = warped_points[:, :, None, :].repeat(1, 1, self.nsample, 1)  # B,N,K,C

        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new  # B,N,K, 3

        pc_euc_diff = torch.sqrt(torch.sum(torch.mul(pc_xyz_diff, pc_xyz_diff), dim=3,
                                           keepdim=True) + 1e-20)  # B,N,K,1

        pc_xyz_diff_concat = torch.cat([pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff],
                                       dim=3)  # B,N,K,10

        pc_xyz_encoding = self.pc_encoding(pc_xyz_diff_concat)  # B,N,K, mlp1[-1]

        # position encoding + center pi features + neighbors pi features
        pc_concat = torch.cat([pc_xyz_encoding, pc_points_new, pc_points_grouped],
                              dim=-1)  # B,N,K, mlp[-1]+3+mlp[-1]

        # mlp3 for generating weights
        for j, conv in enumerate(self.mlp2_convs_2):
            pc_concat = conv(pc_concat)  # B,N,K, mlp2[-1]

        WP = F.softmax(pc_concat, dim=2)

        pc_feat1_new = WP * pc_points_grouped  # B,N,K, mlp2[-1]

        pc_feat1_new = torch.sum(pc_feat1_new, dim=2, keepdim=False)  # B,N, mlp2[-1]

        return pc_feat1_new


class PoseHead(nn.Module):
    class CorrFunc(Enum):
        DIFF = 1
        CONCAT = 2
        NORMALIZED_DIFF = 3

    def __init__(self, in_channels, mlp1, mlp2,
                 hidden, q_dim, t_dim, dropout_rate=0.5
                 , split_dp=False, corr_func=CorrFunc.CONCAT,
                 pos_embed=False, sigmoid=False, maxhead=False):
        """
        take the self attention mask and predictions
        do global attention (global constraint)
        """
        super(PoseHead, self).__init__()
        self.corr_func = corr_func
        self.sigmoid = sigmoid
        self.maxhead = maxhead
        in_channel, l_feature_channel = in_channels
        # take 3+3 as input and get
        self.pos_embed = pos_embed
        if self.pos_embed:
            self.pos_encoder = Conv1d(3 + 3, in_channel, bn=True)

        self.mlps = nn.ModuleList()
        if corr_func == PoseHead.CorrFunc.CONCAT:
            last_dim = in_channel + in_channel  # local,global,pos_embedding
        elif corr_func == PoseHead.CorrFunc.DIFF:
            last_dim = in_channel
        elif corr_func == PoseHead.CorrFunc.NORMALIZED_DIFF:
            last_dim = in_channel
        else:
            raise NotImplementedError
        if self.pos_embed:
            last_dim += in_channel
        for out_dim in mlp1:
            self.mlps.append(Conv1d(last_dim, out_dim, bn=True))
            last_dim = out_dim

        if len(mlp1) > 0:
            self.mlp2s = nn.ModuleList()  # to aggragate the previous mask info
            last_dim = in_channel + mlp1[-1] + l_feature_channel  # previous_mask+cost_volume+layer_feature
            for out_dim in mlp2:
                self.mlp2s.append(Conv1d(last_dim, out_dim, bn=True))
                last_dim = out_dim

        if split_dp:
            self.DP1 = nn.Identity()
            self.DP2 = nn.Dropout(dropout_rate)
        else:
            self.DP1 = nn.Dropout(dropout_rate)
            self.DP2 = nn.Identity()

        self.hidden_layer = Conv1d(in_channel, hidden, use_activation=False)
        self.quat_head = Conv1d(hidden, q_dim, use_activation=False)
        self.trans_head = Conv1d(hidden, t_dim, use_activation=False)

    def forward(self, prediction, mask, xyz, feature, projection_mask):
        """
        Args:
            prediction: [B,N,C]
            mask: [B,N,C]
            xyz: [B,N,3]
            feature: [B,N,C]
        Returns:
            q: [B,4]
            t: [B,3]
        """
        B, N, _ = prediction.shape

        if not self.sigmoid:
            if projection_mask is not None:
                projection_mask = torch.argmax(projection_mask.detach(), dim=-1, keepdim=True).float()
                mask = mask * projection_mask + -1e10 * (1. - projection_mask)
        else:
            prediction = prediction * projection_mask

        # B,1,C
        if self.maxhead:
            mask = torch.max(mask, dim=-1, keepdim=True)[0]
        mask_p = F.softmax(mask, dim=1)
        global_prediction = torch.sum(prediction * mask_p, dim=1, keepdim=True)  # [B,1,64]

        # TODO: less point
        if len(self.mlps) > 0:
            # need global attention
            global_prediction_extend = global_prediction.repeat(1, N, 1)

            if self.pos_embed:
                global_xyz = torch.mean(xyz, dim=1, keepdim=True)
                center_xyz = xyz - global_xyz

                pos_info = torch.cat([xyz, center_xyz], dim=-1)  # B,N,6
                pos_embedding = self.pos_encoder(pos_info)  # B,N,C

            if self.corr_func == PoseHead.CorrFunc.CONCAT:
                # B,N,3C
                global_concat_noembed = torch.cat([prediction, global_prediction_extend], dim=-1)
            elif self.corr_func == PoseHead.CorrFunc.DIFF:
                G_L_diff = prediction - global_prediction
                global_concat_noembed = G_L_diff
            elif self.corr_func == PoseHead.CorrFunc.NORMALIZED_DIFF:
                prediction_norm = (prediction - prediction.mean(dim=-1, keepdim=True)) / (
                        prediction.std(dim=-1, keepdim=True) + 1e-10)
                global_prediction_norm = (global_prediction - global_prediction.mean(dim=-1, keepdim=True)) / (
                        global_prediction.std(dim=-1, keepdim=True) + 1e-10)
                global_concat_noembed = prediction_norm * global_prediction_norm
            else:
                raise NotImplementedError
            global_concat = torch.cat([global_concat_noembed, pos_embedding], -1) \
                if self.pos_embed else global_concat_noembed
            # B,N,C
            for mlp in self.mlps:
                global_concat = mlp(global_concat)

            if len(self.mlp2s) > 0:
                global_embed = torch.cat([global_concat, mask, feature], dim=-1)
                for mlp in self.mlp2s:
                    global_embed = mlp(global_embed)
            else:
                global_embed = global_concat

            weight = torch.softmax(global_embed, dim=1)  # B,N,C

            result = torch.sum(weight * prediction, dim=1, keepdim=True)

        else:
            result = global_prediction

        hidden_feature = self.DP1(self.hidden_layer(result))

        q = self.quat_head(self.DP2(hidden_feature)).squeeze(1)
        t = self.trans_head(self.DP2(hidden_feature)).squeeze(1)

        # normalize q
        q = q / (torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-10) + 1e-10)

        return q, t, mask_p


class ProjectMask(nn.Module):
    def __init__(self, in_channel, mlp, sigmoid=False, drop=0):
        """
        From the prediction, supervising predict the projectedmask
        """
        super(ProjectMask, self).__init__()
        self.mlps = nn.ModuleList()
        last_dim = in_channel
        self.drop = nn.Dropout(p=drop) if drop > 0 else nn.Identity()
        for out_dim in mlp:
            self.mlps.append(Conv1d(last_dim, out_dim, bn=True))
            last_dim = out_dim
        # classification head
        class_dim = 1 if sigmoid else 2
        self.out = Conv1d(mlp[-1], class_dim, use_activation=False)
        self.out_act = nn.Sigmoid() if sigmoid else nn.Identity()

    def forward(self, feature, prediction):
        """
        Args:
            feature: [B,N,C] point-wise feature
            prediction: [B,N,C] point-wise prediction without mask
        Returns:
            projection_mask: [B,N,1] point-wise mask
        """
        if feature is not None:
            feature_concat = torch.cat([feature, prediction], dim=-1)
        else:
            feature_concat = prediction
        for mlp in self.mlps:
            feature_concat = self.drop(mlp(feature_concat))  # B,N,mlp[-1]

        projection_mask = self.out_act(self.out(feature_concat))

        return projection_mask


class DelayWeight(nn.Module):
    def __init__(self, delay_step, delay, ab_delay):
        super(DelayWeight, self).__init__()
        self.delay = delay
        self.ab_delay = ab_delay
        self.delay_step = delay_step
        self.now_step = torch.nn.Parameter(torch.Tensor([delay_step]), requires_grad=False)

    def forward(self, gt, pred):
        if gt is None:
            return pred
        elif pred is None:
            return gt
        elif self.ab_delay:
            if torch.eq(self.now_step, 0).item():
                return pred
            else:
                if self.training:
                    self.now_step.add_(-1).clip_(0)
                return gt
        else:
            pred = F.softmax(pred, dim=-1)  # B,N,2
            weight = self.now_step.item() / (self.delay_step + 1e-10)
            mixed = gt * weight + pred * (1 - weight)
            if self.training and self.delay:
                self.now_step.add_(-1).clip_(0)
            return mixed


class MaskPredictor(nn.Module):
    def __init__(self, in_channel, mlp):
        super(MaskPredictor, self).__init__()

        self.mlp1_modules = nn.ModuleList()

        last_dim = in_channel
        for out_dim in mlp:
            self.mlp1_modules.append(Conv1d(last_dim, out_dim, bn=True))
            last_dim = out_dim

    def forward(self, LF, mask_cv, up_mask_cv=None, global_LF=None, global_RF=None):
        """
        Args:
            LF: [B,N,C]
            mask_cv: [B,N,C]
            up_mask_cv: [B,N,C]
            global_LF: [B,1,C]
            global_RF: [B,1,C]

        Returns:

        """
        B, N, _ = LF.shape
        features = [LF, mask_cv]
        if up_mask_cv is not None:
            features.append(up_mask_cv)
        if global_LF is not None:
            features.append(global_LF.repeat(1, N, 1))
        if global_RF is not None:
            features.append(global_RF.repeat(1, N, 1))

        feature = torch.cat(features, dim=-1)

        for conv in self.mlp1_modules:
            feature = conv(feature)

        return feature


class MaskCostVolume(nn.Module):
    def __init__(self, rgb_in_channels, lidar_in_channels, mlp1, img_size):
        super(MaskCostVolume, self).__init__()

        self.mlp1 = mlp1

        self.mlp1_modules = nn.ModuleList()

        self.img_size = img_size

        last_dim = rgb_in_channels + lidar_in_channels
        for out_dim in mlp1:
            self.mlp1_modules.append(Conv1d(last_dim, out_dim, bn=True))
            last_dim = out_dim
        self.mlp1_modules.append(Conv1d(last_dim, img_size, use_activation=False))

    def forward(self, lidar_features, img_features):
        """
            Input:
                lidar_features: [B,N,C]
                img_features: [B,C,H,W]
        """

        B, N, C = lidar_features.shape

        global_img_feature = img_features.view(B, C, -1).mean(-1).unsqueeze(1)  # B,1,C
        # B,N,2C
        concat_corr = torch.cat([global_img_feature.repeat(1, N, 1), lidar_features], -1)

        # mlp1 processes pi corresponding values
        for i, conv in enumerate(self.mlp1_modules):
            concat_corr = conv(concat_corr)  # B,N,H*W
        concat_corr = concat_corr.unsqueeze(1)  # B,1,N,H*W
        img_features = img_features.reshape(B, -1, 1, self.img_size)  # B,C,1,H*W
        new_feature = torch.mean(concat_corr * img_features, -1).permute(0, 2, 1)

        return new_feature  # B,N,C


class AllCostVolume(nn.Module):
    def __init__(self, nsample, rgb_in_channels, lidar_in_channels, mlp1, mlp2):
        super(AllCostVolume, self).__init__()
        self.nsample = nsample
        self.mlp1 = mlp1
        self.mlp2 = mlp2

        # for correlation embedding forward similarity and backward similarity
        corr_channel = rgb_in_channels
        self.in_channels = corr_channel + 6 + 2
        self.mlp1_convs = nn.ModuleList()
        for i, num_out_channel in enumerate(mlp1):
            self.mlp1_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        # weight estimation
        self.mlp2_convs = nn.ModuleList()
        self.in_channels = 2 * mlp1[-1]
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        self.pi_encoding = Conv2d(6, mlp1[-1], [1, 1], stride=[1, 1], bn=True)

        self.pc_encoding = Conv2d(10, mlp1[-1], [1, 1], stride=[1, 1], bn=True)

        self.mlp2_convs_2 = nn.ModuleList()
        self.in_channels = 2 * mlp1[-1] + lidar_in_channels
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs_2.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

    def forward(self, warped_xyz, warped_points, f2_xyz, f2_points, lidar_z):
        """
            Input:
                warped_xyz: (b,npoint,3)
                warped_points: (b,npoint,c)
                f2_xyz:  (b,ndataset,3)
                f2_points: (b,ndataset,c)
                lidar_z: (b, npoint, 1)

            Output:
                pc_feat1_new: batch_size, npoints, mlp2[-1]
        """

        N2 = f2_xyz.shape[1]
        # B,N1,N2,C

        qi_xyz_grouped, qi_points_grouped = f2_xyz.unsqueeze(1).repeat(1, warped_xyz.shape[1], 1, 1), \
                                            f2_points.unsqueeze(1).repeat(1, warped_xyz.shape[1], 1, 1)

        # important
        warped_xyz = warped_xyz.mul(lidar_z)  # restore depth

        # lidar_z_repeat = torch.unsqueeze(lidar_z, dim=2).repeat(1, 1, self.nsample_q, 1)

        pi_xyz_expanded = warped_xyz[:, :, None, :].repeat(1, 1, N2, 1)  # B,N,K,3
        pi_points_expanded = warped_points[:, :, None, :].repeat(1, 1, N2, 1)  # B,N,K,C

        # position embedding
        pi_xyz_diff_concat = torch.cat([pi_xyz_expanded, qi_xyz_grouped], dim=3)  # B,N,K,6

        # cosine correspondence
        cosine_corr = torch.sum(F.normalize(pi_points_expanded, p=2, dim=-1, eps=1e-12) * \
                                F.normalize(qi_points_grouped, p=2, dim=-1, eps=1e-12), dim=-1, keepdim=True)

        # normalized features
        pi_points_expanded = (pi_points_expanded - torch.mean(pi_points_expanded, -1, keepdim=True)) / torch.std(
            pi_points_expanded, -1, keepdim=True)
        qi_points_grouped = (qi_points_grouped - torch.mean(qi_points_grouped, -1, keepdim=True)) / torch.std(
            qi_points_grouped, -1, keepdim=True)

        # correlation
        pi_feat_diff = pi_points_expanded * qi_points_grouped  # B,N,K,C
        # row maximum
        pi_feature_diff_RM_rate = cosine_corr / (torch.max(cosine_corr, dim=2, keepdim=True)[0] + 1e-10)
        # column maximum
        pi_feature_diff_CM_rate = cosine_corr / (torch.max(cosine_corr, dim=1, keepdim=True)[0] + 1e-10)

        pi_feat1_new = torch.cat([pi_xyz_diff_concat, pi_feat_diff, pi_feature_diff_RM_rate, pi_feature_diff_CM_rate],
                                 dim=3)  # B,N,K, 6+c+2

        # position encoding for generating weights
        pi_xyz_encoding = self.pi_encoding(pi_xyz_diff_concat)  # B,N,K,mlp1[-1]

        # mlp1 processes pi corresponding values
        for i, conv in enumerate(self.mlp1_convs):
            pi_feat1_new = conv(pi_feat1_new)  # B,N,K, mlp1[-1]

        pi_concat = torch.cat([pi_xyz_encoding, pi_feat1_new], dim=3)  # B,N,K,2*mlp1[-1]

        # mlp2 processes the pi features to generate weights
        for j, conv in enumerate(self.mlp2_convs):
            pi_concat = conv(pi_concat)  # B,N,K,mlp2[-1]

        WQ = F.softmax(pi_concat, dim=2)

        pi_feat1_new = WQ * pi_feat1_new  # mlp1[-1]=mlp2[-1]
        pi_feat1_new = torch.sum(pi_feat1_new, dim=2, keepdim=False)  # B,N,mlp1[-1]

        # 3d find 3d grouped features to be weighted
        pc_xyz_grouped, _, pc_points_grouped, idx = grouping(pi_feat1_new, self.nsample, warped_xyz, warped_xyz)

        pc_xyz_new = warped_xyz[:, :, None, :].repeat(1, 1, self.nsample, 1)  # B,N,K,3
        pc_points_new = warped_points[:, :, None, :].repeat(1, 1, self.nsample, 1)  # B,N,K,C

        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new  # B,N,K, 3

        pc_euc_diff = torch.sqrt(torch.sum(torch.mul(pc_xyz_diff, pc_xyz_diff), dim=3,
                                           keepdim=True) + 1e-20)  # B,N,K,1

        pc_xyz_diff_concat = torch.cat([pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff],
                                       dim=3)  # B,N,K,10

        pc_xyz_encoding = self.pc_encoding(pc_xyz_diff_concat)  # B,N,K, mlp1[-1]

        # position encoding + center pi features + neighbors pi features
        pc_concat = torch.cat([pc_xyz_encoding, pc_points_new, pc_points_grouped],
                              dim=-1)  # B,N,K, mlp[-1]+3+mlp[-1]

        # mlp3 for generating weights
        for j, conv in enumerate(self.mlp2_convs_2):
            pc_concat = conv(pc_concat)  # B,N,K, mlp2[-1]

        WP = F.softmax(pc_concat, dim=2)

        pc_feat1_new = WP * pc_points_grouped  # B,N,K, mlp2[-1]

        pc_feat1_new = torch.sum(pc_feat1_new, dim=2, keepdim=False)  # B,N, mlp2[-1]

        return pc_feat1_new
