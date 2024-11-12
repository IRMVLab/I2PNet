import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.basicConv import Conv2d, Conv1d
from src.modules.point_utils import grouping


def calc_cosine_similarity(src, dst):
    src_norm = F.normalize(src, 2, dim=-1, eps=1e-12)
    dst_norm = F.normalize(dst, 2, dim=-1, eps=1e-12)

    return torch.sum(src_norm * dst_norm, dim=-1)


def cal_sim(src_feats, dst_feats):
    """

    Args:
        src_feats: [B,N,C]
        dst_feats: [B,M,C]

    Returns:

    """
    N = src_feats.shape[1]
    M = dst_feats.shape[1]
    # B,M,N,C
    dst_feats_expand = dst_feats.unsqueeze(2). \
        repeat(1, 1, N, 1)
    src_feats_expand = src_feats.unsqueeze(1). \
        repeat(1, M, 1, 1)
    # B,M,N
    similarity = calc_cosine_similarity(dst_feats_expand, src_feats_expand)
    dst_src_nbr_cos_max = torch.max(similarity, dim=2, keepdim=True)[0]  # [B,M,1]
    dst_src_nbr_cos_norm = similarity / (dst_src_nbr_cos_max + 1e-10)  # [B,M,N]

    src_dst_nbr_cos = similarity.permute(0, 2, 1)  # [B,N1,N2]
    src_dst_nbr_cos_max = torch.max(src_dst_nbr_cos, dim=2, keepdim=True)[0]  # [B,N,1]
    src_dst_nbr_cos_norm = src_dst_nbr_cos / (src_dst_nbr_cos_max + 1e-6)  # [B,N,M]

    # dst_src_cos_knn = knn_gather(dst_src_cos_norm, src_knn_idx)  # [B,N1,k,N1]
    # dst_src_cos = torch.zeros(src_knn_xyz.shape[0], src_knn_xyz.shape[1], \
    #                           src_knn_xyz.shape[2]).cuda()  # [B,N1,k]
    # for i in range(src_xyz.shape[1]):
    #     dst_src_cos[:, i, :] = dst_src_cos_knn[:, i, :, i]
    #
    # src_dst_cos_knn = knn_gather(src_dst_cos_norm.permute(0, 2, 1), src_knn_idx)
    # src_dst_cos = torch.zeros(src_knn_xyz.shape[0], src_knn_xyz.shape[1], \
    #                           src_knn_xyz.shape[2]).cuda()  # [B,N1,k]
    # for i in range(src_xyz.shape[1]):
    #     src_dst_cos[:, i, :] = src_dst_cos_knn[:, i, :, i]

    return torch.stack([src_dst_nbr_cos_norm,
                        dst_src_nbr_cos_norm.permute(0, 2, 1)], dim=-1)  # B,N,M,2


class SimCrossVolume(nn.Module):

    def __init__(self, rgb_in_channels, lidar_in_channels,
                 nsample, img_nbr_kernel,
                 mlp3):
        """
        Args:
            rgb_in_channels:
            lidar_in_channels:
            nsample:
            img_nbr_kernel:
            mlp3: the last mlp
        """
        super(SimCrossVolume, self).__init__()

        self.nsample = nsample  # point neighbors
        self.img_nbr_kernel = img_nbr_kernel  # img neighbors

        # for nbr weight aggregation
        self.convs_2 = nn.ModuleList()
        mlp1 = [lidar_in_channels, lidar_in_channels, lidar_in_channels]
        in_dim = 4 + lidar_in_channels
        for num_out_channel in mlp1:
            self.convs_2.append(Conv2d(in_dim, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            in_dim = num_out_channel
        # for nbr weight aggregation
        self.convs_3 = nn.ModuleList()
        mlp2 = [rgb_in_channels, rgb_in_channels, rgb_in_channels]
        in_dim = rgb_in_channels
        kernel_size = img_nbr_kernel
        for num_out_channel in mlp2:
            self.convs_3.append(
                nn.Sequential(nn.Conv2d(in_dim, num_out_channel, kernel_size,
                                        1, kernel_size // 2, bias=False),
                              nn.BatchNorm2d(num_out_channel),
                              nn.LeakyReLU(0.1)))
            kernel_size = 1  # first perform aggregation and then mlp
            in_dim = num_out_channel

        self.convs_1 = nn.ModuleList()
        # mlp3 = [2 * lidar_in_channels, 2 * lidar_in_channels, 2 * lidar_in_channels]
        in_dim = 8 + rgb_in_channels + lidar_in_channels + 4
        for num_out_channel in mlp3:
            self.convs_1.append(Conv2d(in_dim, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            in_dim = num_out_channel

    def forward(self, warped_xyz, warped_points, RF3, RF3_index, lidar_z):
        """

        Args:
            warped_xyz: [B,N,3] (x_n,y_n,1)
            warped_points: [B,N,C]
            RF3: [B,C,H,W]
            RF3_index: [B,3,H,W] (x_n,y_n,1)
            lidar_z: [B,N,1]

        Returns:

        """
        B, N, _ = warped_xyz.shape
        B, Ci, H, W = RF3.shape

        RF3_bnc = RF3.permute(0, 2, 3, 1).reshape(B, H * W, Ci)
        RF3_index_bnc = RF3_index.permute(0, 2, 3, 1).reshape(B, H * W, 3)

        src_xyz_expand = warped_xyz[:, :, None, :].repeat(1, 1, H * W, 1)  # B,N,M,3
        dst_xyz_expand = RF3_index_bnc[:, None, :, :].repeat(1, N, 1, 1)

        src_xyz_euc = (dst_xyz_expand - src_xyz_expand)[:, :, :, :2]
        src_xyz_norm = torch.norm(src_xyz_euc, p=2, dim=-1, keepdim=True)

        # important
        warped_xyz = warped_xyz.mul(lidar_z)  # restore depth

        src_xyz_expand = warped_xyz[:, :, None, :].repeat(1, 1, H * W, 1)  # B,N,M,3

        src_points_expand = warped_points[:, :, None, :].repeat(1, 1, H * W, 1)
        dst_points_expand = RF3_bnc[:, None, :, :].repeat(1, N, 1, 1)

        geom_feats = torch.cat([src_xyz_expand, dst_xyz_expand[:, :, :, :2], src_xyz_euc, src_xyz_norm],
                               dim=-1)  # 3+2+2+1,xyz,xy1
        desc_feats = torch.cat([src_points_expand, dst_points_expand], dim=-1)  # 2c

        src_dst_sim = cal_sim(warped_points, RF3_bnc)

        ###############################
        # important self using neighbors similarity
        # in our structure it can be point knn and pixel kernel similarity
        # our implement
        ###############################
        K = self.nsample
        pc_xyz_grouped, _, pc_points_grouped, idx = grouping(warped_points, self.nsample, warped_xyz, warped_xyz)

        pc_xyz_new = warped_xyz[:, :, None, :].repeat(1, 1, K, 1)  # B,N,K,3

        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new  # B,N,K,3

        pc_xyz_diff_norm = torch.norm(pc_xyz_diff, dim=-1, keepdim=True)  # [B,N,k]
        pc_nbr_feats = torch.cat([pc_points_grouped, pc_xyz_diff, pc_xyz_diff_norm], dim=-1)
        ###################
        pc_nbr_weights = pc_nbr_feats  # 6+c
        for mlp in self.convs_2:
            pc_nbr_weights = mlp(pc_nbr_weights)  # B,N,K,C
        pc_nbr_weights = torch.max(pc_nbr_weights, -1, keepdim=True)[0]
        pc_nbr_weights = F.softmax(pc_nbr_weights, -2)
        pc_feats = torch.sum(pc_nbr_weights * pc_points_grouped, -2)  # B,N,C
        ################
        img_feats = RF3
        for conv in self.convs_3:
            img_feats = conv(img_feats)
        img_feats = img_feats.reshape(B, Ci, H * W).permute(0, 2, 1)

        src_dst_nbr_sim = cal_sim(pc_feats, img_feats)
        #######################

        similarity_feats = torch.cat([src_dst_sim, src_dst_nbr_sim], dim=-1)  # B,N,M,4

        # B,N,M,6+2C+4
        feats = torch.cat([geom_feats, desc_feats, similarity_feats], dim=-1)

        for mlp in self.convs_1:
            feats = mlp(feats)  # B,N,M,C
        attentive_weights = torch.max(feats, dim=-1)[0]  # B,N,M
        attentive_weights = F.softmax(attentive_weights, dim=-1)
        # B,N,C
        attentive_feats = torch.sum(torch.mul(attentive_weights.unsqueeze(-1), feats), dim=-2, keepdim=False)

        return attentive_feats


class SimMask(nn.Module):

    def __init__(self, rgb_in_channels, lidar_in_channels,
                 nsample, img_nbr_kernel,
                 mlp3):
        """
        Args:
            rgb_in_channels:
            lidar_in_channels:
            nsample:
            img_nbr_kernel:
            mlp3: the last mlp
        """
        super(SimMask, self).__init__()

        self.nsample = nsample  # point neighbors
        self.img_nbr_kernel = img_nbr_kernel  # img neighbors

        # for nbr weight aggregation
        # self.convs_2 = nn.ModuleList()
        # mlp1 = [lidar_in_channels, lidar_in_channels, lidar_in_channels]
        # in_dim = 4 + lidar_in_channels
        # for num_out_channel in mlp1:
        #     self.convs_2.append(Conv2d(in_dim, num_out_channel, [1, 1], stride=[1, 1], bn=True))
        #     in_dim = num_out_channel
        # for nbr weight aggregation
        # self.convs_3 = nn.ModuleList()
        # mlp2 = [rgb_in_channels, rgb_in_channels, rgb_in_channels]
        # in_dim = rgb_in_channels
        # kernel_size = img_nbr_kernel
        # for num_out_channel in mlp2:
        #     self.convs_3.append(
        #         nn.Sequential(nn.Conv2d(in_dim, num_out_channel, kernel_size,
        #                                 1, kernel_size // 2, bias=False),
        #                       nn.BatchNorm2d(num_out_channel),
        #                       nn.LeakyReLU(0.1)))
        #     in_dim = num_out_channel

        self.convs_1 = nn.ModuleList()
        # mlp3 = [2 * lidar_in_channels, 2 * lidar_in_channels, 2 * lidar_in_channels]
        in_dim = 10 + rgb_in_channels + lidar_in_channels + 2
        for num_out_channel in mlp3:
            self.convs_1.append(Conv2d(in_dim, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            in_dim = num_out_channel

        self.mlps = nn.ModuleList()
        feats_dim = in_dim
        mlp4 = [feats_dim // 2]
        in_dim = feats_dim  # 64
        for num_out_channel in mlp4:
            self.mlps.append(Conv1d(in_dim, num_out_channel, 1, stride=1, bn=True))
            in_dim = num_out_channel
        self.mlps.append(Conv1d(in_dim, 2, 1, stride=1, use_activation=False))

    def forward(self, warped_xyz, warped_points, RF3, RF3_index, lidar_z):
        """

        Args:
            warped_xyz: [B,N,3] (x_n,y_n,1)
            warped_points: [B,N,C]
            RF3: [B,C,H,W]
            RF3_index: [B,3,H,W] (x_n,y_n,1)
            lidar_z: [B,N,1]

        Returns:

        """
        B, N, _ = warped_xyz.shape
        B, Ci, H, W = RF3.shape

        RF3_bnc = RF3.permute(0, 2, 3, 1).reshape(B, H * W, Ci)
        RF3_index_bnc = RF3_index.permute(0, 2, 3, 1).reshape(B, H * W, 3)

        src_xyz_expand = warped_xyz[:, :, None, :].repeat(1, 1, H * W, 1)  # B,N,M,3
        dst_xyz_expand = RF3_index_bnc[:, None, :, :].repeat(1, N, 1, 1)

        src_xyz_euc = (src_xyz_expand - dst_xyz_expand)[:, :, :, :2]
        src_xyz_norm = torch.sqrt(torch.square(src_xyz_euc[:, :, :, 0:1])
                                  + torch.square(src_xyz_euc[:, :, :, 1:]))

        # important
        warped_xyz = warped_xyz.mul(lidar_z)  # restore depth

        src_xyz_expand_z = warped_xyz[:, :, None, :].repeat(1, 1, H * W, 1)  # B,N,M,3

        src_points_expand = warped_points[:, :, None, :].repeat(1, 1, H * W, 1)
        dst_points_expand = RF3_bnc[:, None, :, :].repeat(1, N, 1, 1)

        geom_feats = torch.cat([src_xyz_expand_z,
                                src_points_expand[:, :, :, :2],
                                dst_xyz_expand[:, :, :, :2],
                                src_xyz_euc,
                                src_xyz_norm],
                               dim=-1)  # 3+2+2+2+1,xyz,xy1
        desc_feats = torch.cat([src_points_expand, dst_points_expand], dim=-1)  # 2c

        src_dst_sim = cal_sim(warped_points, RF3_bnc)

        ###############################
        # important self using neighbors similarity
        # in our structure it can be point knn and pixel kernel similarity
        # our implement
        ###############################
        # K = self.nsample
        # pc_xyz_grouped, _, pc_points_grouped, idx = grouping(warped_points, self.nsample, warped_xyz, warped_xyz)
        #
        # pc_xyz_new = warped_xyz[:, :, None, :].repeat(1, 1, K, 1)  # B,N,K,3
        #
        # pc_xyz_diff = pc_xyz_grouped - pc_xyz_new  # B,N,K,3
        #
        # pc_xyz_diff_norm = torch.norm(pc_xyz_diff, dim=-1, keepdim=True)  # [B,N,k]
        # pc_nbr_feats = torch.cat([pc_points_grouped, pc_xyz_diff, pc_xyz_diff_norm], dim=-1)
        ###################
        # pc_nbr_weights = pc_nbr_feats  # 6+c
        # for mlp in self.convs_2:
        #     pc_nbr_weights = mlp(pc_nbr_weights)  # B,N,K,C
        # pc_nbr_weights = torch.max(pc_nbr_weights, -1, keepdim=True)[0]
        # pc_nbr_weights = F.softmax(pc_nbr_weights, -2)
        # pc_feats = torch.sum(pc_nbr_weights * pc_points_grouped, -2)  # B,N,C
        ################
        # img_feats = RF3
        # for conv in self.convs_3:
        #     img_feats = conv(img_feats)
        # img_feats = img_feats.reshape(B, Ci, H * W).permute(0, 2, 1)
        #
        # src_dst_nbr_sim = cal_sim(pc_feats, img_feats)
        #######################

        similarity_feats = torch.cat([src_dst_sim], dim=-1)  # B,N,M,2

        # B,N,M,8+2C+4
        feats = torch.cat([geom_feats, desc_feats, similarity_feats], dim=-1)

        for mlp in self.convs_1:
            feats = mlp(feats)  # B,N,M,C
        attentive_weights = torch.max(feats, dim=-1)[0]  # B,N,M
        attentive_weights = F.softmax(attentive_weights, dim=-1)
        # B,N,C
        attentive_feats = torch.sum(torch.mul(attentive_weights.unsqueeze(-1), feats), dim=-2, keepdim=False)

        for mlp in self.mlps:
            attentive_feats = mlp(attentive_feats)  # B,N,2
        proj_logits = attentive_feats
        return proj_logits

# class FineReg(nn.Module):
#     '''
#     Params:
#         k: number of candidate keypoints
#         in_channels: input channel number
#     Input:
#         src_xyz: [B,N,3]
#         src_desc: [B,C,N]
#         dst_xyz: [B,N,3]
#         dst_desc: [B,C,N]
#         src_weights: [B,N]
#         dst_weights: [B,N]
#     Output:
#         corres_xyz: [B,N,3]
#         weights: [B,N]
#     '''
#
#     def __init__(self, k, in_channels):
#         super(FineReg, self).__init__()
#         self.k = k
#         out_channels = [in_channels * 2 + 12, in_channels * 2, in_channels * 2, in_channels * 2]
#         layers = []
#         for i in range(1, len(out_channels)):
#             layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], kernel_size=1, bias=False),
#                        nn.BatchNorm2d(out_channels[i]),
#                        nn.ReLU()]
#         self.convs_1 = nn.Sequential(*layers)
#
#         self.mlp1 = nn.Sequential(nn.Conv1d(in_channels * 2, in_channels * 2, kernel_size=1),
#                                   nn.BatchNorm1d(in_channels * 2),
#                                   nn.ReLU())
#         self.mlp2 = nn.Sequential(nn.Conv1d(in_channels * 2, in_channels * 2, kernel_size=1),
#                                   nn.BatchNorm1d(in_channels * 2),
#                                   nn.ReLU())
#         self.mlp3 = nn.Sequential(nn.Conv1d(in_channels * 2, 1, kernel_size=1))
#
#     def forward(self, src_xyz, src_feat, dst_xyz, dst_feat, src_weights, dst_weights):
#         _, src_knn_idx, src_knn_xyz = knn_points(src_xyz, dst_xyz, K=self.k, return_nn=True)
#         src_feat = src_feat.permute(0, 2, 1).contiguous()
#         dst_feat = dst_feat.permute(0, 2, 1).contiguous()
#         src_knn_feat = knn_gather(dst_feat, src_knn_idx)  # [B,N,k,C]
#         src_xyz_expand = src_xyz.unsqueeze(2).repeat(1, 1, self.k, 1)
#         src_feat_expand = src_feat.unsqueeze(2).repeat(1, 1, self.k, 1)
#         src_rela_xyz = src_knn_xyz - src_xyz_expand  # [B,N,k,3]
#         src_rela_dist = torch.norm(src_rela_xyz, dim=-1, keepdim=True)  # [B,N,k,1]
#         src_weights_expand = src_weights.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.k, 1)  # [B,N,k,1]
#         src_knn_weights = knn_gather(dst_weights.unsqueeze(-1), src_knn_idx)  # [B,N,k,1]
#         feats = torch.cat([src_rela_xyz, src_rela_dist, src_xyz_expand, src_knn_xyz, \
#                            src_feat_expand, src_knn_feat, src_weights_expand, src_knn_weights], dim=-1)
#         feats = self.convs_1(feats.permute(0, 3, 1, 2).contiguous())  # [B,C,N,k]
#         attentive_weights = torch.max(feats, dim=1)[0]
#         attentive_weights = F.softmax(attentive_weights, dim=-1)  # [B,N,k]
#         corres_xyz = torch.sum(torch.mul(attentive_weights.unsqueeze(-1), src_knn_xyz), dim=2, keepdim=False)  # [B,N,3]
#         attentive_feats = torch.sum(torch.mul(attentive_weights.unsqueeze(1), feats), dim=-1, keepdim=False)  # [B,N,C]
#         weights = self.mlp3(self.mlp2(self.mlp1(attentive_feats)))  # [B,1,N]
#         weights = torch.sigmoid(weights.squeeze(1))
#
#         return corres_xyz, weights
