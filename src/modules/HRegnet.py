import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.basicConv import Conv2d, Conv1d
from src.modules.point_utils import grouping
from src.WeightedPnP import WeightedPnPHead


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


class SimSVDCrossVolume(nn.Module):

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
        super(SimSVDCrossVolume, self).__init__()

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
        in_dim = 3 + rgb_in_channels
        for num_out_channel in mlp2:
            self.convs_3.append(Conv2d(in_dim, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            in_dim = num_out_channel

        self.convs_1 = nn.ModuleList()
        # mlp3 = [2 * lidar_in_channels, 2 * lidar_in_channels, 2 * lidar_in_channels]
        in_dim = 6 + rgb_in_channels + lidar_in_channels + 4
        for num_out_channel in mlp3:
            self.convs_1.append(Conv2d(in_dim, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            in_dim = num_out_channel

        self.pnp_head = WeightedPnPHead()
        self.mlps = nn.ModuleList()
        mlp4 = [mlp3[-1], mlp3[-3]]
        in_dim = mlp3[-1]
        for num_out_channel in mlp4:
            self.mlps.append(Conv1d(in_dim, num_out_channel, bn=True))
            in_dim = num_out_channel
        self.mlps.append(Conv1d(in_dim, 2, use_activation=False))

    def forward(self, warped_xyz, warped_points, RF3, RF3_index, lidar_z, gt_project):
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
        # important
        warped_xyz = warped_xyz.mul(lidar_z)  # restore depth

        RF3_bnc = RF3.permute(0, 2, 3, 1).reshape(B, H * W, Ci)
        RF3_index_bnc = RF3_index.permute(0, 2, 3, 1).reshape(B, H * W, 3)

        src_xyz_expand = warped_xyz[:, :, None, :].repeat(1, 1, H * W, 1)  # B,N,M,3
        dst_xyz_expand = RF3_index_bnc[:, None, :, :].repeat(1, N, 1, 1)

        src_points_expand = warped_points[:, :, None, :].repeat(1, 1, H * W, 1)
        dst_points_expand = RF3_bnc[:, None, :, :].repeat(1, N, 1, 1)

        geom_feats = torch.cat([src_xyz_expand, dst_xyz_expand], dim=-1)  # 6,xyz,xy1
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
        S = self.img_nbr_kernel

        RF_nbr_index = F.unfold(RF3_index, (S, S), padding=(S // 2, S // 2)). \
            reshape(B, 3, S * S, H * W).permute(0, 3, 2, 1)

        RF_nbr_points_grouped = F.unfold(RF3, (S, S), padding=(S // 2, S // 2)). \
            view(B, Ci, S * S, H * W).permute(0, 3, 2, 1)

        RF_nbr_feats = torch.cat([RF_nbr_index, RF_nbr_points_grouped], dim=-1)
        ####################
        img_nbr_weights = RF_nbr_feats
        for mlp in self.convs_3:
            img_nbr_weights = mlp(img_nbr_weights)  # B,N,K,C
        img_nbr_weights = torch.max(img_nbr_weights, -1, keepdim=True)[0]
        img_nbr_weights = F.softmax(img_nbr_weights, -2)
        img_feats = torch.sum(img_nbr_weights * RF_nbr_points_grouped, -2)  # B,N,C
        src_dst_nbr_sim = cal_sim(pc_feats, img_feats)
        #######################

        similarity_feats = torch.cat([src_dst_sim, src_dst_nbr_sim], dim=-1)  # B,N,M,4

        # B,N,M,6+2C+4
        feats = torch.cat([geom_feats, desc_feats, similarity_feats], dim=-1)

        for mlp in self.convs_1:
            feats = mlp(feats)  # B,N,M,C
        attentive_weights = torch.max(feats, dim=-1)[0]  # B,N,M
        attentive_weights = F.softmax(attentive_weights, dim=-1)

        corres_uv1 = torch.sum(torch.mul(attentive_weights.unsqueeze(-1), dst_xyz_expand), dim=2,
                               keepdim=False)  # [B,N,3]
        attentive_feats = torch.sum(torch.mul(attentive_weights.unsqueeze(-1), feats), dim=-2, keepdim=False)  # [B,N,C]
        for i, conv in enumerate(self.mlps):
            attentive_feats = conv(attentive_feats)
        weights = torch.softmax(attentive_feats, -1)  # B,N,2

        if gt_project is not None:
            weights_input = gt_project.argmax(-1)
        else:
            weights_input = weights.argmax(-1)
        R, t = self.pnp_head(warped_xyz, corres_uv1[:, :, :2], weights_input.float())

        return R, t, weights


class SVDCrossVolume(nn.Module):

    def __init__(self, rgb_in_channels, lidar_in_channels,
                 nsample_q,
                 mlp3):
        """
        Args:
            rgb_in_channels:
            lidar_in_channels:
            nsample:
            img_nbr_kernel:
            mlp3: the last mlp
        """
        super(SVDCrossVolume, self).__init__()

        self.nsample_q = nsample_q

        self.convs_1 = nn.ModuleList()
        # mlp3 = [2 * lidar_in_channels, 2 * lidar_in_channels, 2 * lidar_in_channels]
        in_dim = 6 + rgb_in_channels + lidar_in_channels
        for num_out_channel in mlp3:
            self.convs_1.append(Conv2d(in_dim, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            in_dim = num_out_channel

        self.pnp_head = WeightedPnPHead()
        self.mlps = nn.ModuleList()
        mlp4 = [mlp3[-1], mlp3[-3]]
        in_dim = mlp3[-1]
        for num_out_channel in mlp4:
            self.mlps.append(Conv1d(in_dim, num_out_channel, bn=True))
            in_dim = num_out_channel
        self.mlps.append(Conv1d(in_dim, 2, use_activation=False))

    def forward(self, warped_xyz, warped_points, RF3, RF3_index, lidar_z,
                gt_project):
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

        qi_xyz_grouped, _, qi_points_grouped, idx = grouping(RF3_bnc, self.nsample_q,
                                                             RF3_index_bnc,
                                                             warped_xyz)
        K = qi_xyz_grouped.shape[2]
        # important
        warped_xyz = warped_xyz.mul(lidar_z)  # restore depth

        src_xyz_expand = warped_xyz[:, :, None, :].repeat(1, 1, K, 1)  # B,N,K,3

        src_points_expand = warped_points[:, :, None, :].repeat(1, 1, K, 1)

        geom_feats = torch.cat([src_xyz_expand, qi_xyz_grouped], dim=-1)  # 6,xyz,xy1
        desc_feats = torch.cat([src_points_expand, qi_points_grouped], dim=-1)  # 2c

        # B,N,M,6+2C
        feats = torch.cat([geom_feats, desc_feats], dim=-1)

        for mlp in self.convs_1:
            feats = mlp(feats)  # B,N,M,C
        attentive_weights = torch.max(feats, dim=-1)[0]  # B,N,M
        attentive_weights = F.softmax(attentive_weights, dim=-1)

        corres_uv1 = torch.sum(torch.mul(attentive_weights.unsqueeze(-1), qi_xyz_grouped), dim=2,
                               keepdim=False)  # [B,N,3]
        attentive_feats = torch.sum(torch.mul(attentive_weights.unsqueeze(-1), feats), dim=-2, keepdim=False)  # [B,N,C]
        for i, conv in enumerate(self.mlps):
            attentive_feats = conv(attentive_feats)
        weights = attentive_feats  # B,N,2

        if gt_project is not None:
            weights_input = gt_project.argmax(-1)
        else:
            weights_input = weights.argmax(-1)
        R, t = self.pnp_head(warped_xyz, corres_uv1[:, :, :2], weights_input.float())

        return R, t, weights
