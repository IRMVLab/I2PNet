import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from pprint import pprint
from pointnet2.pointnet2_utils import FurthestPointSampling

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device    
    B = points.shape[0]    
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    #batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(
        device)  # B*npoint的0矩阵
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # randint(low=0, high, size, out=None, dtype=None) 标准正态分布 B*1
    # farthest = 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print(xyz.shape)
        # print(xyz[batch_indices, farthest, :].shape)
        # print(xyz[batch_indices, farthest, :])
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def query_ball_point(radius, nsample, xyz, new_xyz):
    """

    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(
        device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    # torch.cuda.empty_cache()
    group_idx[sqrdists > radius ** 2] = N
    # print("group_idx.shape##",group_idx.shape)# torch.Size([1, 4096, 8192])
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # print("group_idx.shape###",group_idx.shape)# torch.Size([1, 4096, 8192])
    temp_list=group_idx.cpu().numpy().tolist()[0][0]
    # pprint(temp_list)
    # count=0
    # for data in temp_list:
    #     if data==N:
    #         continue
    #     else:
    #         count+=1
    # print(count)
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    # pprint(group_idx.shape)
    # pprint(group_idx.cpu().numpy().tolist())
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, sample_idx=None, raw_feat_point=False, raw_xyz=None, feat_mode=None):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    if sample_idx is not None:
        fps_idx = sample_idx
    else:
        # TODO: faster FPS
        fps_idx = FurthestPointSampling.forward(None,xyz.contiguous(),npoint).long()
        # fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]

    new_xyz = index_points(xyz, fps_idx)
    if raw_feat_point:
        new_raw_xyz = index_points(raw_xyz, fps_idx)
    idx = knn_point(nsample, xyz, new_xyz)

      # [B, npoint, nsample, C]
    #print("new_raw_xyz shape:", new_raw_xyz.shape)
    if raw_feat_point:
        grouped_xyz = index_points(raw_xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_raw_xyz.view(B, S, 1, C)
    else:
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    _, _, N, _ = grouped_xyz.shape

    if feat_mode == "dim10feat": # use 10 dim feat
        ## adding center point: 
        if raw_feat_point:
            center_points = new_raw_xyz.view(B, S, 1, C).repeat(1,1,N,1)
        else:
            center_points = new_xyz.view(B, S, 1, C).repeat(1,1,N,1)

        ## adding e-dis
        dist = torch.norm(grouped_xyz_norm, p=2, dim=3).unsqueeze(3)

        new_points = torch.cat(
            [grouped_xyz_norm, center_points, grouped_xyz, dist], -1)
    elif feat_mode =="dist":
        dist = torch.norm(grouped_xyz_norm, p=2, dim=3).unsqueeze(3)
        #new_points =  torch.cat(
        #    [grouped_xyz_norm, dist], -1)
        new_points = dist
    else:
        if points is not None:
            grouped_points = index_points(points, idx)
            # [B, npoint, nsample, C+D]
            new_points = torch.cat(
                [grouped_xyz_norm, grouped_points], -1)
        else:
            new_points = grouped_xyz_norm
    if returnfps:
        if raw_feat_point:
            return new_xyz, new_points, grouped_xyz, fps_idx, new_raw_xyz #new_xyz, new_points, grouped_xyz, fps_idx
        else:
            return new_xyz, new_points, grouped_xyz, fps_idx, None
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3] B：batchsize N:number of points 3:xyz
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape  # C=3
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(
            B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(
                nn.Conv2d(last_channel, out_channel, 1))

            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points, sample_idx=None, feat_mode=None, raw_feat_point=False, raw_xyz=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, C, N = xyz.shape
        xyz = xyz.permute(0, 2, 1)

        grouped_xyz = []
        fps_idx = []

        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points, grouped_xyz, fps_idx, new_raw_xyz = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points, returnfps=True, sample_idx=sample_idx, 
                raw_feat_point=raw_feat_point, raw_xyz=raw_xyz, feat_mode = feat_mode)

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]

        # point net layer
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        if raw_feat_point:
            return new_xyz, new_points, grouped_xyz, fps_idx, new_raw_xyz
        else:
            return new_xyz, new_points, grouped_xyz, fps_idx, None


# class PointNetSetAbstractionMsg(nn.Module):
#     def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
#         super(PointNetSetAbstractionMsg, self).__init__()
#         self.npoint = npoint
#         self.radius_list = radius_list
#         self.nsample_list = nsample_list
#         self.conv_blocks = nn.ModuleList()
#         self.bn_blocks = nn.ModuleList()
#         for i in range(len(mlp_list)):
#             convs = nn.ModuleList()
#             bns = nn.ModuleList()
#             last_channel = in_channel + 3
#             for out_channel in mlp_list[i]:
#                 convs.append(nn.Conv2d(last_channel, out_channel, 1))
#                 bns.append(nn.BatchNorm2d(out_channel))
#                 last_channel = out_channel
#             self.conv_blocks.append(convs)
#             self.bn_blocks.append(bns)

#     def forward(self, xyz, points):
#         """
#         Input:
#             xyz: input points position data, [B, C, N]
#             points: input points data, [B, D, N]
#         Return:
#             new_xyz: sampled points position data, [B, C, S]
#             new_points_concat: sample points feature data, [B, D', S]
#         """
#         xyz = xyz.permute(0, 2, 1)
#         if points is not None:
#             points = points.permute(0, 2, 1)

#         B, N, C = xyz.shape
#         S = self.npoint
#         new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
#         new_points_list = []
#         for i, radius in enumerate(self.radius_list):
#             K = self.nsample_list[i]
#             group_idx = query_ball_point(radius, K, xyz, new_xyz)
#             grouped_xyz = index_points(xyz, group_idx)
#             grouped_xyz -= new_xyz.view(B, S, 1, C)
#             if points is not None:
#                 grouped_points = index_points(points, group_idx)
#                 grouped_points = torch.cat(
#                     [grouped_points, grouped_xyz], dim=-1)
#             else:
#                 grouped_points = grouped_xyz

#             grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
#             for j in range(len(self.conv_blocks[i])):
#                 conv = self.conv_blocks[i][j]
#                 bn = self.bn_blocks[i][j]
#                 grouped_points = F.relu(bn(conv(grouped_points)))
#             new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
#             new_points_list.append(new_points)

#         new_xyz = new_xyz.permute(0, 2, 1)
#         new_points_concat = torch.cat(new_points_list, dim=1)
#         return new_xyz, new_points_concat


# class PointNetFeaturePropagation(nn.Module):
#     def __init__(self, in_channel, mlp):
#         super(PointNetFeaturePropagation, self).__init__()
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm1d(out_channel))
#             last_channel = out_channel

#     def forward(self, xyz1, xyz2, points1, points2):
#         """
#         Input:
#             xyz1: input points position data, [B, C, N]
#             xyz2: sampled input points position data, [B, C, S]
#             points1: input points data, [B, D, N]
#             points2: input points data, [B, D, S]
#         Return:
#             new_points: upsampled points data, [B, D', N]
#         """
#         xyz1 = xyz1.permute(0, 2, 1)
#         xyz2 = xyz2.permute(0, 2, 1)

#         points2 = points2.permute(0, 2, 1)
#         B, N, C = xyz1.shape
#         _, S, _ = xyz2.shape

#         if S == 1:
#             interpolated_points = points2.repeat(1, N, 1)
#         else:
#             dists = square_distance(xyz1, xyz2)
#             dists, idx = dists.sort(dim=-1)
#             dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

#             dist_recip = 1.0 / (dists + 1e-8)
#             norm = torch.sum(dist_recip, dim=2, keepdim=True)
#             weight = dist_recip / norm
#             interpolated_points = torch.sum(index_points(
#                 points2, idx) * weight.view(B, N, 3, 1), dim=2)

#         if points1 is not None:
#             points1 = points1.permute(0, 2, 1)
#             new_points = torch.cat([points1, interpolated_points], dim=-1)
#         else:
#             new_points = interpolated_points

#         new_points = new_points.permute(0, 2, 1)
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             new_points = F.relu(bn(conv(new_points)))
#         return new_points
