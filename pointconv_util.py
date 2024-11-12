"""
PointConv util functions
Author: Wenxuan Wu
Date: May 2020
"""
import os
import torch
import torch.nn as nn

import torch.nn.functional as F
from time import time
import numpy as np
#import tensorflow as tf
#from sklearn.neighbors.kde import KernelDensity
#from sklearn.neighbors import KDTree
from pointnet2 import pointnet2_utils
#import pointnet2.pointnet2_utils as pointnet2_utils

LEAKY_RATE = 0.1
use_bn = False

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.cuda.set_device(1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_activation=True,
                 use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if use_activation:
            relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        else:
            relu = nn.Identity()

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.composed_module(x)
        x = x.permute(0, 2, 1)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=[1, 1], bn=False, activation_fn=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = bn
        self.activation_fn = activation_fn

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        if bn:
            self.bn_linear = nn.BatchNorm2d(out_channels)

        if activation_fn:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x (b,n,s,c)
        # print('x is ')
        # print('x: ', x.device)
        x = x.permute(0, 3, 2, 1)  # (b,c,s,n)
        # print(self.conv)
        outputs = self.conv(x)
        #print('self conv has be carried out')
        if self.bn:
            outputs = self.bn_linear(outputs)

        if self.activation_fn:
            outputs = self.relu(outputs)

        outputs = outputs.permute(0, 3, 2, 1)  # (b,n,s,c)
        return outputs


def square_distance(src, dst):

    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

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
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def grouping(feature, K, src_xyz, q_xyz, use_xyz=False):
    '''
    Input:
        feature: (batch_size, ndataset, c)
        K: neighbor size
        src_xyz: original point xyz (batch_size, ndataset, 3)
        q_xyz: query point xyz (batch_size, npoint, 3)
    Return:
        grouped_xyz: (batch_size, npoint, K,3)
        xyz_diff: (batch_size, npoint,K, 3)
        new_points: (batch_size, npoint,K, c+3) if use_xyz else (batch_size, npoint,K, c)
        point_indices: (batch_size, npoint, K)
    '''
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_xyz = q_xyz.contiguous()
    src_xyz = src_xyz.contiguous()

    point_indices = knn_point(K, src_xyz, q_xyz)  # (batch_size, npoint, K)

    grouped_xyz = index_points_group(src_xyz, point_indices)  # (batch_size, npoint, K,3)
    #print(grouped_xyz.device,q_xyz.device )
    xyz_diff = grouped_xyz - (q_xyz.unsqueeze(2)).repeat(1, 1, K, 1)  # (batch_size, npoint,K, 3)
    # x' - x : KNN points - centroids
    grouped_feature = index_points_group(feature, point_indices)  # (batch_size, npoint, K,c)
    if use_xyz:
        new_points = torch.cat([xyz_diff, grouped_feature], dim=-1)  # (batch_size, npoint,K, c+3)
    else:
        new_points = grouped_feature  # (batch_size, npoint, K,c)

    return grouped_xyz, xyz_diff, new_points, point_indices

def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)  # [B,S,nsample,C]
    grouped_xyz = index_points_group(s_xyz, idx)  # [B, S, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # [B, S, nsample, C]
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)  # [B, S, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, S, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

#change
def sample_and_group(sample_idx, npoint, radius, nsample, xyz, xyz_raw, label,
                     points,
                     knn=True, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor  channel——是否涉及local point features
        label: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_label: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
    '''
    # print ('sample_and_group_ xyz shape is: ',xyz.shape)
    # print ('sample_and_group_ xyz_raw shape is: ', xyz_raw.shape)
    # print ('sample_and_group_ label shape is: ', label.shape)
    xyz = xyz.contiguous()
    #print(xyz.shape,'=============')
    # print ('sample_and_group_ xyz shape after contiguous is: ', xyz.shape)
    '''if npoint == 2048:
        # new_xyz, _, _, _ = torch.split(xyz, 2048, 1)  # (batch_size, 2048, 3)
        # new_xyz_raw, _, _, _ = torch.split(xyz_raw, 2048, 1)  # (batch_size, 2048, 3)
        # new_label, _, _, _ = torch.split(label, 2048, 1)  # (batch_size, 2048, 3)
        new_xyz = torch.split(xyz, 2048, 1)[0]  # (batch_size, 2048, 3)
        new_xyz_raw = torch.split(xyz_raw, 2048, 1)[0]  # (batch_size, 2048, 3)
        new_label = torch.split(label, 2048, 1)[0]  # (batch_size, 2048, 3)

        if sample_idx is not None:
            sample_idx = sample_idx
        else:
            sample_idx = pointnet2_utils.furthest_point_sample(xyz, npoint)  # (batch_size,npoint)
            #print(sample_idx)
    else:'''
    if sample_idx is not None:
        sample_idx = sample_idx
    else:
        sample_idx = pointnet2_utils.furthest_point_sample(xyz, npoint)  # (batch_size,npoint)

    new_xyz = index_points_gather(xyz, sample_idx)  # (batch_size, npoint, 3)
    new_label = index_points_gather(label, sample_idx)  # (batch_size, npoint, 3)
    if xyz_raw is not None:
        new_xyz_raw = index_points_gather(xyz_raw, sample_idx)

    if points is None:
        grouped_xyz, xyz_diff, grouped_points, idx = grouping(xyz, nsample, xyz,
                                                              new_xyz)  # (b, n,nsample,3) (b, n,nsample. 3) (b, n, nsample, 3) (b,n,nsample)
        new_points = torch.cat([xyz_diff, grouped_points], dim=-1)  # (b, n,nsample,3+3)

    else:
        grouped_xyz, xyz_diff, grouped_points, idx = grouping(points, nsample, xyz,
                                                              new_xyz)  # (b, n,nsample,3) (b, n,nsample. 3) (b, n, nsample, c) (b,n,nsample)
        new_points = torch.cat([xyz_diff, grouped_points], dim=-1)  # (batch_size, npoint, nample, 3+c)

    if xyz_raw is not None:
        return new_xyz, new_label, new_points, new_xyz_raw,sample_idx
    else:
        return new_xyz, new_label, new_points,sample_idx  # (batch_size, npoint, 3) (batch_size, npoint, 3) (batch_size, npoint, nample, 3+c)


class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], bn=use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights = F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights



# is_training is not used in all class
class PointNetSaModule(nn.Module):
    def __init__(self,npoint, radius, nsample, in_channels, mlp, mlp2,
                 group_all, is_training, bn_decay, bn=True, pooling='max', knn=False, use_xyz=True):
        super(PointNetSaModule, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.in_channels = in_channels + 3   #3 + 3
        self.mlp = mlp   #[8 8 16]
        self.mlp2 = mlp2
        self.group_all = group_all
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.use_xyz = use_xyz
        self.num_mlp_layers = len(mlp)
        self.mlp_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()


        for i, num_out_channel in enumerate(mlp):
            self.mlp_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=bn))
            self.in_channels = num_out_channel

        # if pooling == 'max_and_avg':
        # self.in_channels = 2 * mlp[-1]

        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.mlp2_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=bn))
                self.in_channels = num_out_channel

    def forward(self, xyz, xyz_raw, label, points, sample_idx=None):
        """
        PointNetSaModule
        Input:
            xyz: input points position data, [B, N, 3]
            xyz_raw: [B, N,3]
            label: [B, N,3]
            points: input points data, [B, N, C]  Initial point cloud characteristics
        Return:
            new_xyz: sampled points position data, [B, npoint, 3]
            new_label: [B, npoint, 3]
            new_points: (batch_size, npoint, mlp2[-1]) if mlp2 is not None else  (batch_size, npoint, mlp[-1])
        """

        if xyz_raw is not None:
            # [B,npoint,3] [B,npoint,3] [B,npoint,nsample,3+C] [B,npoint,3]
            new_xyz, new_label, new_points, new_xyz_raw,sample_idx = sample_and_group(sample_idx, self.npoint,
                                                                           self.radius,
                                                                           self.nsample, xyz, xyz_raw, label, points,
                                                                           self.knn, self.use_xyz)
            #print('sam can be carried out')
        else:
            new_xyz, new_label, new_points,sample_idx = sample_and_group(sample_idx, self.npoint, self.radius, self.nsample, xyz, xyz_raw,
                                                              label, points, self.knn, self.use_xyz)

        # new_points: (batch_size, npoint, nample, 3+channel)
        # print('try to conv')
        # print('new_points is')
        # print(new_points)
        # print('new_points device: ', new_points.device)
        for i, conv in enumerate(self.mlp_convs):
            new_points = conv(new_points)

        #print('it ok')
        # (batch_size, npoint, nample, mlp[-1])
        if self.pooling == 'max':
            new_points = torch.max(new_points, dim=2, keepdim=True)[0]  # (batch_size, npoint, 1, mlp[-1])
        elif self.pooling == 'avg':
            new_points = torch.mean(new_points, dim=2, keepdim=True)  # (batch_size, npoint, 1, mlp[-1])
        # elif self.pooling == 'max_and_avg':
        # max_points = torch.max(new_points,dim=2,keepdim=True)[0]
        # avg_points = torch.mean(new_points,dim=2,keepdim=True)
        # new_points = torch.cat([avg_points,max_points],dim=-1)

        if self.mlp2 is not None:
            for i, conv in enumerate(self.mlp2_convs):
                new_points = conv(new_points)

        new_points = new_points.squeeze(2)  # (batch_size,npoint, mlp2[-1]) if mlp2 is not None else  (batch_size,npoint, mlp[-1])

        if xyz_raw is not None:
            return new_xyz, new_label, new_points, new_xyz_raw,sample_idx
        else:
            return new_xyz, new_label, new_points,sample_idx



class CostVolume(nn.Module):
    def __init__(self,radius, nsample, nsample_q, in_channels, mlp1, mlp2,
                 is_training, bn_decay, bn=True, pooling='max', knn=True, corr_func='elementwise_product'):
        super(CostVolume, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.nsample_q = nsample_q
        self.in_channels = 2 * in_channels + 10
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.corr_func = corr_func

        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_convs_2 = nn.ModuleList()

        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i, num_out_channel in enumerate(mlp1):
            self.mlp1_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        self.pi_encoding = Conv2d(10, mlp1[-1], [1, 1], stride=[1, 1], bn=True)

        self.in_channels = 2 * mlp1[-1]
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        self.pc_encoding = Conv2d(10, mlp1[-1], [1, 1], stride=[1, 1], bn=True)

        self.in_channels = 2 * mlp1[-1] + in_channels
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs_2.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

    def forward(self, warped_xyz, warped_points, f2_xyz, f2_points):
        '''
            Input:
                warped_xyz: (b,npoint,3)
                warped_points: (b,npoint,c)
                f2_xyz:  (b,ndataset,3)
                f2_points: (b,ndataset,c)

            Output:
                pc_feat1_new: batch_size, npoints, mlp2[-1]
            '''

        qi_xyz_grouped, _, qi_points_grouped, idx = grouping(f2_points, self.nsample_q, f2_xyz,
                                                             warped_xyz)  # (b,npoint,nsample_q,3) (b,npoint,nsample_q,3) (b,npoint,nsample_q,c)
        pi_xyz_expanded = (torch.unsqueeze(warped_xyz, 2)).repeat(
            [1, 1, self.nsample_q, 1])  # batch_size, npoints, nsample_q, 3
        pi_points_expanded = (torch.unsqueeze(warped_points, 2)).repeat(
            [1, 1, self.nsample_q, 1])  # batch_size, npoints, nsample, c

        pi_xyz_diff = qi_xyz_grouped - pi_xyz_expanded  # batch_size, npoints, nsample_q, 3

        pi_euc_diff = torch.sqrt(torch.sum(torch.mul(pi_xyz_diff, pi_xyz_diff), dim=-1,
                                           keepdim=True) + 1e-20)  # batch_size, npoints, nsample_q, 1

        pi_xyz_diff_concat = torch.cat([pi_xyz_expanded, qi_xyz_grouped, pi_xyz_diff, pi_euc_diff],
                                       dim=3)  # batch_size, npoints, nsample_q,10

        pi_feat_diff = torch.cat([pi_points_expanded, qi_points_grouped], dim=-1)  # batch_size, npoints, nsample, 2c
        pi_feat1_new = torch.cat([pi_xyz_diff_concat, pi_feat_diff], dim=3)  # batch_size, npoint, nsample, 10+2c

        for i, conv in enumerate(self.mlp1_convs):
            pi_feat1_new = conv(pi_feat1_new)  # batch_size, npoint, nsample, mlp1[-1]

        pi_xyz_encoding = self.pi_encoding(pi_xyz_diff_concat)  # batch_size, npoints, nsample_q,mlp1[-1]

        pi_concat = torch.cat([pi_xyz_encoding, pi_feat1_new], dim=3)  # batch_size, npoints, nsample_q,2*mlp1[-1]
        #print('start')
        for j, conv in enumerate(self.mlp2_convs):
            pi_concat = conv(pi_concat)  # batch_size, npoints, nsample_q,mlp2[-1]
        #print('end')
        WQ = F.softmax(pi_concat, dim=2)

        pi_feat1_new = WQ * pi_feat1_new  # mlp1[-1]=mlp2[-1]
        pi_feat1_new = torch.sum(pi_feat1_new, dim=2, keepdim=False)  # batch_size, npoint,mlp1[-1]

        pc_xyz_grouped, _, pc_points_grouped, idx = grouping(pi_feat1_new, self.nsample, warped_xyz,
                                                             warped_xyz)  # (b,npoint,nsample,3) (b,npoint,nsample,3) (b,npoint,nsample,mlp1[-1])

        pc_xyz_new = (torch.unsqueeze(warped_xyz, dim=2)).repeat(
            [1, 1, self.nsample, 1])  # batch_size, npoints, nsample, 3
        pc_points_new = (torch.unsqueeze(warped_points, dim=2)).repeat(
            [1, 1, self.nsample, 1])  # batch_size, npoints, nsample, c

        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new  # batch_size, npoints, nsample, 3

        pc_euc_diff = torch.sqrt(torch.sum(torch.mul(pc_xyz_diff, pc_xyz_diff), dim=3,
                                           keepdim=True) + 1e-20)  # batch_size, npoints, nsample, 1

        pc_xyz_diff_concat = torch.cat([pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff],
                                       dim=3)  # batch_size, npoints, nsample, 10
        #print(pc_xyz_diff_concat.shape,"pc_xyz_diff_concat shape")
        pc_xyz_encoding = self.pc_encoding(pc_xyz_diff_concat)  # batch_size, npoints, nsample, mlp1[-1]
        #print(pc_xyz_encoding.shape,"pc_xyz_encoding shape")
        pc_concat = torch.cat([pc_xyz_encoding, pc_points_new, pc_points_grouped],
                              dim=-1)  # batch_size, npoints, nsample, mlp[-1]+3+mlp[-1]
        #print('start')
        for j, conv in enumerate(self.mlp2_convs_2):
            #print(pc_concat.shape,"0000000")
            pc_concat = conv(pc_concat)  # batch_size, npoints, nsample, mlp2[-1]
        #print('end')
        WP = F.softmax(pc_concat, dim=2)

        pc_feat1_new = WP * pc_points_grouped  # batch_size, npoints, nsample, mlp2[-1]

        pc_feat1_new = torch.sum(pc_feat1_new, dim=2, keepdim=False)  # batch_size, npoints, mlp2[-1]

        return pc_feat1_new


class SetUpconvModule(nn.Module):
    def __init__(self, nsample, in_channels, mlp, mlp2, is_training,
                 bn_decay=None, bn=True, pooling='max', radius=None, knn=True):
        super(SetUpconvModule, self).__init__()
        self.nsample = nsample
        self.last_channel = in_channels[-1] + 3
        self.mlp = mlp
        self.mlp2 = mlp2
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.radius = radius
        self.knn = knn

        self.mlp_conv = nn.ModuleList()
        self.mlp2_conv = nn.ModuleList()

        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if mlp is not None:
            for i, num_out_channel in enumerate(mlp):
                self.mlp_conv.append(Conv2d(self.last_channel, num_out_channel, [1, 1], stride=[1, 1], bn=True))
                self.last_channel = num_out_channel

        if len(mlp) is not 0:
            self.last_channel = mlp[-1] + in_channels[0]
        else:
            self.last_channel = self.last_channel + in_channels[0]
        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.mlp2_conv.append(Conv2d(self.last_channel, num_out_channel, [1, 1], stride=[1, 1], bn=True))
                self.last_channel = num_out_channel

    def forward(self, xyz1, xyz2, feat1, feat2):
        '''
            Input:
                xyz1: (batch_size, npoint1,3)
                xyz2: (batch_size, npoint2,3)
                feat1: (batch_size, npoint1,c1) features for xyz1 points (earlier layers, more points)
                feat2: (batch_size, npoint2, c2) features for xyz2 points
            Return:
                (batch_size, npoint1, mlp[-1] or mlp2[-1] or channel1+3)
            '''
        xyz2_grouped, _, feat2_grouped, idx = grouping(feat2, self.nsample, xyz2,
                                                       xyz1)  # (batch_size,npoint1,nsample,3) _ (batch_size,npoint1,nsample,c2)

        xyz1_expanded = torch.unsqueeze(xyz1, 2)  # batch_size, npoint1, 1, 3
        xyz_diff = xyz2_grouped - xyz1_expanded  # batch_size, npoint1, nsample, 3

        net = torch.cat([feat2_grouped, xyz_diff], dim=3)  # batch_size, npoint1, nsample, channel2+3

        if self.mlp is not None:
            for i, conv in enumerate(self.mlp_conv):
                net = conv(net)

        if self.pooling == 'max':
            feat1_new = torch.max(net, dim=2, keepdim=False)[0]  # batch_size, npoint1, mlp[-1]
        elif self.pooling == 'avg':
            feat1_new = torch.mean(net, dim=2, keepdim=False)  # batch_size, npoint1, mlp[-1]

        if feat1 is not None:
            feat1_new = torch.cat([feat1_new, feat1], dim=2)  # batch_size, npoint1, mlp[-1]+channel1

        feat1_new = torch.unsqueeze(feat1_new, 2)  # batch_size, npoint1, 1, mlp[-1]

        if self.mlp2 is not None:
            for i, conv in enumerate(self.mlp2_conv):
                feat1_new = conv(feat1_new)

        feat1_new = torch.squeeze(feat1_new, 2)  # batch_size, npoint1, mlp2[-1]
        return feat1_new


class PointnetFpModule(nn.Module):
    def __init__(self, in_channels, mlp, is_training, bn_decay, bn=True, last_mlp_activation=True):
        super(PointnetFpModule, self).__init__()
        self.in_channels = in_channels
        self.mlp = mlp
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.last_mlp_activation = last_mlp_activation
        self.mlp_conv = nn.ModuleList()
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i, num_out_channel in enumerate(mlp):
            if i == len(mlp) - 1 and not (last_mlp_activation):
                activation_fn = False
            else:
                activation_fn = True
            self.mlp_conv.append(
                Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=bn, activation_fn=activation_fn))
            self.in_channels = num_out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
        """
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist, idx = pointnet2_utils.three_nn(xyz1, xyz2)  # (b,n1,3)
        dist[dist < 1e-10] = 1e-10
        norm = torch.sum((1.0 / dist), dim=2, keepdim=True)
        norm = norm.repeat(1, 1, 3)
        weight = (1.0 / dist) / norm
        points2 = points2.permute(0, 2, 1)
        interpolated_points = pointnet2_utils.three_interpolate(points2, idx, weight)
        interpolated_points = interpolated_points.permute(0, 2, 1)  # (b,n1,c2)

        new_points1 = interpolated_points

        if points1 is not None:
            new_points1 = torch.cat([interpolated_points, points1], dim=2)  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points  # B,ndataset1,nchannel2
        new_points1 = torch.unsqueeze(new_points1, 2)

        for i, conv in enumerate(self.mlp_conv):
            new_points1 = conv(new_points1)

        new_points1 = torch.squeeze(new_points1, 2)  # B,ndataset1,mlp[-1]
        return new_points1


class WarpingLayers(nn.Module):

    def forward(self, xyz1, upsampled_flow):
        return xyz1 + upsampled_flow


class FlowPredictor(nn.Module):

    def __init__(self, in_channels, mlp, is_training, bn_decay, bn=True):
        super(FlowPredictor, self).__init__()
        self.in_channels = in_channels
        self.mlp = mlp
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.mlp_conv = nn.ModuleList()
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i, num_out_channel in enumerate(mlp):
            self.mlp_conv.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=bn))
            self.in_channels = num_out_channel

    def forward(self, points_f1, upsampled_feat, cost_volume):

        '''
                    Input:
                        points_f1: (b,n,c1)
                        upsampled_feat: (b,n,c2)
                        cost_volume: (b,n,c3)

                    Output:
                        points_concat:(b,n,mlp[-1])
                '''
        if upsampled_feat is not None:
            points_concat = torch.cat([points_f1, cost_volume, upsampled_feat], -1)  # b,n,c1+c2+c3
        else:
            points_concat = torch.cat([points_f1, cost_volume], -1)

        points_concat = torch.unsqueeze(points_concat, 2)  # B,n,1,c1+c2+c3

        for i, conv in enumerate(self.mlp_conv):
            points_concat = conv(points_concat)

        points_concat = torch.squeeze(points_concat, 2)

        return points_concat


class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)  # permute将tensor维度换位

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B, N,
                                                                                                                -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points


class PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B,
                                                                                                                self.npoint,
                                                                                                                -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx


class PointConvFlow(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn=use_bn, use_leaky=True):
        super(PointConvFlow, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1)  # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx)  # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim=-1)  # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1

        point_to_patch_cost = torch.sum(weights * new_points, dim=2)  # B C N

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, xyz1, xyz1)  # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1),
                                                         knn_idx)  # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim=2)  # B C N

        return patch_to_patch_cost


class PointWarping(nn.Module):

    def forward(self, xyz1, xyz2, flow1=None):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1)  # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1)  # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        knn_idx = knn_point(3, xyz1_to_2, xyz2)
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C)  # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim=2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1)  # B 3 N2

        return warped_xyz2


class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        # import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1)  # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1)  # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1)  # B S 3
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim=2).permute(0, 2, 1)
        return dense_flow


class SceneFlowEstimatorPointConv(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch=3, channels=[128, 128], mlp=[128, 64], neighbors=9, clamp=[-200, 200],
                 use_leaky=True):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn=True, use_leaky=True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow=None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim=1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim=1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])
