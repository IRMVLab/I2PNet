import torch.nn as nn
import torch
import torch.nn.functional as F
from pointnet2.pointnet2_utils import FurthestPointSampling

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
    new_points = points[batch_indices, idx, :]
    return new_points


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


def sample_and_group(npoint, nsample, xyz, points, returnfps=False, sample_idx=None):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, S, 3]
        new_points: sampled points data, [B, N, K , 3+C]
    """
    B, N, C = xyz.shape
    S = npoint
    if sample_idx is not None:
        fps_idx = sample_idx
    else:
        fps_idx = FurthestPointSampling.forward(None, xyz.contiguous(), npoint).long()

    new_xyz = index_points(xyz, fps_idx)
    idx = knn_point(nsample, xyz, new_xyz)

    grouped_xyz = index_points(xyz, idx)  # [B, N,K, C]

    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)  # [B, N,K, C+D]

        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], -1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, grouped_xyz_norm, fps_idx
    else:
        return new_xyz, new_points


class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super(WeightNet, self).__init__()

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
        # xyz : Bx3xKxN
        # weight BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            weights = F.relu(self.mlp_bns[i](conv(weights)), inplace=True)

        return weights


class PointConv(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, weightnet=16):
        super(PointConv, self).__init__()
        self.in_channel = in_channel
        self.radius = radius
        self.npoint = npoint
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        self.weightnet = WeightNet(3, weightnet)

        out_channel = mlp[-1]

        self.linear = nn.Linear(last_ch * weightnet, out_channel)

        self.bn_linear = nn.BatchNorm1d(out_channel)

    def forward(self, xyz, points, sample_idx=None, use_dim10=False):
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
        if points is not None:
            points = points.permute(0, 2, 1)

        # new_points [B,N,K,C+D] grouped_xyz_norm [B,N,K,3] (-xyz)
        new_xyz, new_points, grouped_xyz, grouped_xyz_norm, fps_idx = \
            sample_and_group(
            self.npoint, self.nsample, xyz, points,
            returnfps=True, sample_idx=sample_idx)

        M = new_points.shape[1]

        if use_dim10: # use 10 dim feat
            ## adding center point: 
            center_points = new_xyz.view(B, self.npoint, 1, C).repeat(1,1,grouped_xyz_norm.shape[2],1)
            ## adding e-dis
            dist = torch.norm(grouped_xyz_norm, p=2, dim=3).unsqueeze(3)

            new_points = torch.cat(
                [grouped_xyz_norm, center_points, grouped_xyz, dist], -1)

        weightNetInput = grouped_xyz_norm.permute(0, 3, 2, 1)  # B,C,K,N

        weights = self.weightnet(weightNetInput)

        # MLP
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+3, K, M]
        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(self.mlp_bns[i](conv(new_points)), inplace=True)

        # Aggregation
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2),  # B,M,C,K
                                  other=weights.permute(0, 3, 2, 1)  # B,M,K,16
                                  ).view(B, M, -1)

        new_points = F.relu(self.bn_linear(self.linear(new_points).permute(0, 2, 1)), inplace=True)  # B,C,M

        # new_xyz [B,3,M] new_points [B,C,M]
        return new_xyz.permute(0, 2, 1), \
               new_points, \
               grouped_xyz, fps_idx
