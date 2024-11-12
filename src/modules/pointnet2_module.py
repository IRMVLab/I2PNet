import torch.nn as nn
import torch

from src.modules.point_utils import grouping
from src.modules.basicConv import Conv2d,Conv1d

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

        if mlp is not None:
            for i, num_out_channel in enumerate(mlp):
                self.mlp_conv.append(Conv2d(self.last_channel, num_out_channel, [1, 1], stride=[1, 1], bn=True))
                self.last_channel = num_out_channel

        # if len(mlp) is not 0:
        if len(mlp) > 0:
            self.last_channel = mlp[-1] + in_channels[0]
        else:
            self.last_channel = self.last_channel + in_channels[0]
        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.mlp2_conv.append(Conv2d(self.last_channel, num_out_channel, [1, 1], stride=[1, 1], bn=True))
                self.last_channel = num_out_channel

    def forward(self, xyz1, xyz2, feat1, feat2, raw_feat_point=False, raw_xyz1=None, raw_xyz2=None):
        '''
            Input:
                xyz1: (batch_size, npoint1,3)
                xyz2: (batch_size, npoint2,3)
                feat1: (batch_size, npoint1,c1) features for xyz1 points (earlier layers, more points)
                feat2: (batch_size, npoint2, c2) features for xyz2 points
                inchannel=[c1,c2]
            Return:
                (batch_size, npoint1, mlp[-1] or mlp2[-1] or channel1+3)
            '''
        xyz2_grouped, _, feat2_grouped, idx, raw_xyz2_grouped = grouping(feat2, self.nsample, xyz2,
                                                       xyz1, raw_feat_point=raw_feat_point, raw_xyz1=raw_xyz2, raw_xyz2=raw_xyz1)  # (batch_size,npoint1,nsample,3) _ (batch_size,npoint1,nsample,c2)

        xyz1_expanded = torch.unsqueeze(xyz1, 2)  # batch_size, npoint1, 1, 3
        if raw_feat_point:
            raw_xyz1_expanded = torch.unsqueeze(raw_xyz1, 2)
            xyz_diff = raw_xyz2_grouped - raw_xyz1_expanded  # batch_size, npoint1, nsample, 3
        else:
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