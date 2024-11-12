import torch.nn as nn
import spconv.pytorch as spconv
import torch
from torch_scatter import scatter_softmax, scatter_sum


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1,
                 indice_key='', subm=False):
        super().__init__()
        self.conv = spconv.SparseConv3d(inc,
                                        outc,
                                        kernel_size=ks,
                                        dilation=dilation,
                                        stride=stride,
                                        bias=False,
                                        indice_key=indice_key,
                                        algo=spconv.ConvAlgo.Native
                                        ) \
            if not subm else \
            spconv.SubMConv3d(inc,
                              outc,
                              kernel_size=ks,
                              dilation=dilation,
                              stride=stride,
                              bias=False,
                              indice_key=indice_key,
                              algo=spconv.ConvAlgo.Native)

        self.bn = nn.BatchNorm1d(outc)
        self.relu = nn.LeakyReLU()

    def forward(self, x: spconv.SparseConvTensor):
        out: spconv.SparseConvTensor = self.conv(x)
        # out = out.replace_feature(self.relu(self.bn(out.features)))

        return out


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1,
                 indice_key=''):
        super().__init__()

        self.conv1 = spconv.SubMConv3d(inc,
                                       outc,
                                       kernel_size=ks,
                                       dilation=dilation,
                                       stride=stride,
                                       bias=False,
                                       indice_key=indice_key + 'down3x3' if stride > 1 else
                                       indice_key,
                                       algo=spconv.ConvAlgo.Native)
        self.bn1 = nn.BatchNorm1d(outc)
        self.relu = nn.LeakyReLU()
        self.conv2 = spconv.SubMConv3d(outc, outc,
                                       kernel_size=ks, dilation=dilation,
                                       bias=False,
                                       stride=1,
                                       indice_key=indice_key,
                                       algo=spconv.ConvAlgo.Native)
        self.bn2 = nn.BatchNorm1d(outc)

        if inc == outc and stride == 1:
            self.downsample = spconv.Identity()
        else:
            self.downsample = spconv.SparseSequential(
                spconv.SubMConv3d(inc, outc, kernel_size=1, dilation=1,
                                  bias=False,
                                  stride=stride, indice_key=indice_key + 'down',
                                  algo=spconv.ConvAlgo.Native),
                spconv.SparseBatchNorm(outc))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = x1.replace_feature(self.relu(self.bn1(x1.features)))
        x2 = self.conv2(x1)
        x2 = x2.replace_feature(self.bn2(x2.features))
        skip = self.downsample(x)
        out = x2.replace_feature(self.relu(x2.features + skip.features))
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, indice_key=''):
        super().__init__()
        # inverse conv
        self.conv = spconv.SparseInverseConv3d(
            inc,
            outc,
            kernel_size=ks,
            bias=False,
            indice_key=indice_key,
            algo=spconv.ConvAlgo.Native)

        self.bn = nn.BatchNorm1d(outc)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)
        out = out.replace_feature(self.relu(self.bn(out.features)))

        return out


class UpSampleLayer(nn.Module):
    def __init__(self, inc1, out_c1, inc2, out_c2, ks=3,
                 indice_key_down='',
                 indice_key=''):
        super().__init__()
        self.upconv = BasicDeconvolutionBlock(inc1, out_c1, ks=ks, indice_key=indice_key_down)

        self.conv = ResidualBlock(out_c1, out_c1, 3, 1, indice_key=indice_key)

        self.mlp = nn.Linear(out_c1 + inc2, out_c2, bias=False)
        self.bn = nn.BatchNorm1d(out_c2)
        self.relu = nn.LeakyReLU()

    def forward(self, x, skip):
        out = self.conv(self.upconv(x))
        return self.relu(self.bn(self.mlp(torch.cat(
            [skip, out.features], dim=-1
        ))))


def nc2bnc(feats, batch_info, batch_size):
    device = feats[0].device
    with torch.no_grad():
        uni, count = torch.unique(batch_info, return_counts=True)
        n = torch.max(count).item()
        offset = torch.cumsum(count, 0) - count
        ind = torch.arange(len(feats[0]), device=device)
        new_count = torch.full_like(count, n)
        new_offset = torch.cumsum(new_count, 0) - new_count
        ind += (new_offset - offset).gather(0, batch_info.long())
    feats_bnc = []
    for feat in feats:
        c = feat.shape[-1]

        feats_bnc.append(torch.zeros(batch_size * n, c, dtype=torch.float32, device=device). \
                         scatter_(0, ind[:, None].repeat(1, c), feat).reshape(batch_size, n, c))

    return feats_bnc, ind


class MLP(nn.Module):
    def __init__(self, c_in, c_out, bn=True):
        super().__init__()
        self.linear = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, bias=not bn)
        if bn:
            self.bn = nn.BatchNorm2d(c_out)
            self.relu = nn.LeakyReLU()
        else:
            self.bn = nn.Identity()
            self.relu = nn.Identity()


    def forward(self, x):
        return self.relu(self.bn(self.linear(x.permute(0, 3, 1, 2)))).permute(0, 2, 3, 1)


class MLP1d(nn.Module):
    def __init__(self, c_in, c_out, bn=True):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out, bias=not bn)
        if bn:
            self.bn = nn.BatchNorm1d(c_out)
            self.relu = nn.LeakyReLU()
        else:
            self.bn = nn.Identity()
            self.relu = nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.linear(x)))


from src.modules.point_utils import grouping, mask_grouping
import torch.nn.functional as F


class CostVolume(nn.Module):

    def __init__(self, nsample, nsample_q, rgb_in_channels, lidar_in_channels, mlp1, mlp2,
                 backward_validation=False
                 ):

        super(CostVolume, self).__init__()

        self.nsample = nsample
        self.nsample_q = nsample_q
        self.mlp1 = mlp1
        self.mlp2 = mlp2

        self.backward_validation = backward_validation

        corr_channel = rgb_in_channels

        if backward_validation:
            corr_channel += lidar_in_channels

        self.in_channels = corr_channel + 6

        self.mlp1_convs = nn.ModuleList()

        self.mlp2_convs = nn.ModuleList()
        self.mlp2_convs_2 = nn.ModuleList()

        for i, num_out_channel in enumerate(mlp1):
            self.mlp1_convs.append(MLP(self.in_channels, num_out_channel))
            self.in_channels = num_out_channel

        self.pi_encoding = MLP(6, mlp1[-1], bn=True)

        self.in_channels = 2 * mlp1[-1]
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs.append(MLP(self.in_channels, num_out_channel))
            self.in_channels = num_out_channel

        self.pc_encoding = MLP(10, mlp1[-1], bn=True)

        self.in_channels = 2 * mlp1[-1] + lidar_in_channels
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs_2.append(MLP(self.in_channels, num_out_channel))
            self.in_channels = num_out_channel

    def forward(self, warped_xyz, warped_points, batch_info, batch_size, f2_xyz, f2_points, lidar_z):
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
        # [num_points, ndim + 1] indice tensor. batch index saved in indices[:, 0]
        bnc, inv = nc2bnc([warped_xyz, warped_points, lidar_z], batch_info, batch_size)
        warped_xyz, warped_points, lidar_z = bnc

        valid_mask = torch.ge(torch.sum(torch.square(warped_xyz), dim=-1), 1e-10).float()  # valid_mask
        if self.nsample_q > 0:
            qi_xyz_grouped, _, qi_points_grouped, idx = grouping(f2_points, self.nsample_q, f2_xyz,
                                                                 warped_xyz)
        else:
            qi_xyz_grouped, qi_points_grouped = f2_xyz.unsqueeze(1).repeat(1, warped_xyz.shape[1], 1, 1), \
                f2_points.unsqueeze(1).repeat(1, warped_xyz.shape[1], 1, 1)

        # important
        warped_xyz = warped_xyz.mul(lidar_z)  # restore depth
        K = qi_xyz_grouped.shape[2]
        pi_xyz_expanded = warped_xyz[:, :, None, :].repeat(1, 1, K, 1)  # B,N,K,3
        pi_points_expanded = warped_points[:, :, None, :].repeat(1, 1, K, 1)  # B,N,K,C

        # position embedding
        pi_xyz_diff_concat = torch.cat([pi_xyz_expanded, qi_xyz_grouped], dim=3)  # B,N,K,6

        pi_points_expanded = (pi_points_expanded - torch.mean(pi_points_expanded, -1, keepdim=True)) / torch.clip(
            torch.std(
                pi_points_expanded, -1, keepdim=True), min=1e-12)
        qi_points_grouped = (qi_points_grouped - torch.mean(qi_points_grouped, -1, keepdim=True)) / torch.clip(
            torch.std(
                qi_points_grouped, -1, keepdim=True), min=1e-12)

        pi_feat_diff = pi_points_expanded * qi_points_grouped  # B,N,K,C

        pi_feat1_new = torch.cat([pi_xyz_diff_concat, pi_feat_diff], dim=3)  # B,N,K, 6+c

        if self.backward_validation:
            # B,N1,N2,C
            repeat_image_feature = qi_points_grouped
            # B,N1,N2,C
            repeat_lidar_feature = pi_points_expanded
            # correlation
            repeat_correlation = repeat_image_feature * repeat_lidar_feature
            mask_inv = valid_mask[:, :, None, None].repeat(1, 1, *repeat_correlation.shape[2:4])

            repeat_correlation = repeat_correlation * mask_inv + -1e10 * (1 - mask_inv)

            image_max_respond = torch.max(repeat_correlation, 1, keepdim=True)[0].repeat(1, warped_xyz.shape[1], 1, 1)

            pi_feat1_new = torch.cat([pi_feat1_new, image_max_respond], dim=-1)
        # mlp1 processes pi corresponding values
        for i, conv in enumerate(self.mlp1_convs):
            pi_feat1_new = conv(pi_feat1_new)  # B,N,K, mlp1[-1], to be weighted sum

        # position encoding for generating weights
        pi_xyz_encoding = self.pi_encoding(pi_xyz_diff_concat)  # B,N,K,mlp1[-1]

        pi_concat = torch.cat([pi_xyz_encoding, pi_feat1_new], dim=3)  # B,N,K,2*mlp1[-1]

        # mlp2 processes the pi features to generate weights
        for j, conv in enumerate(self.mlp2_convs):
            pi_concat = conv(pi_concat)  # B,N,K,mlp2[-1]

        WQ = F.softmax(pi_concat, dim=2)

        pi_feat1_new = WQ * pi_feat1_new  # mlp1[-1]=mlp2[-1]
        pi_feat1_new = torch.sum(pi_feat1_new, dim=2, keepdim=False)  # B,N,mlp1[-1]

        # 3d find 3d grouped features to be weighted

        pc_xyz_grouped, _, pc_points_grouped, idx = mask_grouping(pi_feat1_new, self.nsample, warped_xyz,
                                                                  warped_xyz, valid_mask)

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

        b, n, c = pc_feat1_new.shape
        pc_feat1_new = pc_feat1_new.reshape(b * n, c).gather(0, inv[:, None].repeat(1, c))

        return pc_feat1_new


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
            self.mlp_conv.append(MLP1d(self.in_channels, num_out_channel, bn=bn))
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

        # points_concat = torch.unsqueeze(points_concat, 2)  # B,n,1,c1+c2+c3

        for i, conv in enumerate(self.mlp_conv):
            points_concat = conv(points_concat)

        # points_concat = torch.squeeze(points_concat, 2)

        return points_concat


class PoseHead(nn.Module):

    def __init__(self, in_channels,
                 hidden, q_dim, t_dim, dropout_rate=0.5
                 , split_dp=False):
        """
        take the self attention mask and predictions
        do global attention (global constraint)
        """
        super(PoseHead, self).__init__()

        if split_dp:
            self.DP1 = nn.Identity()
            self.DP2 = nn.Dropout(dropout_rate)
        else:
            self.DP1 = nn.Dropout(dropout_rate)
            self.DP2 = nn.Identity()

        self.hidden_layer = MLP1d(in_channels, hidden, bn=False)
        self.quat_head = MLP1d(hidden, q_dim, bn=False)
        self.trans_head = MLP1d(hidden, t_dim, bn=False)

    def forward(self, prediction, mask, batch_info):
        """
        Args:
            prediction: [N,C]
            mask: [N,C]
        Returns:
            q: [B,4]
            t: [B,3]
        """

        # B,1,C
        batch_info = batch_info.long()

        mask = scatter_softmax(mask, batch_info, dim=0)

        global_prediction = scatter_sum(mask * prediction, batch_info, dim=0)  # 5,64

        hidden_feature = self.DP1(self.hidden_layer(global_prediction))

        q = self.quat_head(self.DP2(hidden_feature))
        t = self.trans_head(self.DP2(hidden_feature))

        # normalize q
        q = q / (torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-10) + 1e-10)

        return q, t


from torch_scatter import scatter_mean


def get_downsample_info(xyz, batch, indice_pairs):
    pair_in, pair_out = indice_pairs[0], indice_pairs[1]
    valid_mask = (pair_in != -1)
    valid_pair_in, valid_pair_out = pair_in[valid_mask].long(), pair_out[valid_mask].long()
    xyz_next = scatter_mean(xyz[valid_pair_in], index=valid_pair_out, dim=0)
    batch_next = scatter_mean(batch.float()[valid_pair_in], index=valid_pair_out, dim=0)
    return xyz_next, batch_next
