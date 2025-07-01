import torch
import torch.nn as nn
from src.modules.basicConv import Conv1d
from src.projectPN.utils import get_stride_idx_cuda, get_sample_idx, get_neighbor_copy, gather_torch, \
    grouping, check_valid, get_neighbor_att
import torch.nn.functional as F
from src.config_proj import I2PNetConfig as cfg_default


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, bn=False, activation_fn=True,
                 leaky_relu=True, use_bn_input=True):
        """
        FC implement
        """
        super(Conv2d, self).__init__()
        if stride is None:
            stride = [1, 1]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = bn
        self.activation_fn = activation_fn
        self.use_bn_input = use_bn_input

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        if bn:
            self.bn_linear = nn.BatchNorm2d(out_channels, track_running_stats=not use_bn_input)
        if activation_fn:
            self.relu = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # x (b,n,s,c)

        x = x.permute(0, 3, 2, 1)  # (b,c,s,n)
        outputs = self.conv(x)
        if self.bn:
            outputs = self.bn_linear(outputs)

        if self.activation_fn:
            outputs = self.relu(outputs)

        outputs = outputs.permute(0, 3, 2, 1)  # (b,n,s,c)
        return outputs

    def set_bn(self):
        if self.bn:
            self.bn_linear.track_running_stats = not self.use_bn_input
            self.bn_linear.training = True


class ProjectPointNet(nn.Module):
    def __init__(self, H, W, out_h, out_w, stride_H, stride_W, kernel_size, nsample, distance, in_channel, mlp,
                 use_trans=False,
                 use_bn_p=True, use_bn_input=True):
        super(ProjectPointNet, self).__init__()
        self.H = H
        self.W = W
        self.out_h = out_h
        self.out_w = out_w
        self.stride_H = stride_H
        self.stride_W = stride_W
        self.kernel_size = kernel_size
        self.distance = distance
        self.nsample = nsample
        self.usetrans = use_trans
        self.mlp_convs = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(
                Conv2d(last_channel, out_channel, kernel_size=(1, 1), bn=use_bn_p, leaky_relu=False,
                       use_bn_input=use_bn_input))
            last_channel = out_channel

    def forward(self, xyz_proj_raw, xyz_proj, feature_proj, sample_idx=None,cfg=cfg_default, raw_feat_point=False):
        """
        Input:
            xyz_proj_raw: B,H,W,3
            xyz_proj: B,H,W,3
            feature_proj: B,H,W,C
            sample_idx: List[B,outh,outw,2]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz_proj.shape[0]
        device = feature_proj.device
        with torch.no_grad():
            if sample_idx is None:
                sample_idx = get_sample_idx(B, self.out_h, self.out_w, self.stride_H, self.stride_W, device)
            idx_n2 = get_stride_idx_cuda(B, self.out_h, self.out_w, self.stride_H, self.stride_W, device)
        new_xyz_proj = gather_torch(xyz_proj, *sample_idx, B, self.H, self.W)
        new_xyz_proj_raw = gather_torch(xyz_proj_raw, *sample_idx, B, self.H, self.W)
        with torch.no_grad():
            xyz_pr = xyz_proj if self.usetrans else xyz_proj_raw

            grouped_idx = get_neighbor_copy(xyz_pr, xyz_pr, idx_n2, self.kernel_size, self.nsample,
                                            distance=self.distance)
        grouped_points = gather_torch(feature_proj, *grouped_idx[:3], B, self.H, self.W)  # B,N,K,C

        if raw_feat_point:
            grouped_xyz = gather_torch(xyz_proj_raw, *grouped_idx[:3], B, self.H, self.W)  # B,N,K,3
            grouped_xyz_norm = grouped_xyz - new_xyz_proj_raw.view(B, self.out_h * self.out_w, 1, 3)
        else:
            grouped_xyz = gather_torch(xyz_proj, *grouped_idx[:3], B, self.H, self.W)  # B,N,K,3
            grouped_xyz_norm = grouped_xyz - new_xyz_proj.view(B, self.out_h * self.out_w, 1, 3)

        if cfg.debug and cfg.debug_count < cfg.debug_storage:
            with torch.no_grad():
                xyz_pr = xyz_proj if self.usetrans else xyz_proj_raw
                grouped_idx = get_neighbor_att(xyz_pr, xyz_pr, idx_n2, self.kernel_size, self.nsample,
                                               distance=self.distance)
                valid_mask = grouped_idx[-1]  # B,N,K,1
                storage = cfg.debug_dict["global_valid_neighbor_num_downsample"]
                storage[0][storage[1]] = (storage[0][storage[1]] * cfg.debug_count + valid_mask.sum(-2).mean().item()) / \
                                         (cfg.debug_count + 1)
                storage[1] = (storage[1] + 1) % len(storage[0])

        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], -1)


        # point net layer
        for i, conv in enumerate(self.mlp_convs):
            new_points = conv(new_points)

        new_points = torch.max(new_points, dim=2)[0].view(B, self.out_h, self.out_w, -1)

        return new_xyz_proj_raw, new_xyz_proj, new_points, grouped_xyz, sample_idx

    def forward_center(self, xyz_proj_raw, xyz_proj, feature_proj, sample_idx=None,cfg=cfg_default, using_intens=False, raw_feat_point=False):
        """
        Input:
            xyz_proj_raw: B,H,W,3
            xyz_proj: B,H,W,3
            feature_proj: B,H,W,C
            sample_idx: List[B,outh,outw,2]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz_proj.shape[0]
        device = feature_proj.device
        with torch.no_grad():
            if sample_idx is None:
                sample_idx = get_sample_idx(B, self.out_h, self.out_w, self.stride_H, self.stride_W, device)
            idx_n2 = get_stride_idx_cuda(B, self.out_h, self.out_w, self.stride_H, self.stride_W, device)
        new_xyz_proj = gather_torch(xyz_proj, *sample_idx, B, self.H, self.W)
        new_xyz_proj_raw = gather_torch(xyz_proj_raw, *sample_idx, B, self.H, self.W)
        with torch.no_grad():
            xyz_pr = xyz_proj if self.usetrans else xyz_proj_raw

            grouped_idx = get_neighbor_copy(xyz_pr, xyz_pr, idx_n2, self.kernel_size, self.nsample,
                                            distance=self.distance)
        grouped_points = gather_torch(feature_proj, *grouped_idx[:3], B, self.H, self.W)  # B,N,K,C

        if raw_feat_point:
            grouped_xyz = gather_torch(xyz_proj_raw, *grouped_idx[:3], B, self.H, self.W)  # B,N,K,3
            grouped_xyz_norm = grouped_xyz - new_xyz_proj_raw.view(B, self.out_h * self.out_w, 1, 3)
        else:
            grouped_xyz = gather_torch(xyz_proj, *grouped_idx[:3], B, self.H, self.W)  # B,N,K,3
            grouped_xyz_norm = grouped_xyz - new_xyz_proj.view(B, self.out_h * self.out_w, 1, 3)

        if cfg.debug and cfg.debug_count < cfg.debug_storage:
            with torch.no_grad():
                xyz_pr = xyz_proj if self.usetrans else xyz_proj_raw
                grouped_idx = get_neighbor_att(xyz_pr, xyz_pr, idx_n2, self.kernel_size, self.nsample,
                                               distance=self.distance)
                valid_mask = grouped_idx[-1]  # B,N,K,1
                storage = cfg.debug_dict["global_valid_neighbor_num_downsample"]
                storage[0][storage[1]] = (storage[0][storage[1]] * cfg.debug_count + valid_mask.sum(-2).mean().item()) / \
                                         (cfg.debug_count + 1)
                storage[1] = (storage[1] + 1) % len(storage[0])

        ## adding center point: 
        center_points = new_xyz_proj.view(B, self.out_h * self.out_w, 1, 3).repeat(1,1,grouped_xyz_norm.shape[2],1)
        ## adding e-dis
        dist = torch.norm(grouped_xyz_norm, p=2, dim=3).unsqueeze(3)

        if using_intens:
            new_points = torch.cat(
                [grouped_xyz_norm, center_points, grouped_xyz, dist, grouped_points], -1)
        else:
            new_points = torch.cat(
                [grouped_xyz_norm, center_points, grouped_xyz, dist], -1)

        # point net layer
        for i, conv in enumerate(self.mlp_convs):
            new_points = conv(new_points)

        new_points = torch.max(new_points, dim=2)[0].view(B, self.out_h, self.out_w, -1)

        return new_xyz_proj_raw, new_xyz_proj, new_points, grouped_xyz, sample_idx

    def set_bn(self):
        for conv in self.mlp_convs:
            conv.set_bn()


class ProjSetUpconvModule(nn.Module):
    def __init__(self, H, W, out_h, out_w, stride_H, stride_W, kernel_size, nsample, distance, in_channels, mlp,
                 mlp2, use_trans=False, use_bn_p=True, use_bn_input=True):
        super(ProjSetUpconvModule, self).__init__()
        self.nsample = nsample
        self.last_channel = in_channels[-1] + 3
        self.mlp = mlp
        self.mlp2 = mlp2
        self.H = H
        self.W = W
        self.out_h = out_h
        self.out_w = out_w
        self.stride_H = stride_H
        self.stride_W = stride_W
        self.kernel_size = kernel_size
        self.distance = distance
        self.use_trans = use_trans

        self.mlp_conv = nn.ModuleList()
        self.mlp2_conv = nn.ModuleList()

        if mlp is not None:
            for i, num_out_channel in enumerate(mlp):
                self.mlp_conv.append(Conv2d(self.last_channel, num_out_channel, [1, 1], stride=[1, 1], bn=use_bn_p,
                                            use_bn_input=use_bn_input))
                self.last_channel = num_out_channel

        # if len(mlp) is not 0:
        if len(mlp) > 0:
            self.last_channel = mlp[-1] + in_channels[0]
        else:
            self.last_channel = self.last_channel + in_channels[0]
        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.mlp2_conv.append(
                    Conv2d(self.last_channel, num_out_channel, [1, 1], stride=[1, 1], bn=use_bn_p,
                           use_bn_input=use_bn_input))
                self.last_channel = num_out_channel

    def forward(self, xyz1_raw, xyz2_raw, xyz1, xyz2, idx_n2, feat1, feat2,cfg=cfg_default, raw_feat_point=False):
        '''
            Input:
                xyz1: (B,out_h,out_w,3)
                xyz2: (B,H,W,3)
                feat1: (batch_size, npoint1,c1) features for xyz1 points (earlier layers, more points)
                feat2: (batch_size, npoint2, c2) features for xyz2 points
                idx_n2: (B,out_h*out_w,2)
            Return:
                (batch_size, npoint1, mlp[-1] or mlp2[-1] or channel1+3)
            '''
        # xyz2_grouped, _, feat2_grouped, idx = grouping(feat2, self.nsample, xyz2,
        #                                                xyz1)  # (batch_size,npoint1,nsample,3) _ (batch_size,npoint1,nsample,c2)
        device = xyz1.device
        B = xyz1.shape[0]
        with torch.no_grad():
            xyz1_pr = xyz1 if self.use_trans else xyz1_raw
            xyz2_pr = xyz2 if self.use_trans else xyz2_raw
            # idx_n2 = get_stride_idx_cuda(B, self.out_h, self.out_w, 1, 1, device)
            grouped_idx = get_neighbor_copy(xyz1_pr, xyz2_pr, idx_n2, self.kernel_size, self.nsample, self.stride_H,
                                            self.stride_W, distance=self.distance)

        if cfg.debug and cfg.debug_count < cfg.debug_storage:
            with torch.no_grad():
                grouped_idx = get_neighbor_att(xyz1_pr, xyz2_pr, idx_n2, self.kernel_size, self.nsample, self.stride_H,
                                               self.stride_W, distance=self.distance)
                valid_mask = grouped_idx[-1]  # B,N,K,1
                storage = cfg.debug_dict["global_valid_neighbor_num_upsample"]
                storage[0][storage[1]] = (storage[0][storage[1]] * cfg.debug_count + valid_mask.sum(-2).mean().item()) / \
                                         (cfg.debug_count + 1)
                storage[1] = (storage[1] + 1) % len(storage[0])

        if raw_feat_point:
            grouped_xyz_proj = gather_torch(xyz2_raw, *grouped_idx[:3], B, self.H, self.W)
            xyz_diff = grouped_xyz_proj - xyz1_raw.view(B, self.out_h * self.out_w, 1, 3)
        else:
            grouped_xyz_proj = gather_torch(xyz2, *grouped_idx[:3], B, self.H, self.W)
            xyz_diff = grouped_xyz_proj - xyz1.view(B, self.out_h * self.out_w, 1, 3)

        grouped_feat2 = gather_torch(feat2, *grouped_idx[:3], B, self.H, self.W)

        upfeats = torch.cat([grouped_feat2, xyz_diff], dim=3)  # B,N,K,3+C

        if self.mlp is not None:
            for i, conv in enumerate(self.mlp_conv):
                upfeats = conv(upfeats)

        feat1_new = torch.max(upfeats, dim=2, keepdim=False)[0].view(B, self.out_h, self.out_w, -1)

        if feat1 is not None:
            feat1_new = torch.cat([feat1_new, feat1], dim=3)

        if self.mlp2 is not None:
            for i, conv in enumerate(self.mlp2_conv):
                feat1_new = conv(feat1_new)
        return feat1_new.reshape(B, self.out_h * self.out_w, -1)

    def set_bn(self):
        for conv in self.mlp_conv:
            conv.set_bn()
        for conv in self.mlp2_conv:
            conv.set_bn()


class CostVolume(nn.Module):
    def __init__(self, H, W, kernel_size, distance, nsample, nsample_q, rgb_in_channels, lidar_in_channels, mlp1, mlp2,
                 backward_validation=False, use_trans=False, use_bn_p=True, use_bn_input=True):
        super(CostVolume, self).__init__()
        self.H = H
        self.W = W
        self.nsample = nsample
        self.nsample_q = nsample_q
        self.distance = distance
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.kernel_size = kernel_size
        self.backward_validation = backward_validation
        self.use_trans = use_trans

        corr_channel = rgb_in_channels

        if backward_validation:
            corr_channel += lidar_in_channels

        self.in_channels = corr_channel + 6

        self.mlp1_convs = nn.ModuleList()

        self.mlp2_convs = nn.ModuleList()

        self.mlp2_convs_2 = nn.ModuleList()

        for i, num_out_channel in enumerate(mlp1):
            self.mlp1_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=use_bn_p,
                                          use_bn_input=use_bn_input))
            self.in_channels = num_out_channel

        self.pi_encoding = Conv2d(6, mlp1[-1], [1, 1], stride=[1, 1], bn=use_bn_p, use_bn_input=use_bn_input)

        self.in_channels = 2 * mlp1[-1]
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=use_bn_p,
                                          use_bn_input=use_bn_input))
            self.in_channels = num_out_channel

        self.pc_encoding = Conv2d(10, mlp1[-1], [1, 1], stride=[1, 1], bn=use_bn_p, use_bn_input=use_bn_input)

        self.in_channels = 2 * mlp1[-1] + lidar_in_channels
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs_2.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=use_bn_p,
                                            use_bn_input=use_bn_input))
            self.in_channels = num_out_channel

    def forward(self, xyz_proj_raw, warped_xyz, warped_points, idx_n2, f2_xyz, f2_points, lidar_z,cfg=cfg_default):
        """
            Input:
                warped_xyz: (B,H*W,3)
                warped_points: (B,H*W,C)
                f2_xyz:  (B,N,C)
                f2_points: (B,N,C)
                lidar_z: (B, H*W, 1)

            Output:
                pc_feat1_new: B,H,W,mlp2[-1]
        """
        B = warped_xyz.shape[0]

        if self.nsample_q > 0:
            qi_xyz_grouped, _, qi_points_grouped, idx = grouping(f2_points, self.nsample_q, f2_xyz,
                                                                 warped_xyz)
        else:
            # B,N1,N2,C
            qi_xyz_grouped, qi_points_grouped = f2_xyz.unsqueeze(1).repeat(1, warped_xyz.shape[1], 1, 1), \
                                                f2_points.unsqueeze(1).repeat(1, warped_xyz.shape[1], 1, 1)
        if self.backward_validation and cfg.debug_time:
            cfg.debug_timing.time("cv1_gather")
        # important
        warped_xyz = warped_xyz.mul(lidar_z)  # restore depth

        # lidar_z_repeat = torch.unsqueeze(lidar_z, dim=2).repeat(1, 1, self.nsample_q, 1)
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
        if self.backward_validation and cfg.debug_time:
            cfg.debug_timing.time("cv1_encode")
        if self.backward_validation:
            # B,N1,N2,C
            repeat_image_feature = qi_points_grouped
            # B,N1,N2,C
            repeat_lidar_feature = pi_points_expanded
            # correlation
            repeat_correlation = repeat_image_feature * repeat_lidar_feature  # B,N,M,1

            valid_mask = check_valid(warped_xyz).unsqueeze(-1)  # B,N,1,1

            masked_corr = repeat_correlation * valid_mask + -1e10 * (1 - valid_mask)

            image_max_respond = torch.max(masked_corr, 1, keepdim=True)[0]. \
                repeat(1, warped_xyz.shape[1], 1, 1)
            pi_feat1_new = torch.cat([pi_feat1_new, image_max_respond], dim=-1)
        if self.backward_validation and cfg.debug_time:
            cfg.debug_timing.time("cv1_bv")
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
        if self.backward_validation and cfg.debug_time:
            cfg.debug_timing.time("cv1_cg")
            # print(WQ.shape)
        # 3d find 3d grouped features to be weighted

        warped_xyz_bhw = warped_xyz.view(B, self.H, self.W, 3)

        with torch.no_grad():
            xyz_pr = warped_xyz_bhw if self.use_trans else xyz_proj_raw
            grouped_idx = get_neighbor_att(xyz_pr, xyz_pr, idx_n2, self.kernel_size, self.nsample,
                                           distance=self.distance)

        if cfg.debug and cfg.debug_count < cfg.debug_storage:
            with torch.no_grad():
                valid_mask = grouped_idx[-1]  # B,N,K,1
                storage = cfg.debug_dict["global_valid_neighbor_num_cv"]
                storage[0][storage[1]] = (storage[0][storage[1]] * cfg.debug_count + valid_mask.sum(-2).mean().item()) / \
                                         (cfg.debug_count + 1)
                storage[1] = (storage[1] + 1) % len(storage[0])

        pc_xyz_grouped = gather_torch(warped_xyz_bhw, *grouped_idx[:3], B, self.H, self.W)

        pc_points_grouped = gather_torch(pi_feat1_new, *grouped_idx[:3], B, self.H, self.W)

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

        # valid mask
        valid_mask = grouped_idx[-1]
        pc_concat = pc_concat * valid_mask + -1e10 * (1 - valid_mask)

        WP = F.softmax(pc_concat, dim=2)

        pc_feat1_new = WP * pc_points_grouped  # B,N,K, mlp2[-1]

        pc_feat1_new = torch.sum(pc_feat1_new, dim=2, keepdim=False)  # B,N, mlp2[-1]
        if self.backward_validation and cfg.debug_time:
            cfg.debug_timing.time("cv1_lst")
        return pc_feat1_new.view(B, self.H, self.W, -1)

    def set_bn(self):
        for conv in self.mlp2_convs:
            conv.set_bn()
        for conv in self.mlp1_convs:
            conv.set_bn()
        for conv in self.mlp2_convs_2:
            conv.set_bn()
        self.pc_encoding.set_bn()
        self.pi_encoding.set_bn()


class PoseHead(nn.Module):
    def __init__(self, in_channels, mlp1, mlp2,
                 hidden, q_dim, t_dim, dropout_rate=0.5
                 , split_dp=False,
                 pos_embed=False, sigmoid=False, maxhead=False):
        """
        take the self attention mask and predictions
        do global attention (global constraint)
        """
        super(PoseHead, self).__init__()
        self.sigmoid = sigmoid
        self.maxhead = maxhead
        in_channel, l_feature_channel = in_channels
        # take 3+3 as input and get
        self.pos_embed = pos_embed
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

        result = global_prediction

        hidden_feature = self.DP1(self.hidden_layer(result))

        q = self.quat_head(self.DP2(hidden_feature)).squeeze(1)
        t = self.trans_head(self.DP2(hidden_feature)).squeeze(1)

        # normalize q
        q = q / (torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-10) + 1e-10)

        return q, t, mask_p


class FlowPredictor(nn.Module):

    def __init__(self, in_channels, mlp, is_training, bn_decay, bn=True, use_bn_input=True):
        super(FlowPredictor, self).__init__()
        self.in_channels = in_channels
        self.mlp = mlp
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.mlp_conv = nn.ModuleList()

        for i, num_out_channel in enumerate(mlp):
            self.mlp_conv.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=bn, use_bn_input=use_bn_input))
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

    def set_bn(self):
        for conv in self.mlp_conv:
            conv.set_bn()
