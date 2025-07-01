import torch
from src.projectPN.fused_conv_select.fused_conv_select_k import fused_conv_select_k, FLAG_COPY, FLAG_SHIFT
import numpy as np
from pointnet2.pointnet2_utils import grouping_operation
from src.config_proj import I2PNetConfig as cfg
import matplotlib.pyplot as plt

def get_idx_cuda(B, H, W, device):
    H_idx = torch.reshape(torch.arange(0, H, device=device, dtype=torch.int),
                          [1, -1, 1, 1]).repeat(B, 1,
                                                W, 1)
    W_idx = torch.reshape(torch.arange(0, W, device=device, dtype=torch.int),
                          [1, 1, -1, 1]).repeat(B, H,
                                                1, 1)
    return torch.reshape(torch.cat([H_idx, W_idx], dim=-1), [B, -1, 2])


def get_sample_idx(batch, out_h, out_w, stride_H, stride_W, device):
    h_idx = torch.arange(0, out_h * stride_H, stride_H, device=device, dtype=torch.int)
    w_idx = torch.arange(0, out_w * stride_W, stride_W, device=device, dtype=torch.int)
    selected_h_idx = (torch.reshape(h_idx, (1, -1, 1))).repeat(batch, 1, out_w).long()
    selected_w_idx = (torch.reshape(w_idx, (1, 1, -1))).repeat(batch, out_h, 1).long()
    selected_b_idx = torch.reshape(torch.arange(batch, device=device), (-1, 1, 1)) \
        .repeat(1, out_h, out_w).long()
    return selected_b_idx, selected_h_idx, selected_w_idx


def get_stride_idx_cuda(B, out_h, out_w, stride_h, stride_w, device):
    H_idx = torch.reshape(torch.arange(0, out_h * stride_h, stride_h, device=device, dtype=torch.int),
                          [1, -1, 1, 1]).repeat(B, 1, out_w, 1)
    W_idx = torch.reshape(torch.arange(0, out_w * stride_w, stride_w, device=device, dtype=torch.int),
                          [1, 1, -1, 1]).repeat(B, out_h, 1, 1)
    return torch.reshape(torch.cat([H_idx, W_idx], dim=-1), [B, -1, 2])


def gather_torch(feature, neigh_b_idx, neigh_h_idx, neigh_w_idx, batch, height, width):
    '''
    Args:
        feature: B,H,W,C
        neigh_*_idx: list[B,H',W']
        batch: batch of feature
        height: origin map size H
        width: origin map size W
    Returns:
        f_neighbors [B,H',W',C]
    '''
    nei_h, nei_w = neigh_h_idx.shape[1:3]
    neighbor_idx = neigh_h_idx * width + neigh_w_idx
    index_input = neighbor_idx.reshape(batch, -1)  # B,N'
    feature_proj = torch.reshape(feature, (batch, height, width, -1))

    H, W, C = feature_proj.shape[1:4]
    gather_feature_proj = feature_proj.reshape(batch, H * W, C)  # B,N,C

    gather_feature_proj = torch.gather(gather_feature_proj, 1, index_input.unsqueeze(-1).repeat(1, 1, C))

    gather_feature_proj = gather_feature_proj.permute(0, 2, 1)  # B,C,N'
    gather_feature_proj = gather_feature_proj.reshape(batch, C, nei_h, nei_w)  # B,C,H',W'
    f_neighbors = gather_feature_proj.permute(0, 2, 3, 1)  # B,H',W',C
    return f_neighbors


def get_neighbor_copy(xyz1_proj, xyz2_proj, idx_n2, kernel_shape, knn_points, stride_h=1, stride_w=1, distance=10):
    """for each point in `xyz1`, find nearest points in `xyz2`

    Args:
        xyz1_proj ([B, H, W, 3]): [pointclouds which search nearest points]
        xyz2_proj ([B, H, W, 3]): [pointclouds to be searched]
        idx_n2 ([B, N, 2]): [by which index each point in xyz1]
        kernel_shape ([int, int]): [the scope of search for each point]
        knn_points ([int]): [the number of points to be searched in `xyz2`]

    Returns:
        [((B, N,K, 3), (B, N, K, 1))]: [the nearest points indices in `xyz2`, valid mask to erase those invalid points]
    """
    batch, height, width, _ = xyz1_proj.shape
    small_h = xyz2_proj.shape[1]  # small == target
    small_w = xyz2_proj.shape[2]

    kernel_total_p = kernel_shape[0] * kernel_shape[1]

    n_points = idx_n2.shape[1]

    random_hw = (torch.arange(0, kernel_total_p, device=xyz1_proj.device, dtype=torch.int))  # kernel idx

    select_b_idx = torch.zeros(batch, n_points, knn_points, 1, device=xyz1_proj.device,
                               dtype=torch.long)  # B,N,K,1
    select_h_idx = torch.zeros(batch, n_points, knn_points, 1, device=xyz1_proj.device, dtype=torch.long)
    select_w_idx = torch.zeros(batch, n_points, knn_points, 1, device=xyz1_proj.device, dtype=torch.long)

    valid_idx = torch.zeros(batch, n_points, kernel_total_p, 1, device=xyz1_proj.device, dtype=torch.float)
    valid_in_dis_idx = torch.zeros(batch, n_points, kernel_total_p, 1, device=xyz1_proj.device, dtype=torch.float)
    # select_mask will become the valid_mask to return
    select_mask = torch.zeros(batch, n_points, knn_points, 1, device=xyz1_proj.device, dtype=torch.float)

    flag = FLAG_SHIFT | FLAG_COPY  # flag to control the behavior of fused_conv
    # valid_mask is just the selected mask
    selected_b_idx, selected_h_idx, selected_w_idx, valid_idx, valid_idx_in_dis, valid_mask = fused_conv_select_k(
        xyz1_proj, xyz2_proj, idx_n2, random_hw, height, width, n_points,
        kernel_shape[0], kernel_shape[1], knn_points, flag, distance, stride_h, stride_w,
        select_b_idx, select_h_idx, select_w_idx, valid_idx, valid_in_dis_idx, select_mask, small_h, small_w
    )
    return selected_b_idx.squeeze(-1), selected_h_idx.squeeze(-1), selected_w_idx.squeeze(-1), valid_mask


def check_valid(xyz):

    return torch.any(torch.ne(xyz, 0), dim=-1, keepdim=True).float()


def project_seq(xyz, features, H, W,use_rank=True,fup=2.0,fdown=-24.8):
    """
    project raw point cloud into spherical ring
    Args:
        PC: B,N,3
        features:List[B,N,D]

    Returns:
        PC_project_final: B,H,W,3 channels are (X,Y,Z)
        Feature_project_final: B,H,W,D the features are just the corresponding features
    """
    B = xyz.shape[0]
    # D = feature.shape[-1]
    #print("xyz shape:", xyz.shape)
    # print("xyz height:", -torch.min(xyz[:,:,2]), -torch.max(xyz[:,:,2]))
    device = xyz.device
    degree2radian = np.pi / 180
    AzimuthResolution = 360.0 / W  # degree
    VerticalViewDown = fdown
    VerticalViewUp = fup

    # specifications of Velodyne-64
    AzimuthResolution = AzimuthResolution * degree2radian  # the original resolution is 0.18
    VerticalViewDown = VerticalViewDown * degree2radian
    VerticalViewUp = VerticalViewUp * degree2radian
    VerticalResolution = (VerticalViewUp - VerticalViewDown) / (H - 1)
    VerticalPixelsOffset = -VerticalViewDown / VerticalResolution

    PI = np.pi
    with torch.no_grad():
        x = xyz[:, :, 0]
        y = xyz[:, :, 1]
        z = xyz[:, :, 2]
        r = torch.norm(xyz, p=2, dim=2)

        # alpha
        iCol = (((PI - torch.atan2(y, x)) / AzimuthResolution).long())

        # beta
        beta = torch.asin(z / r)

        iRow = H - ((beta / VerticalResolution + VerticalPixelsOffset).long())  # top -> down

        iRow = torch.clamp(iRow, min=0, max=H - 1).long()
        iCol = torch.clamp(iCol, min=0, max=W - 1).long()

        # just project the min distance
        if use_rank:
            rank = torch.argsort(r, dim=1, descending=True).long()  # [B,N]
            iRow = torch.gather(iRow, 1, rank)
            iCol = torch.gather(iCol, 1, rank)
    if use_rank:
        xyz = torch.gather(xyz, 1, rank[:, :, None].repeat(1, 1, 3))
        #r_new = torch.norm(xyz, p=2, dim=2)
        reordered_features = [torch.gather(feature, 1, rank[:, :, None].repeat(1, 1, feature.shape[-1])) for feature in features]
    else:
        reordered_features = features
        #r_new = r
    xyz_proj = torch.zeros((B, H, W, 3), dtype=torch.float32, device=device)
    feature_projs = [torch.zeros((B, H, W, feat.shape[-1]), dtype=torch.float32, device=device) for feat in features]

    proj_img = np.zeros((H, W))
    # print(H, W)
    for i in range(B):
        xyz_proj[i, iRow[i], iCol[i]] = xyz[i]
        for j in range(len(features)):
            feature_projs[j][i, iRow[i], iCol[i]] = reordered_features[j][i]
        # if i==0:
        #     # visualize
        #     print("max min:", torch.min(iRow[i]), torch.max(iRow[i]))
        #     depth_vis = ((r_new[i]-torch.min(r[i]))/(torch.max(r_new[i])-torch.min(r_new[i]))*255).detach().cpu().numpy()
        #     plt.figure()
        #     plt.imshow(proj_img)
        #     plt.scatter(iCol[i].detach().cpu().numpy(), iRow[i].detach().cpu().numpy(), c = depth_vis, cmap='jet', alpha=1, s=0.5)
        #     plt.savefig("/data/debug2/4s_proj/proj_img_" + str(i) + ".jpg", dpi=600)
    #print(what)
    return xyz_proj, feature_projs


def project(xyz, feature, H, W,fup=2.0,fdown=-24.8):
    """
    project raw point cloud into spherical ring
    Args:
        PC: B,N,3
        features:B,N,D

    Returns:
        PC_project_final: B,H,W,3 channels are (X,Y,Z)
        Feature_project_final: B,H,W,D the features are just the corresponding features
    """
    B = xyz.shape[0]
    D = feature.shape[-1]

    device = xyz.device
    degree2radian = np.pi / 180
    AzimuthResolution = 360.0 / W  # degree
    VerticalViewDown = fdown
    VerticalViewUp = fup

    # specifications of Velodyne-64
    AzimuthResolution = AzimuthResolution * degree2radian  # the original resolution is 0.18
    VerticalViewDown = VerticalViewDown * degree2radian
    VerticalViewUp = VerticalViewUp * degree2radian
    VerticalResolution = (VerticalViewUp - VerticalViewDown) / (H - 1)
    VerticalPixelsOffset = -VerticalViewDown / VerticalResolution

    PI = np.pi
    with torch.no_grad():
        x = xyz[:, :, 0]
        y = xyz[:, :, 1]
        z = xyz[:, :, 2]
        r = torch.norm(xyz, p=2, dim=2)

        # alpha
        iCol = (((PI - torch.atan2(y, x)) / AzimuthResolution).long())

        # beta
        beta = torch.asin(z / r)

        iRow = H - ((beta / VerticalResolution + VerticalPixelsOffset).long())  # top -> down

        iRow = torch.clamp(iRow, min=0, max=H - 1).long()
        iCol = torch.clamp(iCol, min=0, max=W - 1).long()

        # just project the min distance
        rank = torch.argsort(r, dim=1, descending=True).long()  # [B,N]
        iRow = torch.gather(iRow, 1, rank)
        iCol = torch.gather(iCol, 1, rank)

    xyz = torch.gather(xyz, 1, rank[:, :, None].repeat(1, 1, 3))

    feature = torch.gather(feature, 1, rank[:, :, None].repeat(1, 1, feature.shape[-1]))

    xyz_proj = torch.zeros((B, H, W, 3), dtype=torch.float32, device=device)
    feature_proj = torch.zeros((B, H, W, D), dtype=torch.float32, device=device)

    for i in range(B):
        xyz_proj[i, iRow[i], iCol[i]] = xyz[i]
        feature_proj[i, iRow[i], iCol[i]] = feature[i]

    return xyz_proj, feature_proj

def get_neighbor_att(xyz1_proj, xyz2_proj, idx_n2, kernel_shape, knn_points, stride_h=1, stride_w=1, distance=10):
    """for each point in `xyz1`, find nearest points in `xyz2`

    Args:
        xyz1_proj ([B, H, W, 3]): [pointclouds which search nearest points]
        xyz2_proj ([B, H, W, 3]): [pointclouds to be searched]
        idx_n2 ([B, N, 2]): [by which index each point in xyz1]
        kernel_shape ([int, int]): [the scope of search for each point]
        knn_points ([int]): [the number of points to be searched in `xyz2`]

    Returns:
        [((B, N,K, 3), (B, N, K, 1))]: [the nearest points indices in `xyz2`, valid mask to erase those invalid points]
    """
    batch, height, width, _ = xyz1_proj.shape
    small_h = xyz2_proj.shape[1]  # small == target
    small_w = xyz2_proj.shape[2]

    kernel_total_p = kernel_shape[0] * kernel_shape[1]

    n_points = idx_n2.shape[1]

    random_hw = (torch.arange(0, kernel_total_p, device=xyz1_proj.device, dtype=torch.int))  # kernel idx

    select_b_idx = torch.zeros(batch, n_points, knn_points, 1, device=xyz1_proj.device,
                               dtype=torch.long)  # B,N,K,1
    select_h_idx = torch.zeros(batch, n_points, knn_points, 1, device=xyz1_proj.device, dtype=torch.long)
    select_w_idx = torch.zeros(batch, n_points, knn_points, 1, device=xyz1_proj.device, dtype=torch.long)

    valid_idx = torch.zeros(batch, n_points, kernel_total_p, 1, device=xyz1_proj.device, dtype=torch.float)
    valid_in_dis_idx = torch.zeros(batch, n_points, kernel_total_p, 1, device=xyz1_proj.device, dtype=torch.float)
    # select_mask will become the valid_mask to return
    select_mask = torch.zeros(batch, n_points, knn_points, 1, device=xyz1_proj.device, dtype=torch.float)

    flag = FLAG_SHIFT  # flag to control the behavior of fused_conv
    # valid_mask is just the selected mask
    selected_b_idx, selected_h_idx, selected_w_idx, valid_idx, valid_idx_in_dis, valid_mask = fused_conv_select_k(
        xyz1_proj, xyz2_proj, idx_n2, random_hw, height, width, n_points,
        kernel_shape[0], kernel_shape[1], knn_points, flag, distance, stride_h, stride_w,
        select_b_idx, select_h_idx, select_w_idx, valid_idx, valid_in_dis_idx, select_mask, small_h, small_w
    )
    return selected_b_idx.squeeze(-1), selected_h_idx.squeeze(-1), selected_w_idx.squeeze(-1), valid_mask



def get_pixel_posinfo(rf,K):
    B,H,W,_ = rf.shape
    device = rf.device
    idx_n2 = get_idx_cuda(B,H,W,device).float()
    idx_n2 = torch.cat([idx_n2,
                        torch.ones(B,H*W,1,dtype=torch.float32,device=device)],dim=-1)
    # cuda not support matrix inverse
    intrinsic_inv = torch.inverse(K)  # B,3,3
    RF_index = torch.bmm(intrinsic_inv, idx_n2.permute(0, 2, 1))  # B,3,3 @ B,3,N
    RF_index = RF_index.permute(0, 2, 1).reshape(B,H,W,3)  # [B,418,3]

    return RF_index




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
    q_xyz = q_xyz.contiguous()
    src_xyz = src_xyz.contiguous()

    point_indices = knn_point(K, src_xyz, q_xyz)  # (batch_size, npoint, K)

    grouped_xyz = index_points_group(src_xyz, point_indices)  # (batch_size, npoint, K,3)

    xyz_diff = grouped_xyz - (q_xyz.unsqueeze(2)).repeat(1, 1, K, 1)  # (batch_size, npoint,K, 3)

    grouped_feature = index_points_group(feature, point_indices)  # (batch_size, npoint, K,c)
    if use_xyz:
        new_points = torch.cat([xyz_diff, grouped_feature], dim=-1)  # (batch_size, npoint,K, c+3)
    else:
        new_points = grouped_feature  # (batch_size, npoint, K,c)

    return grouped_xyz, xyz_diff, new_points, point_indices


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
    B = src.shape[0]
    N = src.shape[1]
    M = dst.shape[1]
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)  # BxNx1
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)  # Bx1xM
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

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points