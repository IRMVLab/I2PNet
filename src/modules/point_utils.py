from pointnet2 import pointnet2_utils
import torch


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


def mask_knn_point(nsample, xyz, new_xyz, mask):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    mask = mask[:, None].repeat(1, new_xyz.shape[1], 1)  # B,S,N
    sqrdists = square_distance(new_xyz, xyz)
    sqrdists = sqrdists * mask + 1e10 * (1 - mask)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def mask_grouping(feature, K, src_xyz, q_xyz, mask, use_xyz=False):
    '''
    Input:
        feature: (batch_size, ndataset, c)
        K: neighbor size
        src_xyz: original point xyz (batch_size, ndataset, 3)
        q_xyz: query point xyz (batch_size, npoint, 3)
        mask: B,N
    Return:
        grouped_xyz: (batch_size, npoint, K,3)
        xyz_diff: (batch_size, npoint,K, 3)
        new_points: (batch_size, npoint,K, c+3) if use_xyz else (batch_size, npoint,K, c)
        point_indices: (batch_size, npoint, K)
    '''
    q_xyz = q_xyz.contiguous()
    src_xyz = src_xyz.contiguous()

    # TODO: cuda fasten knn
    point_indices = mask_knn_point(K, src_xyz, q_xyz, mask)  # (batch_size, npoint, K)

    grouped_xyz = index_points_group(src_xyz, point_indices)  # (batch_size, npoint, K,3)
    # print(grouped_xyz.device,q_xyz.device )
    xyz_diff = grouped_xyz - (q_xyz.unsqueeze(2)).repeat(1, 1, K, 1)  # (batch_size, npoint,K, 3)
    # x' - x : KNN points - centroids
    grouped_feature = index_points_group(feature, point_indices)  # (batch_size, npoint, K,c)
    if use_xyz:
        new_points = torch.cat([xyz_diff, grouped_feature], dim=-1)  # (batch_size, npoint,K, c+3)
    else:
        new_points = grouped_feature  # (batch_size, npoint, K,c)

    return grouped_xyz, xyz_diff, new_points, point_indices


def grouping(feature, K, src_xyz, q_xyz, use_xyz=False, raw_feat_point=False, raw_xyz1=None, raw_xyz2=None):
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


    # TODO: cuda fasten knn
    point_indices = knn_point(K, src_xyz, q_xyz)  # (batch_size, npoint, K)

    grouped_xyz = index_points_group(src_xyz, point_indices)  # (batch_size, npoint, K,3)
    if raw_feat_point:
        # print("src_xyz", src_xyz.shape)
        # print("raw_xyz1", raw_xyz1.shape)
        # print("q_xyz", q_xyz.shape)
        # print("raw_xyz2", raw_xyz2.shape)
        # raw_xyz1 = raw_xyz1.contiguous()
        grouped_raw_xyz = index_points_group(raw_xyz1, point_indices)

    # print(grouped_xyz.device,q_xyz.device )
    if raw_feat_point:
        xyz_diff = grouped_raw_xyz - (raw_xyz2.unsqueeze(2)).repeat(1, 1, K, 1)
    else:
        xyz_diff = grouped_xyz - (q_xyz.unsqueeze(2)).repeat(1, 1, K, 1)  # (batch_size, npoint,K, 3)
    # x' - x : KNN points - centroids
    grouped_feature = index_points_group(feature, point_indices)  # (batch_size, npoint, K,c)
    if use_xyz:
        new_points = torch.cat([xyz_diff, grouped_feature], dim=-1)  # (batch_size, npoint,K, c+3)
    else:
        new_points = grouped_feature  # (batch_size, npoint, K,c)
    if raw_feat_point:
        return grouped_xyz, xyz_diff, new_points, point_indices, grouped_raw_xyz
    else:   
        return grouped_xyz, xyz_diff, new_points, point_indices, None


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
    S = new_xyz.shape[1]
    # B, S, C = new_xyz.shape
    # print("S:", S)
    group_idx = torch.arange(N, dtype=torch.long).to(
        device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


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
