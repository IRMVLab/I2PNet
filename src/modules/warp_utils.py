import torch

def conj_quat(q):
    output = q
    helper = torch.Tensor([1, -1, -1, -1]).cuda()
    output = output.mul(helper)
    output = output.reshape(-1)
    return output

def inv_q(q):
    """
    q: [B,1,4] or [B,4]
    """
    B = q.shape[0]
    q = q.reshape(B,4)  # [B,4]
    q_2 = torch.sum(q * q, dim=-1, keepdim=True) + 1e-10
    # this will create a new tensor
    q0 = torch.index_select(q, 1, torch.LongTensor([0]).cuda())
    q_ijk = -torch.index_select(q, 1, torch.LongTensor([1, 2, 3]).cuda())
    q_ = torch.cat([q0, q_ijk], dim=-1)  # conj
    q_inv = q_ / q_2
    return q_inv # [B,4]


def mul_q(q_a, q_b):
    """
    quat multiply
    Args:
        q_a: [B,1 or N,4] could be [0,x,y,z] or any quat
        q_b: [B,1 or N,4] could be [0,x,y,z] or any quat
    Returns:
        q_result: [B,N,4]
    """
    if q_a.ndim == 2: # [B,4]
        q_a = q_a.unsqueeze(1) # [B,1,4]

    if q_b.ndim == 2:
        q_b = q_b.unsqueeze(1)

    # [B,N]
    q_result_0 = q_a[:, :, 0] * q_b[:, :, 0] - q_a[:, :, 1] * q_b[:, :, 1] - \
                 q_a[:, :, 2] * q_b[:, :, 2] - q_a[:, :, 3] * q_b[:, :, 3]

    q_result_1 = q_a[:, :, 0] * q_b[:, :, 1] + q_a[:, :, 1] * q_b[:, :, 0] + \
                 q_a[:, :, 2] * q_b[:, :, 3] - q_a[:, :, 3] * q_b[:, :, 2]

    q_result_2 = q_a[:, :, 0] * q_b[:, :, 2] - q_a[:, :, 1] * q_b[:, :, 3] + \
                 q_a[:, :, 2] * q_b[:, :, 0] + q_a[:, :, 3] * q_b[:, :, 1]

    q_result_3 = q_a[:, :, 0] * q_b[:, :, 3] + q_a[:, :, 1] * q_b[:, :, 2] - \
                 q_a[:, :, 2] * q_b[:, :, 1] + q_a[:, :, 3] * q_b[:, :, 0]

    q_result = torch.stack([q_result_0, q_result_1, q_result_2, q_result_3], dim=-1)

    return q_result  # B,N,4


def warp_quat(lidar_xyz, Hi_quat, H_trans, cam_intrinsic, img_shape, LF):
    lidar_xyz = lidar_xyz.permute(0,2,1) # B,N,3
    B, N, _ = lidar_xyz.shape  # B,N,3

    device = lidar_xyz.device

    H_trans = H_trans.reshape(B, 1, 4)

    c_0 = torch.zeros(B, N, 1).to(device)
    homo_xyz = torch.cat([c_0, lidar_xyz], -1)  # [0,xyz]
    Hi_quat_inv = inv_q(Hi_quat)  # B,4

    homo_xyz = mul_q(Hi_quat, homo_xyz)
    homo_xyz = mul_q(homo_xyz, Hi_quat_inv) + H_trans

    homo_xyz = homo_xyz[:, :, 1:4]
    lidar_z = homo_xyz[:, :, 2:]
    lidar_xyz = homo_xyz / (lidar_z + 1e-10)  # normalized position
    return lidar_xyz, lidar_z, LF

def warp_quat_xyz(lidar_xyz, Hi_quat, H_trans):
    B, N, _ = lidar_xyz.shape  # B,N,3

    device = lidar_xyz.device

    H_trans = H_trans.reshape(B, 1, 4)

    c_0 = torch.zeros(B, N, 1).to(device)
    homo_xyz = torch.cat([c_0, lidar_xyz], -1)  # [0,xyz]
    Hi_quat_inv = inv_q(Hi_quat)  # B,4

    homo_xyz = mul_q(Hi_quat, homo_xyz)
    homo_xyz = mul_q(homo_xyz, Hi_quat_inv) + H_trans

    homo_xyz = homo_xyz[:, :, 1:4]

    return homo_xyz


def warp_quat_EFGH(lidar_xyz, Hi_quat, H_trans, calib, LF):
    """

    Args:
        lidar_xyz:
        (fit Tr_inv@Tji@Tr in j frame lidar coordination)
        Hi_quat:
        H_trans:
        calib: calib P_2@Tr [B,4,4]
        LF:

    process calib@(q@p@q'+t)[1:]
    """
    lidar_xyz = lidar_xyz.permute(0,2,1) # B,N,3
    B, N, _ = lidar_xyz.shape  # B,N,3

    device = lidar_xyz.device

    H_trans = H_trans.reshape(B, 1, 4)

    c_0 = torch.zeros(B, N, 1).to(device)
    homo_xyz = torch.cat([c_0, lidar_xyz], -1)  # [0,xyz]
    Hi_quat_inv = inv_q(Hi_quat)  # B,4

    homo_xyz = mul_q(Hi_quat, homo_xyz)
    homo_xyz = mul_q(homo_xyz, Hi_quat_inv) + H_trans

    homo_xyz = homo_xyz[:, :, 1:4]

    homo_xyz = torch.cat([homo_xyz,torch.ones(B,N,1,device=device)],dim=-1)
    homo_xyz = torch.bmm(calib,homo_xyz.transpose(1,2)).transpose(1,2)

    lidar_z = homo_xyz[:, :, 2:]
    lidar_xyz = homo_xyz / (lidar_z + 1e-10)  # normalized position
    return lidar_xyz, lidar_z, LF

def projection_initial_EFGH(lidar_xyz, calib, LF):

    lidar_xyz = lidar_xyz.permute(0, 2, 1)
    B, N, _ = lidar_xyz.shape  # B, N, 3
    lidar_xyz = lidar_xyz.reshape(B, N, 3)

    lidar_xyz = torch.cat([lidar_xyz,torch.ones(B,N,1,
                                               device=lidar_xyz.device)],dim=-1)
    lidar_xyz = torch.bmm(calib,lidar_xyz.transpose(1,2)).transpose(1,2)

    lidar_z = torch.unsqueeze(lidar_xyz[:, :, 2], dim=2)
    lidar_xyz = lidar_xyz / lidar_xyz[:, :, 2, None]

    return lidar_xyz, lidar_z, LF

def projection_initial(lidar_xyz, Hi, cam_intrinsic, img_shape, LF):
    lidar_xyz = lidar_xyz.permute(0, 2, 1)
    B, N, _ = lidar_xyz.shape  # B, N, 3
    lidar_xyz = lidar_xyz.reshape(B, N, 3)

    lidar_z = torch.unsqueeze(lidar_xyz[:, :, 2], dim=2)
    lidar_xyz = lidar_xyz / lidar_xyz[:, :, 2, None]

    return lidar_xyz, lidar_z, LF
