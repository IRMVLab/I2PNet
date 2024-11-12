import torch
import torch.nn as nn
import numpy as np
import random

class WeightedPnPHead(nn.Module):
    '''
    Input:
        src: [B,N,3]
        src_corres: [B,N,2]
        weights: [B,N]
    Output:
        r: [B,3,3]
        t: [B,3]
    '''

    def __init__(self):
        super(WeightedPnPHead, self).__init__()

    def forward(self, src, src_corres, weights):
        # https://www.jianshu.com/p/a35fa8ac0803
        B, N, _ = src.shape
        homo_src = torch.cat([src, torch.ones(B, N, 1, device=src.device)], dim=-1)  # B,N,4
        xyz_uv = (homo_src.unsqueeze(-2) * (-src_corres.unsqueeze(-1))).reshape(B, 2 * N, 4)
        xyz_pre = torch.cat([torch.zeros(B, N, 4, device=src.device), homo_src], dim=-1).view(B, N, 8)
        xyz_sub = torch.cat([homo_src, torch.zeros(B, N, 4, device=src.device)], dim=-1).view(B, N, 8)
        xyz_pair = torch.cat([xyz_sub, xyz_pre], dim=-1).reshape(B, 2 * N, 8)
        A = torch.cat([xyz_pair, xyz_uv], dim=-1)  # B,N*2,12
        weighted_A = (A.view(B, N, 2, 12) * weights[:, :, None, None]).view(B, N * 2, 12)

        _, _, V = torch.svd(weighted_A)  # solve v 12*12

        tilde_f = V[:, :, -1]  # https://blog.csdn.net/qq_34213260/article/details/120572372

        # tilde_f without scale

        F = tilde_f.view(B, 3, 4)  # T without scale and constraint R \in SO(3)

        u, s, v = torch.svd(F[:, :3, :3])


        beta = 1 / (s.mean(-1) + 1e-10)

        # R = torch.bmm(u, v.transpose(1, 2))
        R = torch.bmm(torch.bmm(u,torch.diag_embed(s*beta.unsqueeze(-1))),v.transpose(1,2))

        # validation
        best_p = homo_src.gather(1,torch.argmax(weights,-1,keepdim=True).
                                 unsqueeze(-1).repeat(1,1,4))
        z = beta[:,None]*(best_p*F[:, 2:]).sum(-1) # B,1
        sign = 2 * torch.ge(z.squeeze(-1), 0).float() - 1

        beta = sign * beta
        R = sign[:, None, None] * R

        t = beta[:, None] * F[:, :, 3]

        return R, t

def angles2rotation_matrix(angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R
def generate_random_transform(P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
                              P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude):
    """

    :param pc_np: pc in NWU coordinate
    :return:
    """
    t = [random.uniform(-P_tx_amplitude, P_tx_amplitude),
         random.uniform(-P_ty_amplitude, P_ty_amplitude),
         random.uniform(-P_tz_amplitude, P_tz_amplitude)]
    angles = [random.uniform(-P_Rx_amplitude, P_Rx_amplitude),
              random.uniform(-P_Ry_amplitude, P_Ry_amplitude),
              random.uniform(-P_Rz_amplitude, P_Rz_amplitude)]

    rotation_mat = angles2rotation_matrix(angles)
    P_random = np.identity(4, dtype=np.float32)
    P_random[0:3, 0:3] = rotation_mat
    P_random[0:3, 3] = t

    return P_random


if __name__ == '__main__':
    N = 1000
    T = generate_random_transform(10,10,10,np.pi,np.pi,np.pi)[:3,:]
    print(T)
    p = np.random.randn(N, 3)
    homo = np.concatenate([p, np.ones((N, 1))], axis=1)
    uv = (T @ (homo.T)).T
    uv_xy = uv[:, :2] / uv[:, 2:]
    weights = torch.from_numpy(uv[:,2:]>0).view(1,N).float()
    # print(uv[:,2:],weights)
    print(torch.sum(weights).item())
    print()
    p_t = torch.from_numpy(p).view(1, N, 3).repeat(2,1,1)
    uv = torch.from_numpy(uv_xy).view(1, N, 2).repeat(2,1,1)

    svd = WeightedPnPHead()
    print(svd(p_t, uv, weights.repeat(2,1)))
