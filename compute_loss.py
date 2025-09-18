import torch
import numpy as np
import torch.nn.functional as F
import src.utils as utils
from src.config import I2PNetConfig as cfg_default


def focalloss(pred, gt,cfg=cfg_default):
    # pred [N,C] gt [N]
    N, C = pred.shape
    softmax_p = F.softmax(pred, dim=-1)  # N,2
    onehot = F.one_hot(gt, C)  # N,2 one hot
    target_p = (softmax_p * onehot).sum(-1)  # N
    # -(1-p_y)^gamma*log p_y
    ce_loss = F.cross_entropy(pred, gt, reduction='none')  # B*N
    fl = torch.pow(1 - target_p, cfg.focal_gamma) * ce_loss
    return fl.mean()


def GetProjectionLoss(pm, intrinsic, img_size, decalib_q, decalib_t,cfg=cfg_default):
    # img_size (H,W)
    if pm is None:
        return None
    if len(pm) == 2:
        with torch.no_grad():
            l_gt = utils.get_projection_gt(pm[1], intrinsic, img_size, decalib_q, decalib_t)
    else:
        with torch.no_grad():
            l_gt = pm[2].argmax(-1).long().detach()
    if cfg.focal_mask_loss:
        criterion = focalloss
    else:
        criterion = F.cross_entropy
    if cfg.mask_sigmoid:
        loss_p = F.binary_cross_entropy(pm[0].reshape(-1), l_gt.float().view(-1))
    else:
        loss_p = criterion(pm[0].reshape(-1, 2), l_gt.view(-1))
    return loss_p


def qt2Ebatch(q, t):
    B = q.shape[0]
    ####
    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]
    r00 = 1 - 2 * qy ** 2 - 2 * qz ** 2
    r01 = 2 * qx * qy - 2 * qz * qw
    r02 = 2 * qx * qz + 2 * qy * qw

    r10 = 2 * qx * qy + 2 * qz * qw
    r11 = 1 - 2 * qx ** 2 - 2 * qz ** 2
    r12 = 2 * qy * qz - 2 * qx * qw

    r20 = 2 * qx * qz - 2 * qy * qw
    r21 = 2 * qy * qz + 2 * qx * qw
    r22 = 1 - 2 * qx ** 2 - 2 * qy ** 2

    R = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
    R = R.view(B, 3, 3)
    E = torch.cat([R, t.view(B, 3, 1)], dim=-1)

    return E


def GetPointwiseReProjectionLoss(p, intrinsic, img_size, out3, out4, decalib_q, decalib_t):
    """
    Args:
        p: [B,N,3]
        intrinsic: [B,3,3]
        img_size: [H,W]
        out3: [B,7]
        out4: [B,7]
        decalib_q: [B,4]
        decalib_t: [B,3]
    """
    p_projected = utils.get_projection_gt(p, intrinsic, img_size, decalib_q, decalib_t).float()
    E_gt = qt2Ebatch(decalib_q, decalib_t)
    E_est3 = qt2Ebatch(out3[:, :4], out3[:, 4:])
    E_est4 = qt2Ebatch(out4[:, :4], out4[:, 4:])
    B, N = p.shape[:2]
    p_pad = torch.cat([p, torch.ones((B, N, 1), device=out3.device)], dim=-1).permute(0, 2, 1)
    intrinsic = intrinsic.float()

    def project(E):
        uv = torch.bmm(intrinsic, torch.bmm(E, p_pad)).transpose(1, 2)
        uv = uv[:, :, :2] / (uv[:, :, 2:] + 1e-10)

        return uv

    p_gt = project(E_gt)
    p_est3 = project(E_est3)
    p_est4 = project(E_est4)

    batch_num = torch.sum(p_projected) + 1e-10
    # B,N,3
    loss3 = (F.l1_loss(p_est3, p_gt, reduction='none').sum(-1) * p_projected).sum() / batch_num
    loss4 = (F.l1_loss(p_est4, p_gt, reduction='none').sum(-1) * p_projected).sum() / batch_num
    return 1.6 * loss3 + 0.8 * loss4


def Get_loss(out3, out4, qq_gt, t_gt, w_x, w_q,cfg=cfg_default):
    l2_q = out3[:, :4]
    l2_t = out3[:, 4:]
    l3_q = out4[:, :4]
    l3_t = out4[:, 4:]

    l2_q_norm = l2_q  # already be normalized
    l2_loss_q = torch.mean(
        torch.sqrt(torch.sum((qq_gt - l2_q_norm) * (qq_gt - l2_q_norm), dim=-1, keepdim=True) + 1e-10))

    if cfg.l1_trans_loss:
        l2_loss_x = F.l1_loss(l2_t, t_gt)  # B,3
    else:
        l2_loss_x = torch.mean(torch.sqrt(torch.sum((l2_t - t_gt) * (l2_t - t_gt), dim=-1, keepdim=True) + 1e-10))
        # l2_loss_x = torch.mean(torch.sqrt((l2_t - t_gt) * (l2_t - t_gt) + 1e-10))
    l2_loss = l2_loss_x * torch.exp(-w_x) + w_x + l2_loss_q * torch.exp(-w_q) + w_q
    l3_q_norm = l3_q  # already be normalized
    l3_loss_q = torch.mean(
        torch.sqrt(torch.sum((qq_gt - l3_q_norm) * (qq_gt - l3_q_norm), dim=-1, keepdim=True) + 1e-10))
    if cfg.l1_trans_loss:
        l3_loss_x = F.l1_loss(l3_t, t_gt)
    else:
        l3_loss_x = torch.mean(torch.sqrt(torch.sum((l3_t - t_gt) * (l3_t - t_gt), dim=-1, keepdim=True) + 1e-10))
        # l3_loss_x = torch.mean(torch.sqrt((l3_t - t_gt) * (l3_t - t_gt) + 1e-10))
    l3_loss = l3_loss_x * torch.exp(-w_x) + w_x + l3_loss_q * torch.exp(-w_q) + w_q
    loss_sum = 1.6 * l3_loss + 0.8 * l2_loss

    real_loss = 1.6 * l3_loss_q + 0.8 * l2_loss_q
    dual_loss = 1.6 * l3_loss_x + 0.8 * l2_loss_x

    # loss_sum = 0.8 * l2_loss + 0.4 * l1_loss + 0.2 * l0_loss
    return loss_sum, real_loss, dual_loss


def quat2R(decalib_q):
    B = decalib_q.shape[0]
    qw = decalib_q[:, 0]
    qx = decalib_q[:, 1]
    qy = decalib_q[:, 2]
    qz = decalib_q[:, 3]
    r00 = 1 - 2 * qy ** 2 - 2 * qz ** 2
    r01 = 2 * qx * qy - 2 * qz * qw
    r02 = 2 * qx * qz + 2 * qy * qw

    r10 = 2 * qx * qy + 2 * qz * qw
    r11 = 1 - 2 * qx ** 2 - 2 * qz ** 2
    r12 = 2 * qy * qz - 2 * qx * qw

    r20 = 2 * qx * qz - 2 * qy * qw
    r21 = 2 * qy * qz + 2 * qx * qw
    r22 = 1 - 2 * qx ** 2 - 2 * qy ** 2

    R = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
    return R.view(B, 3, 3)


def Rt_loss(gt_R, pred_R):
    B = gt_R.shape[0]
    I = torch.eye(3,device=pred_R.device).unsqueeze(0).repeat(B, 1, 1)
    err_R = torch.bmm(gt_R.transpose(1, 2), pred_R) - I

    return torch.square(err_R).view(B, -1).sum(-1).mean()


def Get_loss_Rt(gt_q, gt_t, R3, t3, R4, t4, sq, sx):
    gt_R = quat2R(gt_q)

    real_loss = 1.6 * Rt_loss(gt_R, R4) + 0.8 * Rt_loss(gt_R, R3)
    dual_loss = 1.6 * F.l1_loss(gt_t, t4) + 0.8 * F.l1_loss(gt_t, t3)

    loss = torch.exp(-sq) * real_loss + sq + torch.exp(-sx) * dual_loss + sx

    return real_loss,dual_loss,loss