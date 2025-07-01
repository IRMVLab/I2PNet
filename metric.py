import numpy as np
from scipy.spatial.transform import Rotation
import math
import torch
import src.utils as utils
from src.util.lie_metric.MSEE import SE3_to_se3, cal_metric


def quat_to_rotmat_batch(q):
    """
    Args:
        q: [B,4]
    Returns:
        rotmat: [B,3,3]
    """
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

    rotmat = np.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], axis=-1)

    return rotmat.reshape(-1, 3, 3)


def mult_extrinsic_batch(m1, m2):
    """
    Args:
        m1: [B,3,4]
        m2: [B,3,4]
    Returns:
        mult: [B,3,4]
    """
    B = m1.shape[0]
    padding = np.array([0, 0, 0, 1]).reshape(1, 1, 4).repeat(B, axis=0)
    m1 = np.concatenate([m1, padding], axis=-2)  # B,4,4
    m2 = np.concatenate([m2, padding], axis=-2)  # B,4,4
    mult = m1 @ m2
    return mult[:, :3, :]  # B,3,4


def inv_extrinsic(m):
    B = m.shape[0]
    padding = np.array([0, 0, 0, 1]).reshape(1, 1, 4).repeat(B, axis=0)
    m1 = np.concatenate([m, padding], axis=-2)  # B,4,4
    return np.linalg.inv(m1)[:, :3, :]


def rotmat_to_euler(rotmat, out='rad'):
    """
    Args:
        rotmat: [B,3,3]
        out: "rad" or "deg"
    """
    sy = np.sqrt(rotmat[:, 0, 0] * rotmat[:, 0, 0] + rotmat[:, 1, 0] * rotmat[:, 1, 0])
    singular = sy < 1e-6  # B
    roll = np.empty_like(sy)
    pitch = np.empty_like(sy)
    yaw = np.empty_like(sy)

    nonsingular = np.logical_not(singular)

    roll[nonsingular] = np.arctan2(rotmat[nonsingular, 2, 1], rotmat[nonsingular, 2, 2])
    pitch[nonsingular] = np.arctan2(-rotmat[nonsingular, 2, 0], sy[nonsingular])
    yaw[nonsingular] = np.arctan2(rotmat[nonsingular, 1, 0], rotmat[nonsingular, 0, 0])

    roll[singular] = np.arctan2(-rotmat[singular, 1, 2], rotmat[singular, 1, 1])
    pitch[singular] = np.arctan2(-rotmat[singular, 2, 0], sy[singular])
    yaw[singular] = 0

    if out == 'rad':
        return np.stack([roll, pitch, yaw], -1)
    elif out == 'deg':
        return np.stack([roll, pitch, yaw], -1) * 180 / np.pi


def calibration_error_batch(e1, e2):
    # new setting E_{pred}^{-1} Pr^{-1}<=>E_{gt}
    inv_e1 = inv_extrinsic(e1)
    err_e = mult_extrinsic_batch(inv_e1, e2)
    err_rotmat = err_e[:, :3, :3]
    err_trans = err_e[:, :3, 3]
    err_euler = rotmat_to_euler(err_rotmat, out='deg')

    err_euler = np.abs(err_euler)
    err_trans = np.abs(err_trans)

    return err_euler[:, 0], err_euler[:, 1], err_euler[:, 2], \
           err_trans[:, 0], err_trans[:, 1], err_trans[:, 2]


def getExtrinsic(out3, data_valid,out_raw=False):
    pred_decalib_quat_real = out3[:, :4].cpu().detach().numpy()
    pred_decalib_quat_dual = out3[:, 4:].cpu().detach().numpy().reshape(-1, 3, 1)

    gt_decalib_quat_real = data_valid['decalib_real_gt'].numpy()
    gt_decalib_quat_dual = data_valid['decalib_dual_gt'].numpy().reshape(-1, 3, 1)
    init_extrinsic = data_valid['init_extrinsic']

    pred_decalib_rot = quat_to_rotmat_batch(pred_decalib_quat_real)  # [B,3,3]
    # [B,3,4]
    pred_decalib_extrinsic = np.concatenate([pred_decalib_rot, pred_decalib_quat_dual], axis=-1)
    pred_extrinsic = mult_extrinsic_batch(pred_decalib_extrinsic, init_extrinsic)  # [B,3,4]

    gt_decalib_rot = quat_to_rotmat_batch(gt_decalib_quat_real)
    gt_decalib_extrinsic = np.concatenate([gt_decalib_rot, gt_decalib_quat_dual], axis=-1)
    gt_extrinsic = mult_extrinsic_batch(gt_decalib_extrinsic, init_extrinsic)

    if out_raw:
        return pred_extrinsic, gt_extrinsic,pred_decalib_extrinsic,gt_decalib_extrinsic
    else:
        return pred_extrinsic, gt_extrinsic

def cal_rete_once(out3, data_valid):
    with torch.no_grad():
        pred_decalib_quat_real = out3[:, :4].cpu().detach().numpy()
        pred_decalib_quat_dual = out3[:, 4:].cpu().detach().numpy().reshape(-1, 3, 1)

        gt_decalib_quat_real = data_valid['decalib_real_gt'].numpy()
        gt_decalib_quat_dual = data_valid['decalib_dual_gt'].numpy().reshape(-1, 3, 1)

        pred_decalib_rot = quat_to_rotmat_batch(pred_decalib_quat_real)  # [B,3,3]
        # [B,3,4]
        pred_raw = np.concatenate([pred_decalib_rot, pred_decalib_quat_dual], axis=-1)

        gt_decalib_rot = quat_to_rotmat_batch(gt_decalib_quat_real)
        gt_raw = np.concatenate([gt_decalib_rot, gt_decalib_quat_dual], axis=-1)

        P_diff = mult_extrinsic_batch(inv_extrinsic(pred_raw), gt_raw)
        t_diff = np.linalg.norm(P_diff[:, :3, 3], 2, -1)

        r_diff = P_diff[:, :3, :3]
        R_diff = Rotation.from_matrix(r_diff)
        angles_diff = np.sum(np.abs(R_diff.as_euler('xzy', degrees=True)), -1)

    return angles_diff.mean(),t_diff.mean()


def getExtrinsicRt(R, t, data_valid):
    gt_decalib_quat_real = data_valid['decalib_real_gt'].numpy()
    gt_decalib_quat_dual = data_valid['decalib_dual_gt'].numpy().reshape(-1, 3, 1)
    init_extrinsic = data_valid['init_extrinsic']

    # [B,3,4]
    pred_decalib_extrinsic = np.concatenate([R.detach().cpu().numpy(),
                                             t.unsqueeze(-1).detach().cpu().numpy()], axis=-1)
    pred_extrinsic = mult_extrinsic_batch(pred_decalib_extrinsic, init_extrinsic)  # [B,3,4]

    gt_decalib_rot = quat_to_rotmat_batch(gt_decalib_quat_real)
    gt_decalib_extrinsic = np.concatenate([gt_decalib_rot, gt_decalib_quat_dual], axis=-1)
    gt_extrinsic = mult_extrinsic_batch(gt_decalib_extrinsic, init_extrinsic)

    return pred_extrinsic, gt_extrinsic


def eval_acc(pm, intrinsic, img_size, decalib_q, decalib_t, sigmoid=False):
    if len(pm) == 2:
        l_gt = utils.get_projection_gt(pm[1], intrinsic, img_size, decalib_q, decalib_t)
    else:
        l_gt = pm[2].argmax(-1).long()
    pred = pm[0].argmax(-1) if not sigmoid else torch.ge(pm[0].squeeze(-1), 0.5).long()

    N = pred.shape[1]

    err = torch.sum(torch.abs(pred - l_gt), -1)
    acc = (1 - err.float() / N).cpu().numpy()

    return acc


def eval_msee(out3, gt_se3):
    # TODO: MRR
    # pred [B,3,4] gt_se3 [B,6]
    pred_decalib_quat_real = out3[:, :4].cpu().detach().numpy()
    pred_decalib_quat_dual = out3[:, 4:].cpu().detach().numpy().reshape(-1, 3, 1)

    pred_decalib_rot = quat_to_rotmat_batch(pred_decalib_quat_real)  # [B,3,3]
    # [B,3,4]
    pred = np.concatenate([pred_decalib_rot, pred_decalib_quat_dual], axis=-1)

    pred_se3 = np.stack([SE3_to_se3(np.concatenate([pred[i], np.array([[0, 0, 0, 1]])], axis=0))
                         for i in range(pred.shape[0])], axis=0)  # B,6
    msee = cal_metric(pred_se3, gt_se3)

    return msee


def eval_mrr(msee, gt_se3):
    se3_noise = cal_metric(np.zeros_like(np.array(gt_se3)), gt_se3)
    rr = 1 - msee / se3_noise
    return rr


class RteRreEval(object):
    def __init__(self, threshold=False, rre_th=10., rte_th=5.):
        self.t_diff = []
        self.r_diff = []
        self.t_diff_all = []
        self.r_diff_all = []
        self.threshold = threshold
        self.rre_th = rre_th
        self.rte_th = rte_th
        self.acc_count = 0
        self.all_count = 0

    def reset(self):
        self.t_diff.clear()
        self.r_diff.clear()
        self.acc_count = 0
        self.all_count = 0

    def get_recall(self):
        return self.acc_count / self.all_count

    def addBatch(self, pred_extrinsic, gt_extrinsic):
        """
        Args:
            pred_extrinsic: [B,3,4]
            gt_extrinsic: [B,3,4]
        Returns:
        """
        # change inv(pred)@gt to pred@inv(gt) for gt:(Pc) init: (Pr@Pc)=>pred@Pr
        # P_diff = mult_extrinsic_batch(pred_extrinsic, inv_extrinsic(gt_extrinsic))
        P_diff = mult_extrinsic_batch(inv_extrinsic(pred_extrinsic), gt_extrinsic)
        t_diff = np.linalg.norm(P_diff[:, :3, 3], 2, -1)

        r_diff = P_diff[:, :3, :3]
        R_diff = Rotation.from_matrix(r_diff)
        angles_diff = np.sum(np.abs(R_diff.as_euler('xzy', degrees=True)), -1)

        self.all_count += len(angles_diff)
        if self.threshold:
            mask = np.logical_and(t_diff < self.rte_th, angles_diff < self.rre_th)
            masked_t_diff = t_diff[mask]
            masked_angles_diff = angles_diff[mask]
            self.acc_count += len(masked_t_diff)
            self.t_diff.extend(list(masked_t_diff))
            self.r_diff.extend(list(masked_angles_diff))

        else:
            self.acc_count += len(angles_diff)
            self.t_diff.extend(list(t_diff))
            self.r_diff.extend(list(angles_diff))

        self.t_diff_all.extend(list(t_diff))
        self.r_diff_all.extend(list(angles_diff))

        return list(angles_diff), list(t_diff)

    def evalSeq(self):
        t_diff = np.array(self.t_diff)
        r_diff = np.array(self.r_diff)
        rte_sigma = math.sqrt(np.var(t_diff))
        rre_sigma = math.sqrt(np.var(r_diff))
        rte_mean = t_diff.mean()
        rre_mean = r_diff.mean()

        return rte_mean, rte_sigma, rre_mean, rre_sigma

    def save_metric(self, path):
        np.savez(path, RRE=np.array(self.r_diff_all),
                 RTE=np.array(self.t_diff_all))


def quatmultiply(q, r, device='cpu'):
    """
    Batch quaternion multiplication
    Args:
        q (torch.Tensor/np.ndarray): shape=[Nx4]
        r (torch.Tensor/np.ndarray): shape=[Nx4]
        device (str): 'cuda' or 'cpu'

    Returns:
        torch.Tensor: shape=[Nx4]
    """
    if isinstance(q, torch.Tensor):
        t = torch.zeros(q.shape[0], 4, device=device)
    elif isinstance(q, np.ndarray):
        #print(q.shape[0])
        t = np.zeros((q.shape[0], 4))
    else:
        raise TypeError("Type not supported")
    t[:, 0] = r[:, 0] * q[:, 0] - r[:, 1] * q[:, 1] - r[:, 2] * q[:, 2] - r[:, 3] * q[:, 3]
    t[:, 1] = r[:, 0] * q[:, 1] + r[:, 1] * q[:, 0] - r[:, 2] * q[:, 3] + r[:, 3] * q[:, 2]
    t[:, 2] = r[:, 0] * q[:, 2] + r[:, 1] * q[:, 3] + r[:, 2] * q[:, 0] - r[:, 3] * q[:, 1]
    t[:, 3] = r[:, 0] * q[:, 3] - r[:, 1] * q[:, 2] + r[:, 2] * q[:, 1] + r[:, 3] * q[:, 0]
    return t

def quatinv(q):
    """
    Batch quaternion inversion
    Args:
        q (torch.Tensor/np.ndarray): shape=[Nx4]

    Returns:
        torch.Tensor/np.ndarray: shape=[Nx4]
    """
    if isinstance(q, torch.Tensor):
        t = q.clone()
    elif isinstance(q, np.ndarray):
        t = q.copy()
    else:
        raise TypeError("Type not supported")
    t *= -1
    t[:, 0] *= -1
    return t

def quaternion_distance(q, r):
    """
    Batch quaternion distances, used as loss
    Args:
        q (torch.Tensor): shape=[Nx4]
        r (torch.Tensor): shape=[Nx4]
        device (str): 'cuda' or 'cpu'

    Returns:
        torch.Tensor: shape=[N]
    """
    t = quatmultiply(q, quatinv(r))
    return 2 * np.arctan2(np.linalg.norm(t[:, 1:], axis=1), np.abs(t[:, 0]))
        

