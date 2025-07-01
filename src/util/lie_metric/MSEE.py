from geomstats.special_euclidean_group import SpecialEuclideanGroup
import geomstats.lie_group as lie_group
import numpy as np
import src.util.lie_metric.liegroups as liegroups
import time


def SE3_to_se3(SE3_matrix):
    # This liegroups lib represent se3 with first 3 element as translation, which is different than us
    se3_rot_last = liegroups.SE3.log(liegroups.SE3.from_matrix(SE3_matrix, normalize=True))
    se3 = np.zeros_like(se3_rot_last)
    se3[:3] = se3_rot_last[3:]
    se3[3:] = se3_rot_last[:3]
    return se3


def cal_metric(pred, gt):
    """
    pred: [N,6]
    gt: [N,6]
    """
    group = SpecialEuclideanGroup(3, epsilon=np.finfo(np.float32).eps)  # SE3
    metric = group.left_canonical_metric
    se3_error = lie_group.loss(pred, gt, group, metric)

    return se3_error


if __name__ == '__main__':
    def euler_to_rotmat(roll, pitch, yaw):
        rot_x = [[1, 0, 0],
                 [0, np.cos(roll), -np.sin(roll)],
                 [0, np.sin(roll), np.cos(roll)]]
        rot_x = np.asarray(rot_x)

        rot_y = [[np.cos(pitch), 0, np.sin(pitch)],
                 [0, 1, 0],
                 [-np.sin(pitch), 0, np.cos(pitch)]]
        rot_y = np.asarray(rot_y)

        rot_z = [[np.cos(yaw), -np.sin(yaw), 0],
                 [np.sin(yaw), np.cos(yaw), 0],
                 [0, 0, 1]]
        rot_z = np.asarray(rot_z)

        return rot_z @ rot_y @ rot_x


    pred = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)
    err_r = 15 * np.pi / 180.
    pred[:3, :3] = euler_to_rotmat(err_r, err_r, err_r)

    gt = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=np.float32)
    err_r = 15 * np.pi / 180.
    gt[:3, :3] = euler_to_rotmat(err_r, err_r, err_r)

    pred[:3, :4] = np.array(
        '0.999991599 0.004115324 0.000166180 3.741627481 -0.000066176 0.056396619 -0.998408417 -0.293070543 -0.004118170 0.998400045 0.056396417 -8.945714455'.split(
            ' ')).reshape(3, 4)

    gt[:3, :4] = np.array('0.999855358 0.015505301 -0.006985343 -0.133210091 -0.007872897 0.057939322 -0.998289078 -0.449764716 -0.015074048 0.998199657 0.058053011 -9.358668589'.split(' ')).reshape(3, 4)
    print(np.abs(pred[:3,3]-gt[:3,3]).mean())
    se3_pred = SE3_to_se3(pred)[None].repeat(1, 0)
    se3_gt = SE3_to_se3(gt)[None].repeat(1, 0)
    t1 = time.time()
    print(cal_metric(se3_pred, se3_gt))
    # print((time.time()-t1)*1000)
