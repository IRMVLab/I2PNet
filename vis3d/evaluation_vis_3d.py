import torch
import os
from tqdm import tqdm
import cv2
import numpy as np
import sys

sys.path.append("..")

# from src.kitti_odometry_corr_snr import Kitti_Odometry_Dataset as testdataset
from src.kitti_odometry import read_calib
# from src.dataset_params import  as cfg

from p3d_render import PointCloudRender, PointCloudCameraRender
import open3d

from pytorch3d.renderer import (
    look_at_view_transform,
)


def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == (4,), "Not a valid quaternion"
    if not np.isclose(np.linalg.norm(q), 1.):
        q = q / np.linalg.norm(q)
    mat = np.zeros((3, 3), np.float32)
    mat[0, 0] = 1 - 2 * q[2] ** 2 - 2 * q[3] ** 2
    mat[0, 1] = 2 * q[1] * q[2] - 2 * q[3] * q[0]
    mat[0, 2] = 2 * q[1] * q[3] + 2 * q[2] * q[0]
    mat[1, 0] = 2 * q[1] * q[2] + 2 * q[3] * q[0]
    mat[1, 1] = 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2
    mat[1, 2] = 2 * q[2] * q[3] - 2 * q[1] * q[0]
    mat[2, 0] = 2 * q[1] * q[3] - 2 * q[2] * q[0]
    mat[2, 1] = 2 * q[2] * q[3] + 2 * q[1] * q[0]
    mat[2, 2] = 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2

    return mat


class Evaluator(object):
    def __init__(self):

        pred_path = "./demo.txt"
        with open(pred_path, "r") as f:
            self.lines = f.readlines()
        sections, last = self.calculate_sections(self.lines)

        tag = last

        self.num, self.start, self.pose_t = sections[tag]

        seq = 0
        with open(os.path.join("/dataset", 'kitti_processed_CMRNet', 'poses', f'kitti-{seq:02d}.csv')) as f:
            lines = f.readlines()[1:]
            self.poses = [line.strip('\n').split(',')[1:] for line in lines]  # timestamp,x,y,z,qx,qy,qz,qw
        map_path = os.path.join("/dataset", 'kitti_processed_CMRNet',
                                'sequences', '00', 'map',
                                f'map-{seq:02d}_0.1_0-{len(self.poses)}.pcd')

        print("Load Map...")
        dpath = os.path.join("/dataset", 'kitti_processed_CMRNet',
                             'sequences', '00', 'map',
                             f'map-{seq:02d}_0.1_0-{len(self.poses)}_downsampled_0.7.pcd')
        if os.path.exists(dpath):
            pcd_down = open3d.io.read_point_cloud(dpath)
        else:
            pcd = open3d.io.read_point_cloud(map_path)
            print("Downsampling...")
            pcd_down = pcd.voxel_down_sample(0.2)

            open3d.io.write_point_cloud(dpath, pcd_down)

        dpath = os.path.join("/dataset", 'kitti_processed_CMRNet',
                             'sequences', '00', 'map',
                             f'map-{seq:02d}_0.1_0-{len(self.poses)}_downsampled_1..pcd')
        if os.path.exists(dpath):
            pcd_down2 = open3d.io.read_point_cloud(dpath)
        else:
            pcd = open3d.io.read_point_cloud(map_path)
            print("Downsampling...")
            pcd_down2 = pcd.voxel_down_sample(0.7)

            open3d.io.write_point_cloud(dpath, pcd_down2)

        self.map = np.asarray(pcd_down.points).T  # [3,N]
        self.map2 = np.asarray(pcd_down2.points).T  # [3,N]

        print("Load Done...")

        # self.render_pc = PointCloudRender()

        self.render = PointCloudCameraRender()

        self.path = []

        self.path_gt = []

        self.z0 = 35.

    def decode_meta(self, meta_info):

        seq, seq_i, seq_j = meta_info.strip('\n').split(' ')
        pose = np.array(self.poses[int(seq_i)], np.float32)
        R = quat2mat(pose[[6, 3, 4, 5]]).T
        local_pc = R @ self.map + (-R @ pose[:3, None])  # 3,N

        # local_pc2 = (torch.from_numpy(R).cuda().float()
        #              @ torch.from_numpy(self.map2).cuda().float() +
        #              torch.from_numpy(-R @ pose[:3, None]).cuda().float()).cpu().numpy()  # 3,N
        local_pc2 = self.map2  # .cpu().numpy()
        # indexes = local_pc2[1] > -150.
        # indexes = indexes & (local_pc2[1] < 150.)
        # indexes = indexes & (local_pc2[0] > -50.)
        # indexes = indexes & (local_pc2[0] < 150.)

        pcl = local_pc2.T  # [:, indexes].T  # without visibility filter

        indexes2 = local_pc[1] > -30.
        indexes2 = indexes2 & (local_pc[1] < 30.)
        indexes2 = indexes2 & (local_pc[0] > -5.)
        indexes2 = indexes2 & (local_pc[0] < 100.)

        lpcl = local_pc[:, indexes2].T

        imp = os.path.join("/dataset", 'kitti_processed_DeepI2P', 'data_odometry_color_npy',
                           'sequences',
                           seq, 'image_2', seq_j + ".npy")
        # pcl = np.fromfile(pcp, dtype=np.float32).reshape(-1, 4)[:, :3]
        img = cv2.cvtColor(np.load(imp), cv2.COLOR_RGB2BGR)
        Tr, K, P2 = read_calib(
            os.path.join("/dataset", 'data_odometry_calib', 'dataset', 'sequences', seq,
                         'calib.txt'))
        Tr = np.vstack((Tr, [0, 0, 0, 1]))
        Pc = np.dot(P2, Tr)

        # pcl = pcl[np.random.choice(len(pcl), len(pcl) // 4, replace=False)]

        return pcl, img, K, Pc, lpcl

    def transform(self, pc, Trans, inv=False):
        # path = np.stack(path, axis=0)
        pc = torch.from_numpy(pc).float().cuda()
        Trans = torch.from_numpy(Trans).float().cuda()
        if inv:
            pc = pc @ Trans[:3, :3] - (Trans[:3, :3].T @ Trans[:3, 3])
            return pc.cpu().numpy()
        else:
            return (pc @ Trans[:3, :3].T + Trans[:3, 3]).cpu().numpy()

    def get_color(self, pc, Pc, black=False, cc=None):
        if black:
            if cc is None:
                return np.full_like(pc, 0.1)
            else:
                return np.asarray(cc)[None].repeat(pc.shape[0], 0)

    def get_color_pc(self, pcl, Pc, im, K):
        # pc N,3 Pc
        pc = pcl @ Pc[:3, :3].T + Pc[:3, 3]
        u = (pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2])
        v = (pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2])
        z = pc[:, 2]

        mask = (u > 0) & (u < im.shape[1]) & (v > 0) & (v < im.shape[0]) & (pc[:, 2] > 0.1)

        dist = z[mask]

        pc = pcl[mask]

        color = (dist - dist.min()) / (dist.max() - dist.min() + 1e-10) * 255.
        colors = cv2.applyColorMap(color.astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)[:, :, ::-1] / 255.

        return [pc], [colors[:, 0]]

    def insert_path(self, new_path, old_path=None):
        if old_path is None:
            return [new_path]
        else:
            dist = new_path - old_path
            return [old_path + dist * i / 10. for i in range(10)]

    def get_vis_camera(self):
        R, T = look_at_view_transform(55, 0, -40)
        Z = torch.from_numpy(
            np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
                      [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
                      [0, 0, 1]])
        ).float()
        return R @ Z, T

    def vis(self):
        # step = 100
        if abs(self.start) == len(self.lines) and "section" not in self.lines[self.start]:
            return
        init_start = self.start + 1

        video_project = None

        w, h = 1280, 480

        video_global = cv2.VideoWriter(f"kitti_global.mp4",
                                       cv2.VideoWriter.fourcc(*"mp4v"),
                                       10,
                                       (w, h))

        for vis_t in tqdm(range(4128, self.num)):
            self.start = init_start + self.pose_t * vis_t
            # breakpoint()
            pcl, img, intrinsic, Pc, lpcl = self.decode_meta(self.lines[self.start])

            pred_extrinsic = np.array(self.lines[self.start + self.pose_t - 2].strip('\n').split(' '),
                                      np.float32).reshape(3, 4)
            gt_extrinsic = np.array(self.lines[self.start + self.pose_t - 1].strip('\n').split(' '),
                                    np.float32).reshape(3, 4)

            self.path_gt += self.insert_path(gt_extrinsic[:3, 3], None if len(self.path_gt) == 0 else
            self.path_gt[-1])
            self.path += self.insert_path(pred_extrinsic[:3, 3], None if len(self.path) == 0 else
            self.path[-1])

            # if video_project is None:
            #     video_project = cv2.VideoWriter(f"kitti_project.mp4",
            #                                     cv2.VideoWriter.fourcc(*"mp4v"),
            #                                     10.,
            #                                     (img.shape[1], img.shape[0]))

            Pc_inv = np.eye(4)
            Pc_inv[:3, :3] = Pc[:3, :3].T
            Pc_inv[:3, 3] = -Pc[:3, :3].T @ Pc[:3, 3]

            # self.render.set_camera(intrinsic, Pc_inv, img)

            # ppcl, pcolor = self.get_color_pc(lpcl, Pc, img, intrinsic)

            R, T = self.get_vis_camera()

            Rgt = gt_extrinsic[:3, :3]
            theta = np.arctan2(Rgt[1, 0], Rgt[0, 0])
            if self.z0 is None:
                self.z0 = gt_extrinsic[2, 3]
            gt_view = np.asarray([
                [np.cos(theta), -np.sin(theta), 0, gt_extrinsic[0, 3]],
                [np.sin(theta), np.cos(theta), 0, gt_extrinsic[1, 3]],
                [0, 0, 1, self.z0]
            ])

            # project_im = self.render.rendering(ppcl, pcolor)[0]
            im1 = PointCloudRender(radius=0.01, R=R, T=T, width=w, height=h).rendering([
                self.transform(pcl, gt_view, inv=True)],
                [self.get_color(pcl, None,
                                True,
                                [0.3, 0.3,
                                 0.3])],
                nofilter=False)[0]
            global_im = PointCloudRender(radius=0.015, im=im1, R=R, T=T, width=w, height=h).rendering(
                [self.transform(np.stack(self.path + self.path_gt), gt_view,
                                inv=True)],
                [np.concatenate(
                    [self.get_color(np.stack(self.path), None, True, [0, 1, 0]),
                     self.get_color(np.stack(self.path_gt), None,
                                    True, [1, 0, 0])
                     ]
                )], nofilter=True)[0]
            video_global.write(global_im)
            # video_project.write(project_im)
        # video_project.release()
        video_global.release()

    def calculate_sections(self, lines):
        count = -1
        section = {}
        count2 = 0
        last = None
        while count + len(lines) >= 0:
            # if abs(count) == len(self.lines):
            #     break
            if "section" in lines[count]:
                name = lines[count].strip("[section sign] DEMO on ")[:19]
                if count2 % 3 == 0:  # no coarse:
                    section[name] = (count2 // 3, count, 3)
                last = name
            else:
                count2 += 1
            count -= 1
        # print(section)
        return section, last


if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.vis()
