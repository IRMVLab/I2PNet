import torch
import os
from tqdm import tqdm
import cv2
import numpy as np

# from src.kitti_odometry_corr_snr import Kitti_Odometry_Dataset as testdataset
from src.kitti_odometry import read_calib
# from src.dataset_params import  as cfg

from p3d_render import PointCloudRender, PointCloudCameraRender
import open3d


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
        self.map = np.asarray(open3d.io.read_point_cloud(map_path).points).T  # [3,N]
        print("Load Done...")

        # self.render_pc = PointCloudRender()

        self.render = PointCloudCameraRender()

        self.path = []

        self.path_gt = []

    def decode_meta(self, meta_info):

        seq, seq_i, seq_j = meta_info.strip('\n').split(' ')
        pose = np.array(self.poses[int(seq_i)], np.float32)
        R = quat2mat(pose[[6, 3, 4, 5]]).T
        local_pc = R @ self.map + (-R @ pose[:3, None])  # 3,N

        indexes = local_pc[1] > -25.
        indexes = indexes & (local_pc[1] < 25.)
        indexes = indexes & (local_pc[0] > -10.)
        indexes = indexes & (local_pc[0] < 100.)

        pcl = local_pc[:, indexes].T  # without visibility filter

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

        return pcl, img, K, Pc

    def transform(self, pc, Trans, inv=False):
        # path = np.stack(path, axis=0)
        if inv:
            return pc @ Trans[:3, :3] - (Trans[:3, :3].T @ Trans[:3, 3])
        else:
            return pc @ Trans[:3, :3].T + Trans[:3, 3]

    def get_color(self, pc, Pc, black=False, cc=None):
        if black:
            if cc is None:
                return np.full_like(pc, 0.1)
            else:
                return np.asarray(cc)[None].repeat(0, pc.shape[0])
        # pc N,3 Pc
        pc = pc @ Pc[:3, :3].T + Pc[:3, 3]
        z = pc[:, 2]
        color = np.clip(z / max(z.max(), 1e-10), amin=0) * 90.
        colors = np.full((pc.shape[0], 3), 1)
        colors[:, 0] = color / 255.
        return colors

    def vis(self):
        # step = 100
        if abs(self.start) == len(self.lines) and "section" not in self.lines[self.start]:
            return
        init_start = self.start + 1

        video_project = None
        video_global = cv2.VideoWriter(f"kitti_global.mp4",
                                       cv2.VideoWriter.fourcc(*"mp4v"),
                                       30.,
                                       (1280, 720))

        for vis_t in tqdm(range(10)):
            self.start = init_start + self.pose_t * vis_t
            breakpoint()
            pcl, img, intrinsic, Pc = self.decode_meta(self.lines[self.start])

            pred_extrinsic = np.array(self.lines[self.start + self.pose_t - 2].strip('\n').split(' '),
                                      np.float32).reshape(3, 4)
            gt_extrinsic = np.array(self.lines[self.start + self.pose_t - 1].strip('\n').split(' '),
                                    np.float32).reshape(3, 4)

            self.path_gt.append(gt_extrinsic[:3, 3])
            self.path.append(pred_extrinsic[:3, 3])

            if video_project is None:
                video_project = cv2.VideoWriter(f"kitti_global.mp4",
                                cv2.VideoWriter.fourcc(*"mp4v"),
                                30.,
                                (img.shape[1],img.shape[0]))

            Pc_inv = np.eye(4)
            Pc_inv[:3, :3] = Pc[:3, :3].T
            Pc_inv[:3, 3] = -Pc[:3, :3].T @ Pc[:3, 3]

            self.render.set_camera(intrinsic, Pc_inv, img)

            project_im = self.render.rendering([pcl], [self.get_color(pcl, Pc)])[0]
            im1 = PointCloudRender().rendering([pcl], [self.get_color(pcl, None, True)])
            global_im = PointCloudRender(radius=0.02, im=im1).rendering([np.stack(self.path + self.path_gt)],
                                                                        [np.concatenate(
                                                                            [self.get_color(np.stack(self.path), None,
                                                                                            True, (1, 0, 0)),
                                                                             self.get_color(np.stack(self.path), None,
                                                                                            True, (0, 1, 0))
                                                                             ]
                                                                        )])
            video_global.write(global_im)
            video_project.write(project_im)
        video_project.release()
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
