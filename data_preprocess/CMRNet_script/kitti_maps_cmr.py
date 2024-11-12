# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
import argparse
import os
import sys

sys.path.append("../..")
sys.path.append("..")
from pathlib import Path
import h5py
import numpy as np
import open3d as o3
from pykitti_utils import load_velo_scan, load_calib
import torch
from tqdm import tqdm,trange

from utils import to_rotation_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--sequence', default='00',
                    help='sequence')
parser.add_argument('--device', default='cuda',
                    help='device')
parser.add_argument('--voxel_size', default=0.1, type=float, help='Voxel Size')
parser.add_argument('--start', default=0, help='Starting Frame')
parser.add_argument('--end', default=100000, help='End Frame')
# parser.add_argument('--map', default=None, help='Use map file')
parser.add_argument('--kitti_folder', default='/dataset/data_odometry_velodyne/dataset',
                    help='Folder of the KITTI dataset')
parser.add_argument('--output_folder', default='/dataset/kitti_processed_CMRNet/sequences')

args = parser.parse_args()
sequence = args.sequence
output_folder = args.output_folder

thisfile_dir = str(Path(__file__).parent.resolve())

if __name__ == '__main__':
    print("Sequence: ", sequence)
    version = 0.1

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, sequence), exist_ok=True)
    velodyne_folder = os.path.join(args.kitti_folder, 'sequences', sequence, 'velodyne')
    pose_file = os.path.join(thisfile_dir, "data", f'kitti-{sequence}.csv')

    # load refined poses
    poses = []
    with open(pose_file, 'r') as f:
        for x in f:
            if x.startswith('timestamp'):
                continue
            x = x.split(',')
            T = torch.tensor([float(x[1]), float(x[2]), float(x[3])])
            R = torch.tensor([float(x[7]), float(x[4]), float(x[5]), float(x[6])])
            poses.append(to_rotation_matrix(R, T))

    # map_file = args.map
    first_frame = int(args.start)
    last_frame = min(len(poses), int(args.end))
    # kitti = pykitti.odometry(args.kitti_folder, sequence)

    # create the global map
    os.makedirs(os.path.join(output_folder, sequence, "map"), exist_ok=True)
    mappath = os.path.join(output_folder, sequence, "map",
                           f'map-{sequence}_{args.voxel_size}_{first_frame}-{last_frame}.pcd')

    if not os.path.exists(mappath):

        pc_map = []
        pcl = o3.geometry.PointCloud()
        for i in tqdm(range(first_frame, last_frame)):
            pc = load_velo_scan(os.path.join(velodyne_folder, "%06d.bin" % i))

            valid_indices = pc[:, 0] < -3.
            valid_indices = valid_indices | (pc[:, 0] > 3.)
            valid_indices = valid_indices | (pc[:, 1] < -3.)
            valid_indices = valid_indices | (pc[:, 1] > 3.)
            pc = pc[valid_indices].copy()
            intensity = pc[:, 3].copy()
            pc[:, 3] = 1.
            RT = poses[i].numpy()
            pc_rot = np.matmul(RT, pc.T)
            pc_rot = pc_rot.astype(np.float_).T.copy()

            pcl_local = o3.geometry.PointCloud()
            pcl_local.points = o3.utility.Vector3dVector(pc_rot[:, :3])
            pcl_local.colors = o3.utility.Vector3dVector(np.vstack((intensity, intensity, intensity)).T)

            downpcd = o3.geometry.PointCloud.voxel_down_sample(pcl_local, voxel_size=args.voxel_size)

            pcl.points.extend(downpcd.points)
            pcl.colors.extend(downpcd.colors)
        print("Start to save the map....")
        downpcd_full = o3.geometry.PointCloud.voxel_down_sample(pcl, voxel_size=args.voxel_size)
        downpcd, ind = o3.geometry.PointCloud.remove_statistical_outlier(downpcd_full, nb_neighbors=40, std_ratio=0.3)

        # o3.draw_geometries(downpcd)

        # remove outliers
        o3.io.write_point_cloud(mappath, downpcd)
        print("Save done....")
    else:
        downpcd = o3.io.read_point_cloud(mappath)

    # local map is too large, build runtime
    # exit(0)
    # build the global map
    # construct the local map

    # os.makedirs(os.path.join(output_folder, sequence, "local_maps"), exist_ok=True)
    voxelized = torch.tensor(np.asarray(downpcd.points), dtype=torch.float)
    voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
    voxelized = voxelized.t()
    # voxelized = voxelized.to(args.device)  # (3,N)
    vox_intensity = torch.tensor(np.asarray(downpcd.colors), dtype=torch.float)[:, 0:1].t()  # [1,N]

    # velo2cam2 = torch.from_numpy(load_calib(sequence)).float().to(args.device)

    # save local maps
    if not os.path.exists(os.path.join(output_folder, sequence, f'local_maps_{version}')):
        os.mkdir(os.path.join(output_folder, sequence, f'local_maps_{version}'))
    elif sequence == '08':
        pass
    else:
        print(f"Seq {sequence} Local map have been built.")
        exit(0)
    tbar = tqdm(range(first_frame, last_frame))
    for i in tbar:
        pose = poses[i]
        # pose = pose.to(args.device)
        pose = pose.inverse()

        # warp to the local coordination
        local_map = voxelized.clone()
        local_intensity = vox_intensity.clone()
        local_map = torch.mm(pose, local_map).t()  # (N,4)

        # y \in [-25,25] x \in [-10,100]

        # indexes = local_map[:, 1] > -25.
        # indexes = indexes & (local_map[:, 1] < 25.)
        # indexes = indexes & (local_map[:, 0] > -10.)
        # indexes = indexes & (local_map[:, 0] < 100.)
        # y \in [-10,10] x \in [-5,15]
        indexes = local_map[:, 1] > -10.
        indexes = indexes & (local_map[:, 1] < 10.)
        indexes = indexes & (local_map[:, 0] > -5.)
        indexes = indexes & (local_map[:, 0] < 15.)
        # y \in [-25,25] x \in [-10,100]
        # indexes = local_map[:, 1] > -25.
        # indexes = indexes & (local_map[:, 1] < 25.)
        # indexes = indexes & (local_map[:, 0] > -10.)
        # indexes = indexes & (local_map[:, 0] < 100.)

        # local_map = local_map[indexes].cpu().numpy()[:,:3]  # (N,3)
        # intensity = local_intensity[:, indexes].cpu().numpy().T  # (N,1)

        # local_map = local_map[indexes]  # (N,3)
        # intensity = local_intensity[:, indexes].T  # (N,1)
        local_map = local_map[indexes].t()  # (3,N)
        local_intensity = local_intensity[:, indexes]  # (1,N)

        # We do not perform this to transform it to the camera coordination
        # in the dataset preprocessing

        # local_map = torch.mm(velo2cam2, local_map.t())
        # local_map = local_map[[2, 0, 1, 3], :] # [z,x,y,1]

        # pcd = o3.PointCloud()
        # pcd.points = o3.Vector3dVector(local_map[:,:3].numpy())
        # o3.write_point_cloud(f'{i:06d}.pcd', pcd)

        # pcd = o3.geometry.PointCloud()
        # pcd.points = o3.utility.Vector3dVector(local_map)
        # pcd.colors = o3.utility.Vector3dVector(np.concatenate([intensity,intensity,intensity],axis=1))

        # pcd = pcd.voxel_down_sample(0.3)

        # local_map = torch.from_numpy(np.asarray(pcd.points).T).to(torch.float16)
        # local_intensity = torch.from_numpy(np.asarray(pcd.colors)[:,:1].T).to(torch.float16)

        tbar.set_postfix({"point":local_map.shape[1]})
        # store in the h5 file to save the storage
        file = os.path.join(output_folder, sequence,
                            f'local_maps_{version}', f'{i:06d}.h5')
        # with h5py.File(file, 'w') as hf:
        #     hf.create_dataset('PC', data=local_map, compression='lzf', shuffle=True)
        #     hf.create_dataset('intensity', data=local_intensity, compression='lzf', shuffle=True)
        with h5py.File(file, 'w') as hf:
            hf.create_dataset('PC', data=local_map.cpu().half(), compression='lzf', shuffle=True)
            hf.create_dataset('intensity', data=local_intensity.cpu().half(), compression='lzf', shuffle=True)
