from scipy.spatial import cKDTree
import struct
from multiprocessing import Process
import numpy as np
import open3d
import os
import cv2
from tqdm import tqdm


def transform_pc_np(P, pc_np):
    """

    :param pc_np: 3xN
    :param P: 4x4
    :return:
    """
    pc_homo_np = np.concatenate((pc_np,
                                 np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
                                axis=0)
    P_pc_homo_np = np.dot(P, pc_homo_np)
    return P_pc_homo_np[0:3, :]


def search_for_accumulation(pc_folder, seq_pose_folder,
                            seq_i, seq_sample_num, Pc, P_oi,
                            stride):
    Pc_inv = np.linalg.inv(Pc)
    P_io = np.linalg.inv(P_oi)

    pc_np_list, intensity_np_list, sn_np_list = [], [], []

    accumulation_frame_num = 3

    counter = 0
    while len(pc_np_list) < accumulation_frame_num:
        counter += 1
        seq_j = seq_i + stride * counter
        if seq_j < 0 or seq_j >= seq_sample_num:
            break

        npy_data = np.load(os.path.join(pc_folder, '%06d.npy' % seq_j)).astype(np.float32)
        pc_np = npy_data[0:3, :]  # 3xN
        intensity_np = npy_data[3:4, :]  # 1xN
        sn_np = npy_data[4:7, :]  # 3xN

        P_oj = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_j))['arr_0'].astype(np.float32)  # 4x4
        P_ij = np.dot(P_io, P_oj)

        P_transform = np.dot(Pc_inv, np.dot(P_ij, Pc))
        pc_np = transform_pc_np(P_transform, pc_np)
        P_transform_rot = np.copy(P_transform)
        P_transform_rot[0:3, 3] = 0
        sn_np = transform_pc_np(P_transform_rot, sn_np)

        pc_np_list.append(pc_np)
        intensity_np_list.append(intensity_np)
        sn_np_list.append(sn_np)

    return pc_np_list, intensity_np_list, sn_np_list


def get_accumulated_pc(pc_folder, seq_pose_folder, seq_i, seq_sample_num, Pc):
    pc_path = os.path.join(pc_folder, '%06d.npy' % seq_i)
    npy_data = np.load(pc_path).astype(np.float32)
    # shuffle the point cloud data, this is necessary!
    npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
    pc_np = npy_data[0:3, :]  # 3xN
    intensity_np = npy_data[3:4, :]  # 1xN
    sn_np = npy_data[4:7, :]  # 3xN

    accumulation_frame_skip = 6
    # if self.opt.accumulation_frame_num <= 0.5:
    #     return pc_np, intensity_np, sn_np

    pc_np_list = [pc_np]
    intensity_np_list = [intensity_np]
    sn_np_list = [sn_np]

    # pose of i
    P_oi = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_i))['arr_0'].astype(np.float32)  # 4x4

    # search for previous
    prev_pc_np_list, \
    prev_intensity_np_list, \
    prev_sn_np_list = search_for_accumulation(pc_folder,
                                              seq_pose_folder,
                                              seq_i,
                                              seq_sample_num,
                                              Pc,
                                              P_oi,
                                              -accumulation_frame_skip)
    # search for next
    next_pc_np_list, \
    next_intensity_np_list, \
    next_sn_np_list = search_for_accumulation(pc_folder,
                                              seq_pose_folder,
                                              seq_i,
                                              seq_sample_num,
                                              Pc,
                                              P_oi,
                                              accumulation_frame_skip)

    pc_np_list = pc_np_list + prev_pc_np_list + next_pc_np_list
    intensity_np_list = intensity_np_list + prev_intensity_np_list + next_intensity_np_list
    sn_np_list = sn_np_list + prev_sn_np_list + next_sn_np_list

    pc_np = np.concatenate(pc_np_list, axis=1)
    intensity_np = np.concatenate(intensity_np_list, axis=1)
    sn_np = np.concatenate(sn_np_list, axis=1)

    return pc_np, intensity_np, sn_np


def downsample_with_intensity_sn(pointcloud, intensity, sn, voxel_grid_downsample_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))
    intensity_max = np.max(intensity)

    fake_colors = np.zeros((pointcloud.shape[1], 3))
    fake_colors[:, 0:1] = np.transpose(intensity) / intensity_max

    pcd.colors = open3d.utility.Vector3dVector(fake_colors)
    pcd.normals = open3d.utility.Vector3dVector(np.transpose(sn))

    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
    pointcloud = down_pcd_points

    intensity = np.transpose(np.asarray(down_pcd.colors)[:, 0:1]) * intensity_max
    sn = np.transpose(np.asarray(down_pcd.normals))

    return pointcloud, intensity, sn


def read_calib(calib_file_path):
    with open(calib_file_path, 'r') as f:
        for line in f.readlines():
            key = line[0:2]
            mat = np.fromstring(line[4:], sep=' ').reshape((3, 4)).astype(np.float32)

            if 'Tr' == key:
                Tr = mat
            elif 'P2' == key:
                K = mat[0:3, 0:3]
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                intrinsic = np.asarray([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]])

                tz = mat[2, 3]
                tx = (mat[0, 3] - cx * tz) / fx
                ty = (mat[1, 3] - cy * tz) / fy
                P = np.identity(4)
                P[0:3, 3] = np.asarray([tx, ty, tz])

    return Tr, intrinsic, P


def process_kitti(input_root_path,
                  output_root_path,
                  seq_list,
                  downsample_voxel_size):
    for seq in seq_list:
        pc_folder = os.path.join(input_root_path, 'data_odometry_velodyne_deepi2p_new', 'data_odometry_velodyne_NWU',
                                 'sequences', '%02d' % seq, 'voxel0.1_snr0.6')
        seq_pose_folder = os.path.join(input_root_path, 'kitti_processed_DeepI2P', 'poses', '%02d' % seq)

        output_folder_snr = os.path.join(input_root_path, 'data_odometry_velodyne_deepi2p_new',
                                         "data_odometry_velodyne_NWU",
                                         "sequences", '%02d' % seq,
                                         'voxel%.1f_snr0.6_acc' % downsample_voxel_size)
        output_folder_pc = os.path.join(input_root_path, 'data_odometry_velodyne_deepi2p_new',
                                        "data_odometry_velodyne_NWU",
                                        "sequences", '%02d' % seq,
                                        'voxel%.1f_acc' % downsample_voxel_size)
        calib_path = os.path.join(input_root_path, 'data_odometry_calib', 'dataset', 'sequences', '%02d' % seq,
                                  'calib.txt')

        if not os.path.isdir(output_folder_snr):
            os.makedirs(output_folder_snr)
        if not os.path.isdir(output_folder_pc):
            os.makedirs(output_folder_pc)
        # if not os.path.isdir(output_im_folder):
        #     os.makedirs(output_im_folder)
        sample_num = round(len(os.listdir(pc_folder)))
        # sn_radius = 0.6
        # sn_max_nn = 30
        for i in tqdm(range(sample_num)):
            # processing with calib params
            Tr, cam_intrinsic, P = read_calib(calib_path)
            Pc = np.dot(P, np.vstack((Tr, [0, 0, 0, 1])))
            pc, intensity, sn = get_accumulated_pc(pc_folder, seq_pose_folder, i, sample_num, Pc)
            if pc.shape[1] > 2 * 20480:
                pc, intensity, sn = downsample_with_intensity_sn(pc, intensity, sn, 0.3)

            output_np = np.concatenate((pc, intensity, sn), axis=0).astype(np.float32)
            np.save(os.path.join(output_folder_snr, '%06d.npy' % i), output_np)
            np.save(os.path.join(output_folder_pc, '%06d.npy' % i), pc)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default="/dataset")
    parser.add_argument('--dst', type=str, default="/dataset")
    parser.add_argument("--voxelsize", type=float, default=0.1)
    FLAGS = parser.parse_args()

    input_root_path = FLAGS.src
    output_root_path = FLAGS.dst
    downsample_voxel_size = FLAGS.voxelsize

    os.makedirs(output_root_path, exist_ok=True)

    seq_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    thread_num = len(seq_list)  # One thread for one folder
    kitti_threads = []
    for i in range(thread_num):
        thread_seq_list = [seq_list[i]]
        process_kitti(input_root_path,
                      output_root_path,
                      thread_seq_list,
                      downsample_voxel_size)
        # kitti_threads.append(Process(target=process_kitti,
        #                              args=))

    # for thread in kitti_threads:
    #     thread.start()

    # for thread in kitti_threads:
    #     thread.join()
