from scipy.spatial import cKDTree
import struct
from multiprocessing import Process
import numpy as np
import open3d
import os
import cv2
from tqdm import tqdm


# from data_preprocess.kitti_helper import *

def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*4
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2], point[3]])
    return np.asarray(pc_list, dtype=np.float32).T



def process_kitti_cat(input_root_path,
                  output_root_path,
                  seq_list):
    for seq in seq_list:
        input_folder = os.path.join(input_root_path, "data_odometry_velodyne", "dataset", '%02d' % seq, 'velodyne')

        output_folder = os.path.join(output_root_path, "data_odometry_velodyne_NWU", "sequences", '%02d' % seq,
                                     'snr0.6')

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        sample_num = round(len(os.listdir(input_folder)))
        sn_radius = 0.6
        sn_max_nn = 30
        for i in tqdm(range(sample_num)):
            data_np = read_velodyne_bin(os.path.join(input_folder, '%06d.bin' % i))
            pc_np = data_np[0:3, :]
            intensity_np = data_np[3:, :]
            snr = np.load(os.path.join(os.path.join(output_folder, '%06d.npy' % i)))[-3:,:]

            output_np = np.concatenate((pc_np,intensity_np,snr), axis=0).astype(np.float32)
            np.save(os.path.join(output_folder, '%06d.npy' % i),output_np)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default="/dataset")
    parser.add_argument('--dst', type=str, default="/dataset/data_odometry_velodyne_deepi2p_new")
    parser.add_argument("--voxelsize", type=float, default=0.1)
    FLAGS = parser.parse_args()

    input_root_path = FLAGS.src
    output_root_path = FLAGS.dst
    downsample_voxel_size = FLAGS.voxelsize

    os.makedirs(output_root_path, exist_ok=True)

    seq_list = [0,1,2,3,4,5,6,7,8,9,10]

    thread_num = len(seq_list)  # One thread for one folder
    kitti_threads = []
    for i in range(thread_num):
        thread_seq_list = [seq_list[i]]
        process_kitti_cat(input_root_path,
                      output_root_path,
                      thread_seq_list,
                      )

