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


def process_kitti(input_root_path,
                  output_root_path,
                  seq_list,
                  downsample_voxel_size):
    for seq in seq_list:
        pose_folder = os.path.join(input_root_path, "data_odometry_velodyne",'data_odometry_poses',"dataset","poses",
                                   '%02d.txt'%seq)
        output_folder = os.path.join(output_root_path,"poses",'%02d' % seq)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        with open(pose_folder,'r') as f:
            poses = f.readlines()
            sample_num = len(poses)
            for i in tqdm(range(sample_num)):
                line = poses[i].strip('\n').split(' ')
                pose = np.array([float(s) for s in line]).reshape(3,4)
                pose = np.concatenate([pose,
                                       np.array([0,0,0,1],np.float32).reshape(1,4)],axis=0)
                np.savez(os.path.join(output_folder,"%06d.npz"%i),arr_0=pose)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str,default="/dataset")
    parser.add_argument('--dst', type=str,default="/dataset/data_odometry_velodyne_deepi2p_new")
    parser.add_argument("--voxelsize",type=float,default=0.1)
    FLAGS = parser.parse_args()

    input_root_path = FLAGS.src
    output_root_path = FLAGS.dst
    downsample_voxel_size = FLAGS.voxelsize

    os.makedirs(output_root_path,exist_ok=True)

    seq_list = list(range(11))

    thread_num = 11  # One thread for one folder
    kitti_threads = []
    for i in range(thread_num):
        thread_seq_list = [i]
        kitti_threads.append(Process(target=process_kitti,
                                     args=(input_root_path,
                                           output_root_path,
                                           thread_seq_list,
                                           downsample_voxel_size)))

    for thread in kitti_threads:
        thread.start()

    for thread in kitti_threads:
        thread.join()


