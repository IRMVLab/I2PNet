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
        input_folder = os.path.join(input_root_path, "data_odometry_velodyne", "dataset", '%02d' % seq, 'velodyne')
        # img2_folder = os.path.join(input_root_path, 'data_odometry_color', '%02d' % seq, 'image_2')
        output_folder = os.path.join(output_root_path, "data_odometry_velodyne_NWU", "sequences", '%02d' % seq,
                                     'voxel%.1f_snr0.6' % downsample_voxel_size)
        # output_im_folder = os.path.join(output_root_path,'data_odometry_color_npy', 'sequences', '%02d' % seq, 'image_2')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        # if not os.path.isdir(output_im_folder):
        #     os.makedirs(output_im_folder)
        sample_num = round(len(os.listdir(input_folder)))
        sn_radius = 0.6
        sn_max_nn = 30
        for i in tqdm(range(sample_num)):
            data_np = read_velodyne_bin(os.path.join(input_folder, '%06d.bin' % i))
            pc_np = data_np[0:3, :]
            intensity_np = data_np[3:, :]

            # convert to Open3D point cloud datastructure
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(pc_np.T)
            downpcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)

            # surface normal computation
            downpcd.estimate_normals(search_param=open3d.geometry.
                                     KDTreeSearchParamHybrid(radius=sn_radius, max_nn=sn_max_nn))
            downpcd.orient_normals_to_align_with_direction(np.array([0., 0., 1.]))
            # open3d.visualization.draw_geometries([downpcd])

            # get numpy array from pcd
            pc_down_np = np.asarray(downpcd.points).T  # [N,3]
            pc_down_sn_np = np.asarray(downpcd.normals).T

            # get intensity through 1-NN between downsampled pc and original pc
            kdtree = cKDTree(pc_np.T)
            D, I = kdtree.query(pc_down_np.T, k=1)
            intensity_down_np = intensity_np[:, I]

            # save downampled points, intensity, surface normal to npy
            # output_np = pc_down_np.astype(np.float32)
            output_np = np.concatenate((pc_down_np, intensity_down_np, pc_down_sn_np), axis=0).astype(np.float32)
            np.save(os.path.join(output_folder, '%06d.npy' % i), output_np)

            # img2_path = os.path.join(img2_folder, '%06d.png' % i)
            # img2 = cv2.imread(img2_path)
            # img2 = img2[:, :, ::-1]  # HxWx3

            # img3 = cv2.imread(img3_path)
            # img3 = img3[:, :, ::-1]  # HxWx3

            # np.save(os.path.join(output_im_folder, '%06d.npy' % i), img2)

def process_kitti_novoxel(input_root_path,
                  output_root_path,
                  seq_list):
    for seq in seq_list:
        input_folder = os.path.join(input_root_path, "data_odometry_velodyne", "dataset", '%02d' % seq, 'velodyne')
        # img2_folder = os.path.join(input_root_path, 'data_odometry_color', '%02d' % seq, 'image_2')
        output_folder = os.path.join(output_root_path, "data_odometry_velodyne_NWU", "sequences", '%02d' % seq,
                                     'snr0.6')
        # output_im_folder = os.path.join(output_root_path,'data_odometry_color_npy', 'sequences', '%02d' % seq, 'image_2')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        # if not os.path.isdir(output_im_folder):
        #     os.makedirs(output_im_folder)
        sample_num = round(len(os.listdir(input_folder)))
        sn_radius = 0.6
        sn_max_nn = 30
        for i in tqdm(range(sample_num)):
            data_np = read_velodyne_bin(os.path.join(input_folder, '%06d.bin' % i))
            pc_np = data_np[0:3, :]
            intensity_np = data_np[3:, :]

            # convert to Open3D point cloud datastructure
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(pc_np.T)
            # downpcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)

            # surface normal computation
            pcd.estimate_normals(search_param=open3d.geometry.
                                     KDTreeSearchParamHybrid(radius=sn_radius, max_nn=sn_max_nn))
            pcd.orient_normals_to_align_with_direction(np.array([0., 0., 1.]))
            # open3d.visualization.draw_geometries([downpcd])

            # get numpy array from pcd
            pc_down_np = np.asarray(pcd.points).T  # [N,3]
            pc_down_sn_np = np.asarray(pcd.normals).T

            # get intensity through 1-NN between downsampled pc and original pc
            # kdtree = cKDTree(pc_np.T)
            # D, I = kdtree.query(pc_down_np.T, k=1)
            # intensity_down_np = intensity_np[:, I]

            # save downampled points, intensity, surface normal to npy
            # output_np = pc_down_np.astype(np.float32)
            # output_np = np.concatenate((intensity_np, pc_down_sn_np), axis=0).astype(np.float32)
            np.save(os.path.join(output_folder, '%06d.npy' % i), pc_down_sn_np)

            # img2_path = os.path.join(img2_folder, '%06d.png' % i)
            # img2 = cv2.imread(img2_path)
            # img2 = img2[:, :, ::-1]  # HxWx3

            # img3 = cv2.imread(img3_path)
            # img3 = img3[:, :, ::-1]  # HxWx3

            # np.save(os.path.join(output_im_folder, '%06d.npy' % i), img2)
def process_kitti_cat(input_root_path,
                  output_root_path,
                  seq_list):
    for seq in seq_list:
        input_folder = os.path.join(input_root_path, "data_odometry_velodyne", "dataset", '%02d' % seq, 'velodyne')
        # img2_folder = os.path.join(input_root_path, 'data_odometry_color', '%02d' % seq, 'image_2')
        output_folder = os.path.join(output_root_path, "data_odometry_velodyne_NWU", "sequences", '%02d' % seq,
                                     'snr0.6')
        # output_folder2 = os.path.join(output_root_path, "data_odometry_velodyne_NWU", "sequences", '%02d' % seq,
        #                              'pc_snr0.6')
        # snr_folder = os.path.join(output_root_path, "data_odometry_velodyne_NWU", "sequences", '%02d' % seq,
        #                              'raw_snr0.6')
        # output_im_folder = os.path.join(output_root_path,'data_odometry_color_npy', 'sequences', '%02d' % seq, 'image_2')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        # if not os.path.isdir(output_folder2):
        #     os.makedirs(output_folder2)
        # if not os.path.isdir(output_im_folder):
        #     os.makedirs(output_im_folder)
        sample_num = round(len(os.listdir(input_folder)))
        sn_radius = 0.6
        sn_max_nn = 30
        for i in tqdm(range(sample_num)):
            data_np = read_velodyne_bin(os.path.join(input_folder, '%06d.bin' % i))
            pc_np = data_np[0:3, :]
            intensity_np = data_np[3:, :]
            snr = np.load(os.path.join(os.path.join(output_folder, '%06d.npy' % i)))[-3:,:]

            # convert to Open3D point cloud datastructure
            # pcd = open3d.geometry.PointCloud()
            # pcd.points = open3d.utility.Vector3dVector(pc_np.T)
            # downpcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)

            # surface normal computation
            # pcd.estimate_normals(search_param=open3d.geometry.
            #                          KDTreeSearchParamHybrid(radius=sn_radius, max_nn=sn_max_nn))
            # pcd.orient_normals_to_align_with_direction(np.array([0., 0., 1.]))
            # open3d.visualization.draw_geometries([downpcd])

            # get numpy array from pcd
            # pc_down_np = np.asarray(pcd.points).T  # [N,3]
            # pc_down_sn_np = np.asarray(pcd.normals).T

            # get intensity through 1-NN between downsampled pc and original pc
            # kdtree = cKDTree(pc_np.T)
            # D, I = kdtree.query(pc_down_np.T, k=1)
            # intensity_down_np = intensity_np[:, I]

            # save downampled points, intensity, surface normal to npy
            # output_np = pc_down_np.astype(np.float32)
            output_np = np.concatenate((pc_np,intensity_np,snr), axis=0).astype(np.float32)
            np.save(os.path.join(output_folder, '%06d.npy' % i),output_np)


            # img2_path = os.path.join(img2_folder, '%06d.png' % i)
            # img2 = cv2.imread(img2_path)
            # img2 = img2[:, :, ::-1]  # HxWx3

            # img3 = cv2.imread(img3_path)
            # img3 = img3[:, :, ::-1]  # HxWx3

            # np.save(os.path.join(output_im_folder, '%06d.npy' % i), img2)
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
        # kitti_threads.append(Process(target=process_kitti,
        #                              args=))

    # for thread in kitti_threads:
    #     thread.start()

    # for thread in kitti_threads:
    #     thread.join()
