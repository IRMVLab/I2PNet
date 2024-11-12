import pickle
import json
import numpy as np
import math
import os
from pyquaternion import Quaternion
import time

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3
import torch
from tqdm import tqdm
import h5py
from PIL import Image
import cv2

test_night_scene_tokens = ['e59a4d0cc6a84ed59f78fb21a45cdcb4',
                           '7209495d06f24712a063ac6c4a9b403b',
                           '3d776ea805f240bb925bd9b50b258416',
                           '48f81c548d0148fc8010a73d70b2ef9c',
                           '2ab683f384234dce89800049dec19a30',
                           '7edca4c44eac4f52a3105e1794e56b7e',
                           '81c939ce8c0d4cc7b159cb5ed4c4e712',
                           '24e6e64ecf794be4a51f7454c8b6d0b2',
                           '828ed34a5e0c456fbf0751cabbab3341',
                           'edfd6cfd1805477fbeadbd29f39ed599',
                           '7692a3e112b44b408d191e45954a813c',
                           '58d27a9f83294d99a4ff451dcad5f4d2',
                           'a1573aef0bf74324b373dd8a22b4dd68',
                           'ba06095d4e2e425b8e398668abc301d8',
                           '7c315a1db2ac49439d281605f3cca6be',
                           '732d7a84353f4ada803a9a115728496c',
                           '1630a1d9cf8a46b3843662a23126e3f6',
                           'f437809584344859882bdff7f8784c43']


def get_scene_lidar_token(nusc, scene_token, frame_skip=2):
    sensor = 'LIDAR_TOP'
    scene = nusc.get('scene', scene_token)
    first_sample = nusc.get('sample', scene['first_sample_token'])
    lidar = nusc.get('sample_data', first_sample['data'][sensor])

    lidar_token_list = [lidar['token']]
    counter = 1
    while lidar['next'] != '':
        lidar = nusc.get('sample_data', lidar['next'])
        counter += 1
        if counter % frame_skip == 0:
            lidar_token_list.append(lidar['token'])
    return lidar_token_list


def get_lidar_token_list(nusc, frame_skip):
    daytime_scene_list = []
    for scene in nusc.scene:
        if 'night' in scene['description'] \
                or 'Night' in scene['description'] \
                or scene['token'] in test_night_scene_tokens:
            continue
        else:
            daytime_scene_list.append(scene['token'])

    lidar_token_list = []
    for scene_token in daytime_scene_list:
        lidar_token_list += get_scene_lidar_token(nusc, scene_token, frame_skip=frame_skip)
    return lidar_token_list

def get_scene_daytime_list(nusc, range):
    daytime_scene_list = []
    daytime_scenename_list = []
    for scene in nusc.scene[range[0]:range[1]]:
        if 'night' in scene['description'] \
                or 'Night' in scene['description'] \
                or scene['token'] in test_night_scene_tokens:
            continue
        else:
            daytime_scene_list.append(scene['token'])
            daytime_scenename_list.append(scene['name'])
    return daytime_scene_list, daytime_scenename_list

def get_dayscene_token(nusc, id):
    id = int(id)
    for scene in nusc.scene:
        if 'night' in scene['description'] \
                or 'Night' in scene['description'] \
                or scene['token'] in test_night_scene_tokens:
            continue
        else:
            if id == 0:
                return scene['token'], scene['name']
            else:
                id -= 1

    return None


def get_P_from_Rt(R, t):
    P = np.identity(4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t
    return P


def get_sample_data_ego_pose_P(nusc, sample_data):
    sample_data_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    sample_data_pose_R = np.asarray(Quaternion(sample_data_pose['rotation']).rotation_matrix).astype(np.float32)
    sample_data_pose_t = np.asarray(sample_data_pose['translation']).astype(np.float32)
    sample_data_pose_P = get_P_from_Rt(sample_data_pose_R, sample_data_pose_t)
    return sample_data_pose_P

def get_calibration_P(nusc, sample_data):
    calib = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    R = np.asarray(Quaternion(calib['rotation']).rotation_matrix).astype(np.float32)
    t = np.asarray(calib['translation']).astype(np.float32)
    P = get_P_from_Rt(R, t)
    return P

def get_sample_data_calibrate_pose_P(nusc, sample_data, no_k=False):
    sample_data_pose = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    sample_data_K = None
    if not no_k:
        sample_data_K = sample_data_pose["camera_intrinsic"]

    sample_data_pose_R = np.asarray(Quaternion(sample_data_pose['rotation']).rotation_matrix).astype(np.float32)
    sample_data_pose_t = np.asarray(sample_data_pose['translation']).astype(np.float32)
    sample_data_pose_P = get_P_from_Rt(sample_data_pose_R, sample_data_pose_t)
    return sample_data_pose_P, sample_data_K

def search_nearby_cameras(nusc,
                          init_camera,
                          max_translation,
                          direction,
                          lidar_P_inv,
                          nearby_camera_token_list):
    init_camera_direction_token = init_camera[direction]
    if init_camera_direction_token == '':
        return nearby_camera_token_list

    camera = nusc.get('sample_data', init_camera_direction_token)
    while True:
        camera_token = camera[direction]
        if camera_token == '':
            break
        camera = nusc.get('sample_data', camera_token)
        camera_P = get_sample_data_ego_pose_P(nusc, camera)
        P_lc = np.dot(lidar_P_inv, camera_P)
        t_lc = P_lc[0:3, 3]
        t_lc_norm = np.linalg.norm(t_lc)

        if t_lc_norm < max_translation:
            nearby_camera_token_list.append(camera_token)
        else:
            break
    return nearby_camera_token_list


def get_nearby_camera_token_list(nusc,
                                 lidar_token,
                                 max_translation,
                                 camera_name):
    lidar = nusc.get('sample_data', lidar_token)
    lidar_P = get_sample_data_ego_pose_P(nusc, lidar)
    lidar_P_inv = np.linalg.inv(lidar_P)

    lidar_sample_token = lidar['sample_token']
    lidar_sample = nusc.get('sample', lidar_sample_token)
    
    init_camera_token = lidar_sample['data'][camera_name]
    init_camera = nusc.get('sample_data', init_camera_token)
    nearby_camera_token_list = [init_camera_token]

    # nearby_camera_token_list = search_nearby_cameras(
    #     nusc,
    #     init_camera,
    #     max_translation,
    #     'next',
    #     lidar_P_inv,
    #     nearby_camera_token_list)
    # nearby_camera_token_list = search_nearby_cameras(
    #     nusc,
    #     init_camera,
    #     max_translation,
    #     'prev',
    #     lidar_P_inv,
    #     nearby_camera_token_list)
    #print(len(nearby_camera_token_list))
    return nearby_camera_token_list


def get_nearby_camera(nusc, lidar_token, max_translation):
    cam_list = ['CAM_FRONT',
                'CAM_FRONT_LEFT',
                'CAM_FRONT_RIGHT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT']
    nearby_cam_token_dict = {}
    for camera_name in cam_list:
        nearby_cam_token_dict[camera_name] \
            = get_nearby_camera_token_list(nusc,
                                           lidar_token,
                                           max_translation,
                                           camera_name)
    return nearby_cam_token_dict


def make_nuscenes_dataset_raw(nusc, frame_skip, max_translation):
    dataset = []

    lidar_token_list = get_lidar_token_list(nusc,
                                            frame_skip)
    for i, lidar_token in enumerate(lidar_token_list):
        # begin_t = time.time()
        'CAM_FRONT'
        nearby_camera_token_dict = get_nearby_camera(nusc,
                                                     lidar_token,
                                                     max_translation)

        dataset.append((lidar_token, nearby_camera_token_dict))

        # print('lidar %s takes %f' % (lidar_token, time.time()-begin_t))
        if i % 100 == 0:
            print('%d done...' % i)

    return dataset


def load_dataset_info(filepath):
    with open(filepath, 'rb') as f:
        dataset_read = pickle.load(f)
    return dataset_read

# def make_nuscenes_dataset(root_path):
#     dataset = load_dataset_info(os.path.join(root_path, 'dataset_info.list'))
#     return dataset

def make_nus_localmap_dataset(nusc, scene_name, scene_token, dataset):
    #scene_token, scene_name = get_dayscene_token(nusc, scene_id)
    #print("scene num:", len(daytime_scene_list))
    print("scene_name:", scene_name)
    camera_name = 'CAM_FRONT'

    
    # for scene_token in daytime_scene_list:
    #     lidar_token_list = get_scene_lidar_token(nusc, scene_token, frame_skip=frame_skip)
    

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, scene_name), exist_ok=True)

    lidar_token_list = get_scene_lidar_token(nusc, scene_token, frame_skip=frame_skip)
    ### generating lidar map
    # create the global map
    os.makedirs(os.path.join(output_folder, scene_name, "map"), exist_ok=True)
    mappath = os.path.join(output_folder, scene_name, "map",
                        f'map-{scene_name}_{args.voxel_size}.pcd')
    lidar_num = len(lidar_token_list)
    print("lidar num:", lidar_num)

    if not os.path.exists(mappath):
        
        pcl = o3.geometry.PointCloud()
        for i in tqdm(range(0,lidar_num)):
            # begin_t = time.time()
            lidar_token = lidar_token_list[i]
            lidar = nusc.get('sample_data', lidar_token)
            pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar['filename'])).points.T
            pc_np = pc[:, :3].T
            intensity = pc[:, 3].copy()
            pc[:, 3] = 1.

             # remove point lying on the ego car
            x_inside = np.logical_and(pc_np[0, :] < 0.8, pc_np[0, :] > -0.8)
            y_inside = np.logical_and(pc_np[1, :] < 2.7, pc_np[1, :] > -2.7)
            inside_mask = np.logical_and(x_inside, y_inside)
            outside_mask = np.logical_not(inside_mask)
            pc = pc[outside_mask, :]
            intensity = intensity[outside_mask]

            lidar_calib_P = get_calibration_P(nusc, lidar)
            lidar_pose_P = get_sample_data_ego_pose_P(nusc, lidar)
            RT = np.dot(lidar_pose_P, lidar_calib_P)
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

    voxelized = torch.tensor(np.asarray(downpcd.points), dtype=torch.float)
    voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
    voxelized = voxelized.t()
    # voxelized = voxelized.to(args.device)  # (3,N)
    vox_intensity = torch.tensor(np.asarray(downpcd.colors), dtype=torch.float)[:, 0:1].t()  # [1,N]

    # save local maps
    if not os.path.exists(os.path.join(output_folder, scene_name, f'local_maps_small_{version}')):
        os.mkdir(os.path.join(output_folder, scene_name, f'local_maps_small_{version}'))
    else:
        print(f"{scene_name} Local map have been built. Covering")

    print("generating data:")
    for i in tqdm(range(0,lidar_num)):
        lidar_token = lidar_token_list[i]
        lidar = nusc.get('sample_data', lidar_token)
        lidar_CP = get_calibration_P(nusc, lidar)
        lidar_P = get_sample_data_ego_pose_P(nusc, lidar)
        # begin_t = time.time()
        camera_token_list = get_nearby_camera_token_list(nusc,
                                        lidar_token,
                                        max_translation,
                                        camera_name)
        init_camera_token = camera_token_list[int(np.random.choice(len(camera_token_list), 1))]

        init_camera = nusc.get('sample_data', init_camera_token)

        camera_P = get_sample_data_ego_pose_P(nusc, init_camera)

        camera_CP, camera_K = get_sample_data_calibrate_pose_P(nusc, init_camera)

        camera_CP_inv = np.linalg.inv(camera_CP)
        camera_P_inv = np.linalg.inv(camera_P)

        # T_pc^{cam} @ pc
        #Tr = camera_CP_inv @ camera_P_inv @ lidar_P @ lidar_CP
        # pose = torch.tensor((lidar_P @ lidar_CP), dtype=torch.float).inverse()
        #velo2cam2 = torch.tensor(camera_CP_inv @ camera_P_inv @ lidar_P @ lidar_CP, dtype=torch.float)
        near_sample_token = init_camera['sample_token']
        cam_sample = nusc.get('sample', near_sample_token)
        near_lidar_token = cam_sample['data']['LIDAR_TOP']
        near_lidar = nusc.get('sample_data', near_lidar_token)
        near_lidar_CP = get_calibration_P(nusc, near_lidar)
        P_near_lidar = get_sample_data_ego_pose_P(nusc, near_lidar)
        near_pose = torch.tensor((P_near_lidar @ near_lidar_CP), dtype=torch.float).inverse()
        velo2cam2 = torch.tensor(camera_CP_inv @ camera_P_inv @ P_near_lidar @ near_lidar_CP, dtype=torch.float)

        ### local map
        # warp to the local coordination
        local_map = voxelized.clone()
        local_intensity = vox_intensity.clone()
        local_map = torch.mm(near_pose, local_map).t()  # (N,4)

        # y \in [-25,25] x \in [-10,100]
        indexes = local_map[:, 0] > -10.
        indexes = indexes & (local_map[:, 0] < 10.)
        indexes = indexes & (local_map[:, 1] > -5.)
        indexes = indexes & (local_map[:, 1] < 15.)
        local_map = local_map[indexes].t()  # (3,N)
        local_intensity = local_intensity[:, indexes]  # (1,N)

        #local_map = torch.mm(velo2cam2, local_map)
        
        # ### for debug
        # if i%100 ==0:
        #     print("intrinsic:", camera_K)
        #     local_map_new = torch.mm(velo2cam2, local_map)
        #     img = cv2.imread(os.path.join(args.nus_folder, init_camera['filename']))  # RGB
        #     cv2.imwrite('/data/debug/nus_cmr/raw_gen'+f'{i:06d}.png', img)
        #     #cv2.imwrite(output_folder_list+'/'+scene_name+f'{i:06d}.png', img)
        #     import matplotlib.pyplot as plt
        #     ## project pc
        #     pix_pc = np.transpose(np.dot(camera_K,local_map_new[0:3, :]))
        #     #print(pix_pc)
        #     pix_pc[:, :2] = np.divide(pix_pc[:, :2],  (pix_pc[:, 2])[:,None])
        #     z_ = pix_pc[:, 2]
        #     xy = pix_pc[:, :2]
        #     is_in_picture = (xy[:, 0] >= 0) & (xy[:, 0] <= (1600  - 1)) & (xy[:, 1] >= 0) & (
        #             xy[:, 1] <= (900 - 1)) & (z_ > 0)
        #     z_ = z_[is_in_picture]
        #     #print("y max:", np.max(xy[:,1]))
        #     #print("z:",  z_)
        #     xy = xy[is_in_picture, :]

        #     plt.figure()
        #     plt.imshow(img)
        #     #plt.savefig(file_img_seq+str(ts)+"_img_left"+".jpg")
        #     pc_draw = (z_-np.min(z_)/(np.max(z_)-np.min(z_)))
        #     plt.scatter(xy[:,0], xy[:,1], c=pc_draw, cmap='jet', alpha=0.7, s=1)
        #     plt.savefig('/data/debug/nus_cmr/proj_gen'+f'{i:06d}.png')
        #     #plt.savefig(output_folder_list+'/'+scene_name+f'{i:06d}_proj.png')
        # print(what)
        #local_map = local_map[[2, 0, 1, 3], :] # [z,x,y,1]

        
        #print(what)
        file = os.path.join(output_folder, scene_name,
                            f'local_maps_small_{version}', f'{i:06d}.h5')
        # with h5py.File(file, 'w') as hf:
        #     hf.create_dataset('PC', data=local_map, compression='lzf', shuffle=True)
        #     hf.create_dataset('intensity', data=local_intensity, compression='lzf', shuffle=True)
        with h5py.File(file, 'w') as hf:
            hf.create_dataset('PC', data=local_map.cpu().half(), compression='lzf', shuffle=True)
            hf.create_dataset('intensity', data=local_intensity.cpu().half(), compression='lzf', shuffle=True)

        extra = [camera_P, camera_CP, lidar_CP, lidar_P, near_pose.inverse(), scene_name]
        #print(velo2cam2)
        dataset.append(((file, init_camera['filename']), camera_K, velo2cam2, extra))

    return dataset


def main():
    ### new by wyx
    nuscenes_path = os.path.join(args.nus_folder, 'trainval')
    nus_version = 'v1.0-trainval'
    #dataset = make_nuscenes_dataset(nuscenes_path)
    nusc = NuScenes(version=nus_version, dataroot=nuscenes_path, verbose=True)
    daytime_scene_list, daytime_scenename_list = get_scene_daytime_list(nusc, range=(0,700))
    
    # rand select
    select_idx = np.random.choice(len(daytime_scene_list), 70, replace=False).astype(int)

    scene_num = len(select_idx)#len(daytime_scene_list)
    print("train_num:", scene_num)
    dataset = []
    for id in tqdm(range(0,scene_num)):
        scene_token = daytime_scene_list[select_idx[id]]
        scene_name = daytime_scenename_list[select_idx[id]]
        dataset = make_nus_localmap_dataset(nusc, scene_name, scene_token, dataset)
        print("len dataset:", len(dataset))
    output_file = os.path.join(output_folder_list, f'train_dataset_map_small_short.list')
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    #print(nusc.list_scenes())

def main_val():
    ### new by wyx
    nuscenes_path = os.path.join(args.nus_folder, 'trainval')
    nus_version = 'v1.0-trainval'
    #dataset = make_nuscenes_dataset(nuscenes_path)
    nusc = NuScenes(version=nus_version, dataroot=nuscenes_path, verbose=True)
    daytime_scene_list, daytime_scenename_list = get_scene_daytime_list(nusc, range=(700,850))

    # rand select
    select_idx = np.random.choice(len(daytime_scene_list), 6, replace=False).astype(int)

    scene_num = len(select_idx)
    print("val_num:", scene_num)
    dataset = []
    for id in tqdm(range(0,scene_num)):
        scene_token = daytime_scene_list[select_idx[id]]
        scene_name = daytime_scenename_list[select_idx[id]]
        dataset = make_nus_localmap_dataset(nusc, scene_name, scene_token, dataset)

    output_file = os.path.join(output_folder_list, f'val_dataset_map_demo.list')
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    #print(nusc.list_scenes())

def test():
    ### new by wyx
    nuscenes_path = os.path.join(args.nus_folder, 'test')
    nus_version = 'v1.0-test'
    #dataset = make_nuscenes_dataset(nuscenes_path)
    nusc = NuScenes(version=nus_version, dataroot=nuscenes_path, verbose=True)
    daytime_scene_list, daytime_scenename_list = get_scene_daytime_list(nusc, range=(0,150))

    # rand select
    select_idx = np.random.choice(len(daytime_scene_list), 13, replace=False).astype(int)

    scene_num = len(select_idx)
    print("test_num:", scene_num)
    dataset = []
    for id in tqdm(range(0,scene_num)):
        scene_token = daytime_scene_list[select_idx[id]]
        scene_name = daytime_scenename_list[select_idx[id]]
        dataset = make_nus_localmap_dataset(nusc, scene_name, scene_token, dataset)

    output_file = os.path.join(output_folder_list, f'test_dataset_map_small_short.list')
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    #print(nusc.list_scenes())
    
        

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('--scene_num', default=0, help='Chooese Scene, Max 751')
parser.add_argument('--device', default='cuda',
                    help='device')
parser.add_argument('--voxel_size', default=0.1, type=float, help='Voxel Size')
#parser.add_argument('--start', default=0, help='Starting Frame')
#parser.add_argument('--end', default=100000, help='End Frame')
# parser.add_argument('--map', default=None, help='Use map file')
parser.add_argument('--nus_folder', default='/dataset/nuScenes',
                    help='Folder of the KITTI dataset')
parser.add_argument('--output_folder', default='/dataset/nus_processed_CMRNet')
parser.add_argument('--frame_skip', default=2, help='Frame Skip')
parser.add_argument('--max_translation', default=5, help='Max Trans')


args = parser.parse_args()
#scene_id = args.scene_num
output_folder_list = args.output_folder
output_folder = output_folder_list +"/sequences"
frame_skip = 2
max_translation= 5

if __name__ == '__main__':
    version = 0.1
    #print("in")
    #print(np.random.choice(700, size=70))
    main()
    main_val()
    test()
