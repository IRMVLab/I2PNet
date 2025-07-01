from pathlib import Path
import pickle as pkl
import numpy as np
import os

params = {
    'train1_val': {
        'date': "2011_09_26",
        'num': 24000,  # 22000,2000 for validation
        'drives': [1, 2, 9, 11, 13, 14, 15,
                   17, 18, 19, 20, 22, 23,
                   27, 28, 29, 32, 35, 36, 39,
                   46, 48, 51, 52, 56, 57,
                   59, 60, 61, 64, 79,
                   84, 86, 87, 91, 93, 95,
                   96, 101, 104, 106, 113,
                   117],  # except [5,70]
        'rot': 15.,
        'trans': 0.2
    },
    'train2': {
        'date': "2011_09_26",
        'num': 4000,
        'drives': [1, 2, 9, 11, 13, 14, 15,
                   17, 18, 19, 20, 22, 23,
                   27, 28, 29, 32, 35, 36, 39,
                   46, 48, 51, 52, 56, 57,
                   59, 60, 61, 64, 79,
                   84, 86, 87, 91, 93, 95,
                   96, 101, 104, 106, 113,
                   117],  # except [5,70]
        'rot': 0.,
        'trans': 0.3
    },
    'train3': {
        'date': "2011_09_26",
        'num': 4000,
        'drives': [1, 2, 9, 11, 13, 14, 15,
                   17, 18, 19, 20, 22, 23,
                   27, 28, 29, 32, 35, 36, 39,
                   46, 48, 51, 52, 56, 57,
                   59, 60, 61, 64, 79,
                   84, 86, 87, 91, 93, 95,
                   96, 101, 104, 106, 113,
                   117],  # except [5,70]
        'rot': 20,
        'trans': 0.
    },
    'train_val_ex': {
        'date': "2011_10_03",
        'num': 2000,
        'drives': [27],  # except [5,70]
        'rot': 2.,
        'trans': 0.3
    },
    'T1': {
        'date': "2011_09_26",
        'num': 2000,
        'drives': [5, 70],  # except [5,70]
        'rot': 15.,
        'trans': 0.2
    },
    'T2a': {
        'date': "2011_09_26",
        'num': 2000,
        'drives': [1, 2, 9, 11, 13, 14, 15,
                   17, 18, 19, 20, 22, 23,
                   27, 28, 29, 32, 35, 36, 39,
                   46, 48, 51, 52, 56, 57,
                   59, 60, 61, 64, 79,
                   84, 86, 87, 91, 93, 95,
                   96, 101, 104, 106, 113,
                   117],  # except [5,70]
        'rot': 10.,
        'trans': 0.2
    },
    'T2b': {
        'date': "2011_09_26",
        'num': 2000,
        'drives': [5, 70],  # except [5,70]
        'rot': 10.,
        'trans': 0.2
    },
    'T3': {
        'date': "2011_10_03",
        'num': 2000,
        'drives': [27],  # except [5,70]
        'rot': 2.,
        'trans': 0.3
    },
}


def make_dataset():
    base_path = '/dataset/kitti/raw/'
    save_dir = '/data/I2PNet/rgg_datas'
    os.makedirs(save_dir, exist_ok=True)
    for key in params.keys():
        if "ex" in key:
            continue
        drives = params[key]["drives"]
        date = params[key]["date"]
        N = params[key]["num"]
        rot_range = params[key]["rot"]
        trans_range = params[key]["trans"]
        img_path = []
        lidar_path = []
        for drive in drives:
            cur_img_path = Path(
                base_path) / date / (date + '_drive_{:04d}_sync'.format(drive)) / 'image_02' / 'data'
            cur_lidar_path = Path(
                base_path) / date / (date + '_drive_{:04d}_sync'.format(drive)) / 'velodyne_points' / 'data'
            for file_name in sorted(cur_img_path.glob('*.png')):
                img_path.append(str(file_name))
            for file_name in sorted(cur_lidar_path.glob('*.bin')):
                lidar_path.append(str(file_name))

        M = len(img_path)
        print(M, len(lidar_path))
        if M >= N:
            choice_idx = np.random.choice(M, N, replace=False)
        else:
            fix_idx = np.random.permutation(M)
            while fix_idx.shape[0] + M < N:
                fix_idx = np.concatenate([fix_idx, np.random.permutation(M)], axis=0)
            random_idx = np.random.choice(M, N - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate([fix_idx, random_idx], axis=0)
        img_path = np.array(img_path)[choice_idx]
        lidar_path = np.array(lidar_path)[choice_idx]
        rot = np.random.random((N, 3)) * (2 * rot_range) - rot_range
        trans = np.random.random((N, 3)) * (2 * trans_range) - trans_range
        data = {
            'img': img_path,
            'lidar': lidar_path,
            'rot': rot,
            'trans': trans,
        }
        if 'val' in key:
            data["train_split"] = np.random.choice(N, 22000, replace=False)
        with open(os.path.join(save_dir, f"rgg_data_{key}.pkl"), 'wb') as f:
            pkl.dump(data, f)


def make_dataset_extra():
    base_path = '/dataset/kitti/raw/'
    save_dir = '/data/I2PNet/rgg_datas'
    os.makedirs(save_dir, exist_ok=True)
    for key in params.keys():
        if "ex" not in key:
            continue
        drives = params[key]["drives"]
        date = params[key]["date"]
        N = params[key]["num"]
        rot_range = params[key]["rot"]
        trans_range = params[key]["trans"]
        img_path = []
        lidar_path = []
        for drive in drives:
            cur_img_path = Path(
                base_path) / date / (date + '_drive_{:04d}_sync'.format(drive)) / 'image_02' / 'data'
            cur_lidar_path = Path(
                base_path) / date / (date + '_drive_{:04d}_sync'.format(drive)) / 'velodyne_points' / 'data'
            for file_name in sorted(cur_img_path.glob('*.png')):
                img_path.append(str(file_name))
            for file_name in sorted(cur_lidar_path.glob('*.bin')):
                lidar_path.append(str(file_name))

        M = len(img_path)
        print(M, len(lidar_path))
        if M >= N:
            choice_idx = np.random.choice(M, N, replace=False)
        else:
            fix_idx = np.random.permutation(M)
            while fix_idx.shape[0] + M < N:
                fix_idx = np.concatenate([fix_idx, np.random.permutation(M)], axis=0)
            random_idx = np.random.choice(M, N - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate([fix_idx, random_idx], axis=0)
        img_path = np.array(img_path)[choice_idx]
        lidar_path = np.array(lidar_path)[choice_idx]
        rot = np.random.random((N, 3)) * (2 * rot_range) - rot_range
        trans = np.random.random((N, 3)) * (2 * trans_range) - trans_range
        data = {
            'img': img_path,
            'lidar': lidar_path,
            'rot': rot,
            'trans': trans,
            "train_split": np.random.choice(N, 1800, replace=False)
        }
        with open(os.path.join(save_dir, f"rgg_data_{key}.pkl"), 'wb') as f:
            pkl.dump(data, f)


if __name__ == '__main__':
    make_dataset()
    make_dataset_extra()
