from pathlib import Path

SMALL_SET_2011_09_26 = {
    'base_path': Path('/dataset/kitti/raw/'),
    'date': '2011_09_26',
    'drives': [5],
}

TRAIN_SET_2011_09_26 = {
    # 'base_path': Path('data')/'KITTI',
    'base_path': Path('/dataset/kitti/raw/'),
    'date': '2011_09_26',
    'drives': [1, 2, 9, 11, 13, 14, 15,
               17, 18, 19, 20, 22, 23,
               27, 28, 29, 32, 35, 36, 39,
               46, 48, 51, 52, 56, 57,
               59, 60, 61, 64, 79,
               84, 86, 87, 91, 93, 95,
               96, 101, 104, 106, 113,
               117],
}

# for validation
TEST_SET_2011_09_26 = {
    'base_path': Path('/dataset/kitti/raw/'),
    'date': '2011_09_26',
    'drives': [5, 70],
}

TEST_SET_2011_09_30 = {
    'base_path': Path('/dataset/kitti/raw/'),
    'date': '2011_09_30',
    'drives': [28],
}


# train parameters
class KITTI_ONLINE_CALIB:
    dataset_params = {
        'base_path': TRAIN_SET_2011_09_26['base_path'],
        'date': TRAIN_SET_2011_09_26['date'],
        'drives': TRAIN_SET_2011_09_26['drives'],
        'd_rot': 10,
        'd_trans': 1.0,
        'fixed_decalib': False,
        'resize_w': 1216,
        'resize_h': 352,
    }
    # 'd_rot': 20,
    # 'd_trans': 1.5,
    dataset_params_valid1 = {
        'base_path': TEST_SET_2011_09_26['base_path'],
        'date': TEST_SET_2011_09_26['date'],
        'drives': TEST_SET_2011_09_26['drives'],
        'd_rot': 10,
        'd_trans': 1.0,
        'fixed_decalib': True,
        'resize_w': 1216,
        'resize_h': 352,
    }
    dataset_params_valid2 = {
        'base_path': TEST_SET_2011_09_26['base_path'],
        'date': TEST_SET_2011_09_26['date'],
        'drives': TEST_SET_2011_09_26['drives'],
        'd_rot': 2,
        'd_trans': 0.2,
        'fixed_decalib': True,
        'resize_w': 1216,
        'resize_h': 352,
    }
    dataset_params_valid3 = {
        'base_path': TEST_SET_2011_09_26['base_path'],
        'date': TEST_SET_2011_09_26['date'],
        'drives': TEST_SET_2011_09_26['drives'],
        'd_rot': 5,
        'd_trans': 0.5,
        'fixed_decalib': True,
        'resize_w': 1216,
        'resize_h': 352,
    }
    dataset_params_test = {  # d_rot and d_trans
        'base_path': TEST_SET_2011_09_30['base_path'],
        'date': TEST_SET_2011_09_30['date'],
        'drives': TEST_SET_2011_09_30['drives'],
        # 'd_rot': 5,
        # 'd_trans': 0.5,
        'fixed_decalib': True,
        'resize_w': 1216,
        'resize_h': 352,
    }


class KITTI_ODOMETRY:
    ## training data
    dataset_params = {
        'root_path': '/dataset',
        'mode': 'train',
        'd_rot': 10,
        'd_trans': 1.0,
        'fixed_decalib': False,
        'resize_w': 1216,
        'resize_h': 352,
    }
    # 'd_rot': 20,
    # 'd_trans': 1.5,
    ##validation data
    dataset_params_valid3 = {
        'root_path': '/dataset',
        'mode': 'test',
        'd_rot': 10,
        'd_trans': 1.0,
        'fixed_decalib': False,
        'resize_w': 1216,
        'resize_h': 352,
    }
    dataset_params_test = {
        'root_path': '/dataset',
        'mode': 'val',
        'd_rot': -1,
        'd_trans': -1,
        'fixed_decalib': True,
        'resize_w': 1216,
        'resize_h': 352,
    }


class NUSCENES:
    dataset_params = {
        'root_path': '/dataset/nuScenes',
        'mode': 'train',
        'd_rot': 10,
        'd_trans': 1.0,
        'fixed_decalib': False,
        'resize_w': 1216,
        'resize_h': 352,
    }
    dataset_params_valid3 = {
        'root_path': '/dataset/nuScenes',
        'mode': 'val',
        'd_rot': 10,
        'd_trans': 1.0,
        'fixed_decalib': False,
        'resize_w': 1216,
        'resize_h': 352,
    }
    dataset_params_test = {
        'root_path': '/dataset/nuScenes',
        'mode': 'test',
        'd_rot': -1,
        'd_trans': -1,
        'fixed_decalib': False,
        'resize_w': 1216,
        'resize_h': 352,
    }

class REAL_DATA:
    dataset_params = {
        'root_path': '/dataset/real_localize',
        'mode': 'train',
        'd_rot': 10,
        'd_trans': 1.0,
        'fixed_decalib': False,
        'resize_w': 1216,
        'resize_h': 352,
    }
    dataset_params_valid3 = {
        'root_path': '/dataset/real_localize',
        'mode': 'test',
        'd_rot': 10,
        'd_trans': 1.0,
        'fixed_decalib': False,
        'resize_w': 1216,
        'resize_h': 352,
    }
    dataset_params_test = {
        'root_path': '/dataset/real_localize',
        'mode': 'val',
        'd_rot': -1,
        'd_trans': -1,
        'fixed_decalib': False,
        'resize_w': 1216,
        'resize_h': 352,
    }



