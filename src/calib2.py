import numpy as np
import os

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def _load_calib_rigid(calib_path, filename):
    """Read a rigid transform calibration file as a numpy.array."""
    filepath = os.path.join(calib_path, filename)
    data = read_calib_file(filepath)
    return transform_from_rot_trans(data['R'], data['T'])


def calib(calib_path, velo_to_cam_file, cam_to_cam_file):
    # L -> C0
    Tr = _load_calib_rigid(calib_path, velo_to_cam_file)

    # Load and parse the cam-to-cam calibration data
    cam_to_cam_filepath = os.path.join(calib_path, cam_to_cam_file)
    filedata = read_calib_file(cam_to_cam_filepath)

    P2 = np.reshape(filedata['P_rect_02'], (3, 4))

    K = P2[:3, :3]

    P2 = np.linalg.inv(K) @ P2  # (3,4)

    R_rect_00 = np.eye(4)
    R_rect_00[0:3, 0:3] = np.reshape(filedata['R_rect_00'], (3, 3))

    T2 = P2 @ R_rect_00 @ Tr  # (3,4) L->C2

    return K,T2


class CALIB:
    CALIB_ROOT = "/dataset/kitti/raw"
    PATH_0926 = "2011_09_26"
    PATH_0930 = "2011_09_30"
    PATH_1003 = "2011_10_03"
    VELOCAM = "calib_velo_to_cam.txt"
    CAMCAM = "calib_cam_to_cam.txt"
    def __init__(self):
        self.K_0926,self.TR_0926 = \
        calib(os.path.join(CALIB.CALIB_ROOT,CALIB.PATH_0926),
              CALIB.VELOCAM,
              CALIB.CAMCAM)
        self.K_0930,self.TR_0930 = \
        calib(os.path.join(CALIB.CALIB_ROOT,CALIB.PATH_0930),
              CALIB.VELOCAM,
              CALIB.CAMCAM)
        self.K_1003,self.TR_1003 = \
        calib(os.path.join(CALIB.CALIB_ROOT,CALIB.PATH_1003),
              CALIB.VELOCAM,
              CALIB.CAMCAM)