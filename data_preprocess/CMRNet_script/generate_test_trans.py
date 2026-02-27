import numpy as np
from pathlib import Path
import os
import pickle as pkl

#seed = int(os.environ["SEED"])
#seq = os.environ["SEQ"]
seed = 50000
seq = 10
thisfile_dir = str(Path(__file__).parent.resolve())
np.random.seed(seed)

if __name__ == '__main__':
    # scans = 0
    # for seq in [3,5,6,7,8,9,0]:
    #     with open(os.path.join(thisfile_dir, 'data', f'kitti-{seq:02d}.csv')) as f:
    #         lines = f.readlines()[1:]
    #         poses = [line.strip('\n').split(',')[1:] for line in lines]  # timestamp,x,y,z,qx,qy,qz,qw
    #     scans += len(poses)
    # print(scans*7/1024)
    # exit(0)
    
    dir = os.path.join(thisfile_dir, "test_set_trans")
    os.makedirs(dir, exist_ok=True)
    max_r = 10.
    max_t = 2.
    # max_r = 2.
    # max_t = 1.
    test_sequence = int(seq)
    filename = f'test_RT_seq{test_sequence:02d}_{max_r:.2f}_{max_t:.2f}_{seed}.pkl'

    with open(os.path.join(thisfile_dir, 'data', f'kitti-{test_sequence:02d}.csv')) as f:
        lines = f.readlines()[1:]
        poses = [line.strip('\n').split(',')[1:] for line in lines]  # timestamp,x,y,z,qx,qy,qz,qw
        #print(poses)
    scans = len(poses)

    test_errs = []
    print(scans)
    for i in range(scans):
        rotz = np.random.uniform(-max_r, max_r) * (np.pi / 180.0)
        roty = np.random.uniform(-max_r, max_r) * (np.pi / 180.0)
        rotx = np.random.uniform(-max_r, max_r) * (np.pi / 180.0)
        transl_x = np.random.uniform(-max_t, max_t)
        transl_y = np.random.uniform(-max_t, max_t)
        transl_z = np.random.uniform(-max_t, min(max_t, 1.))
        test_errs.append([rotx, roty, rotz, transl_x, transl_y, transl_z])

    with open(os.path.join(dir,filename),'wb') as f:
        pkl.dump(test_errs,f)
    print("save ", filename)

