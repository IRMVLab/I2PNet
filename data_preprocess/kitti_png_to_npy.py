import os
import os.path
import numpy as np
import cv2


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str,default="/dataset/data_odometry_color")
    parser.add_argument('--dst', type=str,default="/dataset/data_odometry_velodyne_deepi2p_new")
    FLAGS = parser.parse_args()

    seq_list = range(0, 10)

    root_path = FLAGS.src
    dst_path = FLAGS.dst

    for seq in seq_list:
        img2_folder = os.path.join(root_path, 'sequences', '%02d' % seq, 'image_2')
        sample_num = round(len(os.listdir(img2_folder)))

        img2_folder_npy = os.path.join(dst_path, 'data_odometry_color_npy', 'sequences', '%02d' % seq, 'image_2')
    
        for i in range(sample_num):

            img2_path = os.path.join(img2_folder, '%06d.png' % i)

            img2 = cv2.imread(img2_path)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


            np.save(os.path.join(img2_folder_npy, '%06d.npy' % i), img2)

            