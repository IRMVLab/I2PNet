import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
try:
    from moviepy.editor import *
except:
    print("no moviepy")
import imageio
import sys

import src.utils as utils


def show_img(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.save("projection.png")
    plt.show()


def show_pcl(pcl):
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    velo_range = range(0, pcl.shape[0], 100)
    ax.scatter(pcl[velo_range, 0],
               pcl[velo_range, 1],
               pcl[velo_range, 2],
               c=pcl[velo_range, 3],
               cmap='gray')
    plt.show()


def get_projected_img(pts, dist, img,dis_range=90):
    if len(pts)==0:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    dist_norm = utils.max_normalize_pts(dist)*90
    #
    # print(np.histogram(dist))
    # _,bins = np.histogram(dist,bins=90)
    # dist_norm = np.clip(np.abs(dist.reshape(-1,1)-bins.reshape(1,-1)).argmin(-1),0,89)

    color = np.full([1,dist_norm.shape[0],3],255,np.uint8)
    color[:,:,0] = dist_norm.astype(np.uint8)
    color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB).reshape(-1, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(pts.shape[0]):
        cv2.circle(img, (int(pts[i, 0]), int(pts[i, 1])), radius=1, color=
            (int(color[i,0]),int(color[i,1]),int(color[i,2])), thickness=-1)

    # projection = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return img


def get_lidar_img(pts, dist, img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = hsv_img > 0
    hsv_img[mask] = 0

    dist_norm = utils.max_normalize_pts(dist)*90

    for i in range(pts.shape[0]):
        cv2.circle(hsv_img, (int(pts[i, 0]), int(pts[i, 1])), radius=2, color=(
            int(dist_norm[i]), 255, 255), thickness=-1)

    projection = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    # projection = cv2.cvtColor(hsv_img, cv2.COLOR_RGB2GRAY)

    return projection


def show_projection(pts, dist, img):
    projection = get_projected_img(pts, dist, img)
    show_img(projection)

# def write_to_video(load_data, h_fov, v_fov, save_path=Path('media')):
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     img = load_data.load_image(0)
#     vid_size = (img.shape[1], img.shape[0])
#     save_path = str(save_path/'projection')
#     vid = cv2.VideoWriter(save_path + '.avi', fourcc, 25.0, vid_size)

#     num_imgs = len(list(load_data.img_path.glob('*')))

#     print('Writing video to {}'.format(save_path))
#     for i in tqdm(range(num_imgs)):
#         pcl_uv, _, pcl_z, img = load_data.get_projected_pts(i, h_fov, v_fov)
#         projection = get_projected_img(pcl_uv, pcl_z, img)[:, :, ::-1]
#         vid.write(projection)

#     convert_to_gif(save_path)


def convert_to_gif(save_path):
    """https://gist.github.com/michaelosthege/cd3e0c3c556b70a79deba6855deb2cc8"""
    input_path = save_path + '.avi'
    output_path = save_path + '.gif'

    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()['fps']

    print('Writing GIF to {}'.format(output_path))

    writer = imageio.get_writer(output_path, fps=fps)
    for i, im in enumerate(reader):
        sys.stdout.write('\rframe {0}'.format(i))
        sys.stdout.flush()
        writer.append_data(im)

    print('\r\nFinalizing...')
    writer.close()
    print('Done.')
