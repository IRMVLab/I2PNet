import torch
import visibility
import numpy as np
import cv2


def checkvalid(depth_map):
    return torch.eq(depth_map, 1000.)


def pixel_depth(img, depth):
    # img np.ndarray depth torch.tensor BGR
    # depth include nonvalid
    valid_mask = torch.not_equal(depth, 1000.).cpu().numpy()
    depth_color = depth.cpu().numpy() * valid_mask
    # print(depth_color[valid_mask].min(),depth_color[valid_mask].max())
    depth_color = (depth_color - depth_color[valid_mask].min()) / (
            depth_color[valid_mask].max() - depth_color[valid_mask].min())

    depth_color = (depth_color * 90).astype(np.uint8)

    # exit(0)
    depth_color_hsv = np.full_like(img, 255)
    depth_color_hsv[:, :, 0] = depth_color
    depth_color_bgr = cv2.cvtColor(depth_color_hsv, cv2.COLOR_HSV2BGR)
    depth_color_bgr[np.logical_not(valid_mask), :] = img[np.logical_not(valid_mask), :]

    return depth_color_bgr


def depth_image(lidar_uv, lidar_z, img_size, depth_map=None):
    """
    automatically choose the min z to project on the depth map
    Args:
        lidar_uv: [N,2] [u,v] (torch.int)
        lidar_z:  [N,] (You should preserve it's all >0)
        img_size: [H,W]
    Returns:
        depth_map [H,W]
    """
    height, width = img_size
    # assert height % 32 == 0 and width % 32 == 0
    size = lidar_uv.shape[0]
    device = lidar_uv.device
    if depth_map is None:
        depth_map = torch.full((height, width), 1000., dtype=torch.float32, device=device)
    visibility.depth_image(lidar_uv, lidar_z, depth_map, size, width, height)
    # depth_map[torch.eq(depth_map, 1000.)] = 0.
    return depth_map


def depth_image_index(lidar_uv, lidar_z, img_size, depth_map=None):
    """
    automatically choose the min z to project on the depth map
    Args:
        lidar_uv: [N,2] [u,v] (torch.int)
        lidar_z:  [N,] (You should preserve it's all >0)
        img_size: [H,W]
    Returns:
        depth_map [H,W]
    """
    height, width = img_size
    device = lidar_uv.device
    lidar_z, rank = torch.sort(lidar_z, descending=True)  # (N,) (N,)
    if depth_map is None:
        depth_map = torch.full((height, width), 1000., dtype=torch.float32, device=device)
    lidar_u = torch.gather(lidar_uv[:, 0], 0, index=rank).long()
    lidar_v = torch.gather(lidar_uv[:, 1], 0, index=rank).long()

    idx_map = torch.full((height, width), -1, dtype=torch.long, device=device)

    depth_map[lidar_v, lidar_u] = lidar_z
    idx_map[lidar_v, lidar_u] = rank

    return depth_map, idx_map


def visibility2(origin_depth_map,
                intrinsic,
                img_size,
                output_depth_map=None,
                threshold=3.,
                kernel=5
                ):
    """

    Args:
        origin_depth_map:
        intrinsic: [4,] [fx,fy,cx,cy]
        img_size: [H,W]
        threshold: the cos sum in the four submatrix
        kernel: submatrix size [K*K]*4
    Returns:

    """
    # valid mask
    valid_mask = torch.eq(origin_depth_map, 1000.)
    origin_depth_map[valid_mask] = 0.
    height, width = img_size
    # assert height % 32 == 0 and width % 32 == 0
    if intrinsic.shape != (4,):  # (3,3)
        assert intrinsic.shape == (3, 3)
        intrinsic_3x3 = intrinsic
        intrinsic = torch.empty((4,), dtype=torch.float32, device=intrinsic_3x3.device)
        intrinsic[0] = intrinsic_3x3[0, 0]
        intrinsic[1] = intrinsic_3x3[1, 1]
        intrinsic[2] = intrinsic_3x3[0, 2]
        intrinsic[3] = intrinsic_3x3[1, 2]
    if output_depth_map is None:
        output_depth_map = torch.full_like(origin_depth_map, 1000.)
    visibility.visibility2(origin_depth_map, intrinsic, output_depth_map,
                           width, height, threshold, kernel)
    output_depth_map[valid_mask] = 1000.
    return output_depth_map


def reproject_depth_map_to_PC(depth_map, intrinsic):
    h, w = depth_map.shape
    intrinsic_inv = torch.inverse(intrinsic)
    grid_id_h = torch.arange(0, h, device=depth_map.device).view(1, h).repeat(w, 1).float()
    grid_id_w = torch.arange(0, w, device=depth_map.device).view(w, 1).repeat(1, h).float()
    grid_id_1 = torch.ones_like(grid_id_w)
    grid_id = torch.stack([grid_id_w, grid_id_h, grid_id_1], dim=-1).permute(2, 1, 0).reshape(3, -1)
    xy1 = torch.mm(intrinsic_inv, grid_id).permute(1, 0)  # 3,H*W

    xyz = xy1 * (depth_map.view(-1, 1))

    valid_mask = torch.not_equal(depth_map, 1000.).view(-1)
    reproject_xyz = xyz[valid_mask]

    return reproject_xyz


def filter_point(pc_np, cam_intrinsic, h, w, pad_h, pad_w, init_extrinsic):
    """
    Args:
        pc: np.array [3,N]
        cam_intrinsic: [3,3]
        h: img_height
        w: img_width
        pad_h: pad_h
        pad_w: pad_w
        init_extrinsic: [3,4]
    """
    device = torch.device("cuda:0")
    Pc = init_extrinsic

    # padding operation
    cam_intrinsic = cam_intrinsic.copy()

    cam_intrinsic[0, 2] += pad_w
    cam_intrinsic[1, 2] += pad_h

    h = h + 2 * pad_h
    w = w + 2 * pad_w

    pc_np_trans = Pc[:3, :3] @ pc_np + Pc[:3, 3][:, None]

    pc_np_cam = cam_intrinsic @ pc_np_trans

    pc_np_z = pc_np_cam[2:, :]
    pc_np_uv = pc_np_cam[:2, :] / (pc_np_z + 1e-10)

    pc_origin_idx = np.arange(pc_np_cam.shape[1])

    pc_np_uv = pc_np_uv.astype(np.int_)
    pc_fore_mask = pc_np_z[0] > 0
    pc_fore_insidey = np.logical_and(pc_np_uv[1] >= 0, pc_np_uv[1] < h)
    pc_fore_insidex = np.logical_and(pc_np_uv[0] >= 0, pc_np_uv[0] < w)
    pc_fore_inside = np.logical_and(pc_fore_insidey, pc_fore_insidex)
    pc_fore_mask = np.logical_and(pc_fore_mask, pc_fore_inside)
    pc_np_uv = pc_np_uv[:, pc_fore_mask]
    pc_np_z = pc_np_z[:, pc_fore_mask]
    pc_origin_idx = pc_origin_idx[pc_fore_mask]

    lidar_uv = torch.from_numpy(pc_np_uv.T).to(device).int()
    lidar_depth = torch.from_numpy(pc_np_z[0]).to(device).float()
    cam_intrinsic = torch.from_numpy(cam_intrinsic).to(device).float()

    depth_map, idx_map = depth_image_index(lidar_uv, lidar_depth, (h, w))

    new_depth_map = visibility2(depth_map, cam_intrinsic, (h, w))

    idx_map = idx_map[torch.not_equal(new_depth_map, 1000)]  # N'

    return pc_origin_idx[idx_map.cpu().numpy()], pc_np_trans
