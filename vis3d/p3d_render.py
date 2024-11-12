import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    FoVPerspectiveCameras,
    look_at_view_transform,
    PerspectiveCameras
)
import numpy as np
import torch.nn as nn
from pytorch3d.renderer.compositing import alpha_composite



def _add_background_img_to_images(pix_idxs, images, im):
    # Initialize background mask
    background_mask = pix_idxs[:, 0] < 0  # (N, H, W)

    B = background_mask.shape[0]

    # permute so that features are the last dimension for masked_scatter to work
    images_rgb = images.permute(0, 2, 3, 1)[..., :3]  # N,H,W,C
    im_view = (im[None].repeat(B, 1, 1, 1)).permute(0, 2, 3, 1)
    images_rgb[background_mask] = im_view[background_mask]

    return images


class MyCompositor(nn.Module):
    """
    Accumulate points using alpha compositing.
    """

    def __init__(self, im):
        super().__init__()
        self.im = im.float()  # tensor /255. H,W,C RGB

    def forward(self, fragments, alphas, ptclds, **kwargs) -> torch.Tensor:
        images = alpha_composite(fragments, alphas, ptclds)

        # images are of shape (N, C, H, W)
        # check for background color & feature size C (C=4 indicates rgba)
        # im_back = _add_background_color_to_images(fragments, images, background_color)
        images = _add_background_img_to_images(fragments, images, self.im)
        return images


class PointCloudRender:
    def __init__(self, R, T, device="cuda:0", radius=0.005, im=None, height=720, width=1280):
        self.device = torch.device(device)

        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T,
                                        )

        tanHalfFov = np.tan((np.pi / 3 / 2))
        max_y = tanHalfFov * 1.
        min_y = -max_y
        max_x = max_y * 1.
        znear = 1.
        zfar = 100.
        min_x = -max_x
        K = np.eye(3)
        K[0, 0] = 2.0 * znear / (max_x - min_x)
        K[1, 1] = 2.0 * znear / (max_y - min_y)
        K[0, 2] = width / 2
        K[1, 2] = height / 2
        # K[2, 2] = zfar / (zfar - znear)
        # K[2, 3] = -(zfar * znear) / (zfar - znear)

        self.K = K
        self.R = R.clone().numpy()[0]
        self.R[:, 0] = -self.R[:, 0]
        self.R[:, 1] = -self.R[:, 1]
        self.T = T.numpy()

        raster_settings = PointsRasterizationSettings(
            image_size=(height, width),
            radius=radius,
            points_per_pixel=10,
            bin_size=0
        )

        self.renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
            compositor=NormWeightedCompositor(background_color=(1, 1, 1)) if im is None else \
                MyCompositor(torch.from_numpy(im[:, :, ::-1] / 255.)
                             .to(self.device).permute(2, 0, 1))

        )

        self.h = height
        self.w = width

    def rendering(self, points, colors, nofilter=False):
        if not nofilter:
            pts_c = points[0] @ self.R + self.T
            u = (pts_c[:, 0] / pts_c[:, 2] * self.K[0, 0]) + self.K[0, 2]
            v = (pts_c[:, 1] / pts_c[:, 2] * self.K[1, 1]) + self.K[1, 2]
            z = pts_c[:, 2]

            mask = (u > 0) & (u < self.w) & (v > 0) & (v < self.h) & (z > 0.1)

            verts = [torch.Tensor(pts[mask]).to(self.device) for pts in points]
            rgb = [torch.Tensor(crs[mask]).to(self.device) for crs in colors]
        else:
            verts = [torch.Tensor(pts).to(self.device) for pts in points]
            rgb = [torch.Tensor(crs).to(self.device) for crs in colors]
        point_cloud = Pointclouds(verts, features=rgb)

        images = self.renderer(point_cloud).cpu().numpy()

        ims = [(images[i, ..., :3][..., ::-1] * 255.).astype(np.uint8) \
               for i in range(len(points))]

        return ims


class PointCloudCameraRender:
    def __init__(self, device="cuda:0"):
        self.device = device

    def set_camera(self, K, extrinsic, im):
        c2w = extrinsic
        c2w = torch.Tensor(c2w)

        c2w[:, 0] = -c2w[:, 0]
        c2w[:, 1] = -c2w[:, 1]
        R_row = c2w[:3, :3]

        H, W, _ = im.shape
        device = self.device

        focalx = K[0, 0]
        focaly = K[1, 1]
        principalx = K[0, 2]
        principaly = K[1, 2]

        s = min(H, W)

        f_ndcx = focalx * 2.0 / (s - 1)
        f_ndcy = focaly * 2.0 / (s - 1)
        p_ndcx = - (principalx - (W - 1) / 2.0) * 2.0 / (s - 1)
        p_ndcy = - (principaly - (H - 1) / 2.0) * 2.0 / (s - 1)

        self.f_ndc = torch.Tensor([f_ndcx, f_ndcy]).unsqueeze(0).to(device)
        self.p_ndc = torch.Tensor([p_ndcx, p_ndcy]).unsqueeze(0).to(device)

        self.camera = None

        self.width = W
        self.height = H

        self.raster_settings = PointsRasterizationSettings(
            image_size=(self.height, self.width),
            radius=0.005,
            points_per_pixel=10,
            bin_size=0
        )

        w2c = c2w.inverse().to(self.device)

        # 注意pytorch3d的相机需要的T是w2c里的T， R是w2c里的R.T，即c2w里的R（旋转矩阵为正交矩阵）
        T = w2c[:3, -1]

        R_row = R_row.unsqueeze(0).to(self.device)
        T = T.unsqueeze(0).to(self.device)

        cameras = PerspectiveCameras(self.f_ndc, self.p_ndc,
                                     R_row, T, device=self.device)

        self.render = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=cameras,
                                        raster_settings=self.raster_settings),
            compositor=MyCompositor(torch.from_numpy(im[:, :, ::-1] / 255.)
                                    .to(self.device).permute(2, 0, 1))
        )

    def rendering(self, points, colors):
        verts = [torch.Tensor(pts).to(self.device) for pts in points]
        rgb = [torch.Tensor(crs).to(self.device) for crs in colors]
        point_cloud = Pointclouds(verts, features=rgb)

        images = self.render(point_cloud).cpu().numpy()

        ims = [(images[i, ..., :3][..., ::-1] * 255.).astype(np.uint8) \
               for i in range(len(points))]

        return ims


if __name__ == '__main__':
    import cv2
    import open3d as o3d

    imr = cv2.imread(r"003265/SFCNet_pred_sem_003265.png")
    imr = cv2.resize(imr, (1280, 720))

    ply = o3d.io.read_point_cloud(r"003265/gt_003265.ply")

    render = PointCloudRender()

    # render = PointCloudCameraRender(np.array([
    #     600, 0, 1280 / 2,
    #     0, 600, 720 / 2,
    #     0, 0, 1
    # ]).reshape(3, 3), 720, 1280)
    # T = np.eye(4)
    # T[:3, :3] = np.array(
    #     [[0, 0, 1],
    #      [-1, 0, 0],
    #      [0, -1, 0]]
    # )
    # render.set_camera(T, imr)

    im = render.rendering([np.asarray(ply.points)],
                          [np.asarray(ply.colors)])
    render = PointCloudRender(radius=0.01, im=im[0])

    pts = np.zeros((100, 3))
    pts[:, 0] = -np.arange(0, 50, 50 / 100)

    im = render.rendering([pts],
                          [np.asarray([[0, 0, 1]]).repeat(100, 0)])

    cv2.imshow("out", im[0])
    cv2.waitKey(0)
