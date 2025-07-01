# I2PNet: End-to-end 2D-3D Registration between Image and LiDAR Point Cloud for Vehicle Localization

The source code for "End-to-end 2D-3D Registration between Image and LiDAR Point Cloud for Vehicle Localization" is coming soon.

## Citation

```
@article{wang2023end,
  title={End-to-end 2D-3D Registration between Image and LiDAR Point Cloud for Vehicle Localization},
  author={Wang, Guangming and Zheng, Yu and Guo, Yanfeng and Liu, Zhe and Zhu, Yixiang and Burgard, Wolfram and Wang, Hesheng},
  journal={arXiv preprint arXiv:2306.11346},
  year={2023}
}
```

## Abstract

Robot localization using a previously built map is essential for a variety of tasks including highly accurate navigation and mobile manipulation. A popular approach to robot localization is based on image-to-point cloud registration, which combines illumination-invariant LiDAR-based mapping with economical image-based localization. However, the recent works for image-to-point cloud registration either divide the registration into separate modules or project the point cloud to the depth image to register the RGB and depth images. In this paper, we present I2PNet, a novel end-to-end 2D-3D registration network. I2PNet directly registers the raw 3D point cloud with the 2D RGB image using differential modules with a unique target. The 2D-3D cost volume module for differential 2D-3D association is proposed to bridge feature extraction and pose regression. 2D-3D cost volume module implicitly constructs the soft point-to-pixel correspondence on the intrinsic-independent normalized plane of the pinhole camera model. Moreover, we introduce an outlier mask prediction module to filter the outliers in the 2D-3D association before pose regression. Furthermore, we propose the coarse-to-fine 2D-3D registration architecture to increase localization accuracy. We conduct extensive localization experiments on the KITTI Odometry and nuScenes datasets. The results demonstrate that I2PNet outperforms the state-of-the-art by a large margin. In addition, I2PNet has a higher efficiency than the previous works and can perform the localization in real-time. Moreover, we extend the application of I2PNet to the camera-LiDAR online calibration and demonstrate that I2PNet outperforms recent approaches on the online calibration task.



