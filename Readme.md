<h1 align="center"> I2PNet: End-to-end 2D-3D Registration between Image and LiDAR Point Cloud for Vehicle Localization</h1>   
  <h3 align="center">TRO 2025</h3>

[![ieee xplore](https://img.shields.io/badge/IEEE-Xplore-blue)](https://ieeexplore.ieee.org/document/11078010)
[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2306.11346)



Guangming Wang²*, Yu Zheng¹*, Yuxuan Wu¹, Yanfeng Guo⁴, Zhe Liu¹, Yixiang Zhu⁵, Wolfram Burgard³, Hesheng Wang¹†

² University of Cambridge, ¹ Shanghai Jiao Tong University, ³ University of Technology Nuremberg, ⁴ University of California, ⁵ Nanyang Technological University   


## Abstract
Robot localization using a built map is essential for a variety of tasks including accurate navigation and mobile manipulation. A popular approach to robot localization is based on image-to-point cloud registration, which combines illumination-invariant LiDAR-based mapping with economical image-based localization. However, the recent works for image-to-point cloud registration either divide the registration into separate modules or project the point cloud to the depth image to register the RGB and depth images. n this paper, we present I2PNet, a novel end-to-end 2D-3D registration network, which directly registers the raw 3D point cloud with the 2D RGB image using differential modules with a united target. The 2D-3D cost volume module for differential 2D-3D association is proposed to bridge feature extraction and pose regression. The soft point-to-pixel correspondence is implicitly constructed on the intrinsic-independent normalized plane in the 2D-3D cost volume module. Moreover, we introduce an outlier mask prediction module to filter the outliers in the 2D-3D association before pose regression. Furthermore, we propose the coarse-to-fine 2D-3D registration architecture to increase localization accuracy.  Extensive localization experiments are conducted on the KITTI, nuScenes, M2DGR, Argoverse, Waymo, and Lyft5 datasets. The results demonstrate that I2PNet outperforms the state-of-the-art by a large margin and has a higher efficiency than the previous works. Moreover, we extend the application of I2PNet to the camera-LiDAR online calibration and demonstrate that I2PNet outperforms recent approaches on the online calibration task.

## Demo Video
 [![Demo Video](https://img.youtube.com/vi/l2A6temRAg8/0.jpg)](https://www.youtube.com/watch?v=l2A6temRAg8)



## Environment Requirements
install required packages
```bash
pip install -r requirements.txt
```
install pointnet2
```shell
cd pointnet2
python setup.py install
cd ../
```
install projection-aware operators
```shell
cd src/projectPN/fused_conv_select
python setup.py install
cd ../../../
```
## Data Preprocessing
### KITTI Preprocessing
Download [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) sequences 00, 03, 05, 06, 07, 08, 09 and 10. And process the data sequentially as follows: 

- For large-range localization, just run
  ```shell
  cd ./data_preprocess
  python kitti_pc_bin_to_npy_with_downsample_sn.py --src $your_kitti_data_path$ --dst $your_output_path$
  cd ../
  ```
  The output will be put in `/dataset/data_odometry_velodyne_deepi2p_new/data_odometry_velodyne_NWU/sequences/**/snr0.6`.

  **Note:** in our dataloader code, we read data from the folder `data_odometry_velodyne_deepi2p_new`, so you could define your output path as like `/dataset/data_odometry_velodyne_deepi2p_new`

- For small-range localization, just run
  ```shell
  cd ./data_preprocess/CMRNet_script
  python kitti_maps_cmr.py --sequence 00 --kitti_folder ./KITTI_ODOMETRY/
  cd ../
  ```
  **Note:** please uncomment the range constraint at “# y \in [-10,10] x \in [-5,15]” in the [kitti_maps_cmr.py](data_preprocess/CMRNet_script/kitti_maps_cmr.py) code, and comment out lines at “# y \in [-25,25] x \in [-10,100]” 
  
### NuScenes Preprocessing
Download [NuScenes](https://www.nuscenes.org/nuscenes) Full dataset (v1.0) Dataset. 

For **large range localization**, please download our filtered data list [here](https://drive.google.com/file/d/1OXgY8pp3vMfMLr5DsDVPRqHAKEdo-dMG/view?usp=sharing), and put them under the **nuScenes_datasplit** folder.

For **small range localization**, process the data sequentially as follows: 
```shell
python gen_maps_our.py \
    --voxel_size 0.1 \
    --nus_folder /dataset/nuScenes \
    --output_folder /dataset/nus_processed_CMRNet \
    --frame_skip 2 \
    --max_translation 5
```

## Large-Range Localization
### KITTI Dataset Training and evaluation
- Training
```shell
python train20v2learn_wandb_proj.py --gpu 3 --dataset kd_corr_nolidar --modelcfg config_proj_lidarcenter --network modellearn_proj_center --log_dir <LOG> --max_epoch 200 --debug --clip 10
```
- Evaluation
```shell
python evaluation_proj.py --dataset kd_corr_nolidar --modelcfg config_proj_lidarcenter --network modellearn_proj_center --log_dir <LOG>
```
For evaluation with multiple iterations (currently set to 6 iters):
```shell
python evaluation_proj.py --dataset kd_corr_nolidar --modelcfg config_proj_lidarcenter_iter --network modellearn_proj_center_iter --log_dir <LOG>
```

### nuScenes
- Training
```shell
python train20v2learn_wandb_proj.py --gpu 3 --dataset nus_corr_nolidar --modelcfg config_proj_lidarcenter_nus --network modellearn_proj_center --log_dir <LOG> --max_epoch 200 --debug --clip 10
```
- Evaluation
```shell
python evaluation_proj.py --dataset nus_corr_nolidar --modelcfg config_proj_lidarcenter_nus --network modellearn_proj_center --log_dir <LOG>
```

### Final Metrics
```shell
python evaluation_analysis.py --log_dir <LOG> --target <metrics_name_in_'info_test'>
```
### Cross Dataset Validation
For cross dataset evaluation, you could simply change the **dataset** and **modelcfg** args to the according dataset. For example, for evaluation from KITTI Odometry to NuScenes dataset, run:
```shell
python evaluation_proj.py --dataset nus_corr_nolidar --modelcfg config_proj_lidarcenter_nus --network modellearn_proj_center --log_dir <Your_KITTI_training_LOG_dir>
```

## Small-Range Localization
- Training
```shell
python train20v2learn_wandb.py --gpu 1 --dataset kd_cmr_snr --modelcfg config_lidarcenter --log_dir <LOG> --debug --clip 10
```
- Evaluation
```shell
python evaluation_cmr.py --dataset kd_cmr_snr --log_dir <LOG> --gpu 1
```

### nuScenes
- Training
```shell
python train20v2learn_wandb.py --gpu 2 --dataset nus_cmr_snr --modelcfg config_lidarcenter --log_dir <LOG> --debug --clip 10
```
- Evaluation
```shell
python evaluation_cmr.py --dataset nus_cmr_snr --log_dir <LOG> --gpu 1
```

### Final Metrics
```shell
python evaluation_cmrresult.py --log_dir <LOG>
```

# Citation

```
@ARTICLE{wang2023end,
  author={Wang, Guangming and Zheng, Yu and Wu, Yuxuan and Guo, Yanfeng and Liu, Zhe and Zhu, Yixiang and Burgard, Wolfram and Wang, Hesheng},
  journal={IEEE Transactions on Robotics}, 
  title={End-to-End 2D-3D Registration Between Image and LiDAR Point Cloud for Vehicle Localization}, 
  year={2025},
  volume={41},
  number={},
  pages={4643-4662},
  doi={10.1109/TRO.2025.3588454}
  }
```


# Acknowledgements
* Implementation of PointNet++ is based on [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).
* Implementation of data preprocessing inherited from [CMRNet](https://github.com/cattaneod/CMRNet).
* Implementation of projection-aware operators from [EfficientLO-Net](https://github.com/IRMVLab/EfficientLO-Net).