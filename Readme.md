### I2PNet-dev
#### Large-Range Localization
##### KITTI
- kitti训练
```python
python train20v2learn_wandb_proj.py --gpu 3 --dataset kd_corr_nolidar --modelcfg config_proj_lidarcenter --network modellearn_proj_center --log_dir <LOG> --max_epoch 200 --debug --clip 10
```
- kitti测试
```python
python evaluation_proj.py --dataset kd_corr_nolidar --modelcfg config_proj_lidarcenter --network modellearn_proj_center --log_dir <LOG>
```

##### nuScenes
- nuscenes训练
```python
python train20v2learn_wandb_proj.py --gpu 3 --dataset nus_corr_nolidar --modelcfg config_proj_lidarcenter_nus --network modellearn_proj_center --log_dir <LOG> --max_epoch 200 --debug --clip 10
```
- nuscenes测试
```python
python evaluation_proj.py --dataset nus_corr_nolidar --modelcfg config_proj_lidarcenter_nus --network modellearn_proj_center --log_dir <LOG>
```

##### 最终指标计算
```python
python evaluation_analysis.py --log_dir <LOG> --target <metrics_name_in_'info_test'>
```

#### Small-Range Localization
- kitti训练
```python
python train20v2learn_wandb.py --gpu 1 --dataset kd_cmr_snr --modelcfg config_lidarcenter --log_dir <LOG> --debug --clip 10
```
- kitti测试
```python
python evaluation_cmr.py --dataset kd_cmr_snr --log_dir <LOG> --gpu 1
```

##### nuScenes
- nuscenes训练
```python
python train20v2learn_wandb.py --gpu 2 --dataset nus_cmr_snr --modelcfg config_lidarcenter --log_dir <LOG> --debug --clip 10
```
- nuscenes测试
```python
python evaluation_cmr.py --dataset nus_cmr_snr --log_dir <LOG> --gpu 1
```

##### 最终指标计算
```python
python evaluation_cmrresult.py --log_dir <LOG>
```

#### Online Calib

#### TODO
##### 在线标定代码
##### 数据预处理说明