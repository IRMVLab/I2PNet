from src.modules.MainModules import CostVolume, PoseHead
from datetime import datetime
from src.util.tracker import Timings


class I2PNetConfig:
    use_bn_p = True
    use_bn_input = True
    use_trans = True


    dataset_type = 1  # 0:kitti 1:nuscenes 2:real
    ######### rgb encoder ############
    rgb_encoder_channels = [
        # in_channel, channels for 3*3 conv, strides for Maxpooling
        (3, [16, 16, 16, 16, 32], [2, 1, 1, 1, 2]),
        (32, [32, 32, 32, 32, 64], [2, 1, 1, 1, 2]),
        (64, [64, 64, 64, 64, 128], [1, 1, 1, 1, 2])
    ]
    ######### lidar encoder ############
    stride_Hs = [2 ** (2 - dataset_type), 2, 2, 1]
    stride_Ws = [8, 2, 2, 2]

    rank = False

    # for debug
    debug = False

    debug_dict = None if not debug else {
        "global_valid_neighbor_num_downsample": [[0. for _ in range(5)], 0],
        "global_valid_neighbor_num_upsample": [[0 for _ in range(2)], 0],
        "global_valid_neighbor_num_cv": [[0 for _ in range(2)], 0],
        "global_point_sample": [],

    }

    debug_dir = "/data/I2PNet/new_ops_debug"

    debug_no_sample = False

    debug_path = datetime.now().strftime('%Y_%m_%d_') + '_'.join(datetime.now().strftime('%X').split(':')) + '.pkl'

    debug_storage = 3
    debug_count = 0

    debug_time = False
    debug_timing = Timings()
    #############

    down_conv_dis = [0.75, 3.0, 6.0, 12.0] # [100, 100, 100, 100]

    #init_H = 16 * 2 ** (2 - dataset_type)
    init_H = 21
    init_W = 1800

    if dataset_type == 0:
        fup = 2.0
        fdown = -24.8
    elif dataset_type == 1:
        #fup = 10.0
        #fdown = -30.
        fup = 2.0
        fdown = -24.8
    elif dataset_type == 2:
        fup = 15.
        fdown = -15.

    kernel_sizes = [[9, 15], [9, 15], [5, 9], [5, 9]]

    lidar_feature_size = 7
    using_intens = False
    raw_feat_point = True

    lidar_group_samples = [32, 16, 16, 16, 16]

    lidar_encoder_mlps_planA = [
        [8, 16, 32],
        [32, 32, 64],
        [64, 64, 128],
        [128, 128, 128],
        [128, 64, 64]  # set conv for cost volume
    ]

    lidar_encoder_mlps_planB = [
        [16, 16, 32],
        [32, 32, 64],
        [64, 64, 128],
        [128, 128, 256],
        [128, 64, 64]  # set conv for cost volume
    ]
    lidar_encoder_mlps = lidar_encoder_mlps_planB
    ####################################
    # cost volume
    cost_volume_dis = [4.5, 4.5]  # [4.5, 4.5]
    cost_volume_kernel_size = [[3, 5], [3, 5]]
    cost_volume_mlps = [  # mlp1 for pi features processing
        [128, 64, 64],
        # mlp2 (mlp21,mlp22 are the same) for generating weights in pi features and pp features
        [128, 64]]
    cost_volume_nsamples = [4,  # point searches the neighbors
                            [-1, 32]]  # all the pixel in level3
    # [32,32]] # point searches the corresponding image pixel (needs bigger sensing field)
    cost_volume_corr_func = CostVolume.CorrFunc.ELEMENTWISE_PRODUCT  # mean-std normalized and product
    backward_validation = [True, False]
    max_cost = False

    ####################################
    up_conv_dis = [9.0, 9.0]  # [9.0, 9.0]
    up_conv_kernel_size = [[5, 9], [5, 9]]
    setupconv_mlps = [[[128, 64], [64]],  # for mask upsampling
                      [[128, 64], [64]]]  # for embedding upsampling
    setupconv_nsamples = [8,  # for mask upsampling
                          8]  # for embedding upsampling
    ####################################
    flow_predictor_mlps = [[128, 64],  # l4 mask predictor
                           [128, 64],  # l3 refined EM predictor
                           [128, 64]]  # l3 mask
    #####################################
    # pose_head_mlps = [[[128,64],[128,64]] # l4 mlp1 mlp2
    #                     ,[[128,64],[128,64]]] # l3 mlp1 mlp2
    pose_head_mlps = [[[], []]  # l4 mlp1 mlp2
        , [[], []]]  # l3 mlp1 mlp2
    head_hidden_dim = 256
    rotation_quat_head_dim = 4
    transition_vec_head_dim = 3
    head_dropout_rate = 0.5
    head_corr_func = PoseHead.CorrFunc.CONCAT
    head_pos_embedding = False
    split_dp = False  # use one dp for the hidden layer
    max_head = False  # mask pick the maximum in the channel dimension
    #####################################
    # projection_mask
    mask_sigmoid = False

    # one_head_mask_eval = False  # one head mask using the first mlp dim of projection_mask_mlps
    ####################################
    # gt_proj

    ######################################
    sq_init = -2.5
    sx_init = 0.0
    # sx_init = -1.5
    l1_trans_loss = True
    pointwise_reproject_loss = False
    focal_mask_loss = True
    focal_gamma = 2

    ##########################################
    efgh = False
