from src.modules.MainModules import CostVolume, PoseHead


class I2PNetConfig:
    ######### rgb encoder ############
    rgb_encoder_channels = [
        # in_channel, channels for 3*3 conv, strides for Maxpooling
        (3, [16, 16, 16, 16, 32], [2, 1, 1, 1, 2]),
        (32, [32, 32, 32, 32, 64], [2, 1, 1, 1, 2]),
        (64, [64, 64, 64, 64, 128], [1, 1, 1, 1, 2])
    ]
    ######### lidar encoder ############
    lidar_strides = [4, 4, 2, 2]

    lidar_voxel_size = 0.05

    lidar_feature_size = 3

    lidar_channels = [16,  # stem
                      32,  # stage1
                      64,  # stage2
                      128,  # stage3
                      256,  # stage4
                      64  # resample
                      ]

    ####################################
    # cost volume
    cross_cv = False
    sim_cv = False  # Hregnet
    saca_pre = False  # SA & CA
    backward_fc = False
    allcv = False
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
    up_channels = [[64, 64],  # for mask upsampling
                   [64, 64]]  # for embedding upsampling
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
    use_projection_mask = False
    sim_backbone = False
    # all use_proj_mask is False, then all the below is invalid
    layer_mask = [False,  # l4 mask
                  True]  # l3 mask
    projection_mask_mlps = [[128, 64], [128, 64]]
    mask_sigmoid = False
    one_head_mask = False  # one head mask using the first mlp dim of projection_mask_mlps

    # one_head_mask_eval = False  # one head mask using the first mlp dim of projection_mask_mlps
    ####################################
    # gt_proj
    ground_truth_projection_mask = False
    ground_truth_projection_mask_eval = False
    ground_truth_mask_layer = [
        False,  # l4 mask
        True  # l3 mask
    ]
    ab_delay = False
    mask_delay = False  # valid when layer mask and gt_mask_layer are all true
    mask_delay_step = 1904 * 8 * 30  # linear weight debug for the between gt and pred per step

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

    #########################################
    cmr_direct_filter = False
