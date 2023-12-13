import math
import numpy as np


class Config:
    
    '''
    --------LFA--------
    '''
    lfa = 'hu2019'
    # lfa = 'fan2021'
    # lfa_param = 0.1
    # lfa = 'zhao2021'
    # lfa = 'jing2022'
    # lfa_param = 8
    # lfa = 'cheng2023'
    # lfa = 'zhan2023'
    # lfa_param = 1
    # lfa = 'lin_v1'
    
    k_n = 16
    
    '''
    --------RFA--------
    '''
    # rfa = False
    
    axis_k_n = 16
    
    # rfa = 'lin_v1'
    # rfa_param = 1
    # rfa_pooling = 'max'
    # rfa_pooling = 'mean'
    
    # rfa = 'lin_v2'
    # rfa_param = 1
    # rfa_pooling = 'max'
    # rfa_pooling = 'mean'
    
    rfa = 'lin_v3'
    rfa_pooling = 'max'
    # rfa_pooling = 'mean'
    
    '''
    --------GFA-S--------
    '''
    gfa_s = False
    # gfa_s = 'deng2021'
    # gfa_s_param = 20
    # gfa_s = 'li2022'
    # gfa_s_param = 0.1
    # gfa_s = 'liu2022'
    # gfa_s = 'ren2022'
    # gfa_s = 'zhao2023'
    
    
    '''
    --------GFA-L--------
    '''
    gfa_l = False
    # gfa_l = 'deng2021'
    # gfa_l_param = 512
    # gfa_l = 'li2022'
    # gfa_l_param = 0.1
    # gfa_l = 'liu2022'
    # gfa_l = 'ren2022'
    # gfa_l = 'zhao2023'
    
    '''
    --------HPC--------
    '''
    # num_points = 204800
    
    # training_num = 3000
    # validation_num = 600
    # test_num = 1000
    # demo_num = 1
    
    # training_batch_size = 8
    # validation_batch_size = 1
    # test_batch_size = 1
    # demo_batch_size = 1
    
    # num_workers = 32
    
    '''
    --------PC--------
    '''
    num_points = 10240
    
    training_num = 30
    validation_num = 6
    test_num = 1000
    demo_num = 1
    
    training_batch_size = 1
    validation_batch_size = 1
    test_batch_size = 1
    demo_batch_size = 1
    
    num_workers = 16
    
    '''
    --------GENERAL--------
    '''
    num_classes = 7
    
    num_features = 4
    
    data_path = '../Seg2Tunnel/seg2tunnel'
    voxel_size = 0.04
    training_stations = ['1-1', '1-2', '1-3', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '1-11', '1-13', '1-14', '1-16', '1-17', '2-1', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-11', '2-12', '2-13']
    validation_stations = ['1-4', '1-12', '1-15', '2-2', '2-10', '2-14']
    test_stations = ['1-4', '1-12', '1-15', '2-2', '2-10', '2-14']
    demo_stations = ['1-1']
    
    num_layers = 5
    sub_sampling_ratio = [4, 4, 4, 4, 2]
    d_out = [16, 64, 128, 256, 512]
    
    max_epoch = 100
    learning_rate = 0.01

    log_dir = 'log'
    checkpoint_path = 'log/checkpoint.pt'
    result_path = 'result'
    
    '''
    --------LABEL ENCODING--------
    '''
    enc = 'ohe'
    # enc = 'se'
    
    if enc == 'ohe':
        flag_ohe2se = False
        # flag_ohe2se = True
        if flag_ohe2se:
            weight_ohe2se = 1

    if (enc == 'ohe' and flag_ohe2se) or (enc == 'se'):
        cus_enc = [[1, 0, 0],
                   [0, math.cos(1 * math.pi / 3), math.sin(1 * math.pi / 3)],
                   [0, math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)],
                   [0, math.cos(3 * math.pi / 3), math.sin(3 * math.pi / 3)],
                   [0, math.cos(4 * math.pi / 3), math.sin(4 * math.pi / 3)],
                   [0, math.cos(5 * math.pi / 3), math.sin(5 * math.pi / 3)],
                   [0, math.cos(6 * math.pi / 3), math.sin(6 * math.pi / 3)]]
    
    '''
    --------LOSS--------
    '''
    flag_ml = False
    # flag_ml = True
    if flag_ml:
        weight_ml = [1, 1, 1, 1, 1]
        
    if enc == 'ohe':
        loss_func = 'cel'
        weight_cel = np.asarray([1, 1, 1, 1, 1, 1, 1])
        # weight_cel = np.asarray([1 / 0.304, 1 / 0.076, 1 / 0.206, 1 / 0.094, 1 / 0.030, 1 / 0.091, 1 / 0.199])
        # weight_cel = weight_cel / np.sum(weight_cel) * len(weight_cel)