import math
import numpy as np


class Config:
    
    '''
    --------HPC--------
    Keep this part uncommented when using HPC.
    '''
    
    num_points = 204800
    
    training_num = 3000
    validation_num = 600
    test_num = 1000
    demo_num = 1
    
    training_batch_size = 8
    validation_batch_size = 1
    test_batch_size = 1
    demo_batch_size = 1
    
    num_workers = 32
    
    '''
    --------PC--------
    Keep this part uncommented when using PC.
    '''
    
    # num_points = 102400
    
    # training_num = 30
    # validation_num = 6
    # test_num = 1000
    # demo_num = 1
    
    # training_batch_size = 1
    # validation_batch_size = 1
    # test_batch_size = 1
    # demo_batch_size = 1
    
    # num_workers = 16
    
    '''
    --------DATASET--------
    '''
    
    # subset = 'seg2tunnel'
    subset = 'seg2tunnel_dublin'
    
    data_path = '../Seg2Tunnel/' + subset    
    
    voxel_size = 0
    
    if subset == 'seg2tunnel':
        flag_prep = 'ring-wise'
        flag_pipe = 'crop'
        num_raw_features = 4
        training_stations = ['1-1', '1-2', '1-3', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '1-11', '1-13', '1-14', '1-16', '1-17', '2-1', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-11', '2-12', '2-13']
        validation_stations = ['1-4', '1-12', '1-15', '2-2', '2-10', '2-14']
        test_stations = ['1-4', '1-12', '1-15', '2-2', '2-10', '2-14']
        demo_stations = ['1-4']
    elif subset == 'seg2tunnel_dublin':
        flag_prep = 'scene-wise'
        flag_pipe = 'sample_random'
        num_raw_features = 4
        training_stations = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '1-11', '1-12', '1-13', '1-14', '1-15', '1-16']
        validation_stations = ['1-17', '1-18', '1-19', '1-20']
        test_stations = ['1-17', '1-18', '1-19', '1-20', '2-1']
        demo_stations = ['1-17']
    
    '''
    --------NETWORK--------
    '''
    
    num_classes = 3
    num_features = 3
    
    num_layers = 5
    sub_sampling_ratio = [4, 4, 4, 4, 2]
    d_out = [16, 64, 128, 256, 512]
    
    '''
    --------TRAIN--------
    '''
    
    max_epoch = 100
    learning_rate = 0.01

    log_dir = 'log'
    checkpoint_path = 'log/checkpoint.pt'
    result_path = 'result'
    
    '''
    --------LFA--------
    Uncomment the LFA you want to adopt and keep others commented. Make sure the parameters following the adopted LFA, if any, is also uncommented.
    '''
    
    '''best'''
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
    
    rfa = False
    
    '''best'''
    # rfa = 'lin_v1'
    # rfa_param = 1
    '''best'''
    # rfa_pooling = 'max'
    # rfa_pooling = 'mean'
    
    # rfa = 'lin_v2'
    # rfa_param = 1
    # rfa_pooling = 'max'
    # rfa_pooling = 'mean'
    
    # deprecated
    # rfa = 'lin_v3'
    # rfa_pooling = 'max'
    # rfa_pooling = 'mean'
    
    if rfa:
        axis_k_n = 16
    
    '''
    --------GFA-S--------
    '''
    
    gfa_s = False
    
    # gfa_s = 'deng2021'
    # gfa_s_param = 20
    
    # gfa_s = 'li2022'
    # gfa_s_param = 0.1
    
    '''best'''
    # gfa_s = 'liu2022'
    
    # gfa_s = 'ren2022'
    
    # gfa_s = 'liu2023'
    
    '''
    --------GFA-L--------
    '''
    
    gfa_l = False
    
    # gfa_l = 'deng2021'
    # gfa_l_param = 512
    
    '''best'''
    # gfa_l = 'li2022'
    # gfa_l_param = 0.01
    
    # gfa_l = 'liu2022'
    
    # gfa_l = 'ren2022'
    
    # gfa_l = 'liu2023'
    
    '''
    --------LABEL ENCODING--------
    '''
    '''best'''
    enc = 'ohe'
    # enc = 'se'
    
    if enc == 'ohe':
        
        flag_ohe2se = False
        
        '''best'''
        # flag_ohe2se = True
        
        if flag_ohe2se:
            weight_ohe2se = 0.1

    if (enc == 'ohe' and flag_ohe2se) or (enc == 'se'):
        cus_enc = [[1, 0, 0]]
        for i in range(1, num_classes):
            cus_enc.append([0, math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)])
    
    '''
    --------LOSS FUNCTION--------
    '''
    flag_ml = False
    # flag_ml = True
    
    if flag_ml:
        weight_ml = [1] * num_layers
        
    if enc == 'ohe':
        
        loss_func = 'cel'
        if loss_func == 'cel':
            weight_cel = np.ones(num_classes)
    
    '''
    --------FEATURE MAP VISUALISATION--------
    Activate this part only when running test_visualise.py.
    '''
    flag_vis = False
    # flag_vis = True
    
    if flag_vis:
        vis_layers = [0]
        vis_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    
    
    
    
    