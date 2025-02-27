import fnmatch
import math
import numpy as np
import os


class Config:
    
    '''
    --------HPC--------
    Keep this part uncommented when using HPC.
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
    --------Ring--------
    '''
    
    num_points = 12288
    
    training_num = 3000
    validation_num = 600
    test_num = 500
    demo_num = 1
    
    training_batch_size = 8
    validation_batch_size = 1
    test_batch_size = 1
    demo_batch_size = 1
    
    num_workers = 16
    
    '''
    --------DATASET--------
    '''
    
    # subset = 'seg2tunnel'
    subset = 'seg2tunnel_ring'
    # subset = 'seg2tunnel_dublin'
    # subset = 'seg2tunnel_wuxi'
    
    voxel_size = 0.04
    
    if subset == 'seg2tunnel':
        data_path = '../Seg2Tunnel/seg2tunnel' 
        flag_prep = 'ring-wise'
        flag_pipe = 'sphere_crop'
        num_raw_features = 4
        num_classes = 7
        training_stations = ['1-1', '1-2', '1-3', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '1-11', '1-13', '1-14', '1-16', '1-17', '2-1', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-11', '2-12', '2-13']
        validation_stations = ['1-4', '1-12', '1-15', '2-2', '2-10', '2-14']
        test_stations = ['1-4', '1-12', '1-15', '2-2', '2-10', '2-14']
        demo_stations = ['1-4']
    elif subset == 'seg2tunnel_ring':
        data_path = '../Seg2Tunnel/seg2tunnel'
        flag_prep = 'scene-wise'
        flag_pipe = 'sample_random'
        num_raw_features = 4
        num_classes = 8
        training_stations = ['4-1-105', '4-1-106', '4-1-107', '4-1-108', '4-1-109', '4-1-110', '4-1-111', '4-1-112', '4-1-113', '4-1-114', '4-1-115', '4-1-116', '4-1-117', '4-1-118', '4-1-119', '4-1-120', '4-1-121', '4-1-122', '4-1-123', '4-1-124', '4-1-125', '4-2-135', '4-2-136', '4-2-137', '4-2-138', '4-2-139', '4-2-140', '4-2-141', '4-2-142', '4-2-143', '4-2-144', '4-2-145', '4-2-146', '4-2-147', '4-2-148', '4-2-149', '4-2-150', '4-2-151', '4-2-152', '4-2-153', '4-2-154', '4-2-155', '4-3-165', '4-3-166', '4-3-167', '4-3-168', '4-3-169', '4-3-170', '4-3-171', '4-3-172', '4-3-173', '4-3-174', '4-3-175', '4-3-176', '4-3-177', '4-3-178', '4-3-179', '4-3-180', '4-3-181', '4-3-182', '4-3-183', '4-3-184', '4-3-185', '4-5-235', '4-5-236', '4-5-237', '4-5-238', '4-5-239', '4-5-240', '4-5-241', '4-5-242', '4-5-243', '4-5-244', '4-5-245', '4-5-246', '4-5-247', '4-5-248', '4-5-249', '4-5-250', '4-5-251', '4-5-252', '4-5-253', '4-5-254', '4-5-255', '4-6-272', '4-6-273', '4-6-274', '4-6-275', '4-6-276', '4-6-277', '4-6-278', '4-6-279', '4-6-280', '4-6-281', '4-6-282', '4-6-283', '4-6-284', '4-6-285', '4-7-295', '4-7-296', '4-7-297', '4-7-298', '4-7-299', '4-7-300', '4-7-301', '4-7-302', '4-7-303', '4-7-304', '4-7-305', '4-7-306', '4-7-307', '4-7-308', '4-7-309', '4-7-310', '4-7-311', '4-7-312', '4-7-313', '4-7-314', '4-7-315', '4-8-325', '4-8-326', '4-8-327', '4-8-328', '4-8-329', '4-8-330', '4-8-331', '4-8-332', '4-8-333', '4-8-334', '4-8-335', '4-8-336', '4-8-337', '4-8-338', '4-8-339', '4-8-340', '4-8-341', '4-8-342', '4-8-343', '4-8-344', '4-8-345', '4-10-385', '4-10-386', '4-10-387', '4-10-388', '4-10-389', '4-10-390', '4-10-391', '4-10-392', '4-10-393', '4-10-394', '4-10-395', '4-10-396', '4-10-397', '4-10-398', '4-10-399', '4-10-400', '4-10-401', '4-10-402', '4-10-403', '4-10-404', '4-10-405', '4-11-415', '4-11-416', '4-11-417', '4-11-418', '4-11-419', '4-11-420', '4-11-421', '4-11-422', '4-11-423', '4-11-424', '4-11-425', '4-11-426', '4-11-427', '4-11-428', '4-11-429', '4-11-430', '4-11-431', '4-11-432', '4-11-433', '4-11-434', '4-11-435', '4-12-445', '4-12-446', '4-12-447', '4-12-448', '4-12-449', '4-12-450', '4-12-451', '4-12-452', '4-12-453', '4-12-454', '4-12-455', '4-12-456', '4-12-457', '4-12-458', '4-12-459', '4-12-460', '4-12-461', '4-12-462', '4-12-463', '4-12-464', '4-12-465', '4-13-475', '4-13-476', '4-13-477', '4-13-478', '4-13-479', '4-13-480', '4-13-481', '4-13-482', '4-13-483', '4-13-484', '4-13-485', '4-13-486', '4-13-487', '4-13-488', '4-13-489', '4-13-490', '4-13-491', '4-13-492', '4-13-493', '4-13-494', '4-13-495', '4-14-505', '4-14-506', '4-14-507', '4-14-508', '4-14-509', '4-14-510', '4-14-511', '4-14-512', '4-14-513', '4-14-514', '4-14-515', '4-14-516', '4-14-517', '4-14-518', '4-14-519', '4-14-520', '4-14-521', '4-14-522', '4-14-523', '4-14-524', '4-14-525', '4-15-535', '4-15-536', '4-15-537', '4-15-538', '4-15-539', '4-15-540', '4-15-541', '4-15-542', '4-15-543', '4-15-544', '4-15-545', '4-15-546', '4-15-547', '4-15-548', '4-15-549', '4-15-550', '4-15-551', '4-15-552', '4-15-553', '4-15-554', '4-15-555', '4-17-603', '4-17-604', '4-17-605', '4-17-606', '4-17-607', '4-17-608', '4-17-609', '4-17-610', '4-17-611', '4-17-612', '4-17-613', '4-17-614', '4-17-615', '5-2-138', '5-2-139', '5-2-140', '5-2-141', '5-2-142', '5-2-143', '5-2-144', '5-2-145', '5-2-146', '5-2-147', '5-2-148', '5-2-149', '5-2-150', '5-2-151', '5-2-152', '5-3-189', '5-3-190', '5-3-191', '5-3-192', '5-3-193', '5-3-194', '5-3-195', '5-3-196', '5-3-197', '5-3-198', '5-3-199', '5-3-200', '5-3-201', '5-4-217', '5-4-218', '5-4-219', '5-4-220', '5-4-221', '5-4-222', '5-4-223', '5-4-224', '5-4-225', '5-4-226', '5-4-227', '5-4-228', '5-4-229', '5-4-230', '5-4-231', '5-4-232', '5-4-233', '5-5-248', '5-5-249', '5-5-250', '5-5-251', '5-5-252', '5-5-253', '5-5-254', '5-5-255', '5-5-256', '5-5-257', '5-5-258', '5-5-259', '5-5-260', '5-5-261', '5-5-262', '5-6-278', '5-6-279', '5-6-280', '5-6-281', '5-6-282', '5-6-283', '5-6-284', '5-6-285', '5-6-286', '5-6-287', '5-6-288', '5-6-289', '5-6-290', '5-6-291', '5-6-292']
        # training_stations = ['4-1', '4-2', '4-3', '4-5', '4-6', '4-7', '4-8', '4-10', '4-11', '4-12', '4-13', '4-14', '4-15', '4-17', '5-2', '5-3', '5-4', '5-5', '5-6']
        # files = []
        # for station in training_stations:
        #     fs = fnmatch.filter(os.listdir(data_path), station + '-*.txt')
        #     for f in fs:
        #         files.append(f.split('.')[0])
        # training_stations = files
        # print(files)
        validation_stations = ['4-4-205', '4-4-206', '4-4-207', '4-4-208', '4-4-209', '4-4-210', '4-4-211', '4-4-212', '4-4-213', '4-4-214', '4-4-215', '4-4-216', '4-4-217', '4-4-218', '4-4-219', '4-4-220', '4-4-221', '4-4-222', '4-4-223', '4-4-224', '4-4-225', '4-9-355', '4-9-356', '4-9-357', '4-9-358', '4-9-359', '4-9-360', '4-9-361', '4-9-362', '4-9-363', '4-9-364', '4-9-365', '4-9-366', '4-9-367', '4-9-368', '4-9-369', '4-9-370', '4-9-371', '4-9-372', '4-9-373', '4-9-374', '4-9-375', '4-16-565', '4-16-566', '4-16-567', '4-16-568', '4-16-569', '4-16-570', '4-16-571', '4-16-572', '4-16-573', '4-16-574', '4-16-575', '4-16-576', '4-16-577', '4-16-578', '4-16-579', '4-16-580', '4-16-581', '4-16-582', '4-16-583', '4-16-584', '4-16-585', '5-1-107', '5-1-108', '5-1-109', '5-1-110', '5-1-111', '5-1-112', '5-1-113', '5-1-114', '5-1-115', '5-1-116', '5-1-117', '5-1-118', '5-1-119', '5-1-120', '5-1-121', '5-1-122', '5-1-123', '5-7-315', '5-7-316', '5-7-317', '5-7-318', '5-7-319', '5-7-320', '5-7-321', '5-7-322', '5-7-323']
        # validation_stations = ['4-4', '4-9', '4-16', '5-1', '5-7']
        # files = []
        # for station in validation_stations:
        #     fs = fnmatch.filter(os.listdir(data_path), station + '-*.txt')
        #     for f in fs:
        #         files.append(f.split('.')[0])
        # validation_stations = files
        # print(files)
        test_stations = ['4-4-205', '4-4-206', '4-4-207', '4-4-208', '4-4-209', '4-4-210', '4-4-211', '4-4-212', '4-4-213', '4-4-214', '4-4-215', '4-4-216', '4-4-217', '4-4-218', '4-4-219', '4-4-220', '4-4-221', '4-4-222', '4-4-223', '4-4-224', '4-4-225', '4-9-355', '4-9-356', '4-9-357', '4-9-358', '4-9-359', '4-9-360', '4-9-361', '4-9-362', '4-9-363', '4-9-364', '4-9-365', '4-9-366', '4-9-367', '4-9-368', '4-9-369', '4-9-370', '4-9-371', '4-9-372', '4-9-373', '4-9-374', '4-9-375', '4-16-565', '4-16-566', '4-16-567', '4-16-568', '4-16-569', '4-16-570', '4-16-571', '4-16-572', '4-16-573', '4-16-574', '4-16-575', '4-16-576', '4-16-577', '4-16-578', '4-16-579', '4-16-580', '4-16-581', '4-16-582', '4-16-583', '4-16-584', '4-16-585', '5-1-107', '5-1-108', '5-1-109', '5-1-110', '5-1-111', '5-1-112', '5-1-113', '5-1-114', '5-1-115', '5-1-116', '5-1-117', '5-1-118', '5-1-119', '5-1-120', '5-1-121', '5-1-122', '5-1-123', '5-7-315', '5-7-316', '5-7-317', '5-7-318', '5-7-319', '5-7-320', '5-7-321', '5-7-322', '5-7-323']
        # test_stations = ['4-4', '4-9', '4-16', '5-1', '5-7']
        # files = []
        # for station in test_stations:
        #     fs = fnmatch.filter(os.listdir(data_path), station + '-*.txt')
        #     for f in fs:
        #         files.append(f.split('.')[0])
        # test_stations = files
        # print(files)
        demo_stations = ['4-4-205', '4-4-206', '4-4-207', '4-4-208', '4-4-209', '4-4-210', '4-4-211', '4-4-212', '4-4-213', '4-4-214', '4-4-215', '4-4-216', '4-4-217', '4-4-218', '4-4-219', '4-4-220', '4-4-221', '4-4-222', '4-4-223', '4-4-224', '4-4-225']
        # demo_stations = ['4-4']
        # files = []
        # for station in demo_stations:
        #     fs = fnmatch.filter(os.listdir(data_path), station + '-*.txt')
        #     for f in fs:
        #         files.append(f.split('.')[0])
        # demo_stations = files
        # print(files)
    elif subset == 'seg2tunnel_dublin':
        data_path = '../Seg2Tunnel/seg2tunnel_dublin'
        flag_prep = 'scene-wise'
        flag_pipe = 'sample_random'
        num_raw_features = 4
        num_classes = 3
        training_stations = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '1-11', '1-12', '1-13', '1-14', '1-15', '1-16']
        validation_stations = ['1-17', '1-18', '1-19', '1-20']
        test_stations = ['2-1']
        demo_stations = ['1-17']
    elif subset == 'seg2tunnel_wuxi':
        data_path = '../Seg2Tunnel/seg2tunnel_wuxi'
        flag_prep = 'scene-wise'
        flag_pipe = 'sample_random'
        num_raw_features = 4
        num_classes = 2
        training_stations = ['0-1', '0-2', '0-3', '0-4', '0-5', '0-6', '0-7', '0-8', '0-9', '0-10', '0-11', '0-12', '0-13', '0-14', '0-15', '0-16']
        validation_stations = ['0-17', '0-18', '0-19', '0-20']
        test_stations = ['1-4']
        demo_stations = ['0-17']
    
    '''
    --------NETWORK--------
    '''
    
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
        
        # flag_ohe2se = False
        '''best'''
        flag_ohe2se = True
        
        if flag_ohe2se:
            weight_ohe2se = 0

    if (enc == 'ohe' and flag_ohe2se) or (enc == 'se'):
        cus_enc = [[1, 0, 0]]
        for i in range(1, num_classes):
            cus_enc.append([0, math.cos(i * math.pi / ((num_classes - 1) / 2)), math.sin(i * math.pi / ((num_classes - 1) / 2))])
    
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