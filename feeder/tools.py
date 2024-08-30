import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import sin, cos


transform_order = {
    'ntu': [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]
}  


def shear(data_numpy, r=0.5):
    s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1        ]])

    R = R.transpose()
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R)
    data_numpy = data_numpy.transpose(3, 0, 1, 2)
    return data_numpy


def temperal_crop(data_numpy, temperal_padding_ratio=6):
    C, T, V, M = data_numpy.shape
    padding_len = T // temperal_padding_ratio
    frame_start = np.random.randint(0, padding_len * 2 + 1)
    data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],
                                 data_numpy,
                                 data_numpy[:, -padding_len:][:, ::-1]),
                                axis=1)
    data_numpy = data_numpy[:, frame_start:frame_start + T]
    return data_numpy


def random_spatial_flip(seq, p=0.5):
    if random.random() < p:
        # Do the left-right transform C,T,V,M
        index = transform_order['ntu']
        trans_seq = seq[:, :, index, :] 
        return trans_seq
    else:
        return seq


def random_time_flip(seq, p=0.5):
    T = seq.shape[1]
    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        
        return seq[:, time_range_reverse, :, :]
    else:
        return seq


def random_rotate(seq):  
    def rotate(seq, axis, angle):
        # x
        if axis == 0:
            R = np.array([[1, 0, 0],
                              [0, cos(angle), sin(angle)],
                              [0, -sin(angle), cos(angle)]])
        # y
        if axis == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                              [0, 1, 0],
                              [sin(angle), 0, cos(angle)]])

        # z
        if axis == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                              [-sin(angle), cos(angle), 0],
                              [0, 0, 1]])
        R = R.T
        temp = np.matmul(seq, R)
        return temp

    new_seq = seq.copy()
    # C, T, V, M -> T, V, M, C
    new_seq = np.transpose(new_seq, (1, 2, 3, 0))
    total_axis = [0, 1, 2]
    main_axis = random.randint(0, 2)
    for axis in total_axis:
        if axis == main_axis:
            rotate_angle = random.uniform(0, 30)
            rotate_angle = math.radians(rotate_angle) 
            new_seq = rotate(new_seq, axis, rotate_angle)
        else:
            rotate_angle = random.uniform(0, 1)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)

    new_seq = np.transpose(new_seq, (3, 0, 1, 2))

    return new_seq


def gaus_noise(data_numpy, mean= 0, std=0.01, p=0.5): 
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(mean, std, size=(C, T, V, M))
        return temp + noise
    else:
        return data_numpy


def gaus_filter(data_numpy): 
    g = GaussianBlurConv(3)
    return g(data_numpy)


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel = 15, sigma = [0.1, 2]):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)

    def __call__(self, x):
        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        # kernel =  kernel.float()
        kernel = kernel.double()
        kernel = kernel.repeat(self.channels, 1, 1, 1) 
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        prob = np.random.random_sample()
        x = torch.from_numpy(x)
        if prob < 0.5:
            x = x.permute(3,0,2,1) # M,C,V,T
            x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2 )),   groups=self.channels)
            x = x.permute(1,-1,-2, 0) #C,T,V,M

        return x.numpy()

class Zero_out_axis(object):
    def __init__(self, axis = None):
        self.first_axis = axis


    def __call__(self, data_numpy):
        if self.first_axis != None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0,2)

        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((T, V, M))
        temp[axis_next] = x_new
        return temp

def axis_mask(data_numpy, p=0.5): 
    am = Zero_out_axis()
    if random.random() < p:
        return am(data_numpy)
    else:
        return data_numpy

head_ori_index = [3, 4]
trunk_ori_index = [1, 2, 21]
left_arm_ori_index = [9, 10]
right_arm_ori_index = [5, 6]
left_hand_ori_index = [11, 12, 24, 25]
right_hand_ori_index = [7, 8, 22, 23]
left_thigh_ori_index = [17, 18]
right_thigh_ori_index = [13, 14]
left_foot_ori_index = [19, 20]
right_foot_ori_index = [15, 16]

head = [i - 1 for i in head_ori_index]
trunk = [i - 1 for i in trunk_ori_index]
left_arm = [i - 1 for i in left_arm_ori_index]
right_arm = [i - 1 for i in right_arm_ori_index]
left_hand = [i - 1 for i in left_hand_ori_index]
right_hand = [i - 1 for i in right_hand_ori_index]
left_thigh = [i - 1 for i in left_thigh_ori_index]
right_thigh = [i - 1 for i in right_thigh_ori_index]
left_foot = [i - 1 for i in left_foot_ori_index]
right_foot = [i - 1 for i in right_foot_ori_index]

body_parts = [head, trunk, left_arm, right_arm, left_hand, right_hand, left_thigh, right_thigh, left_foot, right_foot]

def random_bodypart_mask(data_numpy,spa_l=1, spa_u=1, tem_l=7, tem_u=11, spatial_mode='semantic',swap_mode='Gaussian'):
    '''
    swap a batch skeleton
    T   64 --> 32 --> 16    # 8n
    S   25 --> 25 --> 25 (5 parts)
    '''
    C, T, V, M = data_numpy.shape
    tem_downsample_ratio = 4

    # generate swap swap idx
    # idx = torch.arange(N)
    # n = torch.randint(1, N - 1, (1,))
    # randidx = (idx + n) % N

    # ------ Spatial ------ #
    if spatial_mode == 'semantic':
        Cs = random.randint(spa_l, spa_u)
        # sample the parts index
        parts_idx = random.sample(body_parts, Cs)
        # generate spa_idx
        spa_idx = []
        for part_idx in parts_idx:
            spa_idx += part_idx
        spa_idx.sort()
    elif spatial_mode == 'random':
        spa_num = random.randint(2, 4)  
        spa_idx = random.sample(list(range(V)), spa_num)
        spa_idx.sort()
    else:
        raise ValueError('Not supported operation {}'.format(spatial_mode))
    # spa_idx = torch.tensor(spa_idx, dtype=torch.long).cuda()

    # ------ Temporal ------ #
    Ct = random.randint(tem_l, tem_u)
    tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
    rt = Ct * tem_downsample_ratio

    xst = data_numpy.copy()
    # begin swap
    if swap_mode == 'swap':
        xst[ :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
            xst[:, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :]  
    elif swap_mode == 'zeros':
        xst[:, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = 0
    elif swap_mode == 'Gaussian':
        xst[:, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
            torch.randn(C, rt, len(spa_idx), M).cuda()  # N, 
    else:
        raise ValueError('Not supported operation {}'.format(swap_mode))
    # generate mask
    # mask = torch.zeros(T // tem_downsample_ratio, V).cuda()
    # mask[tem_idx:tem_idx + Ct, spa_idx] = 1

    return xst  # randidx, , mask.bool()

def random_bodypart_gauss(data_numpy,spa_l=1, spa_u=1, tem_l=7, tem_u=11, spatial_mode='semantic',swap_mode='Gaussian'):
    '''
    swap a batch skeleton
    T   64 --> 32 --> 16    # 8n
    S   25 --> 25 --> 25 (5 parts)
    '''
    C, T, V, M = data_numpy.shape
    tem_downsample_ratio = 4

    # generate swap swap idx
    # idx = torch.arange(N)
    # n = torch.randint(1, N - 1, (1,))
    # randidx = (idx + n) % N

    # ------ Spatial ------ #
    if spatial_mode == 'semantic':
        Cs = random.randint(spa_l, spa_u)
        # sample the parts index
        parts_idx = random.sample(body_parts, Cs)
        # generate spa_idx
        spa_idx = []
        for part_idx in parts_idx:
            spa_idx += part_idx
        spa_idx.sort()
    elif spatial_mode == 'random':
        spa_num = random.randint(2, 4)  # (10, 15)
        spa_idx = random.sample(list(range(V)), spa_num)
        spa_idx.sort()
    else:
        raise ValueError('Not supported operation {}'.format(spatial_mode))
    # spa_idx = torch.tensor(spa_idx, dtype=torch.long).cuda()

    # ------ Temporal ------ #
    Ct = random.randint(tem_l, tem_u)
    tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
    rt = Ct * tem_downsample_ratio

    xst = data_numpy.copy()
    # begin swap
    if swap_mode == 'swap':
        xst[ :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
            xst[:, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :]  
    elif swap_mode == 'zeros':
        xst[:, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = 0
    elif swap_mode == 'Gaussian':
        xst[:, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
            torch.randn(C, rt, len(spa_idx), M)  # N, .cuda()
    else:
        raise ValueError('Not supported operation {}'.format(swap_mode))
    # generate mask
    # mask = torch.zeros(T // tem_downsample_ratio, V).cuda()
    # mask[tem_idx:tem_idx + Ct, spa_idx] = 1

    return xst  # randidx, , mask.bool()

if __name__ == '__main__':
    data_seq = np.ones((3, 50, 25, 2))
    data_seq = random_bodypart_mask(data_seq)
    # data_seq = axis_mask(data_seq)
    print(data_seq.shape)
