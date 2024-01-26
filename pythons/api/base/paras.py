import math
import numpy as np
import torch

# 多线程生成原始数据集
num_thread = 3
num_planning_time_ref = 1.0 * 60

# 参数设定
num_state = 3
num_step = 4
num_anchor_per_step = 30

# 基础参数
paras_base = {
    'Freespace_X': 10.0,  # matlab里是16，是为了方便做最优化求解，这里还是按照实际情况来
    'Freespace_Y': 8.0,
    'Parking_X': 2.5,
    'Parking_Y': 5.5,
    'Car_Length': 2.8,
    'Car_Width': 1.9,
    'Car_L': 2.5,
    'tf': 240,
    'end': [0.0, 1.5, np.pi / 2, 0.0, 0.0],
    'limits': np.array([[-(2.5 / 2 + 10.0), 2.5 / 2 + 10.0],  # matlab里是16，是为了方便做最优化求解，这里还是按照实际情况来
                        [0.0, (5.5 + 8.0)],
                        [-np.pi / 2, np.pi / 2],
                        [-0.8, 0.8],
                        [-5 / 3.6, 5 / 3.6]])
}

# 地图
point_split = 0.25
#  ————
x = np.linspace(-paras_base['Freespace_X'], paras_base['Freespace_X'], math.floor(paras_base['Freespace_X'] * 2 / point_split) + 1)
y = np.ones(x.shape) * (paras_base['Parking_Y'] + paras_base['Freespace_Y'])
map_np = np.stack([x, y], axis=0)

#  ————
#      |
y = np.linspace(paras_base['Parking_Y'], paras_base['Parking_Y'] + paras_base['Freespace_Y'], math.floor(paras_base['Freespace_Y'] / point_split) + 1)
x = np.ones(y.shape) * paras_base['Freespace_X']
map_np = np.append(map_np, np.stack([x, np.flip(y)], axis=0), axis=1)

#  ————
#     _|
x = np.linspace(paras_base['Freespace_X'], paras_base['Parking_X'] / 2, math.floor((paras_base['Freespace_X'] - paras_base['Parking_X'] / 2) / point_split) + 1)
y = np.ones(x.shape) * paras_base['Parking_Y']
map_np = np.append(map_np, np.stack([np.flip(x), y], axis=0), axis=1)

#  ————
#     _|
#     |
y = np.linspace(0.0, paras_base['Parking_Y'], math.floor(paras_base['Parking_Y'] / point_split) + 1)
x = np.ones(y.shape) * paras_base['Parking_X'] / 2
map_np = np.append(map_np, np.stack([x, np.flip(y)], axis=0), axis=1)

#  ————
#     _|
#   __|
x = np.linspace(-paras_base['Parking_X'] / 2, paras_base['Parking_X'] / 2, math.floor(paras_base['Parking_X'] / point_split) + 1)
y = np.zeros(x.shape)
map_np = np.append(map_np, np.stack([np.flip(x), y], axis=0), axis=1)

#  ————
#     _|
#  |__|
y = np.linspace(0.0, paras_base['Parking_Y'], math.floor(paras_base['Parking_Y'] / point_split) + 1)
x = np.ones(y.shape) * (-paras_base['Parking_X'] / 2)
map_np = np.append(map_np, np.stack([x, y], axis=0), axis=1)

#  ————
# _   _|
#  |__|
x = np.linspace(-paras_base['Freespace_X'], -paras_base['Parking_X'] / 2, math.floor((paras_base['Freespace_X'] - paras_base['Parking_X'] / 2) / point_split) + 1)
y = np.ones(x.shape) * paras_base['Parking_Y']
map_np = np.append(map_np, np.stack([np.flip(x), y], axis=0), axis=1)

#  ————
# |_   _|
#  |__|
y = np.linspace(paras_base['Parking_Y'], paras_base['Parking_Y'] + paras_base['Freespace_Y'], math.floor(paras_base['Freespace_Y'] / point_split) + 1)
x = np.ones(y.shape) * (-paras_base['Freespace_X'])
map_np = np.append(map_np, np.stack([x, y], axis=0), axis=1)

paras_Parking_Trajectory_Planner = {
    'max_epochs': 100,
    'lr_init': 1e-3,
    'size_middle': 16,
    'map_range': 5,
    'map_num_max': 200,
    'map': map_np,
    'num_layers': 2,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_anchor_per_step': num_anchor_per_step,
    'num_step': num_step,
    'len_info_loc': 3,
    'delta_limit_mean': np.array([1.0, 1.0, math.pi / 3]),
    'delta_limit_var': 10.0,
    'end_point': np.array([0.0, 1.5, math.pi / 2]),
    'car_length': paras_base['Car_Length']
}
