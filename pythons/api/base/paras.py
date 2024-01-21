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
    'Freespace_X': 8.0,  # matlab里是16，是为了方便做最优化求解，这里还是按照实际情况来
    'Freespace_Y': 8.0,
    'Parking_X': 2.5,
    'Parking_Y': 5.5,
    'Car_Length': 4.0,
    'Car_Width': 1.9,
    'Car_L': 2.5,
    'tf': 240,
    'end': [0.0, 1.5, np.pi / 2, 0.0, 0.0],
    'limits': np.array([[-(2.5 / 2 + 8.0), 2.5 / 2 + 8.0],  # matlab里是16，是为了方便做最优化求解，这里还是按照实际情况来
                        [0.0, (5.5 + 8.0)],
                        [-np.pi / 2, np.pi / 2],
                        [-0.8, 0.8],
                        [-5 / 3.6, 5 / 3.6]])
}

paras_Parking_Trajectory_Planner = {
    'max_epochs': 100,
    'lr_init': 1e-3,
    'size_middle': 16,
    'size_occupy': 10,  # 单方向
    'dis_occupy': 5,  # 单方向
    'num_layers': 2,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_anchor_per_step': num_anchor_per_step,
    'num_step': num_step,
    'len_info_loc': 3,
    'delta_limit_mean': torch.tensor([1.0, 1.0, math.pi]),
    'delta_limit_var': 5.0,
    'end_point': torch.tensor([0.0, 1.5, math.pi / 2]),
    'car_length': paras_base['Car_Length']
}
