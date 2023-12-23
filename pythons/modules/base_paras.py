import math
import numpy as np

# 多线程生成数据的线程数
num_samples = 3
num_planning_time_ref = 1.0 * 60

# 相关参数设定
num_step = 4
num_anchor_state = 3
num_anchor_per_step = 30
num_anchor_inp = 8

# 数据集分布
ratio_train = 0.65
ratio_test = 0.3
ratio_val = 1 - ratio_train - ratio_test

paras = {
    "Freespace_X": 8.0,  # matlab里是16，是为了方便做最优化求解，这里还是按照实际情况来
    "Freespace_Y": 8.0,
    "Parking_X": 2.5,
    "Parking_Y": 5.5,
    "Car_Length": 4.0,
    "Car_Width": 1.9,
    "Car_L": 2.5,
    "tf": 240,
    "end": [0.0, 1.5, math.pi / 2, 0.0, 0.0],
    "limits": np.array([[-(2.5 / 2 + 8.0), 2.5 / 2 + 8.0],  # matlab里是16，是为了方便做最优化求解，这里还是按照实际情况来
                        [0.0, (5.5 + 8.0)],
                        [-math.pi / 2, math.pi / 2],
                        [-0.8, 0.8],
                        [-5 / 3.6, 5 / 3.6]])
}
