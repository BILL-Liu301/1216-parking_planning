import numpy as np

# 多线程生成原始数据集
num_thread = 3
num_planning_time_ref = 1.0 * 60

# 参数设定
num_state = 3
num_step = 4
num_anchor_per_step = 30

# 基础参数
paras_base = {
    "Freespace_X": 8.0,  # matlab里是16，是为了方便做最优化求解，这里还是按照实际情况来
    "Freespace_Y": 8.0,
    "Parking_X": 2.5,
    "Parking_Y": 5.5,
    "Car_Length": 4.0,
    "Car_Width": 1.9,
    "Car_L": 2.5,
    "tf": 240,
    "end": [0.0, 1.5, np.pi / 2, 0.0, 0.0],
    "limits": np.array([[-(2.5 / 2 + 8.0), 2.5 / 2 + 8.0],  # matlab里是16，是为了方便做最优化求解，这里还是按照实际情况来
                        [0.0, (5.5 + 8.0)],
                        [-np.pi / 2, np.pi / 2],
                        [-0.8, 0.8],
                        [-5 / 3.6, 5 / 3.6]])
}