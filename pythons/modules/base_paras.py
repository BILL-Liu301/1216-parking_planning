import math
import numpy as np

samples = 5

num_step = 4
num_anchors_all = 2 * num_step + 1
num_anchors_pre = 2 * num_step
num_anchors_per_step = 3
num_anchor_state = 3
planning_time_ref = 1.0 * 60


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
