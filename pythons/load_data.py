import os
import math
import numpy as np
import pickle
from tqdm import tqdm

from api.base.paths import path_solutions, path_dataset_pkl
from api.base.paras import num_step, num_state, num_anchor_per_step, paras_base


def read_txt(path_txt):
    solution_np_origin = np.loadtxt(path_txt)  # 原始数据
    solution_np = np.zeros([num_step, num_state, num_anchor_per_step])

    # 分阶段提取数据
    for step in np.linspace(start=1, stop=num_step, num=num_step):
        solution_np_per_step = solution_np_origin[solution_np_origin[:, 0] == step, 2:5].transpose()

        # 按照xy极限进行等分
        samples_xy = np.array(
            [np.linspace(start=solution_np_per_step[0, 0], stop=solution_np_per_step[0, -1], num=num_anchor_per_step),
             np.linspace(start=solution_np_per_step[1, 0], stop=solution_np_per_step[1, -1], num=num_anchor_per_step),
             np.zeros([num_anchor_per_step])])
        # 按照间距大的轴进行等分，并搜寻另一轴对应的数
        if abs(samples_xy[0, 0] - samples_xy[0, -1]) >= abs(samples_xy[1, 0] - samples_xy[1, -1]):
            # 按照x轴等分，并搜索对应的y
            for sample_y_id in range(samples_xy.shape[1]):
                # 计算和当前x相距最小的x所对应的id
                id_ = np.argsort(np.abs(solution_np_per_step[0, :] - samples_xy[0, sample_y_id]))[0]
                if id_ == 0 or id_ == (solution_np_per_step.shape[1] - 1):
                    # 当前为此阶段的起点或者终点，无需额外的运算
                    samples_xy[1, sample_y_id] = solution_np_per_step[1, id_]
                    samples_xy[2, sample_y_id] = solution_np_per_step[2, id_]
                else:
                    samples_xy[1, sample_y_id] = (solution_np_per_step[1, id_ - 1] + solution_np_per_step[1, id_] +
                                                  solution_np_per_step[1, id_ + 1]) / 3
                    samples_xy[2, sample_y_id] = (solution_np_per_step[2, id_ - 1] + solution_np_per_step[2, id_] +
                                                  solution_np_per_step[2, id_ + 1]) / 3
        else:
            for sample_x_id in range(samples_xy.shape[1]):
                id_ = np.argsort(np.abs(solution_np_per_step[1, :] - samples_xy[1, sample_x_id]))[0]
                if id_ == 0 or id_ == (solution_np_per_step.shape[1] - 1):
                    samples_xy[0, sample_x_id] = solution_np_per_step[0, id_]
                    samples_xy[2, sample_x_id] = solution_np_per_step[2, id_]
                else:
                    samples_xy[0, sample_x_id] = (solution_np_per_step[0, id_ - 1] + solution_np_per_step[0, id_] +
                                                  solution_np_per_step[0, id_ + 1]) / 3
                    samples_xy[2, sample_x_id] = (solution_np_per_step[2, id_ - 1] + solution_np_per_step[2, id_] +
                                                  solution_np_per_step[2, id_ + 1]) / 3
        solution_np[int(step - 1)] = samples_xy
    return solution_np


if __name__ == '__main__':

    solutions = os.listdir(path_solutions)
    dataset = {
        'solutions': list(),
        'map': None
    }

    for solution_txt in tqdm(solutions, desc='Dataset', leave=False, ncols=100, disable=False):
        solution = read_txt(path_solutions + solution_txt)
        dataset['solutions'].append(solution.transpose(0, 2, 1))

    #  ————
    x = np.linspace(-paras_base['Freespace_X'], paras_base['Freespace_X'], math.floor(paras_base['Freespace_X'] * 2 / 0.1) + 1)
    y = np.ones(x.shape) * (paras_base['Parking_Y'] + paras_base['Freespace_Y'])
    map_np = np.stack([x, y], axis=0)

    #  ————
    #      |
    y = np.linspace(paras_base['Parking_Y'], paras_base['Parking_Y'] + paras_base['Freespace_Y'], math.floor(paras_base['Freespace_Y'] / 0.1) + 1)
    x = np.ones(y.shape) * paras_base['Freespace_X']
    map_np = np.append(map_np, np.stack([x, np.flip(y)], axis=0), axis=1)

    #  ————
    #     _|
    x = np.linspace(paras_base['Freespace_X'], paras_base['Parking_X'] / 2, math.floor((paras_base['Freespace_X'] - paras_base['Parking_X'] / 2) / 0.1) + 1)
    y = np.ones(x.shape) * paras_base['Parking_Y']
    map_np = np.append(map_np, np.stack([np.flip(x), y], axis=0), axis=1)

    #  ————
    #     _|
    #     |
    y = np.linspace(0.0, paras_base['Parking_Y'], math.floor(paras_base['Parking_Y'] / 0.1) + 1)
    x = np.ones(y.shape) * paras_base['Parking_X'] / 2
    map_np = np.append(map_np, np.stack([x, np.flip(y)], axis=0), axis=1)

    #  ————
    #     _|
    #   __|
    x = np.linspace(-paras_base['Parking_X'] / 2, paras_base['Parking_X'] / 2, math.floor(paras_base['Parking_X'] / 0.1) + 1)
    y = np.zeros(x.shape)
    map_np = np.append(map_np, np.stack([np.flip(x), y], axis=0), axis=1)

    #  ————
    #     _|
    #  |__|
    y = np.linspace(0.0, paras_base['Parking_Y'], math.floor(paras_base['Parking_Y'] / 0.1) + 1)
    x = np.ones(y.shape) * (-paras_base['Parking_X'] / 2)
    map_np = np.append(map_np, np.stack([x, y], axis=0), axis=1)

    #  ————
    # _   _|
    #  |__|
    x = np.linspace(-paras_base['Freespace_X'], -paras_base['Parking_X'] / 2, math.floor((paras_base['Freespace_X'] - paras_base['Parking_X'] / 2) / 0.1) + 1)
    y = np.ones(x.shape) * paras_base['Parking_Y']
    map_np = np.append(map_np, np.stack([np.flip(x), y], axis=0), axis=1)

    #  ————
    # |_   _|
    #  |__|
    y = np.linspace(paras_base['Parking_Y'], paras_base['Parking_Y'] + paras_base['Freespace_Y'], math.floor(paras_base['Freespace_Y'] / 0.1) + 1)
    x = np.ones(y.shape) * (-paras_base['Freespace_X'])
    map_np = np.append(map_np, np.stack([x, y], axis=0), axis=1)

    dataset['map'] = map_np

    with open(path_dataset_pkl, 'wb') as pkl:
        pickle.dump(dataset, pkl)
        pkl.close()
    print(f"\nThe dataset has saved into {path_dataset_pkl}")
