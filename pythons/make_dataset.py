import os
import math
import pickle
import torch
import numpy as np
from tqdm import tqdm

from modules.base_path import path_solutions, path_anchors_failed, path_dataset_pkl
from modules.base_paras import num_step, num_anchor_state, num_anchor_per_step, num_inp_1, num_inp_2, paras
from modules.base_paras import ratio_train, ratio_test
from modules.base_settings import device


def cal_inp1(point_s, point_e):
    inp1 = np.zeros([num_step, num_anchor_state, num_inp_1])
    for step in range(num_step):
        inp1[step, :, 0] = point_s
        inp1[step, :, 1] = point_e
    return inp1


def cal_inp2(point_s):
    x_center = np.array([-(paras["Parking_X"] / 2 + paras["Freespace_X"]) / 2, (paras["Parking_X"] / 2 + paras["Freespace_X"]) / 2])
    y_center = np.array([paras["Parking_Y"] + paras["Freespace_Y"] * 3 / 4, paras["Parking_Y"] + paras["Freespace_Y"] / 4])
    x_limit = np.array([paras['limits'][0, 0], paras['limits'][0, 1]])
    y_limit = np.array([paras['limits'][1, 0], paras['limits'][1, 1]])
    inp2 = np.zeros([num_step, num_anchor_state, num_inp_2])
    for step in range(num_step):
        inp2[step, :, 0] = np.array([step / (num_step - 1), point_s[0] - x_center[0], point_s[1] - y_center[0]])
        inp2[step, :, 1] = np.array([step / (num_step - 1), point_s[0] - x_center[0], point_s[1] - y_center[1]])
        inp2[step, :, 2] = np.array([step / (num_step - 1), point_s[0] - x_center[1], point_s[1] - y_center[1]])
        inp2[step, :, 3] = np.array([step / (num_step - 1), point_s[0] - x_center[1], point_s[1] - y_center[1]])
        inp2[step, :, 4] = np.array([step / (num_step - 1), point_s[0] - x_limit[0], point_s[1] - y_limit[0]])
        inp2[step, :, 5] = np.array([step / (num_step - 1), point_s[0] - x_limit[0], point_s[1] - y_limit[1]])
        inp2[step, :, 6] = np.array([step / (num_step - 1), point_s[0] - x_limit[1], point_s[1] - y_limit[1]])
        inp2[step, :, 7] = np.array([step / (num_step - 1), point_s[0] - x_limit[1], point_s[1] - y_limit[1]])
    return inp2


# 读取txt内的数据
def read_txt(path_txt):
    solution_np_origin = np.loadtxt(path_txt)  # 原始数据
    point_s = solution_np_origin[0, 2:5]  # 起点
    point_e = solution_np_origin[-1, 2:5]  # 终点
    solution_np_steps = np.zeros([num_step, num_anchor_state, num_anchor_per_step])

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
        solution_np_steps[int(step - 1)] = samples_xy
    return point_s, point_e, solution_np_steps


if __name__ == '__main__':
    # 加载txt， 并预设相关结构
    solutions = os.listdir(path_solutions)
    num_solutions = len(solutions)
    anchors_failed = np.loadtxt(path_anchors_failed)

    # 数据初始化
    dataset_inp1 = np.zeros([num_solutions, num_step, num_anchor_state, num_inp_1])
    dataset_inp2 = np.zeros([num_solutions, num_step, num_anchor_state, num_inp_2])
    dataset_oup = np.zeros([num_solutions, num_step, num_anchor_state, num_anchor_per_step])
    dataset_inp1_failed = np.zeros([anchors_failed.shape[0], num_step, num_anchor_state, num_inp_1])
    dataset_inp2_failed = np.zeros([anchors_failed.shape[0], num_step, num_anchor_state, num_inp_2])
    dataset_oup_failed = np.zeros([anchors_failed.shape[0], num_step, num_anchor_state, num_anchor_per_step])

    # 训练集和测试集
    # 逐文件读取，并在每个文件按照阶段分隔
    #            阶段       时间戳       x    y    偏航角  前轮转角  速度
    # column = ['step', 'time_stamp', 'x', 'y', 'theta', 'phi', 'v']
    pbar = tqdm(total=num_solutions)
    for solution_id, solution_txt in enumerate(solutions):
        # 提取txt的有用数据
        point_start, point_end, solution_np = read_txt(path_solutions + solution_txt)

        # 计算inp和oup
        inp1 = cal_inp1(point_start, point_end)
        inp2 = cal_inp2(point_start)
        oup = solution_np

        dataset_inp1[solution_id], dataset_inp2[solution_id], dataset_oup[solution_id] = inp1, inp2, oup
        pbar.update(1)
    pbar.close()

    # 失败集
    for failed_id in range(anchors_failed.shape[0]):
        point_start = anchors_failed[failed_id]
        point_end = paras['end'][0:3]

        # 计算inp和oup
        inp1 = cal_inp1(point_start, point_end)
        inp2 = cal_inp2(point_start)

        dataset_inp1_failed[failed_id], dataset_inp2_failed[failed_id] = inp1, inp2

    # 数据分装
    dataset_inp1_train = dataset_inp1[0:math.floor(ratio_train * num_solutions)]
    dataset_inp2_train = dataset_inp2[0:math.floor(ratio_train * num_solutions)]
    dataset_oup_train = dataset_oup[0:math.floor(ratio_train * num_solutions)]

    dataset_inp1_test = dataset_inp1[math.floor(ratio_train * num_solutions):math.floor((ratio_train + ratio_test) * num_solutions)]
    dataset_inp2_test = dataset_inp2[math.floor(ratio_train * num_solutions):math.floor((ratio_train + ratio_test) * num_solutions)]
    dataset_oup_test = dataset_oup[math.floor(ratio_train * num_solutions):math.floor((ratio_train + ratio_test) * num_solutions)]

    dataset_inp1_val = dataset_inp1[math.floor((ratio_train + ratio_test) * num_solutions):]
    dataset_inp2_val = dataset_inp2[math.floor((ratio_train + ratio_test) * num_solutions):]
    dataset_oup_val = dataset_oup[math.floor((ratio_train + ratio_test) * num_solutions):]

    dataset = {
        'dataset_inp1_train': torch.from_numpy(dataset_inp1_train).to(torch.float32).to(device),
        'dataset_inp2_train': torch.from_numpy(dataset_inp2_train).to(torch.float32).to(device),
        'dataset_oup_train': torch.from_numpy(dataset_oup_train).to(torch.float32).to(device),
        'dataset_inp1_test': torch.from_numpy(dataset_inp1_test).to(torch.float32).to(device),
        'dataset_inp2_test': torch.from_numpy(dataset_inp2_test).to(torch.float32).to(device),
        'dataset_oup_test': torch.from_numpy(dataset_oup_test).to(torch.float32).to(device),
        'dataset_inp1_val': torch.from_numpy(dataset_inp1_val).to(torch.float32).to(device),
        'dataset_inp2_val': torch.from_numpy(dataset_inp2_val).to(torch.float32).to(device),
        'dataset_oup_val': torch.from_numpy(dataset_oup_val).to(torch.float32).to(device),
        'dataset_inp1_all': torch.from_numpy(dataset_inp1).to(torch.float32).to(device),
        'dataset_inp2_all': torch.from_numpy(dataset_inp2).to(torch.float32).to(device),
        'dataset_oup_all': torch.from_numpy(dataset_oup).to(torch.float32).to(device),
        'dataset_inp1_failed': torch.from_numpy(dataset_inp1_failed).to(torch.float32).to(device),
        'dataset_inp2_failed': torch.from_numpy(dataset_inp2_failed).to(torch.float32).to(device),
        'dataset_oup_failed': torch.from_numpy(dataset_oup_failed).to(torch.float32).to(device)
    }

    # 数据保存
    with open(path_dataset_pkl, 'wb') as pkl:
        pickle.dump(dataset, pkl)
        pkl.close()
    print(f"\nThe dataset has saved into {path_dataset_pkl}")
