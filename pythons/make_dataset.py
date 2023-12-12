import numpy as np
import torch
import math
import pickle

from modules.base_path import path_anchors, path_dataset_pkl, path_anchors_failed
from modules.base_paras import num_anchors_pre, num_anchor_state, paras, num_step, num_anchors_per_step
from modules.base_settings import ratio_train, ratio_test, device

anchors = np.loadtxt(path_anchors)
anchors_failed = np.loadtxt(path_anchors_failed)

dataset_inp1 = np.zeros([anchors.shape[0], num_anchor_state, num_anchors_pre])
dataset_inp2 = np.zeros([anchors.shape[0], num_anchor_state, num_anchors_pre])
dataset_oup = np.zeros([anchors.shape[0], num_anchor_state, num_anchors_pre])
dataset_inp1_failed = np.zeros([anchors_failed.shape[0], num_anchor_state, num_anchors_pre])
dataset_inp2_failed = np.zeros([anchors_failed.shape[0], num_anchor_state, num_anchors_pre])
dataset_oup_failed = np.zeros([anchors_failed.shape[0], num_anchor_state, num_anchors_pre])

print("anchors")
for i in range(anchors.shape[0]):
    dataset_inp1[i, :, 0] = anchors[i, 0:3]  # 起点
    dataset_inp1[i, :, 1] = paras["end"][0:3]  # 终点
    dataset_inp1[i, :, 2] = np.array([paras["limits"][0, 0], paras["limits"][1, 1], 0.0])
    dataset_inp1[i, :, 3] = np.array([paras["limits"][0, 1], paras["limits"][1, 1], 0.0])
    dataset_inp1[i, :, 4] = np.array([paras["limits"][0, 0], paras["Parking_Y"], 0.0])
    dataset_inp1[i, :, 5] = np.array([paras["limits"][0, 1], paras["Parking_Y"], 0.0])
    dataset_inp1[i, :, 6] = np.array([-paras["Parking_X"]/2, paras["Parking_Y"], 0.0])
    dataset_inp1[i, :, 7] = np.array([paras["Parking_X"]/2, paras["Parking_Y"], 0.0])

    x_center = np.array([-(paras["Parking_X"]/2 + paras["Freespace_X"])/2, (paras["Parking_X"]/2+paras["Freespace_X"])/2])
    y_center = np.array([paras["Parking_Y"] + paras["Freespace_Y"]*3/4, paras["Parking_Y"] + paras["Freespace_Y"]/4])
    dataset_inp2[i, 0:2, 0] = np.array([x_center[0], y_center[0]])
    dataset_inp2[i, 0:2, 1] = anchors[i, 0:2] - np.array([x_center[0], y_center[0]])
    dataset_inp2[i, 0:2, 2] = np.array([x_center[0], y_center[1]])
    dataset_inp2[i, 0:2, 3] = anchors[i, 0:2] - np.array([x_center[0], y_center[1]])
    dataset_inp2[i, 0:2, 4] = np.array([x_center[1], y_center[0]])
    dataset_inp2[i, 0:2, 5] = anchors[i, 0:2] - np.array([x_center[1], y_center[0]])
    dataset_inp2[i, 0:2, 6] = np.array([x_center[1], y_center[1]])
    dataset_inp2[i, 0:2, 7] = anchors[i, 0:2] - np.array([x_center[1], y_center[1]])

    anchors_step = np.zeros([num_step, num_anchors_per_step, num_anchor_state])
    for j in range(num_step):
        for k in range(num_anchors_per_step):
            anchors_step[j, k, :] = anchors[i, (j*(num_anchors_per_step-1)*num_anchor_state+k*num_anchor_state):
                                               (j*(num_anchors_per_step-1)*num_anchor_state+k*num_anchor_state+num_anchor_state)]

    for j in range(num_anchors_pre):
        dataset_oup[i, :, j] = anchors[i, (j + 1) * 3:((j + 1) * 3 + 3)]

    # for j in range(0, num_anchors_pre, 2):
    #     k = int(j / 2)
    #     dataset_oup[i, :, j] = anchors_step[k, 1, :] / np.array([paras["limits"][0, 1], paras["limits"][1, 1], math.pi/2])
    #     dataset_oup[i, :, j+1] = anchors_step[k, 2, :] / np.array([paras["limits"][0, 1], paras["limits"][1, 1], math.pi/2])
    #     dataset_oup[i, :, j+1] = dataset_oup[i, :, j+1] - dataset_oup[i, :, j]

    schedule = "▋" * math.floor((i + 1) / anchors.shape[0] * 10)
    print(f"\rDatasets: {schedule}, {(i + 1)}/{anchors.shape[0]}", end='')

print("\nanchors_failed")
for i in range(anchors_failed.shape[0]):
    dataset_inp1_failed[i, :, 0] = anchors_failed[i, 0:3]  # 起点
    dataset_inp1_failed[i, :, 1] = paras["end"][0:3]  # 终点
    dataset_inp1_failed[i, :, 2] = np.array([paras["limits"][0, 0], paras["limits"][1, 1], 0.0])
    dataset_inp1_failed[i, :, 3] = np.array([paras["limits"][0, 1], paras["limits"][1, 1], 0.0])
    dataset_inp1_failed[i, :, 4] = np.array([paras["limits"][0, 0], paras["Parking_Y"], 0.0])
    dataset_inp1_failed[i, :, 5] = np.array([paras["limits"][0, 1], paras["Parking_Y"], 0.0])
    dataset_inp1_failed[i, :, 6] = np.array([-paras["Parking_X"] / 2, paras["Parking_Y"], 0.0])
    dataset_inp1_failed[i, :, 7] = np.array([paras["Parking_X"] / 2, paras["Parking_Y"], 0.0])

    x_center = np.array([-(paras["Parking_X"]/2 + paras["Freespace_X"])/2, (paras["Parking_X"]/2+paras["Freespace_X"])/2])
    y_center = np.array([paras["Parking_Y"] + paras["Freespace_Y"]*3/4, paras["Parking_Y"] + paras["Freespace_Y"]/4])
    dataset_inp2_failed[i, 0:2, 0] = np.array([x_center[0], y_center[0]])
    dataset_inp2_failed[i, 0:2, 1] = anchors_failed[i, 0:2] - np.array([x_center[0], y_center[0]])
    dataset_inp2_failed[i, 0:2, 2] = np.array([x_center[0], y_center[1]])
    dataset_inp2_failed[i, 0:2, 3] = anchors_failed[i, 0:2] - np.array([x_center[0], y_center[1]])
    dataset_inp2_failed[i, 0:2, 4] = np.array([x_center[1], y_center[0]])
    dataset_inp2_failed[i, 0:2, 5] = anchors_failed[i, 0:2] - np.array([x_center[1], y_center[0]])
    dataset_inp2_failed[i, 0:2, 6] = np.array([x_center[1], y_center[1]])
    dataset_inp2_failed[i, 0:2, 7] = anchors_failed[i, 0:2] - np.array([x_center[1], y_center[1]])

    schedule = "▋" * math.floor((i + 1) / anchors_failed.shape[0] * 10)
    print(f"\rDatasets: {schedule}, {(i + 1)}/{anchors_failed.shape[0]}", end='')

dataset_inp1_train = dataset_inp1[0:math.floor(ratio_train*dataset_inp1.shape[0])]
dataset_inp2_train = dataset_inp2[0:math.floor(ratio_train*dataset_inp1.shape[0])]
dataset_oup_train = dataset_oup[0:math.floor(ratio_train*dataset_inp1.shape[0])]

dataset_inp1_test = dataset_inp1[math.floor(ratio_train*dataset_inp1.shape[0]):math.floor((ratio_train + ratio_test)*dataset_inp1.shape[0])]
dataset_inp2_test = dataset_inp2[math.floor(ratio_train*dataset_inp1.shape[0]):math.floor((ratio_train + ratio_test)*dataset_inp1.shape[0])]
dataset_oup_test = dataset_oup[math.floor(ratio_train*dataset_inp1.shape[0]):math.floor((ratio_train + ratio_test)*dataset_inp1.shape[0])]

dataset_inp1_val = dataset_inp1[math.floor((ratio_train + ratio_test)*dataset_inp1.shape[0]):]
dataset_inp2_val = dataset_inp2[math.floor((ratio_train + ratio_test)*dataset_inp1.shape[0]):]
dataset_oup_val = dataset_oup[math.floor((ratio_train + ratio_test)*dataset_inp1.shape[0]):]

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

with open(path_dataset_pkl, 'wb') as pkl:
    pickle.dump(dataset, pkl)
    pkl.close()
print(f"\nThe dataset has saved into {path_dataset_pkl}")
