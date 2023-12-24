import math
import os.path
import pickle
import time
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import psutil
import pynvml
from tqdm import tqdm

from modules.base_settings import lr_init, device, epoch_max, batch_size
from modules.model_Pre_Anchors import Pre_Anchors
from modules.base_path import path_pts, path_pt_best, path_dataset_pkl, path_result, path_figs
from modules.base_size import sizes
from modules.util_train_test_val import mode_test, ModeTrain
from modules.util_plot import plot_during_train, plot_check_once_test, plot_during_test, plot_trajectories
from modules.util_criterion import Criterion_Train, Criterion_Test
from modules.util_make_trajectory import make_trajectory
from modules.base_paras import paras
from modules.analyse_oup_loss import analyse_distribution_anchors, analyse_distribution_loss, analyse_cloud_loss
from modules.base_paras import num_anchor_state, num_anchor_per_step
from modules.util_sending_data import Sending_email

torch.manual_seed(2023)

# 模型
PA = Pre_Anchors(device=device, multi_head_size=sizes['multi_head_size'],
                 sequence_length_inp=sizes['sequence_length_inp'],
                 sequence_length_oup=sizes['sequence_length_oup'],
                 encoder_input_size=sizes['encoder_input_size'],
                 encoder_middle_size=sizes['encoder_middle_size'],
                 encoder_output_size=sizes['encoder_output_size'],
                 decoder_input_size=sizes['decoder_input_size'],
                 decoder_middle_size=sizes['decoder_middle_size'],
                 decoder_output_size=sizes['decoder_output_size'],
                 paras=paras).to(device)
criterion_train = Criterion_Train()
criterion_test = Criterion_Test()

# 加载数据集
with open(path_dataset_pkl, 'rb') as pkl:
    dataset = pickle.load(pkl)
    pkl.close()

# 模式选择器
mode_switch = {
    "Train": True,
    "Test": False,
    "Make Trajectory": False,
    "Analyse_Distribution_Position_Origin": False,
    "Analyse_Distribution_Position_train&test": False,
    "Analyse_Planning_FailAtMatlab": False,
    "Plan_Once": False
}

if __name__ == '__main__':
    plt.figure(figsize=[13.072, 7.353])
    if mode_switch["Train"]:
        plt.clf()
        print("\n正在进行模型训练")

        optimizer = optim.Adam(PA.parameters(), lr=lr_init)
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=3, eta_min=1e-5)
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr_init, total_steps=epoch_max, pct_start=0.1)
        loss_all = np.zeros([epoch_max, 2])
        lr_all = np.zeros([epoch_max, 1])

        # 发送结果
        sending = Sending_email()
        sending.start()
        for epoch in tqdm(range(epoch_max), desc='Epoch', leave=False, ncols=80):
            train = ModeTrain(model=PA, batch_size=batch_size,
                              datas_inp1=dataset["dataset_inp1_train"],
                              datas_inp2=dataset["dataset_inp2_train"],
                              datas_oup=dataset["dataset_oup_train"],
                              criterion=criterion_train, optimizer=optimizer)
            train.run()
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            while True:
                time.sleep(0.5)
                with open(path_result, 'w') as result:
                    result.write(f"Now: {time.asctime(time.localtime())}")
                    result.write(f"\nEpoch: {epoch + 1}/{epoch_max}, Lr:{scheduler.get_last_lr()[0] * 1e4:.2f}*1e-4")
                    result.write(f"\nLoss_min: {train.loss_min:.2f}, Loss_max: {train.loss_max:.2f}")
                    result.write(f"\nCPU Percent: {psutil.cpu_percent()}%")
                    for d in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(d)
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        result.write(f"\nGPU: {d}, Used: {info.used/1048576:.2f}MB, Free:{info.free/1048576:.2f}MB, Temperature: {temperature}℃")
                    result.flush()
                if train.flag_finish:
                    with open(path_result, 'w') as result:
                        result.write(f"Now: {time.asctime(time.localtime())}")
                        result.write(f"\nEpoch: {epoch + 1}/{epoch_max}, Lr:{scheduler.get_last_lr()[0] * 1e4:.2f}*1e-4")
                        result.write(f"\nLoss_min: {train.loss_min:.2f}, Loss_max: {train.loss_max:.2f}")
                        result.write(f"\nCPU Percent: {psutil.cpu_percent()}%")
                        for d in range(device_count):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(d)
                            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            result.write(f"\nGPU: {d}, Used: {info.used / 1048576}MB, Free:{info.free / 1048576}MB, Temperature: {temperature}℃")
                        result.flush()
                    pynvml.nvmlShutdown()
                    break

            rand_para = torch.randperm(dataset["dataset_inp1_train"].shape[0])
            dataset["dataset_inp1_train"] = dataset["dataset_inp1_train"][rand_para]
            dataset["dataset_inp2_train"] = dataset["dataset_inp2_train"][rand_para]
            dataset["dataset_oup_train"] = dataset["dataset_oup_train"][rand_para]

            loss_all[epoch, :] = np.array([train.loss_min, train.loss_max])
            lr_all[epoch, 0] = scheduler.get_last_lr()[0]
            scheduler.step()
            # plot_during_train(epoch, loss_all, lr_all)
            torch.save(PA, os.path.join(path_pts, f"{epoch}_{loss_all[epoch, 1]:.2f}.pt"))
            if loss_all[epoch, 1] == loss_all[0:(epoch + 1), 1].min():
                torch.save(PA, path_pt_best)

        sending.flag_sending = False

        plot_during_train(epoch_max - 1, loss_all, lr_all)
        plt.savefig(f"{path_figs}/train.png")
    if mode_switch["Test"]:
        plt.clf()
        print("\n正在进行模型测试")

        PA = torch.load(path_pt_best)

        dataset_base = "test"
        datas_inp1 = dataset["dataset_inp1_" + dataset_base]
        datas_inp2 = dataset["dataset_inp2_" + dataset_base]
        datas_oup = dataset["dataset_oup_" + dataset_base]

        loss_xy_all = np.zeros([datas_inp1.shape[0], 8])
        loss_theta_all = np.zeros([datas_inp1.shape[0], 8])
        for i in range(datas_inp1.shape[0]):
            anchors, loss_xy, loss_theta = mode_test(model=PA, criterion=criterion_test,
                                                     data_inp1=datas_inp1[i], data_inp2=datas_inp2[i],
                                                     data_oup=datas_oup[i])
            loss_xy_all[i] = loss_xy
            loss_theta_all[i] = loss_theta
            # plot_check_once_test(anchors, datas_oup[i].cpu().detach().numpy(), loss_xy, loss_theta)
            schedule = "▋" * math.floor((i + 1) / datas_inp1.shape[0] * 10)
            print(f"\rDatasets: {schedule}, {(i + 1)}/{datas_inp1.shape[0]}", end='')
        print()
        plot_during_test(loss_xy_all, loss_theta_all)
        plt.savefig(f"{path_figs}/{dataset_base}.png")
    if mode_switch["Make Trajectory"]:
        plt.clf()
        print("\n正在进行生成轨迹")

        PA = torch.load(path_pt_best)

        dataset_base = "test"
        datas_inp1 = dataset["dataset_inp1_" + dataset_base]
        datas_inp2 = dataset["dataset_inp2_" + dataset_base]
        datas_oup = dataset["dataset_oup_" + dataset_base]

        mode = 1

        for i in range(datas_inp1.shape[0]):
            anchors, _, _ = mode_test(model=PA, criterion=criterion_test,
                                      data_inp1=datas_inp1[i], data_inp2=datas_inp2[i],
                                      data_oup=datas_oup[i])
            anchors = np.append(datas_inp1[i, :, 0:1].cpu().numpy(), anchors, axis=1)
            anchors = np.append(anchors, np.asarray([paras["end"][0:3]]).transpose(), axis=1)
            trajectories = make_trajectory(anchors, mode_switch=mode)
            plot_trajectories(trajectories, anchors)
            plt.savefig(f"{path_figs}/{dataset_base}/{datas_inp1[i, 0, 0]:.2f}_{datas_inp1[i, 1, 0]:.2f}_{datas_inp1[i, 2, 0]:.2f}_{mode}.png")
            schedule = "▋" * math.floor((i + 1) / datas_inp1.shape[0] * 10)
            print(f"\rDatasets: {schedule}, {(i + 1)}/{datas_inp1.shape[0]}", end='')
    if mode_switch["Analyse_Distribution_Position_Origin"]:
        plt.clf()
        print("\n正在分析原始节点数据分布")

        datas_oup = dataset["dataset_oup_all"].cpu().numpy()
        analyse_distribution_anchors(datas_oup)
        plt.savefig(f"{path_figs}/analyse_distribution_anchors_origin.png")
    if mode_switch["Analyse_Distribution_Position_train&test"]:
        plt.clf()
        print("\n正在分析模型输出数据分布")

        PA = torch.load(path_pt_best)

        dataset_base = "test"
        datas_inp1 = dataset["dataset_inp1_" + dataset_base]
        datas_inp2 = dataset["dataset_inp2_" + dataset_base]
        datas_oup = dataset["dataset_oup_" + dataset_base]

        anchors_oup = np.zeros(datas_oup.shape)
        loss_oup = np.zeros([datas_oup.shape[0], 2, datas_oup.shape[2]])

        for i in range(datas_inp1.shape[0]):
            anchors, loss_xy, loss_theta = mode_test(model=PA, criterion=criterion_test,
                                                     data_inp1=datas_inp1[i], data_inp2=datas_inp2[i],
                                                     data_oup=datas_oup[i])
            anchors_oup[i] = anchors
            loss_oup[i] = np.array([loss_xy, loss_theta])

            schedule = "▋" * math.floor((i + 1) / datas_inp1.shape[0] * 10)
            print(f"\rDatasets: {schedule}, {(i + 1)}/{datas_inp1.shape[0]}", end='')

        analyse_cloud_loss(loss_oup, datas_inp1.cpu().numpy()[:, :, 0], sample_num=30)
        plt.savefig(f"{path_figs}/analyse_cloud_loss_{dataset_base}.png")
        analyse_distribution_anchors(anchors_oup)
        plt.savefig(f"{path_figs}/analyse_distribution_anchors_{dataset_base}.png")
        analyse_distribution_loss(loss_oup)
        plt.savefig(f"{path_figs}/analyse_distribution_loss_{dataset_base}.png")
    if mode_switch["Analyse_Planning_FailAtMatlab"]:
        plt.clf()
        print("\n正在分析模型对于matlab失败的数据的输出")

        PA = torch.load(path_pt_best)

        dataset_base = "failed"
        datas_inp1 = dataset["dataset_inp1_" + dataset_base]
        datas_inp2 = dataset["dataset_inp2_" + dataset_base]
        datas_oup = dataset["dataset_oup_" + dataset_base]

        anchors_oup = np.zeros(datas_oup.shape)

        for i in range(datas_inp1.shape[0]):
            anchors, _, _ = mode_test(model=PA, criterion=criterion_test,
                                      data_inp1=datas_inp1[i], data_inp2=datas_inp2[i],
                                      data_oup=datas_oup[i])
            anchors_oup[i] = anchors

            schedule = "▋" * math.floor((i + 1) / datas_inp1.shape[0] * 10)
            print(f"\rDatasets: {schedule}, {(i + 1)}/{datas_inp1.shape[0]}", end='')

        analyse_distribution_anchors(anchors_oup)
        plt.savefig(f"{path_figs}/analyse_distribution_anchors_{dataset_base}.png")
        print()
    if mode_switch["Plan_Once"]:
        plt.clf()
        print("\n正在进行单点规划")

        PA = torch.load(path_pt_best)

        x, y, yaw = -4.965, 7.745, 0.0069

        anchor_start = np.array([x, y, yaw])
        anchor_end = paras["end"][0:3]

        dataset_inp1 = np.zeros([num_anchor_state, num_anchor_per_step])
        dataset_inp2 = np.zeros([num_anchor_state, num_anchor_per_step])

        dataset_inp1[:, 0] = anchor_start
        dataset_inp1[:, 1] = anchor_end
        dataset_inp1[:, 2] = np.array([paras["limits"][0, 0], paras["limits"][1, 1], 0.0])
        dataset_inp1[:, 3] = np.array([paras["limits"][0, 1], paras["limits"][1, 1], 0.0])
        dataset_inp1[:, 4] = np.array([paras["limits"][0, 0], paras["Parking_Y"], 0.0])
        dataset_inp1[:, 5] = np.array([paras["limits"][0, 1], paras["Parking_Y"], 0.0])
        dataset_inp1[:, 6] = np.array([-paras["Parking_X"] / 2, paras["Parking_Y"], 0.0])
        dataset_inp1[:, 7] = np.array([paras["Parking_X"] / 2, paras["Parking_Y"], 0.0])

        x_center = np.array([-(paras["Parking_X"] / 2 + paras["Freespace_X"]) / 2, (paras["Parking_X"] / 2 + paras["Freespace_X"]) / 2])
        y_center = np.array([paras["Parking_Y"] + paras["Freespace_Y"] * 3 / 4, paras["Parking_Y"] + paras["Freespace_Y"] / 4])
        dataset_inp2[0:2, 0] = np.array([x_center[0], y_center[0]])
        dataset_inp2[0:2, 1] = anchor_start[0:2] - np.array([x_center[0], y_center[0]])
        dataset_inp2[0:2, 2] = np.array([x_center[0], y_center[1]])
        dataset_inp2[0:2, 3] = anchor_start[0:2] - np.array([x_center[0], y_center[1]])
        dataset_inp2[0:2, 4] = np.array([x_center[1], y_center[0]])
        dataset_inp2[0:2, 5] = anchor_start[0:2] - np.array([x_center[1], y_center[0]])
        dataset_inp2[0:2, 6] = np.array([x_center[1], y_center[1]])
        dataset_inp2[0:2, 7] = anchor_start[0:2] - np.array([x_center[1], y_center[1]])

        dataset_inp1 = torch.from_numpy(dataset_inp1).to(torch.float32).to(device)
        dataset_inp2 = torch.from_numpy(dataset_inp2).to(torch.float32).to(device)

        mode = 0
        anchors, _, _ = mode_test(model=PA, criterion=criterion_test,
                                  data_inp1=dataset_inp1, data_inp2=dataset_inp2,
                                  data_oup=None)
        anchors = np.append(dataset_inp1[:, 0:1].cpu().numpy(), anchors, axis=1)
        anchors = np.append(anchors, np.asarray([paras["end"][0:3]]).transpose(), axis=1)
        trajectories = make_trajectory(anchors, mode_switch=mode)
        plot_trajectories(trajectories, anchors)
        plt.savefig(f"{path_figs}/Once/{dataset_inp1[0, 0]:.2f}_{dataset_inp1[1, 0]:.2f}_{dataset_inp1[2, 0]:.2f}_{mode}.png")

