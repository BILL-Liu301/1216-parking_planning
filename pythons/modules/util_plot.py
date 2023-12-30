import math
import os.path

import matplotlib.pyplot as plt
import numpy as np

from .base_paras import paras
from .base_path import path_solutions


def plot_base():
    plt.plot([paras["limits"][0, 0], paras["limits"][0, 1]], [paras["limits"][1, 1], paras["limits"][1, 1]], "k")
    plt.plot([paras["limits"][0, 0], -paras["Parking_X"] / 2], [paras["Parking_Y"], paras["Parking_Y"]], "k")
    plt.plot([paras["Parking_X"] / 2, paras["limits"][0, 1]], [paras["Parking_Y"], paras["Parking_Y"]], "k")
    plt.plot([-paras["Parking_X"] / 2, -paras["Parking_X"] / 2], [paras["limits"][1, 0], paras["Parking_Y"]], "k")
    plt.plot([paras["Parking_X"] / 2, paras["Parking_X"] / 2], [paras["limits"][1, 0], paras["Parking_Y"]], "k")
    plt.plot([-paras["Parking_X"] / 2, paras["Parking_X"] / 2], [paras["limits"][1, 0], paras["limits"][1, 0]], "k")


def plot_during_train(epoch, loss_all, lr_all):
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, epoch + 1, 1), loss_all[0:(epoch + 1), 0], "g", label="min")
    plt.plot(np.arange(0, epoch + 1, 1), loss_all[0:(epoch + 1), 1], "r", label="max")
    plt.plot([0, epoch], [0.0, 0.0], "k--")
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, epoch + 1, 1), lr_all[0:(epoch + 1), 0], "k", label="lr_init")
    plt.legend(loc='upper right')

    # plt.pause(0.01)


def plot_check_once_test(pre, ref, loss_xy, loss_theta):
    Car_Length = paras['Car_L']

    pre_xy1 = pre[0:2, :]
    ang = np.stack([np.cos(pre[2, :]), np.sin(pre[2, :])], axis=0)
    pre_xy2 = pre_xy1 + Car_Length * ang
    pre_xy = np.append(pre_xy1, pre_xy2, axis=0)

    ref_xy1 = ref[0:2, :]
    ang = np.stack([np.cos(ref[2, :]), np.sin(ref[2, :])], axis=0)
    ref_xy2 = ref_xy1 + Car_Length * ang
    ref_xy = np.append(ref_xy1, ref_xy2, axis=0)

    for i in range(ref_xy.shape[1]):
        plt.plot(ref_xy[0, i], ref_xy[1, i], "bo")
        plt.plot(ref_xy[2, i], ref_xy[3, i], "bo")
        plt.plot([ref_xy[0, i], ref_xy[2, i]], [ref_xy[1, i], ref_xy[3, i]], "b")
        plt.plot(pre_xy[0, i], pre_xy[1, i], "r+")
        plt.plot(pre_xy[2, i], pre_xy[3, i], "r+")
        plt.plot([pre_xy[0, i], pre_xy[2, i]], [pre_xy[1, i], pre_xy[3, i]], "r--")
        plt.text((ref_xy[0, i] + pre_xy[0, i]) / 2, (ref_xy[1, i] + pre_xy[1, i]) / 2,
                 f"loss_xy = {loss_xy[i]:.2f}m, loss_theta = {loss_theta[i]:.2f}°", fontsize=10)
    plt.show()


def plot_during_test(loss_xy, loss_theta):
    plt.clf()

    max_loss_xy = np.argmax(loss_xy, 2)
    max_loss_theta = np.argmax(np.abs(loss_theta), 2)

    colors = ['k', 'r', 'g', 'b']

    # 误差曲线
    plt.subplot(2, 2, 1)
    for i, loss in enumerate(loss_xy.max(0)):
        plt.plot(loss, colors[i], label=f'Step{i+1}')
    plt.legend()

    # 误差分布
    max_loss = max_loss_xy
    loss = loss_xy
    for step in range(max_loss.shape[1]):
        plt.subplot(2*loss.shape[1], 2, 2*(step+1))
        for i in range(loss.shape[2]):
            loss_rate = len(np.where(max_loss[:, step] == i)[0]) / max_loss.shape[0]
            plt.bar(i, loss_rate, width=1)
            if len(np.where(max_loss[:, step] == i)[0]) != 0:
                plt.text(i, loss_rate, f'{loss_rate:.2f}', fontsize=5, ha='center', va='bottom')
        plt.tick_params(axis='x', pad=0.03, labelsize=7)
        plt.tick_params(axis='y', pad=0.03, labelsize=10)
        plt.xlim(-1, loss.shape[2] + 1)
        plt.ylim(0, 1.1)
        plt.grid(True)

    # 误差曲线
    plt.subplot(2, 2, 3)
    for i, loss in enumerate(loss_theta.max(0)):
        plt.plot(loss, colors[i], label=f'Step{i+1}')
    plt.legend()

    # 误差分布
    max_loss = max_loss_theta
    loss = loss_theta
    for step in range(max_loss.shape[1]):
        ax = plt.subplot(2*loss.shape[1], 2, 2*(step+max_loss.shape[1]+1))
        for i in range(loss.shape[2]):
            loss_rate = len(np.where(max_loss[:, step] == i)[0]) / max_loss.shape[0]
            plt.bar(i, loss_rate, width=1)
            if len(np.where(max_loss[:, step] == i)[0]) != 0:
                plt.text(i, loss_rate, f'{loss_rate:.2f}', fontsize=5, ha='center', va='bottom')
        plt.tick_params(axis='x', pad=0.03, labelsize=7)
        plt.tick_params(axis='y', pad=0.03, labelsize=10)
        plt.xlim(-1, loss.shape[2] + 1)
        plt.ylim(0, 1.1)
        ax.axvspan(xmin=plt.gca().get_xlim()[0], xmax=plt.gca().get_xlim()[1], ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], color='gray', alpha=0.1)
        plt.grid(True)
    # plt.show()


def plot_trajectories(ref, tra_pre, anchors_steps):

    plt.clf()

    plot_base()
    txt = os.path.join(path_solutions, f"{ref[0, 0, 0]:.4f}_{ref[0, 1, 0]:.4f}_{ref[0, 2, 0]:.4f}.txt")
    if os.path.exists(txt):
        xy = np.loadtxt(txt)
        plt.plot(xy[:, 2], xy[:, 3], "k--")
    for anchors in anchors_steps:
        plt.plot(anchors[0], anchors[1], 'b')
    for tra in tra_pre:
        plt.plot(tra_pre[tra][:, 0], tra_pre[tra][:, 1], 'r')
    # plt.show()
    plt.grid(True)
    plt.pause(0.1)


def plot_anchors(anchors):
    for anchor in range(anchors.shape[1]):
        # plt.plot([anchors[0, anchor], anchors[0, anchor] + math.cos(anchors[2, anchor]) * 0.5],
        #          [anchors[1, anchor], anchors[1, anchor] + math.sin(anchors[2, anchor]) * 0.5], "k--")
        plt.plot(anchors[0, anchor], anchors[1, anchor], "k.")
