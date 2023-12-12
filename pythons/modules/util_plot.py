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
    plt.plot(np.arange(0, epoch + 1, 1), loss_all[0:(epoch + 1), 0], "g", label="log(min)")
    plt.plot(np.arange(0, epoch + 1, 1), loss_all[0:(epoch + 1), 1], "r", label="log(max)")
    plt.plot([0, epoch], [0.0, 0.0], "k--")
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, epoch + 1, 1), lr_all[0:(epoch + 1), 0], "k", label="lr_init")
    plt.legend(loc='upper right')

    plt.pause(0.01)


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

    max_loss_xy = np.argmax(loss_xy, 1)
    max_loss_theta = np.argmax(np.abs(loss_theta), 1)

    plt.subplot(2, 2, 1)
    plt.title("loss_xy")
    plt.plot(loss_xy.max(1))

    plt.subplot(2, 2, 2)
    plt.title("loss_xy_distribution")
    for i in range(loss_xy.shape[1]):
        plt.bar(i, len(np.where(max_loss_xy == i)[0]), width=1)
        plt.text(i, len(np.where(max_loss_xy == i)[0]), len(np.where(max_loss_xy == i)[0]), fontsize=10)

    plt.subplot(2, 2, 3)
    plt.title("loss_theta")
    plt.plot(loss_theta.max(1))

    plt.subplot(2, 2, 4)
    plt.title("loss_theta_distribution")
    for i in range(loss_theta.shape[1]):
        plt.bar(i, len(np.where(max_loss_theta == i)[0]), width=1)
        plt.text(i, len(np.where(max_loss_theta == i)[0]), len(np.where(max_loss_theta == i)[0]), fontsize=10)


def plot_trajectories(tra_pre, anchors):

    plt.clf()

    plot_base()
    txt = os.path.join(path_solutions, f"{anchors[0, 0]:.2f}_{anchors[1, 0]:.2f}_{anchors[2, 0]:.2f}.txt")
    if os.path.exists(txt):
        xy = np.loadtxt(txt)
        plt.plot(xy[:, 2], xy[:, 3], "k--")
    for tra in tra_pre:
        plt.plot(tra_pre[tra][:, 0], tra_pre[tra][:, 1])
    for anchor in range(anchors.shape[1]):
        plt.plot([anchors[0, anchor], anchors[0, anchor] + math.cos(anchors[2, anchor]) * 0.5],
                 [anchors[1, anchor], anchors[1, anchor] + math.sin(anchors[2, anchor]) * 0.5], "b")
    # plt.show()
    # plt.pause(0.1)


def plot_anchors(anchors):
    for anchor in range(anchors.shape[1]):
        # plt.plot([anchors[0, anchor], anchors[0, anchor] + math.cos(anchors[2, anchor]) * 0.5],
        #          [anchors[1, anchor], anchors[1, anchor] + math.sin(anchors[2, anchor]) * 0.5], "k--")
        plt.plot(anchors[0, anchor], anchors[1, anchor], "k.")
