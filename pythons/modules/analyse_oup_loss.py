import numpy as np
import matplotlib.pyplot as plt

from .util_plot import plot_base, plot_anchors
from .base_paras import paras


def analyse_distribution_anchors(anchors_oup):
    plt.clf()

    anchors_front = np.stack([anchors_oup[:, 0, :] + paras["Car_L"] * np.cos(anchors_oup[:, 2, :]),
                              anchors_oup[:, 1, :] + paras["Car_L"] * np.sin(anchors_oup[:, 2, :]),
                              anchors_oup[:, 2, :]], axis=1)

    anchors_rear_dis, anchors_rear_dis_min, anchors_rear_dis_mean, anchors_rear_dis_max = cal_dis(anchors_oup)
    anchors_front_dis, anchors_front_dis_min, anchors_front_dis_mean, anchors_front_dis_max = cal_dis(anchors_front)

    anchors_rear_dis_trans = [anchors_rear_dis[:, i] for i in range(anchors_rear_dis.shape[1])]
    anchors_front_dis_trans = [anchors_front_dis[:, i] for i in range(anchors_front_dis.shape[1])]

    plt.subplot(2, 2, 1)
    plt.grid(True, color='c', linewidth=0.3)
    plt.title("anchors_rear_dis")
    plt.ylabel("dis")
    plt.boxplot(anchors_rear_dis_trans, showfliers=False, showmeans=True, meanline=True)

    plt.subplot(2, 2, 2)
    plt.grid(True, color='c', linewidth=0.3)
    plt.title("anchors_front_dis")
    plt.ylabel("dis")
    plt.boxplot(anchors_front_dis_trans, showfliers=False, showmeans=True, meanline=True)

    plt.subplot(2, 2, 3)
    plt.grid(True, color='c', linewidth=0.3)
    plt.title("anchors_rear_dis")
    plt.ylabel("dis")
    x = [i+1 for i in range(anchors_oup.shape[2])]
    plt.bar(x, anchors_rear_dis_max, width=0.7, color=['pink'])
    plt.plot(x, anchors_rear_dis_max, "r*")
    plt.plot(x, anchors_rear_dis_max, "r--")
    plt.plot(x, anchors_rear_dis_mean, "g*")
    plt.plot(x, anchors_rear_dis_mean, "g--")
    plt.plot(x, anchors_rear_dis_min, "b*")
    plt.plot(x, anchors_rear_dis_min, "b--")

    plt.subplot(2, 2, 4)
    plt.grid(True, color='c', linewidth=0.3)
    plt.title("anchors_front_dis")
    plt.ylabel("dis")
    x = [i+1 for i in range(anchors_oup.shape[2])]
    plt.bar(x, anchors_front_dis_max, width=0.7, color=['pink'])
    plt.plot(x, anchors_front_dis_max, "r*")
    plt.plot(x, anchors_front_dis_max, "r--")
    plt.plot(x, anchors_front_dis_mean, "g*")
    plt.plot(x, anchors_front_dis_mean, "g--")
    plt.plot(x, anchors_front_dis_min, "b*")
    plt.plot(x, anchors_front_dis_min, "b--")


def analyse_distribution_loss(loss_oup):
    plt.clf()

    loss_mean = abs(loss_oup).mean(axis=0)
    loss_max = abs(loss_oup).max(axis=0)
    loss_min = abs(loss_oup).min(axis=0)

    loss_position_trans = [loss_oup[:, 0, i] for i in range(loss_oup.shape[2])]
    loss_theta_trans = [loss_oup[:, 1, i] for i in range(loss_oup.shape[2])]

    plt.subplot(2, 2, 1)
    plt.grid(True, color='c', linewidth=0.3)
    plt.title("loss_position")
    plt.ylabel("loss")
    plt.boxplot(loss_position_trans, showfliers=False, showmeans=True, meanline=True)

    plt.subplot(2, 2, 2)
    plt.grid(True, color='c', linewidth=0.3)
    plt.title("loss_theta")
    plt.ylabel("loss")
    plt.boxplot(loss_theta_trans, showfliers=False, showmeans=True, meanline=True)

    plt.subplot(2, 2, 3)
    plt.grid(True, color='c', linewidth=0.3)
    plt.title("loss_position")
    plt.ylabel("loss")
    x = [i+1 for i in range(loss_oup.shape[2])]
    plt.bar(x, loss_max[0, :], width=0.7, color=['pink'])
    plt.plot(x, loss_max[0, :], "r*")
    plt.plot(x, loss_max[0, :], "r--")
    plt.plot(x, loss_mean[0, :], "g*")
    plt.plot(x, loss_mean[0, :], "g--")
    plt.plot(x, loss_min[0, :], "b*")
    plt.plot(x, loss_min[0, :], "b--")

    plt.subplot(2, 2, 4)
    plt.grid(True, color='c', linewidth=0.3)
    plt.title("loss_position")
    plt.ylabel("loss")
    x = [i+1 for i in range(loss_oup.shape[2])]
    plt.bar(x, loss_max[1, :], width=0.7, color=['pink'])
    plt.plot(x, loss_max[1, :], "r*")
    plt.plot(x, loss_max[1, :], "r--")
    plt.plot(x, loss_mean[1, :], "g*")
    plt.plot(x, loss_mean[1, :], "g--")
    plt.plot(x, loss_min[1, :], "b*")
    plt.plot(x, loss_min[1, :], "b--")


def analyse_cloud_loss(loss_oup, anchors_init, sample_num=20):
    loss_oup_mean = loss_oup.sum(axis=2)

    samples_x = np.linspace(anchors_init[:, 0].min(), anchors_init[:, 0].max(), sample_num)
    samples_y = np.linspace(anchors_init[:, 1].min(), anchors_init[:, 1].max(), sample_num)
    loss_average = cal_average_loss(samples_x, samples_y, anchors_init, loss_oup_mean[:, 0:1])

    plot_base()
    plt.contourf(samples_x, samples_y, loss_average)
    plt.colorbar()


def cal_dis(anchors):
    anchors_mean = anchors.mean(axis=0)
    anchors_dis = np.sqrt(np.sum(np.power(anchors - np.array([anchors_mean]), 2)[:, 0:2, :], axis=1))
    anchors_dis_mean = anchors_dis.mean(axis=0)
    anchors_dis_max = anchors_dis.max(axis=0)
    anchors_dis_min = anchors_dis.min(axis=0)
    return anchors_dis, anchors_dis_min, anchors_dis_mean, anchors_dis_max


def cal_average_loss(samples_x, samples_y, anchors_init, loss):

    loss_average = np.zeros([samples_x.shape[0], samples_y.shape[0]])
    for index_x, x in enumerate(samples_x):
        for index_y, y in enumerate(samples_y):
            loss_cal = np.array([])
            loss_cal_index = np.where((anchors_init[:, 0] < samples_x[min(index_x + 1, samples_x.shape[0] - 1)]) &
                                      (anchors_init[:, 0] > samples_x[max(index_x - 1, 0)]))
            if loss_cal_index[0].shape[0] == 0:
                loss_cal = loss[np.argmin(np.linalg.norm(anchors_init[:, 0:2] - np.array([x, y]), axis=1))]
            else:
                loss_cal = loss[loss_cal_index]

            loss_average[index_x, index_y] = loss_cal.mean(axis=0)
    return loss_average
