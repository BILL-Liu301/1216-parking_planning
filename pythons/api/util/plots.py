import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon, Circle, Rectangle
import matplotlib.transforms as transforms

from api.base.paras import paras_base
from api.base.paths import path_figs_init


def plot_base():
    plt.plot([paras_base['limits'][0, 0], paras_base['limits'][0, 1]], [paras_base['limits'][1, 1], paras_base['limits'][1, 1]], 'k')
    plt.plot([paras_base['limits'][0, 0], -paras_base['Parking_X'] / 2], [paras_base['Parking_Y'], paras_base['Parking_Y']], 'k')
    plt.plot([paras_base['Parking_X'] / 2, paras_base['limits'][0, 1]], [paras_base['Parking_Y'], paras_base['Parking_Y']], 'k')
    plt.plot([-paras_base['Parking_X'] / 2, -paras_base['Parking_X'] / 2], [paras_base['limits'][1, 0], paras_base['Parking_Y']], 'k')
    plt.plot([paras_base['Parking_X'] / 2, paras_base['Parking_X'] / 2], [paras_base['limits'][1, 0], paras_base['Parking_Y']], 'k')
    plt.plot([-paras_base['Parking_X'] / 2, paras_base['Parking_X'] / 2], [paras_base['limits'][1, 0], paras_base['limits'][1, 0]], 'k')
    plt.plot([paras_base['limits'][0, 0], paras_base['limits'][0, 0]], [paras_base['Parking_Y'], paras_base['limits'][1, 1]], 'k')
    plt.plot([paras_base['limits'][0, 1], paras_base['limits'][0, 1]], [paras_base['Parking_Y'], paras_base['limits'][1, 1]], 'k')


def plot_failed_init(init_data):
    plt.figure(figsize=(13.072, 7.353))
    xmid_r = init_data[0]
    ymid_r = init_data[1]
    xmid_f = xmid_r + np.cos(init_data[2]) * paras_base['Car_L']
    ymid_f = ymid_r + np.sin(init_data[2]) * paras_base['Car_L']
    xmid_r = xmid_r - np.cos(init_data[2]) * (paras_base['Car_Length'] - paras_base['Car_L']) / 2
    ymid_r = ymid_r - np.sin(init_data[2]) * (paras_base['Car_Length'] - paras_base['Car_L']) / 2
    xmid_f = xmid_f + np.cos(init_data[2]) * (paras_base['Car_Length'] - paras_base['Car_L']) / 2
    ymid_f = ymid_f + np.sin(init_data[2]) * (paras_base['Car_Length'] - paras_base['Car_L']) / 2

    xr_r = xmid_r + np.cos(init_data[2] - np.pi / 2) * paras_base['Car_Width'] / 2
    yr_r = ymid_r + np.sin(init_data[2] - np.pi / 2) * paras_base['Car_Width'] / 2
    xr_l = xmid_r + np.cos(init_data[2] + np.pi / 2) * paras_base['Car_Width'] / 2
    yr_l = ymid_r + np.sin(init_data[2] + np.pi / 2) * paras_base['Car_Width'] / 2
    xf_r = xmid_f + np.cos(init_data[2] - np.pi / 2) * paras_base['Car_Width'] / 2
    yf_r = ymid_f + np.sin(init_data[2] - np.pi / 2) * paras_base['Car_Width'] / 2
    xf_l = xmid_f + np.cos(init_data[2] + np.pi / 2) * paras_base['Car_Width'] / 2
    yf_l = ymid_f + np.sin(init_data[2] + np.pi / 2) * paras_base['Car_Width'] / 2

    plt.title('Init Data', fontsize=10)
    plot_base()
    plt.plot([xr_r, xr_l], [yr_r, yr_l], 'b')
    plt.plot([xr_l, xf_l], [yr_l, yf_l], 'b')
    plt.plot([xf_l, xf_r], [yf_l, yf_r], 'b')
    plt.plot([xf_r, xr_r], [yf_r, yr_r], 'b')

    plt.savefig(f'{path_figs_init}{init_data[0]:.2f}_{init_data[1]:.2f}_{init_data[2]:.2f}.png')
    plt.close()


def plot_for_results_macro(_result, paras):
    plt.clf()

    colors = ['r', 'g', 'b', 'k']
    labels = ['step_1', 'step_2', 'step_3', 'step_4']

    plot_base()
    xlim, ylim = plt.xlim(), plt.ylim()
    pre, ref, std = _result['pre'], _result['ref'], np.sqrt(_result['pre_var'])
    for step in range(paras['num_step']):
        plt.plot(pre[step, 0], pre[step, 1], colors[step] + '-')
        x1 = pre[step, 0, :] - std[step, 0, :]
        x2 = pre[step, 0, :] + std[step, 0, :]
        y1 = pre[step, 1, :] - std[step, 1, :]
        y2 = pre[step, 1, :] + std[step, 1, :]
        for p in range(pre.shape[2]):
            plt.fill_between([x1[p], x2[p]], [y1[p], y1[p]], [y2[p], y2[p]], color=colors[step], alpha=0.5)
        plt.plot(ref[step, 0], ref[step, 1], colors[step] + '--', label=labels[step])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc='lower right')


def plot_for_results_micro_x_y_theta(plot_id, data_id, name, colors, labels, _result, paras):
    pre, ref, std, loss_l1 = _result['pre'], _result['ref'], np.sqrt(_result['pre_var']), _result['loss_l1']

    ax = plt.subplot(6, 1, plot_id)
    x = np.linspace(1, paras['num_anchor_per_step'], paras['num_anchor_per_step'])
    colors_sigma = ['r', 'b', 'k']
    _temp = np.append(pre[:, data_id].reshape([1, -1]), ref[:, data_id].reshape([1, -1]))
    for step in range(paras['num_step']):
        plt.plot(x + step * paras['num_anchor_per_step'], pre[step, data_id], colors[step] + '-', label=labels[step])
        plt.plot(x + step * paras['num_anchor_per_step'], ref[step, data_id], colors[step] + '-.')
        plt.plot([(step + 1) * paras['num_anchor_per_step'], (step + 1) * paras['num_anchor_per_step']],
                 [np.min(_temp), np.max(_temp)], 'k--')

        xx = np.append(x + step * paras['num_anchor_per_step'], np.flip(x + step * paras['num_anchor_per_step']), axis=0)
        for i in range(3):
            yy = np.append(pre[step, data_id] + std[step, data_id] * (3 - i), np.flip(pre[step, data_id] - std[step, data_id] * (3 - i)), axis=0)
            polygon = Polygon(np.column_stack((xx, yy)), color=colors_sigma[i], alpha=0.3)
            ax.add_patch(polygon)
    plt.ylabel(name, fontsize=15)
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(6, 1, plot_id + 1)
    x = np.linspace(1, paras['num_anchor_per_step'], paras['num_anchor_per_step'])
    _temp = loss_l1[:, data_id]
    for step in range(paras['num_step']):
        plt.plot(x + step * paras['num_anchor_per_step'], loss_l1[step, data_id], colors[step] + '-', label=labels[step])
        plt.plot([(step + 1) * paras['num_anchor_per_step'], (step + 1) * paras['num_anchor_per_step']],
                 [np.min(_temp), np.max(_temp)], 'k--')
    plt.ylabel(name + '_loss', fontsize=15)
    plt.legend(loc='lower right')
    plt.grid(True)


def plot_for_results_micro(_result, paras):
    plt.clf()

    colors = ['c', 'm', 'y', 'k']
    labels = ['step_1', 'step_2', 'step_3', 'step_4']

    # 分析x
    plot_for_results_micro_x_y_theta(1, 0, 'x', colors, labels, _result, paras)
    # 分析y
    plot_for_results_micro_x_y_theta(3, 1, 'y', colors, labels, _result, paras)
    # 分析theta
    plot_for_results_micro_x_y_theta(5, 2, 'theta', colors, labels, _result, paras)


# def plot_for_results_dynamic(_result, paras, map_x=None, map_y=None, view_type='ogm'):
#     colors = ['r', 'g', 'b', 'k']
#     labels = ['step_1', 'step_2', 'step_3', 'step_4']
#
#     pre = np.concatenate([_result['pre'][i] for i in range(paras['num_step'])], axis=-1)
#     ref = _result['ref']
#     view = np.concatenate([_result['view'][i] for i in range(paras['num_step'])], axis=-1)
#     state = np.concatenate([_result['state'][i] for i in range(paras['num_step'])], axis=-1)
#     map_np = paras['map']
#
#     for stamp in range(pre.shape[-1]):
#         plt.clf()
#
#         ax = plt.subplot(1, 2, 1)
#         # 基本场景
#         # plot_base()
#         plt.plot(map_np[0], map_np[1], "ko")
#         xlim, ylim = plt.xlim(), plt.ylim()
#         for step in range(paras['num_step']):
#             plt.plot(ref[step, 0], ref[step, 1], colors[step] + '--', label=labels[step])
#             plt.plot(ref[step, 0], ref[step, 1], colors[step] + '.')
#
#         # 车辆
#         plt.plot(pre[0, stamp], pre[1, stamp], 'ko')
#         plt.arrow(pre[0, stamp], pre[1, stamp], paras['car_length'] * np.cos(pre[2, stamp]), paras['car_length'] * np.sin(pre[2, stamp]), head_width=0.3, head_length=0.5)
#         rect = Rectangle(xy=(pre[0, stamp] - paras['map_width_half'] * paras['map_interval'], pre[1, stamp] - paras['map_height_half'] * paras['map_interval']),
#                          width=2 * paras['map_width_half'] * paras['map_interval'], height=2 * paras['map_height_half'] * paras['map_interval'],
#                          angle=pre[2, stamp] * 180 / math.pi, rotation_point='center', linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_artist(rect)
#         plt.plot(pre[0, 0:(stamp + 1)], pre[1, 0:(stamp + 1)], 'k-')
#         plt.xlim(xlim)
#         plt.ylim(ylim)
#         plt.legend(loc='lower right')
#
#         ax = plt.subplot(1, 2, 2)
#         for w in range(view.shape[0]):
#             for h in range(view.shape[1]):
#                 if view[w, h, stamp] == 1:
#                     plt.plot(map_x[w], map_y[h], 'ks')
#         plt.arrow(0.0, 0.0, paras['car_length'], 0.0, head_width=0.3, head_length=0.5)
#         plt.arrow(paras['car_length'], 0.0, np.cos(state[1, stamp]), np.sin(state[1, stamp]), head_width=0.3, head_length=0.5)
#         plt.xlim([-paras['map_range'], paras['map_range']])
#         plt.ylim([-paras['map_range'], paras['map_range']])
#
#         plt.pause(0.01)
def plot_for_results_dynamic(_result, map_x, map_y, paras):
    colors = ['r', 'g', 'b', 'k']
    labels = ['step_1', 'step_2', 'step_3', 'step_4']

    pre = np.concatenate([_result['pre'][i] for i in range(paras['num_step'])], axis=-1)
    ref = _result['ref']
    view = np.concatenate([_result['view'][i] for i in range(paras['num_step'])], axis=-1)
    state = np.concatenate([_result['state'][i] for i in range(paras['num_step'])], axis=-1)
    map_np = paras['map']

    for stamp in range(pre.shape[-1]):
        ax = plt.subplot(1, 2, 1)
        # 基本场景
        # plot_base()
        plt.plot(map_np[0], map_np[1], "ko")
        xlim, ylim = plt.xlim(), plt.ylim()
        for step in range(paras['num_step']):
            plt.plot(ref[step, 0], ref[step, 1], colors[step] + '--', label=labels[step])
            plt.plot(ref[step, 0], ref[step, 1], colors[step] + '.')

        # 车辆
        plt.plot(pre[0, stamp], pre[1, stamp], 'ko')
        plt.arrow(pre[0, stamp], pre[1, stamp], paras['car_length'] * np.cos(pre[2, stamp]), paras['car_length'] * np.sin(pre[2, stamp]), head_width=0.3, head_length=0.5)
        circle = Circle((pre[0, stamp], pre[1, stamp]), paras['map_range'], fill=False)
        ax.add_artist(circle)
        plt.plot(pre[0, 0:(stamp + 1)], pre[1, 0:(stamp + 1)], 'k-')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend(loc='lower right')

        ax = plt.subplot(1, 2, 2)
        # 基本场景
        # plt.plot(view[0, :, stamp], view[1, :, stamp], 'ko')
        for w in range(view.shape[0]):
            for h in range(view.shape[1]):
                if view[w, h, stamp] == 1:
                    plt.plot(map_x[w], map_y[h], 'ks')
        plt.arrow(0.0, 0.0, paras['car_length'], 0.0, head_width=0.3, head_length=0.5)
        # plt.arrow(paras['car_length'], 0.0, np.cos(state[1, stamp]), np.sin(state[1, stamp]), head_width=0.3, head_length=0.5)
        # circle = Circle((0.0, 0.0), paras['map_range'], fill=False)
        # ax.add_artist(circle)
        plt.xlim([-paras['map_range'], paras['map_range']])
        plt.ylim([-paras['map_range'], paras['map_range']])

        plt.pause(0.1)
        plt.clf()
