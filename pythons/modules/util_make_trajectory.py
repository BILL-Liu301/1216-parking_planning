import math
import numpy as np
import matplotlib.pyplot as plt

from .util_plot import plot_base
from .base_paras import num_step, paras


def make_trajectory(anchors_steps, points_per_search=10, r=0.1, eps=1e-3, mode_switch=0):
    trajectories = {}
    for step, anchors in enumerate(anchors_steps):

        # # 旋转坐标系，避免90°的点
        # theta = None
        # if max(abs(anchors[2, 0]), abs(anchors[2, -1])) >= 80 * math.pi / 180:
        #     theta = -(anchors[2, 0] + anchors[2, -1]) / 2
        #     rotate = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        #     points = np.append(np.dot(rotate, anchors[0:2]), anchors[2:3] + theta, axis=0)

        # 最小二乘拟合
        if mode_switch == 0:
            P = np.polyfit(anchors[0], anchors[1], 3)
        else:
            P = None

        trajectory = np.array([[anchors[0, 0], anchors[1, 0]]])
        while True:
            # plot_base()
            # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b*')
            # plt.plot(anchors[0], anchors[1], 'k')
            # plt.pause(0.1)
            # plt.clf()
            x_0, y_0 = trajectory[-1, 0], trajectory[-1, 1]
            if abs(x_0 - anchors[0, -1]) <= r:
                x_1, y_1 = anchors[0, -1], anchors[1, -1]
                trajectory = np.append(trajectory, np.array([[x_1, y_1]]), axis=0)
                # if not theta is None:
                #     rotate = np.array([[math.cos(-theta), -math.sin(-theta)], [math.sin(-theta), math.cos(-theta)]])
                #     trajectory = np.dot(rotate, trajectory.transpose()).transpose()
                break
            else:
                x = np.arange(min(x_0, x_0 + math.copysign(r * 2, anchors[0, -1] - anchors[0, 0])),
                              max(x_0, x_0 + math.copysign(r * 2, anchors[0, -1] - anchors[0, 0])),
                              r / points_per_search)
                y = np.poly1d(P)(x)
                dis = abs(np.sqrt(np.power(x - x_0, 2) + np.power(y - y_0, 2)) - r)
                x_1 = x[np.where(dis == dis.min())[0][0]]
                y_1 = y[np.where(dis == dis.min())[0][0]]
                trajectory = np.append(trajectory, np.array([[x_1, y_1]]), axis=0)
        trajectories[f"{step}"] = trajectory

    # 补上最后那段直线
    trajectory = trajectories[f"{num_step - 1}"]
    x_start, y_start = trajectory[-1, 0], trajectory[-1, 1]
    x_end, y_end = paras['end'][0], paras['end'][1]
    num = round((y_start - y_end) / r)
    trajectory_append = np.stack([np.linspace(x_start, x_end, num)[1:],
                                  np.linspace(y_start, y_end, num)[1:]], axis=0).transpose()
    trajectory = np.append(trajectory, trajectory_append, axis=0)
    trajectories[f"{num_step - 1}"] = trajectory

    return trajectories


def y_data_1(x):
    return np.array([x ** 5, x ** 4, x ** 3, x ** 2, x, 1])


def y_hat_1(x):
    return np.array([5 * x ** 4, 4 * x ** 3, 3 * x ** 2, 2 * x, 1, 0])


def func(x, p):
    return p[0] * x ** 5 + p[1] * x ** 4 + p[2] * x ** 3 + p[3] * x ** 2 + p[4] * x + p[5]


def func_hat(x, p):
    return 5 * p[0] * x ** 4 + 4 * p[1] * x ** 3 + 3 * p[2] * x ** 2 + 2 * p[3] * x + p[4]


def cal_abcdef(points):
    x0, y0, k0, c0 = points[0, 0], points[1, 0], math.tan(points[2, 0]), 0.0
    x1, y1, k1, c1 = points[0, 2], points[1, 2], math.tan(points[2, 2]), 0.0

    a = ((2 * x0 ** 5 * y1 - 2 * x1 ** 5 * y0 + 2 * k0 * x0 * x1 ** 5 - 2 * k1 * x0 ** 5 * x1 + 10 * x0 * x1 ** 4 * y0 - 10 * x0 ** 4 * x1 * y1 - c0 * x0 ** 2 * x1 ** 5 + 2 * c0 * x0 ** 3 * x1 ** 4 - c0 * x0 ** 4 * x1 ** 3 + c1 * x0 ** 3 * x1 ** 4 - 2 * c1 * x0 ** 4 * x1 ** 3 + c1 * x0 ** 5 * x1 ** 2 - 10 * k0 * x0 ** 2 * x1 ** 4 + 8 * k0 * x0 ** 3 * x1 ** 3 - 8 * k1 * x0 ** 3 * x1 ** 3 + 10 * k1 * x0 ** 4 * x1 ** 2 - 20 * x0 ** 2 * x1 ** 3 * y0 + 20 * x0 ** 3 * x1 ** 2 * y1) /
         (2 * (x0 - x1) ** 2 * (x0 ** 3 - 3 * x0 ** 2 * x1 + 3 * x0 * x1 ** 2 - x1 ** 3)))
    b = ((2 * k1 * x0 ** 5 - 2 * k0 * x1 ** 5 + 2 * c0 * x0 * x1 ** 5 - 2 * c1 * x0 ** 5 * x1 + 10 * k0 * x0 * x1 ** 4 - 10 * k1 * x0 ** 4 * x1 - c0 * x0 ** 2 * x1 ** 4 - 4 * c0 * x0 ** 3 * x1 ** 3 + 3 * c0 * x0 ** 4 * x1 ** 2 - 3 * c1 * x0 ** 2 * x1 ** 4 + 4 * c1 * x0 ** 3 * x1 ** 3 + c1 * x0 ** 4 * x1 ** 2 + 16 * k0 * x0 ** 2 * x1 ** 3 - 24 * k0 * x0 ** 3 * x1 ** 2 + 24 * k1 * x0 ** 2 * x1 ** 3 - 16 * k1 * x0 ** 3 * x1 ** 2 + 60 * x0 ** 2 * x1 ** 2 * y0 - 60 * x0 ** 2 * x1 ** 2 * y1) /
         (2 * (x0 - x1) ** 2 * (x0 ** 3 - 3 * x0 ** 2 * x1 + 3 * x0 * x1 ** 2 - x1 ** 3)))
    c = (-(c0 * x1 ** 5 - c1 * x0 ** 5 + 4 * c0 * x0 * x1 ** 4 + 3 * c0 * x0 ** 4 * x1 - 3 * c1 * x0 * x1 ** 4 - 4 * c1 * x0 ** 4 * x1 + 36 * k0 * x0 * x1 ** 3 - 24 * k0 * x0 ** 3 * x1 + 24 * k1 * x0 * x1 ** 3 - 36 * k1 * x0 ** 3 * x1 + 60 * x0 * x1 ** 2 * y0 + 60 * x0 ** 2 * x1 * y0 - 60 * x0 * x1 ** 2 * y1 - 60 * x0 ** 2 * x1 * y1 - 8 * c0 * x0 ** 2 * x1 ** 3 + 8 * c1 * x0 ** 3 * x1 ** 2 - 12 * k0 * x0 ** 2 * x1 ** 2 + 12 * k1 * x0 ** 2 * x1 ** 2) /
         (2 * (x0 - x1) ** 2 * (x0 ** 3 - 3 * x0 ** 2 * x1 + 3 * x0 * x1 ** 2 - x1 ** 3)))
    d = ((c0 * x0 ** 4 + 3 * c0 * x1 ** 4 - 3 * c1 * x0 ** 4 - c1 * x1 ** 4 - 8 * k0 * x0 ** 3 - 12 * k1 * x0 ** 3 + 12 * k0 * x1 ** 3 + 8 * k1 * x1 ** 3 + 20 * x0 ** 2 * y0 - 20 * x0 ** 2 * y1 + 20 * x1 ** 2 * y0 - 20 * x1 ** 2 * y1 + 4 * c0 * x0 ** 3 * x1 - 4 * c1 * x0 * x1 ** 3 + 28 * k0 * x0 * x1 ** 2 - 32 * k0 * x0 ** 2 * x1 + 32 * k1 * x0 * x1 ** 2 - 28 * k1 * x0 ** 2 * x1 - 8 * c0 * x0 ** 2 * x1 ** 2 + 8 * c1 * x0 ** 2 * x1 ** 2 + 80 * x0 * x1 * y0 - 80 * x0 * x1 * y1) /
         (2 * (x0 ** 2 - 2 * x0 * x1 + x1 ** 2) * (x0 ** 3 - 3 * x0 ** 2 * x1 + 3 * x0 * x1 ** 2 - x1 ** 3)))
    e = (-(30 * x0 * y0 - 30 * x0 * y1 + 30 * x1 * y0 - 30 * x1 * y1 + 2 * c0 * x0 ** 3 + 3 * c0 * x1 ** 3 - 3 * c1 * x0 ** 3 - 2 * c1 * x1 ** 3 - 14 * k0 * x0 ** 2 - 16 * k1 * x0 ** 2 + 16 * k0 * x1 ** 2 + 14 * k1 * x1 ** 2 - 4 * c0 * x0 * x1 ** 2 - c0 * x0 ** 2 * x1 + c1 * x0 * x1 ** 2 + 4 * c1 * x0 ** 2 * x1 - 2 * k0 * x0 * x1 + 2 * k1 * x0 * x1) /
         (2 * (x0 - x1) * (x0 ** 4 - 4 * x0 ** 3 * x1 + 6 * x0 ** 2 * x1 ** 2 - 4 * x0 * x1 ** 3 + x1 ** 4)))
    f = ((12 * y0 - 12 * y1 - 6 * k0 * x0 - 6 * k1 * x0 + 6 * k0 * x1 + 6 * k1 * x1 + c0 * x0 ** 2 + c0 * x1 ** 2 - c1 * x0 ** 2 - c1 * x1 ** 2 - 2 * c0 * x0 * x1 + 2 * c1 * x0 * x1) /
         (2 * (x0 - x1) ** 2 * (x0 ** 3 - 3 * x0 ** 2 * x1 + 3 * x0 * x1 ** 2 - x1 ** 3)))

    return np.array([[f], [e], [d], [c], [b], [a]])
