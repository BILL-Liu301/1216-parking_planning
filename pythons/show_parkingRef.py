import os
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from modules.base_path import path_solutions, path_figs_all
from modules.base_paras import paras

if __name__ == '__main__':
    solutions = os.listdir(path_solutions)
    solutions.sort(key=lambda x: int(x.split(".")[0]))

    plt.figure(figsize=[13.072, 7.353])
    pbar = tqdm(total=len(solutions))
    for index, solution in enumerate(solutions):
        sol = np.loadtxt(os.path.join(path_solutions, solution))

        xmid_r = sol[:, 2]
        ymid_r = sol[:, 3]
        xmid_f = xmid_r + np.cos(sol[:, 4]) * paras["Car_L"]
        ymid_f = ymid_r + np.sin(sol[:, 4]) * paras["Car_L"]
        xmid_r = xmid_r - np.cos(sol[:, 4]) * (paras["Car_Length"] - paras["Car_L"]) / 2
        ymid_r = ymid_r - np.sin(sol[:, 4]) * (paras["Car_Length"] - paras["Car_L"]) / 2
        xmid_f = xmid_f + np.cos(sol[:, 4]) * (paras["Car_Length"] - paras["Car_L"]) / 2
        ymid_f = ymid_f + np.sin(sol[:, 4]) * (paras["Car_Length"] - paras["Car_L"]) / 2

        xr_r = xmid_r + np.cos(sol[:, 4] - np.pi / 2) * paras["Car_Width"] / 2
        yr_r = ymid_r + np.sin(sol[:, 4] - np.pi / 2) * paras["Car_Width"] / 2
        xr_l = xmid_r + np.cos(sol[:, 4] + np.pi / 2) * paras["Car_Width"] / 2
        yr_l = ymid_r + np.sin(sol[:, 4] + np.pi / 2) * paras["Car_Width"] / 2
        xf_r = xmid_f + np.cos(sol[:, 4] - np.pi / 2) * paras["Car_Width"] / 2
        yf_r = ymid_f + np.sin(sol[:, 4] - np.pi / 2) * paras["Car_Width"] / 2
        xf_l = xmid_f + np.cos(sol[:, 4] + np.pi / 2) * paras["Car_Width"] / 2
        yf_l = ymid_f + np.sin(sol[:, 4] + np.pi / 2) * paras["Car_Width"] / 2

        plt.subplot(2, 2, 1)
        plt.title("Parking Trajectory", fontsize=10)
        plt.plot([paras["limits"][0, 0], paras["limits"][0, 1]], [paras["limits"][1, 1], paras["limits"][1, 1]], "k")
        plt.plot([paras["limits"][0, 0], -paras["Parking_X"] / 2], [paras["Parking_Y"], paras["Parking_Y"]], "k")
        plt.plot([paras["Parking_X"] / 2, paras["limits"][0, 1]], [paras["Parking_Y"], paras["Parking_Y"]], "k")
        plt.plot([-paras["Parking_X"] / 2, -paras["Parking_X"] / 2], [paras["limits"][1, 0], paras["Parking_Y"]], "k")
        plt.plot([paras["Parking_X"] / 2, paras["Parking_X"] / 2], [paras["limits"][1, 0], paras["Parking_Y"]], "k")
        plt.plot([-paras["Parking_X"] / 2, paras["Parking_X"] / 2], [paras["limits"][1, 0], paras["limits"][1, 0]], "k")

        # 显示车辆运行轨迹
        plt.plot(sol[:, 2], sol[:, 3], "k--")

        ## 显示车辆
        # for i in range(0, sol.shape[0], 10):
        #     plt.plot([xr_r[i], xr_l[i]], [yr_r[i], yr_l[i]], "b")
        #     plt.plot([xr_l[i], xf_l[i]], [yr_l[i], yf_l[i]], "b")
        #     plt.plot([xf_l[i], xf_r[i]], [yf_l[i], yf_r[i]], "b")
        #     plt.plot([xf_r[i], xr_r[i]], [yf_r[i], yr_r[i]], "b")

        plt.subplot(2, 2, 2)
        plt.title("Duration", fontsize=10)
        plt.bar(0.5, sol[np.where(sol[:, 0] == 1.0)[0][-1], 1] - 0.0,
                color="c", edgecolor='k', width=1)
        plt.text(0.5, sol[np.where(sol[:, 0] == 1.0)[0][-1], 1] - 0.0,
                 f"{sol[np.where(sol[:, 0] == 1.0)[0][-1], 1] - 0.0:.2f}s",
                 horizontalalignment='center', verticalalignment='bottom', fontsize=7)
        for j in range(1, 4, 1):
            plt.bar(j * 1 + 0.5, sol[np.where(sol[:, 0] == j + 1)[0][-1], 1] - sol[np.where(sol[:, 0] == j)[0][-1], 1],
                    color="c", edgecolor='k', width=1)
            plt.text(j * 1 + 0.5, sol[np.where(sol[:, 0] == j + 1)[0][-1], 1] - sol[np.where(sol[:, 0] == j)[0][-1], 1],
                     f"{sol[np.where(sol[:, 0] == j + 1)[0][-1], 1]:.1f}"
                     f"({sol[np.where(sol[:, 0] == j + 1)[0][-1], 1] - sol[np.where(sol[:, 0] == j)[0][-1], 1]:.1f})s",
                     horizontalalignment='center', verticalalignment='bottom', fontsize=7)

        plt.subplot(2, 2, 3)
        plt.title("Theta & Delta", fontsize=10)
        for j in range(4):
            plt.plot([sol[np.where(sol[:, 0] == j + 1)[0][-1], 1], sol[np.where(sol[:, 0] == j + 1)[0][-1], 1]],
                     [min(sol[:, 4].min(), sol[:, 5].min()), max(sol[:, 4].max(), sol[:, 5].max())], "k--")
        plt.plot([0.0, sol[-1, 1]], [0.0, 0.0], "k--")
        plt.plot(sol[:, 1], sol[:, 4], label="theta")
        plt.plot(sol[:, 1], sol[:, 5], label="delta")
        plt.legend(loc='upper right')

        plt.subplot(2, 2, 4)
        plt.title("Velocity", fontsize=10)
        for j in range(4):
            plt.plot([sol[np.where(sol[:, 0] == j + 1)[0][-1], 1], sol[np.where(sol[:, 0] == j + 1)[0][-1], 1]],
                     [min(sol[:, 4].min(), sol[:, 5].min()), max(sol[:, 4].max(), sol[:, 5].max())], "k--")
        plt.plot([0.0, sol[-1, 1]], [0.0, 0.0], "k--")
        plt.plot(sol[:, 1], sol[:, 6], label="v")
        plt.legend(loc='upper right')

        plt.savefig(f"{path_figs_all}{solution[:-4]}.png")
        plt.clf()

        pbar.update(1)
    pbar.close()
