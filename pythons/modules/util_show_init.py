import matplotlib.pyplot as plt
import numpy as np

from .base_paras import paras
from .base_path import path_figs


def show_init(init_data):
    plt.figure(figsize=[13.072, 7.353])
    xmid_r = init_data[0]
    ymid_r = init_data[1]
    xmid_f = xmid_r + np.cos(init_data[2]) * paras["Car_L"]
    ymid_f = ymid_r + np.sin(init_data[2]) * paras["Car_L"]
    xmid_r = xmid_r - np.cos(init_data[2]) * (paras["Car_Length"] - paras["Car_L"]) / 2
    ymid_r = ymid_r - np.sin(init_data[2]) * (paras["Car_Length"] - paras["Car_L"]) / 2
    xmid_f = xmid_f + np.cos(init_data[2]) * (paras["Car_Length"] - paras["Car_L"]) / 2
    ymid_f = ymid_f + np.sin(init_data[2]) * (paras["Car_Length"] - paras["Car_L"]) / 2

    xr_r = xmid_r + np.cos(init_data[2] - np.pi / 2) * paras["Car_Width"] / 2
    yr_r = ymid_r + np.sin(init_data[2] - np.pi / 2) * paras["Car_Width"] / 2
    xr_l = xmid_r + np.cos(init_data[2] + np.pi / 2) * paras["Car_Width"] / 2
    yr_l = ymid_r + np.sin(init_data[2] + np.pi / 2) * paras["Car_Width"] / 2
    xf_r = xmid_f + np.cos(init_data[2] - np.pi / 2) * paras["Car_Width"] / 2
    yf_r = ymid_f + np.sin(init_data[2] - np.pi / 2) * paras["Car_Width"] / 2
    xf_l = xmid_f + np.cos(init_data[2] + np.pi / 2) * paras["Car_Width"] / 2
    yf_l = ymid_f + np.sin(init_data[2] + np.pi / 2) * paras["Car_Width"] / 2

    plt.title("Init Data", fontsize=10)
    plt.plot([paras["limits"][0, 0], paras["limits"][0, 1]], [paras["limits"][1, 1], paras["limits"][1, 1]], "k")
    plt.plot([paras["limits"][0, 0], -paras["Parking_X"] / 2], [paras["Parking_Y"], paras["Parking_Y"]], "k")
    plt.plot([paras["Parking_X"] / 2, paras["limits"][0, 1]], [paras["Parking_Y"], paras["Parking_Y"]], "k")
    plt.plot([-paras["Parking_X"] / 2, -paras["Parking_X"] / 2], [paras["limits"][1, 0], paras["Parking_Y"]], "k")
    plt.plot([paras["Parking_X"] / 2, paras["Parking_X"] / 2], [paras["limits"][1, 0], paras["Parking_Y"]], "k")
    plt.plot([-paras["Parking_X"] / 2, paras["Parking_X"] / 2], [paras["limits"][1, 0], paras["limits"][1, 0]], "k")

    plt.plot([xr_r, xr_l], [yr_r, yr_l], "b")
    plt.plot([xr_l, xf_l], [yr_l, yf_l], "b")
    plt.plot([xf_l, xf_r], [yf_l, yf_r], "b")
    plt.plot([xf_r, xr_r], [yf_r, yr_r], "b")

    plt.savefig(f"{path_figs}/init/{init_data[0]:.2f}_{init_data[1]:.2f}_{init_data[2]:.2f}.png")
    plt.close()
