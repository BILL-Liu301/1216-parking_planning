import math
import numpy as np
from .paras import paras_base

    #  ————
    x = np.linspace(-paras_base['Freespace_X'], paras_base['Freespace_X'], math.floor(paras_base['Freespace_X'] * 2 / 0.1) + 1)
    y = np.ones(x.shape) * (paras_base['Parking_Y'] + paras_base['Freespace_Y'])
    map_np = np.stack([x, y], axis=0)

    #  ————
    #      |
    y = np.linspace(paras_base['Parking_Y'], paras_base['Parking_Y'] + paras_base['Freespace_Y'], math.floor(paras_base['Freespace_Y'] / 0.1) + 1)
    x = np.ones(y.shape) * paras_base['Freespace_X']
    map_np = np.append(map_np, np.stack([x, np.flip(y)], axis=0), axis=1)

    #  ————
    #     _|
    x = np.linspace(paras_base['Freespace_X'], paras_base['Parking_X'] / 2, math.floor((paras_base['Freespace_X'] - paras_base['Parking_X'] / 2) / 0.1) + 1)
    y = np.ones(x.shape) * paras_base['Parking_Y']
    map_np = np.append(map_np, np.stack([np.flip(x), y], axis=0), axis=1)

    #  ————
    #     _|
    #     |
    y = np.linspace(0.0, paras_base['Parking_Y'], math.floor(paras_base['Parking_Y'] / 0.1) + 1)
    x = np.ones(y.shape) * paras_base['Parking_X'] / 2
    map_np = np.append(map_np, np.stack([x, np.flip(y)], axis=0), axis=1)

    #  ————
    #     _|
    #   __|
    x = np.linspace(-paras_base['Parking_X'] / 2, paras_base['Parking_X'] / 2, math.floor(paras_base['Parking_X'] / 0.1) + 1)
    y = np.zeros(x.shape)
    map_np = np.append(map_np, np.stack([np.flip(x), y], axis=0), axis=1)

    #  ————
    #     _|
    #  |__|
    y = np.linspace(0.0, paras_base['Parking_Y'], math.floor(paras_base['Parking_Y'] / 0.1) + 1)
    x = np.ones(y.shape) * (-paras_base['Parking_X'] / 2)
    map_np = np.append(map_np, np.stack([x, y], axis=0), axis=1)

    #  ————
    # _   _|
    #  |__|
    x = np.linspace(-paras_base['Freespace_X'], -paras_base['Parking_X'] / 2, math.floor((paras_base['Freespace_X'] - paras_base['Parking_X'] / 2) / 0.1) + 1)
    y = np.ones(x.shape) * paras_base['Parking_Y']
    map_np = np.append(map_np, np.stack([np.flip(x), y], axis=0), axis=1)

    #  ————
    # |_   _|
    #  |__|
    y = np.linspace(paras_base['Parking_Y'], paras_base['Parking_Y'] + paras_base['Freespace_Y'], math.floor(paras_base['Freespace_Y'] / 0.1) + 1)
    x = np.ones(y.shape) * (-paras_base['Freespace_X'])
    map_np = np.append(map_np, np.stack([x, y], axis=0), axis=1)