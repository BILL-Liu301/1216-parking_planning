import math
import threading

import torch
from tqdm import tqdm
from threading import Thread


class ModeTrain(Thread):
    def __init__(self, model, batch_size,
                 datas_inp1, datas_inp2, datas_oup,
                 criterion, optimizer):
        Thread.__init__(self)
        self.model = model
        self.batch_size = batch_size
        self.datas_inp1 = datas_inp1
        self.datas_inp2 = datas_inp2
        self.datas_oup = datas_oup
        self.criterion = criterion
        self.optimizer = optimizer
        self.flag_finish = False
        self.schedule = 0.0
        self.loss = 0.0
        self.loss_min = math.inf
        self.loss_max = 0.0

    def run(self):

        for i in tqdm(range(math.floor(self.datas_inp1.shape[0] / self.batch_size) + 1), desc='Train', leave=False, ncols=80):
            anchors = []
            for j in tqdm(range(min(self.batch_size, self.datas_inp1.shape[0] - i * self.batch_size)), desc='Batch', leave=False, ncols=80):
                anchors.append(self.model(self.datas_inp1[i * self.batch_size + j], self.datas_inp2[i * self.batch_size + j]))
            self.loss, loss_xy = self.criterion(anchors, self.datas_oup[i * self.batch_size:(i * self.batch_size + len(anchors))])
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            self.criterion.reinit()

            self.loss = self.loss.item()
            self.loss_min = min(self.loss_min, loss_xy.min())
            self.loss_max = max(self.loss_max, loss_xy.max())

            self.schedule = (i * self.batch_size + len(anchors)) / self.datas_inp1.shape[0]
        self.flag_finish = True


def mode_test(model, criterion,
              data_inp1, data_inp2, data_oup):
    anchors = model(data_inp1, data_inp2)
    if data_oup is None:
        return anchors.cpu().detach().numpy(), 0.0, 0.0
    loss_xy, loss_theta = criterion(anchors, data_oup)
    return anchors.cpu().detach().numpy(), loss_xy, loss_theta
