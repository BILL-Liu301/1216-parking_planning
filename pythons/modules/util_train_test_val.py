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
        self.grad_max = 0.0
        self.grad_max_name = ''
        self.turn_off = 'encoder'
        self.turn_on = 'decoder'

    def read_grad_max(self, model, prefix):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    self.grad_max = max(self.grad_max, param.grad.abs().max())
                    self.grad_max_name = prefix + '-' + name

    def turn_off_turn_on(self, model, mode):
        if mode == 'on':
            grad = True
        else:
            grad = False
        for name, param in model.named_parameters():
            param.requires_grad = grad

    def run(self):

        for i in tqdm(range(math.floor(self.datas_inp1.shape[0] / self.batch_size) + 1), desc='Train', leave=False, ncols=80, disable=False):

            # # 交替进行优化
            # if i % 5 == 0:
            #     # print(f"[On: {self.turn_on}, Off: {self.turn_off}]")
            #     self.turn_off_turn_on(self.model.get_model(self.turn_off), 'off')
            #     self.turn_off_turn_on(self.model.get_model(self.turn_on), 'on')
            #     self.turn_on, self.turn_off = self.turn_off, self.turn_on

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

            # 判断是否梯度收敛
            self.grad_max, self.grad_max_name = 0.0, ''
            self.read_grad_max(self.model.encoder_1, 'encoder_1')
            self.read_grad_max(self.model.encoder_2, 'encoder_2')
            self.read_grad_max(self.model.decoder, 'decoder')
            # print(self.grad_max, self.grad_max_name)
            if self.grad_max <= 0.01:
                break

            self.schedule = (i * self.batch_size + len(anchors)) / self.datas_inp1.shape[0]
        self.flag_finish = True


def mode_test(model, criterion,
              data_inp1, data_inp2, data_oup):
    anchors = model(data_inp1, data_inp2)
    if data_oup is None:
        return anchors.cpu().detach().numpy(), 0.0, 0.0
    loss_xy, loss_theta = criterion(anchors, data_oup)
    return anchors.cpu().detach().numpy(), loss_xy, loss_theta
