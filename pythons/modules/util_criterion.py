import math
import numpy as np
import torch
import torch.nn as nn

from .base_paras import paras
from .base_settings import device


class Criterion_Train(nn.Module):
    def __init__(self):
        super(Criterion_Train, self).__init__()
        self.weight = 0.8
        self.loss_xy = 0.0
        self.Car_Length = paras['Car_L']

    def reinit(self):
        self.loss_xy = 0.0

    def forward(self, inps, refs):
        losses = []
        for index_inp, inp in enumerate(inps):
            ref = refs[index_inp]
            losses_temp = torch.tensor([]).to(device)
            for step in range(inp.shape[0]):
                inp_xy1 = inp[step, 0:2, :]
                ang = torch.cat((torch.cos(inp[step, 2:3, :]), torch.sin(inp[step, 2:3, :])), 0)
                inp_xy2 = inp_xy1 + self.Car_Length * ang
                inp_xy = torch.cat((inp_xy1, inp_xy2), 0)

                ref_xy1 = ref[step, 0:2, :]
                ang = torch.cat((torch.cos(ref[step, 2:3, :]), torch.sin(ref[step, 2:3, :])), 0)
                ref_xy2 = ref_xy1 + self.Car_Length * ang
                ref_xy = torch.cat((ref_xy1, ref_xy2), 0)

                delta = ref_xy - inp_xy
                loss_xy1 = torch.sqrt(torch.sum(torch.pow(delta[0:2, :], 2), 0))
                loss_xy2 = torch.sqrt(torch.sum(torch.pow(delta[2:4, :], 2), 0))

                losses_temp = torch.cat([losses_temp, (loss_xy1 * self.weight + loss_xy2 * (1 - self.weight)).unsqueeze(dim=0).mean(1)])
                # losses_temp = losses_temp + (loss_xy1 * self.weight + loss_xy2 * (1 - self.weight)).mean()
            losses.append(losses_temp)
        self.loss_xy = sum(losses) / len(losses)
        return self.loss_xy.sum(), self.loss_xy.cpu().detach().numpy().sum()


class Criterion_Test(nn.Module):
    def __init__(self):
        super(Criterion_Test, self).__init__()
        self.loss_xy = 0.0
        self.loss_theta = 0.0

    def forward(self, inp, ref):
        inp_xy = inp[:, 0:2]
        ref_xy = ref[:, 0:2]
        delta = ref_xy - inp_xy
        self.loss_xy = torch.sqrt(torch.sum(torch.pow(delta, 2), 1))
        self.loss_theta = ref[:, -1, :] - inp[:, -1, :]
        return self.loss_xy.cpu().detach().numpy(), self.loss_theta.cpu().detach().numpy()


