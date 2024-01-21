import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl

from .criterion_dis import Criterion_Dis


class Parking_Trajectory_Planner(nn.Module):
    def __init__(self, paras: dict):
        super(Parking_Trajectory_Planner, self).__init__()
        # 基础参数
        self.bias = False
        self.lstm_bidirectional = False
        if self.lstm_bidirectional:
            self.D = 2
        else:
            self.D = 1
        self.num_anchor_per_step = paras['num_anchor_per_step']
        self.num_step = paras['num_step']
        self.size_middle = paras['size_middle']
        self.num_layers = paras['num_layers']
        self.len_info_loc = paras['len_info_loc']
        self.device = paras['device']
        self.delta_limit_mean = paras['delta_limit_mean'].to(self.device)
        self.delta_limit_var = paras['delta_limit_var']
        self.end_point = paras['end_point'].to(self.device)

        # 模型设计
        self.planners = nn.ModuleList()
        self.h0 = torch.ones([self.D * self.num_layers, 1, self.size_middle]).to(self.device)
        self.c0 = torch.ones([self.D * self.num_layers, 1, self.size_middle]).to(self.device)
        for step in range(self.num_step):
            model_per_step = nn.ModuleDict()
            # 对上一时刻的x, y, theta进行编码
            model_per_step.update({'encoder_history': nn.Sequential(nn.Linear(in_features=self.len_info_loc * 2, out_features=self.size_middle, bias=self.bias),
                                                                    nn.LayerNorm(normalized_shape=self.size_middle, elementwise_affine=False))})
            # 时序预测核心
            model_per_step.update({'main_lstm': nn.LSTM(input_size=self.size_middle, hidden_size=self.size_middle, num_layers=self.num_layers,
                                                        bidirectional=self.lstm_bidirectional, bias=self.bias, batch_first=True)})
            model_per_step.update({'main_norm': nn.LayerNorm(normalized_shape=self.D * self.size_middle, elementwise_affine=False)})
            # 对时序预测核心的输出进行解码，输出max/mean/min和var
            model_per_step.update({'decoder_mean': nn.Sequential(nn.ReLU(), nn.Linear(in_features=self.D * self.size_middle, out_features=self.size_middle, bias=self.bias),
                                                                 nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.len_info_loc, bias=self.bias),
                                                                 nn.Tanh())})
            model_per_step.update({'decoder_var': nn.Sequential(nn.ReLU(), nn.Linear(in_features=self.D * self.size_middle, out_features=self.size_middle, bias=self.bias),
                                                                nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.len_info_loc, bias=self.bias),
                                                                nn.Sigmoid())})
            self.planners.append(model_per_step)

    def forward(self, inp_start_point):
        # inp_start_point: [B, num_anchor_per_step, len_info_loc]

        batch_size = inp_start_point.shape[0]
        pre_mean, pre_var = list(), list()
        anchor_last = inp_start_point.clone()
        var_last = torch.ones(inp_start_point.shape).to(self.device)
        for step, model_per_step in enumerate(self.planners):
            decoder_mean, decoder_var = [anchor_last], [var_last]
            for anchor in range(self.num_anchor_per_step - 1):
                encode_history = model_per_step['encoder_history'](torch.cat([anchor_last, anchor_last - self.end_point], dim=2))
                main_lstm, _ = model_per_step['main_lstm'](encode_history, (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1)))
                main_norm = model_per_step['main_norm'](main_lstm)
                decoder_mean.append(model_per_step['decoder_mean'](main_norm) * self.delta_limit_mean + anchor_last)
                decoder_var.append(model_per_step['decoder_var'](main_norm) * self.delta_limit_var)
                anchor_last = decoder_mean[-1].clone()
            pre_mean.append(torch.cat(decoder_mean, dim=1).unsqueeze(1))
            pre_var.append(torch.cat(decoder_var, dim=1).unsqueeze(1))
        pre_mean = torch.cat(pre_mean, dim=1)
        pre_var = torch.cat(pre_var, dim=1)
        return pre_mean, pre_var


class Parking_Trajectory_Planner_LightningModule(pl.LightningModule):
    def __init__(self, paras: dict):
        super(Parking_Trajectory_Planner_LightningModule, self).__init__()
        self.save_hyperparameters('paras')

        self.model = Parking_Trajectory_Planner(paras)
        self.criterion_train = nn.GaussianNLLLoss(reduction='mean')
        self.criterion_val = Criterion_Dis(car_length=4.0, weight=0.8, reduction='max')
        self.criterion_test = Criterion_Dis(car_length=4.0, weight=0.8, reduction='none')
        self.optimizer = optim.Adam(self.parameters(), paras['lr_init'])
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=paras['lr_init'], total_steps=paras['max_epochs'], pct_start=0.1)

        self.test_results = list()
        self.test_losses = {
            'mean': list(),
            'max': list()
        }

    def forward(self):
        pass

    def run_base(self, batch, mode='test'):
        inp_start_point = batch[:, 0, 0:1]
        pre_mean, pre_var = self.model(inp_start_point)
        ref_mean = batch
        # if mode == 'train' or mode == 'val':
        #     pre_mean = torch.cat([pre_mean[:, i] for i in range(pre_mean.shape[1])], dim=1)
        #     pre_var = torch.cat([pre_var[:, i] for i in range(pre_var.shape[1])], dim=1)
        #     ref_mean = torch.cat([ref_mean[:, i] for i in range(ref_mean.shape[1])], dim=1)
        return pre_mean, ref_mean, pre_var

    def training_step(self, batch, batch_idx):
        pre_mean, ref_mean, pre_var = self.run_base(batch, batch_idx)
        loss_train = self.criterion_train(pre_mean, ref_mean, pre_var)

        self.log('loss_train', loss_train, prog_bar=True)
        self.log('lr', self.scheduler.get_last_lr()[0], prog_bar=True)
        return loss_train

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            pre_mean, ref_mean, pre_var = self.run_base(batch, batch_idx)
        losses = self.criterion_test(pre_mean, ref_mean)
        for b in range(len(batch)):
            self.test_results.append({
                'pre': pre_mean[b].cpu().numpy().transpose(0, 2, 1),
                'ref': ref_mean[b].cpu().numpy().transpose(0, 2, 1),
                'pre_var': pre_var[b].cpu().numpy().transpose(0, 2, 1),
                'loss': losses[b].cpu().numpy().transpose(0, 2, 1)
            })
            loss = losses[b]
            self.test_losses['mean'].append(loss.mean().unsqueeze(0))
            self.test_losses['max'].append(loss.max().unsqueeze(0))

    def validation_step(self, batch, batch_idx):
        pre_mean, ref_mean, pre_var = self.run_base(batch, 'val')
        loss_nll = self.criterion_train(pre_mean, ref_mean, pre_var)
        loss_mse = self.criterion_val(pre_mean, ref_mean)

        self.log('loss_val_nll', loss_nll, prog_bar=True)
        self.log('loss_val_dis', loss_mse, prog_bar=True)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
