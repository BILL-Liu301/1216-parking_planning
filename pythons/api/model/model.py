import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from .criterion_dis import Criterion_Dis


class Parking_Trajectory_Planner(nn.Module):
    def __init__(self, paras: dict):
        super(Parking_Trajectory_Planner, self).__init__()

        # 基础参数
        self.bias = False
        self.lstm_bidirectional = False
        self.num_anchor_per_step = paras['num_anchor_per_step']
        self.num_step = paras['num_step']
        self.size_middle = paras['size_middle']
        self.num_layers = paras['num_layers']
        self.len_info_loc = paras['len_info_loc']
        self.len_info_state = paras['len_info_state']
        self.car_length = paras['car_length']
        self.device = paras['device']
        self.delta_limit_mean = torch.from_numpy(paras['delta_limit_mean']).to(torch.float32).to(self.device)
        self.delta_limit_var = paras['delta_limit_var']
        self.end_point = torch.from_numpy(paras['end_point']).to(torch.float32).to(self.device)
        self.map_range = paras['map_range']
        self.map_num_max = paras['map_num_max']
        self.map_width_half = paras['map_width_half']
        self.map_height_half = paras['map_height_half']
        self.map_interval = paras['map_interval']
        self.map_in_channels = paras['map_in_channels']
        self.map_out_channels = paras['map_out_channels']
        self.map_kernel_size = paras['map_kernel_size']
        self.map_stride = paras['map_stride']
        self.map_padding = paras['map_padding']

        # 地图
        self.map, self.map_local, self.map_x, self.map_y = self.init_map(paras['map'], self.map_width_half, self.map_height_half, self.map_interval)

        # 模型设计
        self.planners = self.init_model()
        self.h0 = torch.ones([self.num_layers, 1, self.size_middle]).to(self.device)
        self.c0 = torch.ones([self.num_layers, 1, self.size_middle]).to(self.device)

    def init_map(self, map_points, map_width, map_height, map_interval):
        # 初始化地图数据
        map_global = torch.from_numpy(map_points).to(torch.float32).to(self.device)  # 全局地图
        map_local = torch.zeros([map_width * 2, map_height * 2]).to(self.device)  # 局部地图
        map_x = torch.tensor([(x * map_interval + map_interval / 2) for x in range(-map_width, map_width)]).to(self.device)  # 局部地图x核心
        map_y = torch.tensor([(y * map_interval + map_interval / 2) for y in range(-map_width, map_width)]).to(self.device)  # 局部地图y核心
        return map_global, map_local, map_x, map_y

    def init_model(self):
        # 初始化模型数据
        planners = nn.ModuleList()
        for step in range(self.num_step):
            model_per_step = nn.ModuleDict()
            # 对上一时刻的[x, y, theta]进行编码
            model_per_step.update({'encode_last_anchor': nn.Sequential(nn.Linear(in_features=self.len_info_loc, out_features=self.size_middle, bias=self.bias),
                                                                       nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.size_middle, bias=self.bias))})
            model_per_step.update({'encode_last_map_conv': nn.Sequential(nn.Conv2d(in_channels=self.map_in_channels, out_channels=self.map_out_channels, kernel_size=self.map_kernel_size,
                                                                                   padding=self.map_padding, stride=self.map_stride, bias=False),
                                                                         nn.Tanh())})
            model_per_step.update({'encode_last_map_linear': nn.Sequential(nn.Linear(in_features=(math.floor((self.map_width_half * 2 + self.map_padding * 2 - self.map_kernel_size) / self.map_stride) + 1) ** 2,
                                                                                     out_features=self.size_middle, bias=self.bias),
                                                                           nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.size_middle, bias=self.bias))})
            model_per_step.update({'encode_before_lstm': nn.Sequential(nn.Linear(in_features=self.size_middle * 2, out_features=self.size_middle, bias=self.bias),
                                                                       nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.size_middle, bias=self.bias))})
            # 时序预测核心
            model_per_step.update({'main_lstm': nn.LSTM(input_size=self.size_middle, hidden_size=self.size_middle, num_layers=self.num_layers,
                                                        bidirectional=self.lstm_bidirectional, bias=self.bias, batch_first=True)})
            model_per_step.update({'main_norm': nn.LayerNorm(normalized_shape=self.size_middle, elementwise_affine=False)})
            # 对时序预测核心的输出进行解码，输出max/mean/min和var
            model_per_step.update({'decode_mean': nn.Sequential(nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.size_middle, bias=self.bias),
                                                                nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.len_info_loc, bias=self.bias),
                                                                nn.Tanh())})
            model_per_step.update({'decode_var': nn.Sequential(nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.size_middle, bias=self.bias),
                                                               nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.len_info_loc, bias=self.bias),
                                                               nn.Sigmoid())})
            planners.append(model_per_step)
        return planners

    def points2ogm(self, batch_size, map_points):
        # map_points = [2, points]
        map_ogm = self.map_local.clone().repeat(batch_size, 1, 1)

        # 排除OGM之外的点数据
        flag_out_range = (map_points[:, 0].abs() > (self.map_width_half * self.map_interval)) + (map_points[:, 1].abs() > (self.map_height_half * self.map_interval))
        map_points[:, 0][flag_out_range] = 0
        map_points[:, 1][flag_out_range] = 0

        # 将点图转化为占据栅格地图
        x, y = map_points[:, 0], map_points[:, 1]
        x_id = (self.map_x.repeat(batch_size, x.shape[1], 2) - x.reshape(batch_size, -1, 1)).abs().argmin(axis=2)
        y_id = (self.map_y.repeat(batch_size, y.shape[1], 2) - y.reshape(batch_size, -1, 1)).abs().argmin(axis=2)
        for i in range(map_points.shape[0]):
            map_ogm[i, x_id[i], y_id[i]] = 1

        return map_ogm

    def cal_map_last(self, batch_size, anchor_last):
        # 计算平移和旋转后的map
        map_ = (self.map.T.repeat(batch_size, 1, 1) - anchor_last[:, :, 0:2]).transpose(1, 2)
        rotate_matrix = torch.cat([torch.cat([torch.cos(anchor_last[:, :, -1:]), torch.sin(anchor_last[:, :, -1:])], dim=2),
                                   torch.cat([-torch.sin(anchor_last[:, :, -1:]), torch.cos(anchor_last[:, :, -1:])], dim=2)], dim=1)
        map_points = torch.matmul(rotate_matrix, map_)

        # 点地图转换为占据栅格地图
        map_ogm = self.points2ogm(batch_size, map_points)

        return map_ogm

    def encode_last_map(self, batch_size, anchor_last, model_per_step):
        with torch.no_grad():
            map_ogm_last = self.cal_map_last(batch_size, anchor_last).unsqueeze(1)
        encode_last_map_conv = model_per_step['encode_last_map_conv'](map_ogm_last.repeat(1, self.map_in_channels, 1, 1)).sum(dim=1)
        encode_last_map_flatten = encode_last_map_conv.reshape(batch_size, 1, -1)
        encode_last_map = model_per_step['encode_last_map_linear'](encode_last_map_flatten)

        return encode_last_map, map_ogm_last

    def cal_state(self, network, main_norm):
        state = network(main_norm)
        state[:, :, 0:1] = F.sigmoid(state[:, :, 0:1]) * self.delta_limit_mean[0]
        state[:, :, 1:2] = F.tanh(state[:, :, 1:2]) * self.delta_limit_mean[1]
        return state

    def pre_from_state(self, pre_state, anchors, direction):
        s = direction * pre_state[:, :, 0:1]
        phi = pre_state[:, :, 1:2]
        delta_theta = s * torch.tan(phi) / self.car_length
        k = self.car_length / torch.tan(phi)
        # delta_theta = direction * pre_state[:, :, 0:1] * torch.tan(pre_state[:, :, 1:2]) / self.car_length
        # k = self.car_length / torch.tan(pre_state[:, :, 1:2])
        anchors_now = anchors.clone()
        anchors_now[:, :, 0:1] += k * (torch.sin(anchors[:, :, 2:3] + delta_theta) - torch.sin(anchors[:, :, 2:3]))
        anchors_now[:, :, 1:2] -= k * (torch.cos(anchors[:, :, 2:3] + delta_theta) - torch.cos(anchors[:, :, 2:3]))
        anchors_now[:, :, 2:3] += delta_theta
        return anchors_now

    def forward(self, inp_start_point, train_step=None):
        # inp_start_point: [B, 1, len_info_loc]

        if train_step is None:
            train_step = [0, 1, 2, 3]

        batch_size = inp_start_point.shape[0]
        pre_mean, pre_var = list(), list()

        anchor_last, var_last = inp_start_point.clone(), (torch.ones(inp_start_point.shape) * 0.01).to(self.device)
        views, states = list(), list()
        direction = -1

        # 分阶段预测
        for step, model_per_step in enumerate(self.planners):
            direction = -1 * direction

            # 计算数据
            decode_mean, decode_var = [anchor_last], [var_last]
            if not self.training:
                views_temp, states_temp = [self.cal_map_last(batch_size, anchor_last).unsqueeze(1).cpu()], []

            # 逐点预测
            h, c = self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1)
            for _ in range(self.num_anchor_per_step - 1):
                # 编码上一时刻轨迹点
                encode_last_anchor = model_per_step['encode_last_anchor'](anchor_last)

                # 编码上一时刻观察到的地图范围
                encode_last_map, map_ogm_last = self.encode_last_map(batch_size, anchor_last, model_per_step)
                encode_before_lstm = model_per_step['encode_before_lstm'](torch.cat([encode_last_anchor, encode_last_map], dim=2))

                # 将数据输入lstm进行编码
                main_lstm, (h, c) = model_per_step['main_lstm'](encode_before_lstm, (h, c))
                main_norm = model_per_step['main_norm'](main_lstm) + encode_before_lstm

                # 分别对下一时刻的均值与方差进行解码，并更新“上一时刻”
                # state_now = self.cal_state(model_per_step['decode_mean'], main_norm)
                # anchors_now = self.pre_from_state(state_now, anchor_last, direction)
                # var_now = model_per_step['decode_var'](main_norm) * self.delta_limit_var
                # var_now += 0.1 * torch.sign(var_now)
                anchors_now = model_per_step['decode_mean'](main_norm) * self.delta_limit_mean + anchor_last
                var_now = model_per_step['decode_var'](main_norm) * self.delta_limit_var

                anchor_last, var_last = anchors_now.clone(), var_now.clone()
                decode_mean.append(anchors_now)
                decode_var.append(var_now)

                # 中间变量
                if not self.training:
                    views_temp.append(map_ogm_last.cpu())
                    # states_temp.append(state_now.cpu())
            if not self.training:
                states_temp.append(torch.zeros([batch_size, 1, self.len_info_state]))

            # 变量整合
            pre_mean.append(torch.cat(decode_mean, dim=1).unsqueeze(1))
            pre_var.append(torch.cat(decode_var, dim=1).unsqueeze(1))
            if not self.training:
                views.append(torch.cat(views_temp, dim=1).unsqueeze(1))
                states.append(torch.cat(states_temp, dim=1).unsqueeze(1))

        pre_mean = torch.cat(pre_mean, dim=1)
        pre_var = torch.cat(pre_var, dim=1)
        if not self.training:
            views = torch.cat(views, dim=1)
            states = torch.cat(states, dim=1)
        return pre_mean, pre_var, views, states


class Parking_Trajectory_Planner_LightningModule(pl.LightningModule):
    def __init__(self, paras: dict, train_step=None):
        super(Parking_Trajectory_Planner_LightningModule, self).__init__()
        # 保存超参
        self.save_hyperparameters('paras')

        # 加载基本模型
        self.model = Parking_Trajectory_Planner(paras)

        # 设定评价函数
        self.criterion_train = nn.GaussianNLLLoss(reduction='mean')
        self.criterion_val = Criterion_Dis(car_length=paras['car_length'], weight=0.9, reduction='max')
        self.criterion_test_dis = Criterion_Dis(car_length=paras['car_length'], weight=0.9, reduction='none')
        self.criterion_test_L1 = nn.L1Loss(reduction='none')

        # 设定训练阶段
        self.train_step = [0, 1, 2, 3] if train_step is None else train_step
        self.choose_parameters_train(self.train_step)

        # 设定optimizer和scheduler
        self.optimizer = optim.Adam(self.parameters(), paras['lr_init'])
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=paras['lr_init'], total_steps=paras['max_epochs'], pct_start=0.02)
        # self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=0.1)

        # 设定results
        self.test_results = list()
        self.test_losses = {
            'mean': list(),
            'max': list()
        }

    def choose_parameters_train(self, planner: list):
        planner = [str(item) for item in planner]
        for name, para in self.named_parameters():
            if name.split('.')[2] in planner:
                para.requires_grad = True
            else:
                para.requires_grad = False

    def run_base(self, batch, batch_idx):
        inp_start_point = batch[:, 0, 0:1]
        pre_mean, pre_var, views, states = self.model(inp_start_point, self.train_step)
        ref_mean = batch

        pre_mean = pre_mean[:, self.train_step]
        pre_var = pre_var[:, self.train_step]
        if not self.model.training:
            views = views[:, self.train_step]
            states = states[:, self.train_step]
        ref_mean = ref_mean[:, self.train_step]

        return pre_mean, ref_mean, pre_var, views, states

    def training_step(self, batch, batch_idx):
        pre_mean, ref_mean, pre_var, _, _ = self.run_base(batch, batch_idx)
        loss_nll = self.criterion_train(pre_mean, ref_mean, pre_var)
        loss_dis = self.criterion_val(pre_mean, ref_mean)

        self.log('loss_train', loss_nll, prog_bar=True)
        self.log('lr', self.scheduler.get_last_lr()[0], prog_bar=True)
        return loss_nll

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            pre_mean, ref_mean, pre_var, views, states = self.run_base(batch, batch_idx)
        losses_dis = self.criterion_test_dis(pre_mean, ref_mean)
        losses_l1 = self.criterion_test_L1(pre_mean, ref_mean)
        for b in range(len(batch)):
            self.test_results.append({
                'pre': pre_mean[b].cpu().numpy().transpose(0, 2, 1),
                'view': views[b].numpy().transpose(0, 2, 3, 1),
                'state': states[b].numpy().transpose(0, 2, 1),
                'ref': ref_mean[b].cpu().numpy().transpose(0, 2, 1),
                'pre_var': pre_var[b].cpu().numpy().transpose(0, 2, 1),
                'loss_dis': losses_dis[b].cpu().numpy().transpose(0, 2, 1),
                'loss_l1': losses_l1[b].cpu().numpy().transpose(0, 2, 1)
            })
            loss_dis = losses_dis[b]
            self.test_losses['mean'].append(loss_dis.mean().unsqueeze(0))
            self.test_losses['max'].append(loss_dis.max().unsqueeze(0))

    def validation_step(self, batch, batch_idx):
        pre_mean, ref_mean, pre_var, _, _ = self.run_base(batch, 'val')
        loss_nll = self.criterion_train(pre_mean, ref_mean, pre_var)
        loss_dis = self.criterion_val(pre_mean, ref_mean)

        self.log('loss_val_nll', loss_nll, prog_bar=True)
        self.log('loss_val_dis', loss_dis, prog_bar=True)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
