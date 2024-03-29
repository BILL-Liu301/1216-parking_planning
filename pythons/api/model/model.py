import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.len_info_state = paras['len_info_state']
        self.car_length = paras['car_length']
        self.device = paras['device']
        self.delta_limit_mean = torch.from_numpy(paras['delta_limit_mean']).to(torch.float32).to(self.device)
        self.delta_limit_var = paras['delta_limit_var']
        self.end_point = torch.from_numpy(paras['end_point']).to(torch.float32).to(self.device)
        self.map = torch.from_numpy(paras['map']).to(torch.float32).to(self.device)
        self.map_range = paras['map_range']
        self.map_num_max = paras['map_num_max']

        # 模型设计
        self.planners = nn.ModuleList()
        self.h0 = torch.ones([self.D * self.num_layers, 1, self.size_middle]).to(self.device)
        self.c0 = torch.ones([self.D * self.num_layers, 1, self.size_middle]).to(self.device)
        for step in range(self.num_step):
            model_per_step = nn.ModuleDict()
            # 对上一时刻的[x, y, theta]，和终点的距离，上一时刻的var进行编码
            model_per_step.update({'encode_last_anchor': nn.Sequential(nn.Linear(in_features=self.len_info_loc * 3, out_features=self.size_middle, bias=self.bias),
                                                                       nn.LayerNorm(normalized_shape=self.size_middle, elementwise_affine=False))})
            model_per_step.update({'encode_last_map_linear_q': nn.Linear(in_features=2, out_features=self.size_middle, bias=self.bias)})
            model_per_step.update({'encode_last_map_linear_k': nn.Linear(in_features=2, out_features=self.size_middle, bias=self.bias)})
            model_per_step.update({'encode_last_map_linear_v': nn.Linear(in_features=2, out_features=self.size_middle, bias=self.bias)})
            model_per_step.update({'encode_last_map_attention': nn.MultiheadAttention(embed_dim=self.size_middle, num_heads=1, bias=self.bias, batch_first=True)})
            model_per_step.update({'encode_last_map_linear': nn.Sequential(nn.ReLU(), nn.Linear(in_features=self.map_num_max, out_features=self.size_middle, bias=self.bias),
                                                                           nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=1, bias=self.bias))})
            model_per_step.update({'encode_last_map_norm': nn.LayerNorm(normalized_shape=self.size_middle, elementwise_affine=False)})
            # 时序预测核心
            model_per_step.update({'main_lstm': nn.LSTM(input_size=self.size_middle * 2, hidden_size=self.size_middle, num_layers=self.num_layers,
                                                        bidirectional=self.lstm_bidirectional, bias=self.bias, batch_first=True)})
            model_per_step.update({'main_norm': nn.LayerNorm(normalized_shape=self.D * self.size_middle, elementwise_affine=False)})
            # 对时序预测核心的输出进行解码，输出max/mean/min和var
            model_per_step.update({'decode_mean': nn.Sequential(nn.ReLU(), nn.Linear(in_features=self.D * self.size_middle, out_features=self.len_info_state, bias=self.bias))})
            model_per_step.update({'decode_var': nn.Sequential(nn.ReLU(), nn.Linear(in_features=self.D * self.size_middle, out_features=self.len_info_loc, bias=self.bias),
                                                               nn.Sigmoid())})
            # model_per_step.update({'decode_mean': nn.Sequential(nn.ReLU(), nn.Linear(in_features=self.D * self.size_middle, out_features=self.size_middle, bias=self.bias),
            #                                                     nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.len_info_loc, bias=self.bias),
            #                                                     nn.Tanh())})
            # model_per_step.update({'decode_var': nn.Sequential(nn.ReLU(), nn.Linear(in_features=self.D * self.size_middle, out_features=self.size_middle, bias=self.bias),
            #                                                    nn.ReLU(), nn.Linear(in_features=self.size_middle, out_features=self.len_info_loc, bias=self.bias),
            #                                                    nn.Sigmoid())})
            self.planners.append(model_per_step)

    def cal_map_last(self, batch_size, anchor_last):
        # 计算平移和旋转后的map
        map_ = (self.map.T.repeat(batch_size, 1, 1) - anchor_last[:, :, 0:2]).transpose(1, 2)
        rotate_matrix = torch.cat([torch.cat([torch.cos(anchor_last[:, :, -1:]), torch.sin(anchor_last[:, :, -1:])], dim=2),
                                   torch.cat([-torch.sin(anchor_last[:, :, -1:]), torch.cos(anchor_last[:, :, -1:])], dim=2)], dim=1)
        map_last_all = torch.matmul(rotate_matrix, map_)

        # 计算点距，并将大于map_range的赋为1e10，用于排序的时候放在最后，并删掉
        map_dis = torch.sqrt(torch.sum(torch.pow(map_last_all, 2), dim=1, keepdim=True)).repeat(1, 2, 1)
        map_last_all[map_dis >= self.map_range] = 1e10

        index = torch.argsort(map_last_all[:, 0, :])

        # 下面这一行代码会降低程序运行效率，是可以优化的对象
        map_last_all = torch.stack([map_last_all[b, :, index[b]] for b in range(map_last_all.shape[0])], dim=0)

        map_last_all[map_last_all >= 1e9] = 0.0

        # pad操作，并将序列长度截取为map_num_max
        if (map_last_all.nonzero().shape[0] / (map_last_all.shape[0] * 2)) > (self.map_num_max + 20):
            # 截去的点数过多
            print(f'{map_last_all.nonzero().shape[0] / (map_last_all.shape[0] * 2)} is out range of {self.map_num_max}')
            raise Exception
        map_last_in_range = F.pad(map_last_all, [0, self.map_num_max - map_last_all.shape[2]], mode='constant', value=0.0).transpose(1, 2)
        return map_last_in_range

    def encode_last_map(self, batch_size, anchor_last, model_per_step):
        with torch.no_grad():
            map_last = self.cal_map_last(batch_size, anchor_last)
        encode_last_map_linear_q = model_per_step['encode_last_map_linear_q'](map_last)
        encode_last_map_linear_k = model_per_step['encode_last_map_linear_k'](map_last)
        encode_last_map_linear_v = model_per_step['encode_last_map_linear_v'](map_last)
        encode_last_map_attention, _ = model_per_step['encode_last_map_attention'](encode_last_map_linear_q, encode_last_map_linear_k, encode_last_map_linear_v)
        encode_last_map_linear = model_per_step['encode_last_map_linear'](encode_last_map_attention.transpose(1, 2)).transpose(1, 2)
        encode_last_map_norm = model_per_step['encode_last_map_norm'](encode_last_map_linear)
        return encode_last_map_norm

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

    def forward(self, inp_start_point):
        # inp_start_point: [B, num_anchor_per_step, len_info_loc]

        batch_size = inp_start_point.shape[0]
        pre_mean, pre_var = list(), list()

        anchor_last, var_last = inp_start_point.clone(), (torch.ones(inp_start_point.shape) * 0.01).to(self.device)
        views, states = list(), list()
        direction = -1
        for step, model_per_step in enumerate(self.planners):
            decode_mean, decode_var = [anchor_last], [var_last]
            views_temp, states_temp = [self.cal_map_last(batch_size, anchor_last).cpu().transpose(1, 2).unsqueeze(1)], []
            direction = -1 * direction

            h, c = self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1)
            for anchor in range(self.num_anchor_per_step - 1):
                # 编码上一时刻轨迹点
                encode_last_anchor = model_per_step['encode_last_anchor'](torch.cat([anchor_last, anchor_last - self.end_point, var_last], dim=2))
                # 编码上一时刻观察到的地图范围
                encode_last_map = self.encode_last_map(batch_size, anchor_last, model_per_step)
                # 将数据输入lstm进行编码
                main_lstm, (h, c) = model_per_step['main_lstm'](torch.cat([encode_last_anchor, encode_last_map], dim=2), (h, c))
                main_norm = model_per_step['main_norm'](main_lstm)
                # 分别对下一时刻的均值与方差进行解码，并更新“上一时刻”
                state_now = self.cal_state(model_per_step['decode_mean'], main_norm)
                anchors_now = self.pre_from_state(state_now, anchor_last, direction)
                var_now = model_per_step['decode_var'](main_norm)
                var_now = var_now * self.delta_limit_var + 0.1 * torch.sign(var_now)

                anchor_last, var_last = anchors_now.clone(), var_now.clone()
                decode_mean.append(anchors_now)
                decode_var.append(var_now)

                # 中间变量
                views_temp.append(self.cal_map_last(batch_size, anchor_last).cpu().transpose(1, 2).unsqueeze(1))
                states_temp.append(state_now.cpu())
            states_temp.append(torch.zeros([batch_size, 1, self.len_info_state]))

            # 变量整合
            pre_mean.append(torch.cat(decode_mean, dim=1).unsqueeze(1))
            pre_var.append(torch.cat(decode_var, dim=1).unsqueeze(1))
            views.append(torch.cat(views_temp, dim=1).unsqueeze(1))
            states.append(torch.cat(states_temp, dim=1).unsqueeze(1))

        pre_mean = torch.cat(pre_mean, dim=1)
        pre_var = torch.cat(pre_var, dim=1)
        views = torch.cat(views, dim=1)
        states = torch.cat(states, dim=1)
        return pre_mean, pre_var, views, states


class Parking_Trajectory_Planner_LightningModule(pl.LightningModule):
    def __init__(self, paras: dict):
        super(Parking_Trajectory_Planner_LightningModule, self).__init__()
        self.save_hyperparameters('paras')

        self.model = Parking_Trajectory_Planner(paras)
        self.criterion_train = nn.GaussianNLLLoss(reduction='mean')
        self.criterion_val = Criterion_Dis(car_length=paras['car_length'], weight=0.5, reduction='max')
        self.criterion_test_dis = Criterion_Dis(car_length=paras['car_length'], weight=0.5, reduction='none')
        self.criterion_test_L1 = nn.L1Loss(reduction='none')
        # self.choose_parameters_train([0])
        self.optimizer = optim.Adam(self.parameters(), paras['lr_init'])
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=paras['lr_init'], total_steps=paras['max_epochs'], pct_start=0.1)
        self.test_results = list()
        self.test_losses = {
            'mean': list(),
            'max': list()
        }

    def choose_parameters_train(self, planner: list):
        planner = [str(planner[item]) for item in planner]
        for name, para in self.named_parameters():
            if name.split('.')[2] in planner:
                para.requires_grad = True
            else:
                para.requires_grad = False

    def run_base(self, batch, batch_idx):
        inp_start_point = batch[:, 0, 0:1]
        pre_mean, pre_var, views, states = self.model(inp_start_point)
        ref_mean = batch
        return pre_mean, ref_mean, pre_var, views, states

    def training_step(self, batch, batch_idx):
        pre_mean, ref_mean, pre_var, _, _ = self.run_base(batch, batch_idx)
        loss_train = self.criterion_train(pre_mean, ref_mean, pre_var)

        self.log('loss_train', loss_train, prog_bar=True)
        self.log('lr', self.scheduler.get_last_lr()[0], prog_bar=True)
        return loss_train

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
