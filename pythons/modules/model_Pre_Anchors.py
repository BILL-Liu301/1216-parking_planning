import math
import numpy as np
import torch
import torch.nn as nn

from .base_paras import num_step


class Encoder(nn.Module):
    def __init__(self, seq_length_inp, seq_length_middle, seq_length_oup):
        super(Encoder, self).__init__()
        self.norm = nn.LayerNorm(seq_length_oup, eps=1e-6)
        self.bias = False

        self.linear_layers_1 = nn.Sequential(nn.Tanh(), nn.Linear(seq_length_inp, seq_length_middle, bias=self.bias),
                                             nn.ReLU(), nn.Linear(seq_length_middle, seq_length_middle, bias=self.bias),
                                             nn.ReLU(), nn.Linear(seq_length_middle, seq_length_oup, bias=self.bias))

        self.linear_layers_2 = nn.Sequential(nn.Tanh(), nn.Linear(seq_length_oup, seq_length_middle, bias=self.bias),
                                             nn.ReLU(), nn.Linear(seq_length_middle, seq_length_middle, bias=self.bias),
                                             nn.ReLU(), nn.Linear(seq_length_middle, seq_length_oup, bias=self.bias))

    def forward(self, inp):
        lineared_1 = self.linear_layers_1(inp)
        lineared_2 = self.linear_layers_2(lineared_1)
        oup = self.norm(lineared_1 + lineared_2)
        return oup


class Decoder(nn.Module):
    def __init__(self, device, state_size, multi_head_size, seq_length_inp, seq_length_middle, seq_length_oup):
        super(Decoder, self).__init__()
        self.norm = nn.LayerNorm(seq_length_oup, eps=1e-6)
        self.bias = False
        self.softmax_switch = False
        self.multi_head_size = multi_head_size
        self.softmax = nn.Softmax()

        self.w_qkv_1 = nn.Parameter(torch.normal(mean=0, std=2, size=(3, state_size, state_size)))
        self.w_qkv_2 = nn.Parameter(torch.normal(mean=0, std=2, size=(3, state_size, state_size)))
        self.multi_head_layers = nn.ModuleList([])
        for i in range(multi_head_size):
            multi_head_layers_temp = nn.ModuleList([])
            for j in range(3):
                multi_head_layers_temp.append(nn.Sequential(nn.ReLU(), nn.Linear(seq_length_inp, seq_length_middle, bias=self.bias),
                                                            nn.ReLU(), nn.Linear(seq_length_middle, seq_length_inp, bias=self.bias)).to(device))
            self.multi_head_layers.append(multi_head_layers_temp)

        self.concat_layers = nn.Sequential(nn.Tanh(), nn.Linear(state_size * multi_head_size, seq_length_middle, bias=self.bias),
                                           nn.ReLU(), nn.Linear(seq_length_middle, state_size, bias=self.bias))

        self.linear_layers = nn.Sequential(nn.Tanh(), nn.Linear(seq_length_inp, seq_length_middle, bias=self.bias),
                                           nn.ReLU(), nn.Linear(seq_length_middle, seq_length_oup, bias=self.bias),
                                           nn.Tanh())

    def dot_production_attention(self, scaled_qkv):
        scaled_attention = []
        for i in range(self.multi_head_size):
            if self.softmax_switch:
                scaled_attention.append(torch.matmul(scaled_qkv[i][2], self.softmax(torch.matmul(scaled_qkv[i][1].T, scaled_qkv[i][0]))))
            else:
                scaled_attention.append(torch.matmul(scaled_qkv[i][2], torch.matmul(scaled_qkv[i][1].T, scaled_qkv[i][0])))
        return scaled_attention

    def concat(self, inp, concat_layers):
        inp = torch.concat(inp, dim=0)
        oup = concat_layers(inp.T)
        return oup.T

    def self_attention(self, w_qkv, inp):
        q = torch.matmul(w_qkv[0], inp)
        k = torch.matmul(w_qkv[1], inp)
        v = torch.matmul(w_qkv[2], inp)

        b = torch.matmul(k.T, q)
        if self.softmax_switch:
            b = self.softmax(b)
        oup = torch.matmul(v, b)
        return oup, q, k, v

    def multi_head_attention(self, qkv, multi_head_layers, concat_layers):
        scaled_qkv = []
        for multi_head in multi_head_layers:
            scaled_qkv_temp = []
            for index_qkv, linear_layers in enumerate(multi_head):
                scaled_qkv_temp.append(linear_layers(qkv[index_qkv]))
            scaled_qkv.append(scaled_qkv_temp)
        scaled_attention = self.dot_production_attention(scaled_qkv)
        oup = self.concat(scaled_attention, concat_layers)
        return oup

    def forward(self, encoded_1, encoded_2):
        _, q, _, _ = self.self_attention(self.w_qkv_1, encoded_1)
        _, _, k, v = self.self_attention(self.w_qkv_2, encoded_2)
        attentioned = self.multi_head_attention([q, k, v], self.multi_head_layers, self.concat_layers)
        oup = self.linear_layers(attentioned)
        return oup


class Pre_Anchors(nn.Module):
    def __init__(self, device,sequence_length_inp_1, sequence_length_inp_2, sequence_length_middle, sequence_length_oup,
                 multi_head_size, state_size, paras):
        super().__init__()
        self.device = device
        self.encoder_1 = Encoder(seq_length_inp=sequence_length_inp_1, seq_length_middle=sequence_length_middle, seq_length_oup=sequence_length_oup - 1).to(device)
        self.encoder_2 = Encoder(seq_length_inp=sequence_length_inp_2, seq_length_middle=sequence_length_middle, seq_length_oup=sequence_length_oup - 1).to(device)
        self.decoder = Decoder(device=device, state_size=state_size, multi_head_size=multi_head_size,
                               seq_length_inp=sequence_length_oup - 1, seq_length_middle=sequence_length_middle, seq_length_oup=sequence_length_oup - 1).to(device)

        self.scale_ref = torch.from_numpy(np.array([[paras["limits"][0, 1], paras["limits"][1, 1], math.pi / 2]])).mT.to(device)
        self.num_step = num_step
        self.start_dis = torch.zeros([state_size, 1]).to(device)

    def get_model(self, name):
        if name == 'encoder_1':
            return self.encoder_1
        elif name == 'encoder_2':
            return self.encoder_2
        elif name == 'decoder':
            return self.decoder
        else:
            print(f'{name} is not defined!!')
            return

    def oup2anchors(self, start, oup):
        oup = torch.cumsum(oup, dim=2)
        anchors = oup * self.scale_ref + start
        return anchors

    def forward(self, inp1, inp2):
        anchors = torch.tensor([]).to(self.device)
        encoded_1 = self.encoder_1(inp1[0])
        for step in range(self.num_step):
            oup = torch.cat([self.start_dis, self.decoder(encoded_1.clone(), self.encoder_2(inp2[step]))], dim=1).unsqueeze(dim=0)
            if step == 0:
                start = inp1[0:1, :, 0:1]
            else:
                start = anchors[-1:, :, -1:]
            anchors = torch.cat([anchors, self.oup2anchors(start, oup)])
        return anchors
