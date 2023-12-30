import math
import numpy as np
import torch
import torch.nn as nn

from .base_paras import num_step


class Encoder(nn.Module):
    def __init__(self, device, multi_head_size, seq_length_inp, seq_length_oup,
                 input_size, middle_size, output_size):
        super(Encoder, self).__init__()
        self.device = device
        self.multi_head_size = multi_head_size
        self.norm = nn.LayerNorm(seq_length_inp, eps=1e-6)
        self.softmax = nn.Softmax(dim=0)
        self.bias = False
        self.softmax_switch = False

        self.w_qkv_1 = nn.Parameter(torch.normal(mean=0, std=2, size=(3, input_size, input_size)))
        self.w_qkv_2 = nn.Parameter(torch.normal(mean=0, std=2, size=(3, input_size, input_size)))

        self.multi_head_layers_1, self.multi_head_layers_2 = nn.ModuleList([]), nn.ModuleList([])
        for i in range(multi_head_size):
            multi_head_layers_1_temp, multi_head_layers_2_temp = nn.ModuleList([]), nn.ModuleList([])
            for j in range(3):
                multi_head_layers_1_temp.append(nn.Sequential(nn.ReLU(), nn.Linear(seq_length_inp, middle_size, bias=self.bias),
                                                              nn.ReLU(), nn.Linear(middle_size, seq_length_inp, bias=self.bias)).to(device))
                multi_head_layers_2_temp.append(nn.Sequential(nn.ReLU(), nn.Linear(seq_length_inp, middle_size, bias=self.bias),
                                                              nn.ReLU(), nn.Linear(middle_size, seq_length_inp, bias=self.bias)).to(device))
            self.multi_head_layers_1.append(multi_head_layers_1_temp)
            self.multi_head_layers_2.append(multi_head_layers_2_temp)

        self.concat_layers_1 = nn.Sequential(nn.ReLU(), nn.Linear(input_size * multi_head_size, middle_size, bias=self.bias),
                                             nn.ReLU(), nn.Linear(middle_size, input_size, bias=self.bias))
        self.concat_layers_2 = nn.Sequential(nn.ReLU(), nn.Linear(input_size * multi_head_size, middle_size, bias=self.bias),
                                             nn.ReLU(), nn.Linear(middle_size, input_size, bias=self.bias))

        self.linear_layers_1 = nn.Sequential(nn.ReLU(), nn.Linear(input_size, middle_size, bias=self.bias),
                                             nn.ReLU(), nn.Linear(middle_size, output_size, bias=self.bias))

        self.linear_layers_2 = nn.Sequential(nn.ReLU(), nn.Linear(input_size, middle_size, bias=self.bias),
                                             nn.ReLU(), nn.Linear(middle_size, output_size, bias=self.bias))

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

    def forward(self, inp):
        _, q1, k1, v1 = self.self_attention(self.w_qkv_1, inp)
        multi_head_attention_1 = self.multi_head_attention([q1, k1, v1], self.multi_head_layers_1, self.concat_layers_1)
        multi_head_attention_1 = self.norm(torch.add(inp, multi_head_attention_1))

        _, q2, k2, v2 = self.self_attention(self.w_qkv_2, multi_head_attention_1)
        multi_head_attention_2 = self.multi_head_attention([q2, k2, v2], self.multi_head_layers_2, self.concat_layers_2)
        multi_head_attention_2 = self.norm(torch.add(inp, multi_head_attention_2))

        oup = [self.linear_layers_1(multi_head_attention_2.T).T + multi_head_attention_2,
               self.linear_layers_2(multi_head_attention_2.T).T + multi_head_attention_2]
        return oup


class Decoder(nn.Module):
    def __init__(self, device, multi_head_size,
                 seq_length_inp, seq_length_oup,
                 input_size, middle_size,
                 output_size, encoded_size):
        super(Decoder, self).__init__()
        self.device = device
        self.multi_head_size = multi_head_size
        self.norm = nn.LayerNorm(seq_length_inp, eps=1e-6)
        self.softmax = nn.Softmax(dim=0)
        self.bias = False
        self.softmax_switch = False

        self.w_qkv_1 = nn.Parameter(torch.normal(mean=0, std=2, size=(3, input_size, input_size)))
        self.w_qkv_2 = nn.Parameter(torch.normal(mean=0, std=2, size=(3, input_size, input_size)))

        self.multi_head_layers_1, self.multi_head_layers_2 = nn.ModuleList([]), nn.ModuleList([])
        for i in range(multi_head_size):
            multi_head_layers_1_temp, multi_head_layers_2_temp = nn.ModuleList([]), nn.ModuleList([])
            for j in range(3):
                multi_head_layers_1_temp.append(nn.Sequential(nn.ReLU(), nn.Linear(seq_length_inp, middle_size, bias=self.bias),
                                                              nn.ReLU(), nn.Linear(middle_size, seq_length_inp, bias=self.bias)).to(device))
                multi_head_layers_2_temp.append(nn.Sequential(nn.ReLU(), nn.Linear(seq_length_inp, middle_size, bias=self.bias),
                                                              nn.ReLU(), nn.Linear(middle_size, seq_length_inp, bias=self.bias)).to(device))
            self.multi_head_layers_1.append(multi_head_layers_1_temp)
            self.multi_head_layers_2.append(multi_head_layers_2_temp)

        self.concat_layers_1 = nn.Sequential(nn.ReLU(), nn.Linear(input_size * multi_head_size, middle_size, bias=self.bias),
                                             nn.ReLU(), nn.Linear(middle_size, input_size, bias=self.bias))
        self.concat_layers_2 = nn.Sequential(nn.ReLU(), nn.Linear(input_size * multi_head_size, middle_size, bias=self.bias),
                                             nn.ReLU(), nn.Linear(middle_size, input_size, bias=self.bias))

        self.linear_layers_1 = nn.Sequential(nn.ReLU(), nn.Linear(seq_length_inp, middle_size, bias=self.bias),
                                             nn.ReLU(), nn.Linear(middle_size, seq_length_oup, bias=self.bias))

        self.linear_layers_2 = nn.Sequential(nn.ReLU(), nn.Linear(seq_length_oup, middle_size, bias=self.bias),
                                             nn.ReLU(), nn.Linear(middle_size, seq_length_oup, bias=self.bias),
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

    def linear_layers(self, inp):
        linear_1 = self.linear_layers_1(inp)
        linear_2 = self.linear_layers_2(linear_1)
        # return torch.add(linear_1, linear_2)
        return linear_2

    def angel_normal(self, oup):
        oup[-1, :] = oup[-1, :] % (2 * math.pi)
        return oup

    def forward(self, encoded_kv, inp):
        _, q1, k1, v1 = self.self_attention(self.w_qkv_1, inp)
        multi_head_attention_1 = self.multi_head_attention([q1, k1, v1], self.multi_head_layers_1, self.concat_layers_1)
        multi_head_attention_1 = self.norm(torch.add(inp, multi_head_attention_1))

        _, q2, _, _ = self.self_attention(self.w_qkv_2, multi_head_attention_1)
        k2, v2 = encoded_kv
        multi_head_attention_2 = self.multi_head_attention([q2, k2, v2], self.multi_head_layers_2, self.concat_layers_2)
        multi_head_attention_2 = self.norm(torch.add(inp, multi_head_attention_2))

        oup = self.linear_layers(multi_head_attention_2)
        # oup = self.angel_normal(oup)
        return oup


class Pre_Anchors(nn.Module):
    def __init__(self, device, sequence_length_inp, sequence_length_oup, multi_head_size,
                 encoder_input_size, encoder_middle_size, encoder_output_size,
                 decoder_input_size, decoder_middle_size, decoder_output_size,
                 paras):
        super().__init__()
        self.device = device
        self.encoder = Encoder(device=device, multi_head_size=multi_head_size,
                               seq_length_inp=sequence_length_inp, seq_length_oup=sequence_length_oup-1,
                               input_size=encoder_input_size, middle_size=encoder_middle_size,
                               output_size=encoder_output_size).to(device)
        self.decoder = Decoder(device=device, multi_head_size=multi_head_size,
                               seq_length_inp=sequence_length_inp, seq_length_oup=sequence_length_oup-1,
                               input_size=decoder_input_size, middle_size=decoder_middle_size,
                               output_size=decoder_output_size, encoded_size=encoder_output_size).to(device)

        self.scale_ref = torch.from_numpy(np.array([[paras["limits"][0, 1], paras["limits"][1, 1], math.pi / 2]])).mT.to(device)
        self.num_step = num_step
        self.start_dis = torch.zeros([decoder_output_size, 1]).to(device)

    def get_model(self, name):
        if name == 'encoder':
            return self.encoder
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
        for step in range(self.num_step):
            oup = torch.cat([self.start_dis, self.decoder(self.encoder(inp1[step]), inp2[step])], dim=1).unsqueeze(dim=0)
            if step == 0:
                start = inp1[0:1, :, 0:1]
            else:
                start = anchors[-1:, :, -1:]
            anchors = torch.cat([anchors, self.oup2anchors(start, oup)])
        return anchors
