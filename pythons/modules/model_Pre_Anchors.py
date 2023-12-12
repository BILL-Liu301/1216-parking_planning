import math

import numpy as np
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.W_q = torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda()
        self.W_k = torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda()
        self.W_v = torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, inp):
        Q = torch.matmul(self.W_q, inp)
        K = torch.matmul(self.W_k, inp)
        V = torch.matmul(self.W_v, inp)

        A = torch.matmul(K.T, Q)
        A = self.softmax(A)
        oup = torch.matmul(V, A)
        return oup, Q, K, V


class Encoder(nn.Module):
    def __init__(self, device, multi_head_size, seq_length,
                 input_size, middle_size, output_size):
        super(Encoder, self).__init__()
        self.device = device
        self.attention1 = SelfAttention(input_size).to(device)
        self.attention2 = SelfAttention(input_size).to(device)
        self.multi_head_size = multi_head_size
        self.norm = nn.LayerNorm(seq_length, eps=1e-6)
        self.softmax = nn.Softmax(dim=0)

        self.w_qkv_1 = [torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda(),
                        torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda(),
                        torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda()]
        self.w_qkv_2 = [torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda(),
                        torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda(),
                        torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda()]

        self.multi_head_layers_1, self.multi_head_layers_2 = [], []
        for i in range(multi_head_size):
            multi_head_layers_1_temp, multi_head_layers_2_temp = [], []
            for j in range(3):
                multi_head_layers_1_temp.append(nn.Sequential(nn.ReLU(), nn.Linear(seq_length, middle_size).to(device),
                                                              nn.ReLU(), nn.Linear(middle_size, middle_size).to(device),
                                                              nn.ReLU(), nn.Linear(middle_size, middle_size).to(device),
                                                              nn.ReLU(), nn.Linear(middle_size, seq_length).to(device)))
                multi_head_layers_2_temp.append(nn.Sequential(nn.ReLU(), nn.Linear(seq_length, middle_size).to(device),
                                                              nn.ReLU(), nn.Linear(middle_size, middle_size).to(device),
                                                              nn.ReLU(), nn.Linear(middle_size, middle_size).to(device),
                                                              nn.ReLU(), nn.Linear(middle_size, seq_length).to(device)))
            self.multi_head_layers_1.append(multi_head_layers_1_temp)
            self.multi_head_layers_2.append(multi_head_layers_2_temp)

        self.concat_layers_1 = nn.Sequential(nn.ReLU(), nn.Linear(input_size * multi_head_size, middle_size),
                                             nn.ReLU(), nn.Linear(middle_size, input_size))
        self.concat_layers_2 = nn.Sequential(nn.ReLU(), nn.Linear(input_size * multi_head_size, middle_size),
                                             nn.ReLU(), nn.Linear(middle_size, input_size))

        self.linear_layers_1 = nn.Sequential(nn.ReLU(), nn.Linear(input_size, middle_size),
                                             nn.ReLU(), nn.Linear(middle_size, middle_size),
                                             nn.ReLU(), nn.Linear(middle_size, output_size))

        self.linear_layers_2 = nn.Sequential(nn.ReLU(), nn.Linear(input_size, middle_size),
                                             nn.ReLU(), nn.Linear(middle_size, middle_size),
                                             nn.ReLU(), nn.Linear(middle_size, output_size))

    def dot_production_attention(self, scaled_qkv):
        scaled_attention = []
        for i in range(self.multi_head_size):
            scaled_attention.append(torch.matmul(scaled_qkv[i][2],
                                                 self.softmax(torch.matmul(scaled_qkv[i][1].T, scaled_qkv[i][0]))))
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
    def __init__(self, device, multi_head_size, seq_length,
                 input_size, middle_size,
                 output_size, encoded_size):
        super(Decoder, self).__init__()
        self.device = device
        self.multi_head_size = multi_head_size
        self.attention1 = SelfAttention(input_size).to(device)
        self.attention2 = SelfAttention(input_size).to(device)
        self.norm = nn.LayerNorm(seq_length, eps=1e-6)
        self.softmax = nn.Softmax(dim=0)

        self.w_qkv_1 = [torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda(),
                        torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda(),
                        torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda()]
        self.w_qkv_2 = [torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda(),
                        torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda(),
                        torch.normal(mean=0, std=2, size=(input_size, input_size), requires_grad=True).cuda()]

        self.multi_head_layers_1, self.multi_head_layers_2 = [], []
        for i in range(multi_head_size):
            multi_head_layers_1_temp, multi_head_layers_2_temp = [], []
            for j in range(3):
                multi_head_layers_1_temp.append(nn.Sequential(nn.ReLU(), nn.Linear(seq_length, middle_size).to(device),
                                                              nn.ReLU(), nn.Linear(middle_size, middle_size).to(device),
                                                              nn.ReLU(), nn.Linear(middle_size, middle_size).to(device),
                                                              nn.ReLU(), nn.Linear(middle_size, seq_length).to(device)))
                multi_head_layers_2_temp.append(nn.Sequential(nn.ReLU(), nn.Linear(seq_length, middle_size).to(device),
                                                              nn.ReLU(), nn.Linear(middle_size, middle_size).to(device),
                                                              nn.ReLU(), nn.Linear(middle_size, middle_size).to(device),
                                                              nn.ReLU(), nn.Linear(middle_size, seq_length).to(device)))
            self.multi_head_layers_1.append(multi_head_layers_1_temp)
            self.multi_head_layers_2.append(multi_head_layers_2_temp)

        self.concat_layers_1 = nn.Sequential(nn.ReLU(), nn.Linear(input_size * multi_head_size, middle_size),
                                             nn.ReLU(), nn.Linear(middle_size, input_size))
        self.concat_layers_2 = nn.Sequential(nn.ReLU(), nn.Linear(input_size * multi_head_size, middle_size),
                                             nn.ReLU(), nn.Linear(middle_size, input_size))

        self.linear_layers_1 = nn.Sequential(nn.ReLU(), nn.Linear(seq_length, middle_size),
                                             nn.ReLU(), nn.Linear(middle_size, middle_size),
                                             nn.ReLU(), nn.Linear(middle_size, seq_length))

        self.linear_layers_2 = nn.Sequential(nn.ReLU(), nn.Linear(seq_length, middle_size),
                                             nn.ReLU(), nn.Linear(middle_size, middle_size),
                                             nn.ReLU(), nn.Linear(middle_size, seq_length),
                                             nn.Tanh())

    def dot_production_attention(self, scaled_qkv):
        scaled_attention = []
        for i in range(self.multi_head_size):
            scaled_attention.append(torch.matmul(scaled_qkv[i][2],
                                                 self.softmax(torch.matmul(scaled_qkv[i][1].T, scaled_qkv[i][0]))))
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
        return torch.add(linear_1, linear_2)

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
    def __init__(self, device, sequence_length, multi_head_size,
                 encoder_input_size, encoder_middle_size, encoder_output_size,
                 decoder_input_size, decoder_middle_size, decoder_output_size,
                 paras):
        super().__init__()
        self.encoder = Encoder(device=device, multi_head_size=multi_head_size, seq_length=sequence_length,
                               input_size=encoder_input_size, middle_size=encoder_middle_size,
                               output_size=encoder_output_size).to(device)
        self.decoder = Decoder(device=device, multi_head_size=multi_head_size, seq_length=sequence_length,
                               input_size=decoder_input_size, middle_size=decoder_middle_size,
                               output_size=decoder_output_size, encoded_size=encoder_output_size).to(device)

        self.scale_ref = torch.from_numpy(np.array([[paras["limits"][0, 1], paras["limits"][1, 1], math.pi / 2]])).mT.to(device)

    def oup2anchors(self, oup, inp1, inp2):
        for i in range(0, oup.shape[1], 2):
            oup[:, i + 1] = torch.add(oup[:, i + 1], oup[:, i])
        anchors = oup * self.scale_ref
        return anchors

    def forward(self, inp1, inp2):
        oup = self.decoder(self.encoder(inp1), inp2)
        anchors = self.oup2anchors(oup, inp1, inp2)
        return anchors
