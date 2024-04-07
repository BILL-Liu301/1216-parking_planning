# import torch
# import torch.nn as nn
# import numpy as np
#
# class Criterion_Dis(nn.Module):
#     r"""Criterion the Distance both of the front and rear.
#
#     Input Shape: [pre] and [ref] should have the same shape : [B, num_anchor_per_step, len_info_loc]
#
#     Args:
#         car_length (float, optional): The length between front wear and rear wear. Default: ``'4.0'``
#
#         weight (float, optional): The weight of Rear Anchor. Default: ``'0.5'``
#
#         reduction (str, optional): Specifies the reduction to apply to the output:
#             ``'none'`` | ``'mean'`` | ``'sum'`` | ``'max'`` | ``'min'``. Default: ``'mean'``
#         """
#     def __init__(self, car_length=4.0, weight=0.5, reduction='mean'):
#         super(Criterion_Dis, self).__init__()
#         self.car_length = car_length
#         self.reduction = reduction
#         self.weight = weight
#
#     def cal_front_from_rear(self, inp):
#         inp_xy1 = inp[:, :, :, 0:2]
#         ang = torch.cat((torch.cos(inp[:, :, :, -1:]), torch.sin(inp[:, :, :, -1:])), dim=-1)
#         inp_xy2 = inp_xy1 + self.car_length * ang
#         oup = torch.cat((inp_xy1, inp_xy2), dim=-1)
#         return oup
#
#     def cal_dis(self, pre_x, pre_y, ref_x, ref_y):
#         # 计算各点对应插值
#         delta_x = pre_x - ref_x.repeat(1, 1, 1, ref_x.shape[2]).transpose(2, 3)
#         delta_y = pre_y - ref_y.repeat(1, 1, 1, ref_y.shape[2]).transpose(2, 3)
#
#         # 计算各点对应距离
#         dis = torch.sqrt(delta_x ** 2 + delta_y ** 2).min(dim=-1, keepdim=True).values
#
#         return dis
#
#     def forward(self, pre, ref):
#         # 提取前后轮的xy坐标
#         pre_xy = self.cal_front_from_rear(pre)
#         ref_xy = self.cal_front_from_rear(ref)
#         pre_x_f, pre_y_f, pre_x_r, pre_y_r = pre_xy[:, :, :, 0:1], pre_xy[:, :, :, 1:2], pre_xy[:, :, :, 2:3], pre_xy[:, :, :, 3:4]
#         ref_x_f, ref_y_f, ref_x_r, ref_y_r = ref_xy[:, :, :, 0:1], ref_xy[:, :, :, 1:2], ref_xy[:, :, :, 2:3], ref_xy[:, :, :, 3:4]
#
#         # 分别计算每个点到参考轨迹最近的距离
#         dis_f = self.cal_dis(pre_x_f, pre_y_f, ref_x_f, ref_y_f)
#         dis_r = self.cal_dis(pre_x_r, pre_y_r, ref_x_r, ref_y_r)
#
#         loss = dis_f * self.weight + dis_r * (1 - self.weight)
#         if self.reduction == 'none':
#             return loss
#         elif self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         elif self.reduction == 'max':
#             return loss.max()
#         elif self.reduction == 'min':
#             return loss.min()
#         else:
#             print(f'The reduction [{self.reduction}] is useless. Waste my time!!!')
#


import torch
import torch.nn as nn


class Criterion_Dis(nn.Module):
    r"""Criterion the Distance both of the front and rear.

    Input Shape: [pre] and [ref] should have the same shape : [B, num_anchor_per_step, len_info_loc]

    Args:
        car_length (float, optional): The length between front wear and rear wear. Default: ``'4.0'``

        weight (float, optional): The weight of Rear Anchor. Default: ``'0.5'``

        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'`` | ``'max'`` | ``'min'``. Default: ``'mean'``
        """
    def __init__(self, car_length=4.0, weight=0.5, reduction='mean'):
        super(Criterion_Dis, self).__init__()
        self.car_length = car_length
        self.reduction = reduction
        self.weight = weight

    def cal_front_from_rear(self, inp):
        inp_xy1 = inp[:, :, :, 0:2]
        ang = torch.cat((torch.cos(inp[:, :, :, -1:]), torch.sin(inp[:, :, :, -1:])), dim=-1)
        inp_xy2 = inp_xy1 + self.car_length * ang
        oup = torch.cat((inp_xy1, inp_xy2), dim=-1)
        return oup

    def forward(self, pre, ref):
        if pre.shape != ref.shape or pre.ndim != 4 or ref.ndim != 4:
            print(f'The shape of [pre] or [ref] in Criterion_Dis is not right')
            print(f'[pre]: {pre.shape}, [ref]: {ref.shape}')
            raise Exception
        pre_xy = self.cal_front_from_rear(pre)
        ref_xy = self.cal_front_from_rear(ref)

        delta = ref_xy - pre_xy
        loss_xy1 = torch.sqrt(torch.sum(torch.pow(delta[:, :, :, 0:2], 2), dim=-1, keepdim=True))
        loss_xy2 = torch.sqrt(torch.sum(torch.pow(delta[:, :, :, 2:4], 2), dim=-1, keepdim=True))
        loss = loss_xy1 * self.weight + loss_xy2 * (1 - self.weight)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'max':
            return loss.max()
        elif self.reduction == 'min':
            return loss.min()
        else:
            print(f'The reduction [{self.reduction}] is useless. Waste my time!!!')
