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

