import numpy as np
import torch
import torch.nn as nn

ratio_train = 0.7
ratio_test = 0.28
ratio_val = 1 - ratio_train - ratio_test

lr_init = 1e-3
epoch_max = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 64
BEGIN = np.array([[-1], [0], [0]])
END = np.array([[-1], [0], [0]])

criterion_train = nn.MSELoss(reduction='sum')
criterion_test = nn.MSELoss(reduction='none')

