import numpy as np
import torch
import torch.nn as nn

lr_init = 1e-2
epoch_max = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 32
BEGIN = np.array([[-1], [0], [0]])
END = np.array([[-1], [0], [0]])

criterion_train = nn.MSELoss(reduction='sum')
criterion_test = nn.MSELoss(reduction='none')

