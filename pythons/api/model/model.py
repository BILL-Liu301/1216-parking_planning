import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from scipy.stats import norm
import pytorch_lightning as pl


class Parking_Trajectory_Planner(nn.Module):
    def __init__(self, paras: dict):
        super(Parking_Trajectory_Planner, self).__init__()

    def forward(self):
        pass


class Parking_Trajectory_Planner_LightningModule(pl.LightningModule):
    def __init__(self, paras: dict):
        super(Parking_Trajectory_Planner_LightningModule, self).__init__()

    def forward(self):
        pass

    def run_base(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
    