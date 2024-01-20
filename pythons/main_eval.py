import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import warnings
from tqdm import tqdm

from lightning_fabric.utilities.warnings import PossibleUserWarning

from api.base.paras import paras_Parking_Trajectory_Planner
from api.dataset.split_dataset import Parking_Trajectory_Planner_Dataset
from api.model.model import Parking_Trajectory_Planner_LightningModule
from api.base.paths import path_ckpt_best_version, path_ckpts, path_figs_test
# from pythons.api.util.plots import plot_for_prediction_seq2seq_val_test

if __name__ == '__main__':
    pl.seed_everything(2024)
    plt.figure(figsize=(20, 11.25))

    # # 为了好看，屏蔽warning，没事别解注释这个
    # warnings.filterwarnings('ignore', category=PossibleUserWarning)
    # warnings.filterwarnings('ignore', category=UserWarning)

    # 找到ckpt
    path_version = path_ckpts + 'lightning_logs/version_0/checkpoints/'
    # path_version = path_ckpt_best_version + 'version_0/checkpoints/'
    ckpt = path_version + os.listdir(path_version)[0]

    # 设置训练器
    # trainer = pl.Trainer(default_root_dir=path_ckpts, accelerator='gpu', devices=1)
    # model_lighting = Parking_Trajectory_Planner_LightningModule.load_from_checkpoint(checkpoint_path=ckpt, paras=paras_Parking_Trajectory_Planner)
    # dataloaders = Parking_Trajectory_Planner_Dataset['dataset_loader_test']
    # trainer.test(model=model_lighting, dataloaders=dataloaders)

    # 结果展示
    # loss_mean, loss_max, loss_min = torch.cat(model_lighting.test_losses['mean'], dim=0), torch.cat(model_lighting.test_losses['max'], dim=0), torch.cat(model_lighting.test_losses['min'], dim=0)
    # print(f'总计有{len(dataloaders.dataset)}组数据')
    # print(f'平均均值误差：{loss_mean.mean()}K')
    # print(f'平均最大误差：{loss_max.mean()}K')
    # print(f'平均最小误差：{loss_min.mean()}K')
    # for i in tqdm(range(0, len(model.test_results), 1), desc='Test', leave=False, ncols=100, disable=False):
    #     test_results = model.test_results[i]
    #     plot_for_prediction_seq2seq_val_test(test_results, paras_Prediction_Seq2Seq)
    #     plt.savefig(path_figs_test + f'{i}.png')
