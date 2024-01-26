import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import warnings
from tqdm import tqdm
from lightning_fabric.utilities.warnings import PossibleUserWarning

from api.base.paras import paras_Parking_Trajectory_Planner
from api.dataset.split_dataset import paras_Parking_Trajectory_Planner_dataset
from api.model.model import Parking_Trajectory_Planner_LightningModule
from api.base.paths import path_ckpt_best_version, path_ckpts, path_figs_test, path_dataset
from pythons.api.util.plots import plot_for_results_macro, plot_for_results_micro, plot_for_results_dynamic

if __name__ == '__main__':
    pl.seed_everything(2024)
    plt.figure(figsize=(20, 11.25))

    # 为了好看，屏蔽warning，没事别解注释这个
    warnings.filterwarnings('ignore', category=PossibleUserWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    # 找到ckpt
    path_version = path_ckpts + 'version_1/checkpoints/'
    # path_version = path_ckpt_best_version + 'version_1/checkpoints/'
    ckpt = path_version + os.listdir(path_version)[0]

    # 设置训练器
    trainer = pl.Trainer(default_root_dir=path_dataset, accelerator='gpu', devices=1)
    model_lighting = Parking_Trajectory_Planner_LightningModule.load_from_checkpoint(checkpoint_path=ckpt)
    dataloaders = paras_Parking_Trajectory_Planner_dataset['dataset_loader_test']
    trainer.test(model=model_lighting, dataloaders=dataloaders)

    # 结果展示
    loss_mean, loss_max = torch.cat(model_lighting.test_losses['mean'], dim=0), torch.cat(model_lighting.test_losses['max'], dim=0)
    print(f'总计有{len(dataloaders.dataset)}组数据')
    print(f'平均均值误差：{loss_mean.mean()}m')
    print(f'平均最大误差：{loss_max.mean()}m')
    for i in tqdm(range(0, len(model_lighting.test_results), 1), desc='Test', leave=False, ncols=100, disable=False):
        test_results = model_lighting.test_results[i]
        # 动态分析
        plot_for_results_dynamic(test_results, paras_Parking_Trajectory_Planner)
        # 全局分析
        plot_for_results_macro(test_results, paras_Parking_Trajectory_Planner)
        plt.savefig(path_figs_test + f'{i}_macro.png')
        # 局部误差分析
        plot_for_results_micro(test_results, paras_Parking_Trajectory_Planner)
        plt.savefig(path_figs_test + f'{i}_micro.png')
