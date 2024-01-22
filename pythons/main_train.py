import os
import shutil
import warnings
import pytorch_lightning as pl
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary, GradientAccumulationScheduler, Timer

from api.base.paras import paras_Parking_Trajectory_Planner
from api.dataset.split_dataset import paras_Parking_Trajectory_Planner_dataset
from api.model.model import Parking_Trajectory_Planner_LightningModule
from api.base.paths import path_ckpts, path_dataset

if __name__ == '__main__':
    pl.seed_everything(2024)

    # 为了好看，屏蔽warning，没事别解注释这个
    warnings.filterwarnings('ignore', category=PossibleUserWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    # 加载pytorch_lighting模型
    model_lighting = Parking_Trajectory_Planner_LightningModule(paras=paras_Parking_Trajectory_Planner)

    # 清空logs
    if os.path.exists(path_ckpts):
        shutil.rmtree(path_ckpts)

    # 设置训练器
    early_stop_callback = EarlyStopping(monitor='loss_val_nll', min_delta=0.001, patience=3, verbose=False, mode='min', check_on_train_epoch_end=False)
    model_checkpoint = ModelCheckpoint(monitor='loss_train', save_top_k=1, mode='min', verbose=False)
    model_summery = ModelSummary(max_depth=3)
    gradient_accumulation_scheduler = GradientAccumulationScheduler({10: 2})
    timer = Timer(duration='00:01:00:00', verbose=True)
    trainer = pl.Trainer(log_every_n_steps=1, max_epochs=paras_Parking_Trajectory_Planner['max_epochs'], check_val_every_n_epoch=1,
                         default_root_dir=path_dataset, accelerator='gpu', devices=1,
                         callbacks=[early_stop_callback, model_checkpoint, model_summery, timer, gradient_accumulation_scheduler])
    trainer.fit(model=model_lighting, train_dataloaders=paras_Parking_Trajectory_Planner_dataset['dataset_loader_train'],
                val_dataloaders=paras_Parking_Trajectory_Planner_dataset['dataset_loader_val'])
