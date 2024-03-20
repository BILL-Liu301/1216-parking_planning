from torch.utils.data import DataLoader, random_split

from .load_dataset import Parking_Trajectory_Planner_Dataset
from api.base.paths import path_dataset_pkl

dataset_base = Parking_Trajectory_Planner_Dataset(path_dataset_pkl=path_dataset_pkl)

# 分割train, test, val，并进行数据加载
train_valid_set_size = int(len(dataset_base) * 0.8)
test_set_size = len(dataset_base) - train_valid_set_size
train_valid_set, test_set = random_split(dataset_base, [train_valid_set_size, test_set_size])

train_set_size = int(len(train_valid_set) * 0.8)
valid_set_size = len(train_valid_set) - train_set_size
train_set, valid_set = random_split(train_valid_set, [train_set_size, valid_set_size])

dataset_loader_train = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True, num_workers=0)
dataset_loader_val = DataLoader(valid_set, batch_size=64, shuffle=True, pin_memory=True, num_workers=0)
dataset_loader_test = DataLoader(test_set, batch_size=512, shuffle=False, pin_memory=True, num_workers=0)
paras_Parking_Trajectory_Planner_dataset = {
    'dataset_loader_train': dataset_loader_train,
    'dataset_loader_val': dataset_loader_val,
    'dataset_loader_test': dataset_loader_test
}
