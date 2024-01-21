import pickle
import torch
from torch.utils.data import Dataset


class Parking_Trajectory_Planner_Dataset(Dataset):
    def __init__(self, path_dataset_pkl):
        self.data = self.load_from_pkl(path_dataset_pkl)

    def load_from_pkl(self, path_data_origin_pkl):
        with open(path_data_origin_pkl, 'rb') as pkl:
            data_pkl = pickle.load(pkl)
            pkl.close()
        solutions = data_pkl['solutions']
        return solutions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.from_numpy(self.data[item]).to(torch.float32)
