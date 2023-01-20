import torch
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, path, train):
        datas = []
        if train:
            datas = []
            for i in range(5):
                datas.append(np.load((path + str(i) + ".npz"), allow_pickle=True))
            self.imgs = torch.tensor(np.concatenate([c['images'] for c in datas])).reshape(-1, 1, 28, 28)
            self.labels = torch.tensor(np.concatenate([c['labels'] for c in datas]))
        else:
            data = np.load(path)
            self.imgs = data['images']
            self.imgs = torch.tensor(self.imgs).reshape(-1, 1, 28, 28)
            self.labels = data['labels']

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]