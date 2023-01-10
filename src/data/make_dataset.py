import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import json
import dill


def mnist(path):
    """ Using corrupted dataset """
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

    train_path = path + "/train_"
    test_path = path + "/test.npz"
    
    train_data = MyDataset(train_path, train=True)
    test_data = MyDataset(test_path, train=False)
  
    return train_data, test_data

if __name__ == "__main__":
    input_path = "data/external/"
    output_path = "data/processed/"

    train_data, test_data = mnist(input_path)
    torch.save(train_data, output_path + 'train_tensor.pt', pickle_protocol=True, pickle_module=dill)
    torch.save(test_data, output_path + 'test_tensor.pt', pickle_protocol=True, pickle_module=dill)
