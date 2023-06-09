# Lightly does not allow you to grab data using .pt extension. We are going to write our own thing.

import os
import torch
from torch.utils.data import Dataset

class MySSLDataset(Dataset):
    def __init__(self, directory, transform=None):
        super().__init__()
        self.directory = directory
        self.file_list = os.listdir(directory)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.file_list[idx])
        sample = torch.load(file_path)
        if self.transform:
            sample = self.transform(sample)
        return sample
