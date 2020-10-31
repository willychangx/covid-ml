import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

import os

# TODO: Construct your data in the following baseline structure: 1) ./Dataset/Train/image/, 2) ./Dataset/Train/label, 3) ./Dataset/Test/image, and 4) ./Dataset/Test/label
class DataSet(Dataset):
    def __init__(self, root):
        
        currDir = os.getcwd()

        folderNames = root.split('/')
        currPath = currDir
        for name in folderNames:
            if name != '.' and name != '':
                try:
                    os.chdir(name)
                except:
                    os.mkdir(name)
                    os.chdir(name)

        os.mkdir('image')
        os.mkdir('label')

        os.chdir(currDir)
        
        self.ROOT = root
        self.images = read_images(root + "/image")
        self.labels = read_labels(root + "/label")

    def __len__(self):
        # Return number of points in the dataset

        return len(self.images)

    def __getitem__(self, idx):
        # Here we have to return the item requested by `idx`. The PyTorch DataLoader class will use this method to make an iterable for training/validation loop.

        img = images[idx]
        label = labels[idx]

        return img, label

# Load the dataset and train and test splits
print("Loading datasets...")

# Data path
DATA_train_path = DataSet('./Dataset/Train')
DATA_test_path = DataSet('./Dataset/Test')

print(DATA_test_path)