import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
import src.config as config
from sklearn.model_selection import train_test_split


class Dataet():
    def __init__(self,reviews,labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return (len(self.reviews)-1)

    def __getitem__(self, item):
        data = self.reviews.iloc[item]
        label = self.labels[item]
        return data,label




