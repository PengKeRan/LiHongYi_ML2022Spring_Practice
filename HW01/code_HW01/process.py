import csv
from torch.utils.data import Dataset, DataLoader, random_split
import torch


class COVID_Dataset(Dataset):
    def __init__(self, features, target=None):
        self.features = torch.FloatTensor(features)
        if target is None:
            self.target = target
        else:
            self.target = torch.FloatTensor(target)

    def __getitem__(self, idx):
        if self.target is None:
            return self.features[idx]
        else:
            return self.features[idx], self.target[idx]

    def __len__(self):
        return len(self.features)

def getTrainData():
    trainData = []
    with open('data/covid.train_new.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == 'id':
                continue
            trainData.append([float(d) for d in row][1:])
    return trainData


def getTestData():
    testData = []
    with open('data/covid.test_un.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == 'id':
                continue
            testData.append([float(d) for d in row][1:])
    return testData

