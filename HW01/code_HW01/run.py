import torch
import torch.nn as nn
import numpy as np
import tqdm
import pandas as pd
import os
import csv
import time
from torch.utils.tensorboard import SummaryWriter
from model import hw01_model
from process import COVID_Dataset, getTrainData, getTestData
from tool import train_valid_split, select_feat
from torch.utils.data import DataLoader



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': True,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 3000,     # Number of epochs.
    'batch_size': 256,
    'learning_rate': 5e-6,
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}

train_data = pd.read_csv('data/covid.train_new.csv').values
test_data = pd.read_csv('data/covid.test_un.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])
print(f"""train_data size: {np.array(train_data).shape} 
valid_data size: {np.array(valid_data).shape} 
test_data size: {np.array(test_data).shape}""")
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

train_dataset, valid_dataset, test_dataset = COVID_Dataset(x_train, y_train), \
                                            COVID_Dataset(x_valid, y_valid), \
                                            COVID_Dataset(x_test)


train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, config['batch_size'], shuffle=False, pin_memory=True)

def trainer(model, train_loader, valid_loader, config, device):
    writer = SummaryWriter("logs")
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    loss_fn = nn.MSELoss(reduction='mean')

    for epoch in range(config['n_epochs']):
        step = 0
        model.train()
        loss_arr = []
        for x, y in train_loader:
            step+=1
            x, y = x.to(device), y.to(device)
            predict = model(x)
            loss = loss_fn(predict, y)
            loss_arr.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = sum(loss_arr) / len(loss_arr)
        writer.add_scalar('loss', mean_loss, epoch)

        model.eval()
        valid_loss_arr = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                predict = model(x)
                valid_loss = loss_fn(predict, y)
            valid_loss_arr.append(valid_loss)
        mean_valid_loss = sum(valid_loss_arr) / len(valid_loss_arr)
        writer.add_scalar('valid_loss', mean_valid_loss, epoch)

        print(f"第{epoch}轮：loss：{mean_loss}, valid_loss：{mean_valid_loss}")


def run():
    model = hw01_model(inputDim=x_train.shape[1], outputDim=1).to(device)
    trainer(model, train_loader, valid_loader, config, device)

if __name__ == "__main__":
    start_time = time.time()
    run()
    end_time = time.time()
    total_time = time.strftime("%H:%M:%S:", time.gmtime(end_time-start_time))
    print(f"总耗时:{total_time}")













# def run():
#     inputDim = 116
#     outputDim = 1
#     train_epoch = 1
#     model = hw01_model(inputDim=inputDim, outputDim=outputDim)
#     optim = torch.optim.Adam(model.parameters(), lr=0.0001)
#     for epoch in range(train_epoch):
#         train_loss = train(model, optim)
#         # predict = test(model)
#         # writer.add_scalar("test_loss", test_loss, epoch)
#
#
# def train(model, optim):
#     dataset = getTrainData()
#     i = 0
#     for data in dataset:
#         target = data[-1]
#         train_data = data[0:-1]
#         predict = model(train_data)
#         train_loss = torch.pow(target - predict, 2)
#         optim.zero_grad()
#         train_loss.backward()
#         optim.step()
#         writer.add_scalar("train_loss", train_loss, i)
#         i += 1
#     return train_loss
#
# def test(model):
#     dataset = getTestData()
#     for data in dataset:
#         # target = data[-1]
#         test_data = data
#         predict = model(test_data)
#         # test_loss = torch.pow(target - predict, 2)
#     return predict, predict



















