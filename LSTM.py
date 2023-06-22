# LSTM on dataset 1

import torch
from torch import nn
import lib
import time
import visdom
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,  TensorDataset

def train(net, train_X, train_Y, test_X, test_Y, num_epochs, lr, batch_size, updater, device, wind=None):
    start = time.time()
    loss = nn.MSELoss()
    train_X, train_Y = train_X.to(device), train_Y.to(device)
    test_X, test_Y = test_X.to(device), test_Y.to(device)
    dataSet = TensorDataset(train_X, train_Y)
    dataLoader = DataLoader(dataSet, batch_size=batch_size, shuffle=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(updater, max_lr=lr, steps_per_epoch=len(dataLoader),
                                                    epochs=num_epochs, pct_start=0.3, div_factor=25)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(updater, 0.9)
    for epoch in range(num_epochs):
        temp_loss = []
        net.train()
        for X, y in dataLoader:
            # X, y = X.to(device), y.to(device)
            updater.zero_grad()
            y_hat = net(X)
            l = loss(y_hat.reshape(y.shape), y)
            temp_loss.append(l.detach().cpu().numpy())
            l.backward()
            grad_clipping(net, 5)
            updater.step()
            scheduler.step()
        if epoch == num_epochs-1:
            net.eval()
            with torch.no_grad():
                predict1 = [net(x).detach() for x in lib.data_iter2(test_X, bs=4096)]
                predict1 = torch.concat(predict1, dim=0).reshape(test_Y.shape)
                MAPE1 = lib.t_MAPE(predict1, test_Y).cpu()
                RMSE1 = lib.t_RMSE(predict1, test_Y).cpu()
                R_21 = lib.t_R_square(predict1, test_Y).cpu()
                print('Test  epoch:%s, MAPE:%.4f, RMSE：%.4f, R_2:%.4f'%(epoch, MAPE1, RMSE1, R_21))
            # 绘制曲线
            if wind:
                predict0 = [net(x).detach() for x in lib.data_iter2(train_X, bs=4096)]
                predict0 = torch.concat(predict0, dim=0).reshape(train_Y.shape)
                MAPE0 = lib.t_MAPE(predict0, train_Y).cpu()
                RMSE0 = lib.t_RMSE(predict0, train_Y).cpu()
                R_2 = lib.t_R_square(predict0, train_Y).cpu()
                wind.line([np.mean(temp_loss)], [epoch], win='train', update='append')
                wind.line([[RMSE0, RMSE1]], [epoch], win='test', update='append')
                print('Train epoch:%s, MAPE:%.4f, RMSE：%.4f, R_2:%.4f' % (num_epochs, MAPE0, RMSE0, R_2))
    end = time.time()
    return end-start, [num_epochs, MAPE1.numpy(), RMSE1.numpy(), R_21.numpy()]


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


path = 'Dataset_1_NCA_battery.xlsx'
data = lib.Load_Data(path)                                         # data is a dictionary that stores the characteristics and labels of each cell
cells_list = [[11,12,28,29,30,31,32], list(range(1, 11)) + list(range(13, 22)), [22,23,24,25,26,27,33,34,35],
              [36,37,38], np.arange(39, 67)]
n = [1, 4, 2, 1, 6]

dropout = 0.01
lr, num_epochs = 2e-3, 100
batch_size = 512
total = []
for random_seed in range(20):
    temp = -1
    while temp < 0:
        test_list = [np.random.RandomState(seed=random_seed).choice(a, n[i], replace=False) for i, a in enumerate(cells_list)]
        b = []
        for i in test_list:
            b.extend(i)
        test_list = b
        train_X_dict, train_Y_dict, test_X_dict, test_Y_dict = lib.diviseData(data, test_list)
        train_X, test_X = lib.normalize(train_X_dict, test_X_dict)
        train_Y = torch.concat([data for data in train_Y_dict.values()])
        indices = np.arange(len(train_X))
        np.random.RandomState(seed=random_seed).shuffle(indices)
        train_X, train_Y = train_X[indices], train_Y[indices]
        test_Y = torch.concat([y for y in test_Y_dict.values()], dim=0)
        device = lib.try_gpu()
        net = lib.LSTM_NN(dropout=dropout)
        net = net.to(device)
        weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
        parameters = [{'params': weight_decay_list},
                      {'params': no_decay_list, 'weight_decay': 0}]
        updater = torch.optim.Adam(parameters, lr, weight_decay=1e-5)
        cost_time, Evals = train(net, train_X, train_Y, test_X, test_Y, num_epochs, lr, batch_size, updater, device)
        temp = Evals[-1]
    total.append([random_seed, cost_time, *Evals])

total = pd.DataFrame(total, columns=['序号', '程序运行时间', 'num_epochs', 'MAPE', 'RMSE', 'R_2'])
total.to_csv('LSTM_Results/%s.csv'%'results', index=False, encoding = "utf-8")