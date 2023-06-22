## Some helper functions for model training and testing

import mmd
import numpy as np
import torch
from torch import nn
import os
import pandas as pd
import time
from torch.utils.data import DataLoader,  TensorDataset

# read dataset
def Load_Data(data_file_name, dataset=-1):
    dir_name = 'dataset2/'
    res_dict = {}
    for i, ii in enumerate(os.listdir(dir_name)):
        if ii not in data_file_name:
            continue
        path = os.path.join(dir_name, ii)
        data = pd.read_excel(path)
        last_cycle, counts = 0, 0
        temp, lens = [], len(data)
        for j in range(lens):
            c = data.loc[j, 'cycle']                   # cycle number
            if i==2:
                Q = data.loc[j, 'Capacity'] / 2500         # the cycle SOH
            else:
                Q = data.loc[j, 'Capacity'] / 3500
            V = data.loc[j, 'Voltages'].replace('\n', '')[1:-1].split(' ')
            while '' in V:
                V.remove('')
            V = np.array(list(map(float, V)))
            if V[-1] < 4.1 or any(np.diff(V, n=1) > 0.001) or any(np.diff(V, n=2) < -0.001):
                continue
            if c < last_cycle or j==lens-1:            # Iterate to a new cell or end of traversal
                counts += 1
                if i == 2:
                    rate, Tem = data.loc[j - 1, 'D_rate'], data.loc[j - 1, 'Tem']
                else:
                    rate, Tem = data.loc[j - 1, 'rate'], data.loc[j - 1, 'Tem']
                cell_name = 'D' + str(i + 1) + '_' + str(rate) + '_' + str(Tem) + '_' + str(counts)
                res_dict[cell_name] = {'cycle': np.array([i[0] for i in temp]), 'V': np.stack([i[1] for i in temp]),
                                       'Q': np.array([i[2] for i in temp])}
                temp = []
            last_cycle = c

            if dataset == 3:
                temp.append([c, V[3::4], Q])
            else:
                temp.append([c, V, Q])
    return res_dict


def getDataLoder(train_X, train_Y, test_X, test_Y, batch_size):
    dataSet2 = TensorDataset(train_X, train_Y)
    data_tar = DataLoader(dataSet2, batch_size=batch_size, shuffle=True, drop_last=False)
    dataSet3 = TensorDataset(test_X, test_Y)
    data_tar2 = DataLoader(dataSet3, batch_size=batch_size, shuffle=True, drop_last=False)
    return data_tar, data_tar2

# train NN
def train(net, train_X, train_Y, test_X, test_Y, num_epochs, lr, batch_size, updater, device, wind=None):
    start = time.time()
    loss = nn.MSELoss()
    train_X, train_Y = train_X.to(device), train_Y.to(device)
    test_X, test_Y = test_X.to(device), test_Y.to(device)
    dataSet = TensorDataset(train_X, train_Y)
    dataLoader = DataLoader(dataSet, batch_size=batch_size, shuffle=True, )
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
            updater.step()
            scheduler.step()
        if epoch == num_epochs - 1:
            print('--' * 20)
            net.eval()
            with torch.no_grad():
                predict1 = [net(x).detach() for x in data_iter2(test_X, bs=4096)]
                predict1 = torch.concat(predict1, dim=0).reshape(test_Y.shape)
                MAPE1 = t_MAPE(predict1, test_Y).cpu()
                RMSE1 = t_RMSE(predict1, test_Y).cpu()
                R_21 = t_R_square(predict1, test_Y).cpu()
                print('Test  epoch:%s, MAPE:%.4f, RMSE：%.4f, R_2:%.4f'%(epoch, MAPE1, RMSE1, R_21))

            if wind:
                predict0 = [net(x).detach() for x in data_iter2(train_X, bs=4096)]
                predict0 = torch.concat(predict0, dim=0).reshape(train_Y.shape)
                MAPE0 = t_MAPE(predict0, train_Y).cpu()
                RMSE0 = t_RMSE(predict0, train_Y).cpu()
                R_2 = t_R_square(predict0, train_Y).cpu()
                wind.line([np.mean(temp_loss)], [epoch], win='train', update='append')
                wind.line([[RMSE0, RMSE1]], [epoch], win='test', update='append')
                print('Train epoch:%s, MAPE:%.4f, RMSE：%.4f, R_2:%.4f' % (epoch, MAPE0, RMSE0, R_2))
    end = time.time()
    return end-start, [num_epochs, MAPE1.numpy(), RMSE1.numpy(), R_21.numpy()]

# train CNN-FT
def train_FT(net, train_X, train_Y, test_X, test_Y, num_epochs, lr, batch_size, updater, device, source_param, lambda0=0.05, wind=None):
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
            loss_L2 = cur_L2(source_param, net.state_dict())
            l = loss(y_hat.reshape(y.shape), y) + loss_L2 * lambda0
            temp_loss.append(l.detach().cpu().numpy())
            l.backward()
            updater.step()
            scheduler.step()

        if epoch == num_epochs - 1:
            print('--' * 20)
            net.eval()
            with torch.no_grad():
                predict1 = [net(x).detach() for x in data_iter2(test_X, bs=4096)]
                predict1 = torch.concat(predict1, dim=0).reshape(test_Y.shape)
                MAPE1 = t_MAPE(predict1, test_Y).cpu()
                RMSE1 = t_RMSE(predict1, test_Y).cpu()
                R_21 = t_R_square(predict1, test_Y).cpu()
                print('Test  epoch:%s, MAPE:%.4f, RMSE：%.4f, R_2:%.4f'%(epoch, MAPE1, RMSE1, R_21))
            # 绘制曲线
            if wind:
                predict0 = [net(x).detach() for x in data_iter2(train_X, bs=4096)]
                predict0 = torch.concat(predict0, dim=0).reshape(train_Y.shape)
                MAPE0 = t_MAPE(predict0, train_Y).cpu()
                RMSE0 = t_RMSE(predict0, train_Y).cpu()
                R_2 = t_R_square(predict0, train_Y).cpu()
                wind.line([np.mean(temp_loss)], [epoch], win='train', update='append')
                wind.line([[RMSE0, RMSE1]], [epoch], win='test', update='append')
                print('Train epoch:%s, MAPE:%.4f, RMSE：%.4f, R_2:%.4f' % (epoch, MAPE0, RMSE0, R_2))
    end = time.time()
    return end-start, [num_epochs, MAPE1.numpy(), RMSE1.numpy(), R_21.numpy()]


def cur_L2(A, B):
    res = 0
    for key in A.keys():
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            continue
        x, y = A[key], B[key]
        z = (x - y) ** 2
        res += torch.sum(z)
    return res

# MK-MMD
def train_mmd2(num_epochs, model, optimizer, data_src, data_tar, data_test, device, scheduler, l, wind=None):
    MMD = mmd.MMD_loss(kernel_type='rbf', kernel_mul=2.0, kernel_num=5)
    l2, l3 = l
    for epoch in range(num_epochs):
        sum_loss_a, sum_loss_b, sum_loss_c = 0, 0, 0
        criterion = nn.MSELoss()
        batch_j = 0
        list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))
        for batch_id, (data, target) in enumerate(data_src):
            _, (x_tar, y_target) = list_tar[batch_j]
            data, target = data.to(device), target.to(device)
            x_tar, y_target = x_tar.to(device), y_target.to(device)
            model.train()
            y_src, y_tar, x_src_mmd, x_tar_mmd, x_src_mmd2, x_tar_mmd2 = model(data, x_tar)

            loss_src = criterion(y_src, target)
            loss_tar = criterion(y_tar, y_target)
            loss_mmd = MMD(x_src_mmd, x_tar_mmd)    # Mk-MMD
            loss = loss_src + loss_tar * l2 + loss_mmd * l3
            sum_loss_a += loss_src
            sum_loss_b += loss_tar * l2
            sum_loss_c += loss_mmd * l3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_j += 1
            if batch_j >= len(list_tar):
                batch_j = 0
        sum_loss_a, sum_loss_b = sum_loss_a / len(data_src), sum_loss_b / len(data_src)
        sum_loss_c = sum_loss_c / len(data_src)

        if epoch == num_epochs - 1:
            MAPE0, RMSE0, R_2_0 = test_mmd(model, data_src, device=device)
            MAPE1, RMSE1, R_2_1 = test_mmd(model, data_tar, device=device)
            MAPE2, RMSE2, R_2_2 = test_mmd(model, data_test, device=device)
            # best_model_param = net.state_dict()
            if wind:
                print('---------------epoch:%s---------------' % epoch)
                print('Source       , MAPE:%.4f, RMSE：%.4f, R_2:%.4f' % (MAPE0, RMSE0, R_2_0))
                print('Target_Train , MAPE:%.4f, RMSE：%.4f, R_2:%.4f' % (MAPE1, RMSE1, R_2_1))
                print('Target_Test  , MAPE:%.4f, RMSE：%.4f, R_2:%.4f' % (MAPE2, RMSE2, R_2_2))
                wind.line([[sum_loss_a.detach().cpu(), sum_loss_b.detach().cpu(),
                            sum_loss_c.detach().cpu()]], [epoch], win='train', update='append')
                wind.line([[RMSE0, RMSE1, RMSE2]], [epoch], win='test', update='append')
    return [epoch, MAPE2.numpy(), RMSE2.numpy(), R_2_2.numpy()]


# Evaluation of model predictions on data_tar
def test_mmd(model, data_tar, device):
    model.eval()
    with torch.no_grad():
        pred_y, true_y  = [], []
        for data, target in data_tar:
            data, target = data.to(device), target.to(device)
            ypred = model(data, data)
            pred_y.append(ypred.detach())
            true_y.append(target)
        pred_y = torch.concat(pred_y, dim=0)
        true_y = torch.concat(true_y, dim=0)
        MAPE1 = t_MAPE(pred_y, true_y)                          # 直接求所有测试样本的评估指标
        RMSE1 = t_RMSE(pred_y, true_y)
        R_2 =  t_R_square(pred_y, true_y)
    return MAPE1.cpu(), RMSE1.cpu(), R_2.cpu()


# Dividing the training set and test set by cells
def diviseData(data_dict, test_list=None, train_list=None):
    if train_list:
        l0 = train_list
        is_train = True
    else:
        l0 = test_list
        is_train = False
    train_X, train_Y, test_X, test_Y = {}, {}, {}, {}
    for i, (key, data) in enumerate(data_dict.items()):
        i = int(key.split('_')[-1])
        if (i in l0) ^ is_train:
            test_X[key], test_Y[key] = torch.tensor(data['V'], dtype=torch.float32), torch.tensor(data['Q'], dtype=torch.float32)
        else:
            train_X[key], train_Y[key] = torch.tensor(data['V'], dtype=torch.float32), torch.tensor(data['Q'], dtype=torch.float32)

    return train_X, train_Y, test_X, test_Y


def diviseData1(data_dict, train_list=None, test_list=None):
    if train_list:
        l0 = train_list
        is_train = True
    else:
        l0 = test_list
        is_train = False
    train_X, train_Y, test_X, test_Y = {}, {}, {}, {}
    for i, (key, data) in enumerate(data_dict.items()):
        i = int(key.split('_')[-1])
        if (i in l0) ^ is_train:             # 异或操作
            test_X[key], test_Y[key] = data['V'], data['Q']
        else:                             # 测试集
            train_X[key], train_Y[key] = data['V'], data['Q']
    return train_X, train_Y, test_X, test_Y


def normalize2(train_dict, test_dict):
    train_X0, test_X0 = [], []
    for i, data in enumerate(train_dict.values()):
        train_X0.append(data)

    for i, data in enumerate(test_dict.values()):
        test_X0.append(data)

    train_X, test_X = np.concatenate(train_X0), np.concatenate(test_X0)
    total = np.concatenate((train_X, test_X))
    max0 = np.max(total, axis=0)
    min0 = np.min(total, axis=0)

    train_X = (train_X - min0) / (max0 - min0)
    test_X = (test_X - min0) / (max0 - min0)

    for key, value in test_dict.items():
        test_dict[key] = (value - min0) / (max0 - min0)
    return train_X, test_X



def normalize(train_dict, test_dict):
    train_X0, test_X0 = [], []
    for i, data in enumerate(train_dict.values()):
        train_X0.append(data)

    for i, data in enumerate(test_dict.values()):
        test_X0.append(data)
    train_X, test_X = torch.concat(train_X0), torch.concat(test_X0)

    max0, min0 = torch.tensor(4.2), torch.tensor(4.1)

    train_X = (train_X - min0) / (max0 - min0)
    test_X = (test_X - min0) / (max0 - min0)

    for key, value in test_dict.items():
        test_dict[key] = (value - min0) / (max0 - min0)
    return train_X, test_X


def ML_fit_predict(model, train_X, train_Y, test_X, test_Y):
    t0 = time.time()
    model.fit(train_X, train_Y)
    fit_time = time.time() - t0
    t0 = time.time()
    y_train_predict = model.predict(train_X)
    train_preict_time = time.time() - t0
    t0 = time.time()
    y_test_predict = [model.predict(x) for x in data_iter2(test_X)]
    test_preict_time = time.time() - t0

    train_RMSE = RMSE(y_train_predict, train_Y)
    train_MAPE = MAPE(y_train_predict, train_Y)

    y_test_predict = np.concatenate(y_test_predict, axis=0)
    test_MAPE = MAPE(y_test_predict, test_Y)
    test_RMSE = RMSE(y_test_predict, test_Y)
    R2 = R_square(y_test_predict, test_Y)
    return model, [train_MAPE, train_RMSE, test_MAPE, test_RMSE, R2], [fit_time, train_preict_time, test_preict_time]


def myPrint(str, evals, times):
    print('--' * 15)
    print('%s在训练集上MAPE结果为:%.4f, RMSE为：%.4f' % (str, evals[0], evals[1]))
    print('%s在测试集上MAPE结果为:%.4f, RMSE为：%.4f, R_2为：%.4f' % (str, evals[2], evals[3], evals[4]))
    print('%s训练时间为%.1f' % (str, times[0]))
    print('%s训练集预测时间为：%.1f' % (str, times[1]))
    print('%s测试集预测时间为：%.1f' % (str, times[2]))


def getHumanFeature(data):
    def getFea(x0):
        x = x0.T
        Mean_val = np.mean(x, axis=0)
        max0 = np.max(x, axis=0)
        Var = np.sum((x - Mean_val) ** 2, axis=0) / (x.shape[0] - 1)
        Crooked = np.mean(((x - Mean_val) / np.sqrt(Var)) ** 3, axis=0)
        return np.stack([max0, Var, Crooked]).T


    for key, value in data.items():
        v = value['V']
        x = getFea(v)
        value['V'] = x


# MMD-based domain adaptive transfer learning (rather than domain adversarial transfer learning)
class DaNN(nn.Module):
    def __init__(self, dropout=0.05, **kwargs):
        super(DaNN, self).__init__(**kwargs)
        self.CNN1 = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(16),
                                  nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2, stride=2), nn.BatchNorm1d(32),
                                  nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2, stride=2), nn.BatchNorm1d(64),
                                  nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64))
        self.layer0 = nn.Flatten()
        self.Linear1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(3 * 64, 32), nn.Dropout(dropout), nn.ReLU())
        self.Linear2 = nn.Linear(32, 1)


    def forward(self, src, tar):
        # inputs.shape: (batch_size * step)
        if self.training:
            x_src, x_tar = torch.unsqueeze(src, dim=1), torch.unsqueeze(tar, dim=1)     #（batch_size， channel， step）
            x_src, x_tar = self.CNN1(x_src), self.CNN1(x_tar)
            x_src, x_tar = self.layer0(x_src), self.layer0(x_tar)
            x_src_mmd, x_tar_mmd = self.Linear1(x_src), self.Linear1(x_tar)
            y_src = self.Linear2(x_src_mmd)
            y_tar = self.Linear2(x_tar_mmd)
            return torch.squeeze(y_src), torch.squeeze(y_tar), x_src_mmd, x_tar_mmd, x_src, x_tar
        else:
            x_src = torch.unsqueeze(src, dim=1)
            x_src = self.CNN1(x_src)
            x_src = self.layer0(x_src)
            x_src_mmd = self.Linear1(x_src)
            y_src = self.Linear2(x_src_mmd)
            return torch.squeeze(y_src)


class CNN_model2(nn.Module):
    def __init__(self, dropout=0.001, **kwargs):
        super(CNN_model2, self).__init__(**kwargs)
        self.CNN1 = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(16),
                                  nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2, stride=2), nn.BatchNorm1d(32),
                                  nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2, stride=2), nn.BatchNorm1d(64),
                                  nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64))
        self.layer0 = nn.Flatten()
        self.Linear1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(3 * 64, 32), nn.ReLU())
        self.Linear2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, inputs):

        y = torch.unsqueeze(inputs, dim=1)  #（batch_size， channel,  step）
        y1 = self.CNN1(y)
        y1 = self.layer0(y1)
        y = self.Linear1(y1)
        y = self.Linear2(y)
        return torch.squeeze(y)


class LSTM_NN(nn.Module):
    def __init__(self, rnn_hidden=32, dropout=0.1):
        super(LSTM_NN, self).__init__()
        self.LSTM = nn.LSTM(1, rnn_hidden, bidirectional=True, num_layers=2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.Linear1 = nn.Sequential(nn.Linear(rnn_hidden * 2, 1), nn.ReLU())
        self.l = 14
        self.Linear2 = nn.Sequential(nn.Linear(self.l, 1))


    def forward(self, inputs):
        inputs = inputs.unsqueeze(dim=2)
        y1, _ = self.LSTM(inputs)
        y1 = self.Linear1(y1).squeeze()
        y1 = self.Linear2(y1)
        return torch.squeeze(y1)


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().
    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def data_iter2(X, bs=1024):
    lens = len(X)
    l = X.shape[0]
    for i in range((lens-1) // bs + 1):
        train_x = X[i * bs:min((i+1) * bs, l), :]
        yield train_x


def RMSE(hat, value):
    hat, value = np.array(hat), np.array(value)
    res = np.dot(hat - value, hat - value)
    return np.sqrt(res / len(value))


def MAPE(hat, value):
    hat, value = np.array(hat), np.array(value)
    temp = np.abs((hat - value) / (value))
    temp1 = np.sum(temp)
    temp2 = len(value)
    return temp1 / temp2


def MAE(hat, value):
    hat, value = np.array(hat), np.array(value)
    temp = np.abs(hat - value)
    return np.sum(temp) / len(value)


def R_square(hat, value):
    hat, value = np.array(hat), np.array(value)
    m = np.mean(value)
    a1 = np.sum((hat - value) * (hat - value))
    a2 = np.sum((value - m) * (value - m))
    return 1 - a1 / a2


def t_RMSE(hat, value):
    res = torch.sum((hat - value) * (hat - value))
    return torch.sqrt(res / len(value))


def t_MAPE(hat, value):
    temp = torch.abs((hat - value) / (value))
    temp1 = torch.sum(temp)
    temp2 = len(value)
    return temp1 / temp2


def t_MAE(hat, value):
    temp = torch.abs(hat - value)
    return torch.sum(temp) / len(value)


def t_R_square(hat, value):
    m = torch.mean(value)
    a1 = torch.sum((hat - value) * (hat - value))
    a2 = torch.sum((value - m) * (value - m))
    return 1 - a1 / a2


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid