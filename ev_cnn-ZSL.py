# The model trained on the source domain dataset1 is used directly for the estimation of dataset3
import pandas as pd
import torch
import lib
import numpy as np
import pickle

import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

device = lib.try_gpu()
source_dict = torch.load('model_save/CNN2', map_location=device)

def my_normalize(train_dict, test_dict):
    train_X0, test_X0 = [], []
    for i, data in enumerate(train_dict.values()):
        train_X0.append(data)

    for i, data in enumerate(test_dict.values()):
        test_X0.append(data)
    train_X, test_X = torch.concat(train_X0), torch.concat(test_X0)
    zz = torch.concat([train_X, test_X])
    max0, min0 = torch.max(zz), torch.min(train_X)

    train_X = (train_X - min0) / (max0 - min0)
    test_X = (test_X - min0) / (max0 - min0)
    # 顺便归一化测试集
    for key, value in test_dict.items():
        test_dict[key] = (value - min0) / (max0 - min0)
    return train_X, test_X


with open('dataset/train_sample.pkl', 'rb') as fp:
    sample_x, sample_y = pickle.load(fp)

random_seed = 5
cell_list = ['W5', 'W8', 'W9', 'W10']
# train_list, test_list = ['W5', 'W8', 'W9'], ['W10']
net = lib.CNN_model2(0.0014)
net.load_state_dict(source_dict)
net = net.to(device)
net.eval()

test_res = []
for g in range(4):
    train_list, test_list = cell_list[0:g] + cell_list[g + 1:], [cell_list[g]]
    train_X_dict, train_Y_dict, test_X_dict, test_Y_dict = {}, {}, {}, {}
    for key, data in sample_x.items():
        x, y = data, sample_y[key] / 4.85
        if np.min(x) < 3.8:
            del_indx = []
            for i in range(len(data)):
                if min(data[i]) < 3.8:
                    del_indx.append(i)
            x, y = np.delete(x, del_indx, axis=0), np.delete(y, del_indx, axis=0)
        if key in train_list:
            train_X_dict[key], train_Y_dict[key] = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        elif key in test_list:
            test_X_dict[key], test_Y_dict[key] = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    train_X, test_X = my_normalize(train_X_dict, test_X_dict)
    train_Y = torch.concat([data for data in train_Y_dict.values()])
    train_X, train_Y = train_X[::], train_Y[::]
    indices = np.arange(len(train_X))
    np.random.RandomState(seed=random_seed).shuffle(indices)
    train_X, train_Y = train_X[indices], train_Y[indices]
    test_Y = torch.concat([y for y in test_Y_dict.values()], dim=0)

    predict1 = [net(x.to(device)).detach() for x in lib.data_iter2(test_X)]
    predict1 = torch.concat(predict1, dim=0).cpu()
    MAPE1 = lib.MAPE(predict1, test_Y)
    RMSE1 = lib.RMSE(predict1, test_Y)
    R_2 = lib.R_square(predict1, test_Y)
    test_res.append([cell_list[g]] + [MAPE1, RMSE1, R_2])
print(test_res)
test_pd = pd.DataFrame(test_res, columns=['test_cell', 'MAPE', 'RMSE', 'R_2'])
test_pd.to_csv('ev_res/CNN-ZSL/CNN_test_res.csv')