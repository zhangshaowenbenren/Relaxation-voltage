# MK-MMD on dataset 3

import pandas as pd
import torch
import lib
import time
import visdom
import numpy as np
import optuna
import pickle
from torch.utils.data import DataLoader,  TensorDataset

import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

device = lib.try_gpu()

source_data = lib.Load_Data('Dataset_1_NCA_battery.xlsx')
src_X, src_Y = [], []
for key, data in source_data.items():
    step = 5 if 'D2' in key else 20
    src_X.append(torch.tensor(data['V'][::step], dtype=torch.float32))
    src_Y.append(torch.tensor(data['Q'][::step], dtype=torch.float32))
src_X, src_Y = torch.concat(src_X), torch.concat(src_Y)
src_X = (src_X - torch.tensor(4.1)) / (torch.tensor(4.2) - torch.tensor(4.1))
dataSet = TensorDataset(src_X, src_Y)

def optuna_objective(trial):
    k = 4
    rmse_sum = 0
    for i in range(k):
        train0_X, train0_Y, val_X, val_Y = lib.get_k_fold_data(k, i, train_X, train_Y)
        dataSet2 = TensorDataset(train0_X, train0_Y)
        dataSet3 = TensorDataset(val_X, val_Y)

        batch_size = trial.suggest_int('batch_size', 32, 256, 32)
        lr = trial.suggest_float('lr', 0.001, 0.01, log=True)
        num_epochs = trial.suggest_int('num_epochs', 20, 100, 10)
        l2 = trial.suggest_float('LAMBDA1', 0.1, 50, log=True)
        l3 = trial.suggest_float('LAMBDA2', 0.01, 20, log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.00001, 0.001, log=True)
        LAMBDA = [l2, l3]

        data_src = DataLoader(dataSet, batch_size=batch_size, shuffle=True, drop_last=False)
        data_tar = DataLoader(dataSet2, batch_size=batch_size, shuffle=True, drop_last=False)
        data_test = DataLoader(dataSet3, batch_size=batch_size, shuffle=True, drop_last=False)

        net = lib.DaNN(dropout=0.)
        net = net.to(device)
        weight_decay_list = (param for name, param in net.named_parameters() if
                             name[-4:] != 'bias' and 'bn' not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
        parameters = [{'params': weight_decay_list},
                      {'params': no_decay_list, 'weight_decay': 0}]
        updater = torch.optim.Adam(parameters, lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(updater, max_lr=lr, steps_per_epoch=len(data_src),
                                                        epochs=num_epochs, pct_start=0.3, div_factor=25)
        best_Evals = lib.train_mmd2(num_epochs=num_epochs, model=net, optimizer=updater, data_src=data_src,
                                   data_tar=data_tar,
                                   data_test=data_test, device=device, scheduler=scheduler, l=LAMBDA)
        rmse_sum += best_Evals[2]
    return rmse_sum / k


def optimizer_optuna(n_trials, algo, optuna_objective):
    if algo == 'TPE':
        algo = optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24)
    elif algo == 'GP':
        from optuna.integration import SkoptSampler
        algo = SkoptSampler(skopt_kwargs={'base_estimator': 'GP', 'n_initial_points': 10, 'acq_func': 'EI'})

    study = optuna.create_study(sampler=algo,
                                direction='minimize')

    study.optimize(optuna_objective
                   , n_trials=n_trials
                   , show_progress_bar=True
                   )
    return study.best_trial.params, study.best_trial.values


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
    for key, value in test_dict.items():
        test_dict[key] = (value - min0) / (max0 - min0)
    return train_X, test_X


need_BO, BO_epochs = True, 50
with open('dataset/train_sample.pkl', 'rb') as fp:
    sample_x, sample_y = pickle.load(fp)

random_seed = 5
cell_list = ['W5', 'W8', 'W9', 'W10']
# train_list, test_list = ['W5', 'W8', 'W9'], ['W10']

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
    test_Y = torch.concat([y for y in test_Y_dict.values()], dim=0)
    train_X, train_Y = train_X[::10], train_Y[::10]
    indices = np.arange(len(train_X))
    np.random.RandomState(seed=random_seed).shuffle(indices)
    train_X, train_Y = train_X[indices[::5]], train_Y[indices[::5]]

    save_path = 'ev_res/MK-MMD/MK-MMD_%s'%cell_list[g]

    if need_BO:
        best_params, best_score = optimizer_optuna(BO_epochs, 'GP', optuna_objective)
        print('best_params:%s' % best_params)
        print('best_score%s' % best_score)
        with open('ev_res/MK-MMD/best_hparams_%s.pkl' % cell_list[g], 'wb') as fp:
            pickle.dump(best_params, fp)
    else:
        with open('ev_res/MK-MMD/best_hparams_%s.pkl' % cell_list[g], 'rb') as fp:
            best_params = pickle.load(fp)
        print(best_params)
# ------------------------------------------ model retrained according to optimal hyperparameters -----------------------------------------------
    lr, num_epochs = best_params['lr'], best_params['num_epochs']
    batch_size = best_params['batch_size']
    lambdas = [best_params['LAMBDA1'], best_params['LAMBDA2']]
    weight_decay = best_params['weight_decay']

    # Target domain training set, target domain test set, respectively
    data_tar, data_test = lib.getDataLoder(train_X, train_Y, test_X, test_Y, batch_size)
    data_src = DataLoader(dataSet, batch_size=batch_size, shuffle=True, drop_last=False)

    net = lib.DaNN(dropout=0.)
    net = net.to(device)

    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0}]
    updater = torch.optim.Adam(parameters, lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(updater, max_lr=lr, steps_per_epoch=len(data_src),
                                                    epochs=num_epochs, pct_start=0.3, div_factor=25)
    start1 = time.time()
    Evals = lib.train_mmd2(num_epochs=num_epochs, model=net, optimizer=updater, data_src=data_src, data_tar=data_tar,
                          data_test=data_test, device=device, scheduler=scheduler, l=lambdas, wind=None)
    torch.save(net.state_dict(), save_path)
    test_res.append([cell_list[g]] + list(Evals[1:]))
    print('测试集上:MAPE:{:.4f}, RMSE为：{:.4f}, R_2:{:.4f}'.format(Evals[1], Evals[2], Evals[3]))
print(test_res)
test_pd = pd.DataFrame(test_res, columns=['test_cell', 'MAPE', 'RMSE', 'R_2'])
test_pd.to_csv('ev_res/MK-MMD/MK-MMD_test_res.csv')