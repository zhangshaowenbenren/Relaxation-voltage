# Determine training and test sets on dataset 3,
# using BO to optimize hyperparameters for the training phase of the model

import pandas as pd
import torch
import lib
import time
import visdom
import numpy as np
import optuna
import pickle

# 执行流程
import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

device = lib.try_gpu()

# Given a set of difference hyperparameters, return the RMSE of the model's prediction on the validation set
def optuna_objective(trial):
    k = 4
    batch_size = trial.suggest_int('batch_size', 32, 256, 32)
    lr = trial.suggest_float('lr', 0.00001, 0.01, log=True)
    weight_decay=trial.suggest_float('weight_decay', 0.00001, 0.001, log=True)
    num_epochs = trial.suggest_int('num_epochs', 50, 150, 10)
    dropout = trial.suggest_float('dropout', 0.01, 0.5, log=True)

    rmse_sum = 0
    for i in range(k):
        data = lib.get_k_fold_data(k, i, train_X, train_Y)
        net = lib.CNN_model2(dropout=dropout)
        net = net.to(device)
        weight_decay_list = (param for name, param in net.named_parameters() if
                             name[-4:] != 'bias' and 'bn' not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
        parameters = [{'params': weight_decay_list},
                      {'params': no_decay_list, 'weight_decay': 0}]
        updater = torch.optim.Adam(parameters, lr, weight_decay=weight_decay)

        cost_time, Evals = lib.train(net, *data, num_epochs, lr, batch_size, updater,
                                          device, wind=None)
        rmse_sum += Evals[2]
    return rmse_sum / k

# Define the optimization process
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
# Returns a collection containing voltage and current data for the entire life cycle charging phase of each battery, as well as capacity (label)
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
        if np.min(x) < 3.8:          # Remove outliers
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
    train_X, train_Y = train_X[::10], train_Y[::10]
    indices = np.arange(len(train_X))
    np.random.RandomState(seed=random_seed).shuffle(indices)
    train_X, train_Y = train_X[indices], train_Y[indices]
    test_Y = torch.concat([y for y in test_Y_dict.values()], dim=0)
    save_path = 'ev_res/CNN/CNN_%s'%cell_list[g]

    if need_BO:
        best_params, best_score = optimizer_optuna(BO_epochs, 'GP', optuna_objective)
        print('best_params:%s' % best_params)
        print('best_score%s' % best_score)
        with open('ev_res/CNN/CNN_ev_best_hparams_%s.pkl' % cell_list[g], 'wb') as fp:
            pickle.dump(best_params, fp)
    else:
        with open('ev_res/CNN/CNN_ev_best_hparams_%s.pkl' % cell_list[g], 'rb') as fp:
            best_params = pickle.load(fp)
        # print(best_params)
# ------------------------------------------ model retrained according to optimal hyperparameters -----------------------------------------------
    dropout = best_params['dropout']
    lr, num_epochs = best_params['lr'], best_params['num_epochs']
    batch_size = best_params['batch_size']

    net = lib.CNN_model2(dropout)
    net = net.to(device)
    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0}]
    updater = torch.optim.Adam(parameters, lr, weight_decay=best_params['weight_decay'])
    cost_time, Evals = lib.train(net, train_X, train_Y, test_X, test_Y, num_epochs, lr, batch_size, updater, device, wind=None)
    torch.save(net.state_dict(), save_path)
    start = time.time()
    y = [net(i.reshape(1, 14)) for i in test_X[:1000].to(device)]
    test_res.append([cell_list[g]] + list(Evals[1:]))
    print('测试集上:MAPE:{:.4f}, RMSE为：{:.4f}, R_2:{:.4f}'.format(Evals[1], Evals[2], Evals[3]))
print(test_res)
test_pd = pd.DataFrame(test_res, columns=['test_cell', 'MAPE', 'RMSE', 'R_2'])
test_pd.to_csv('ev_res/CNN/CNN_test_res.csv')