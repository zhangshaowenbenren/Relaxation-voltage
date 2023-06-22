# CNN-FT on dataset3

import pandas as pd
import torch
import lib
import numpy as np
import optuna
import pickle
# 执行流程
import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

device = lib.try_gpu()
source_dict = torch.load('model_save/CNN2', map_location=device)

def optuna_objective(trial):
    k = 4
    batch_size = trial.suggest_int('batch_size', 32, 256, step=32)
    lr = trial.suggest_float('lr', 0.00001, 0.01, log=True)
    num_epochs = trial.suggest_int('num_epochs', 50, 150, step=10)
    num_layers = trial.suggest_categorical('num_layers', [0, 3, 7, 11, 14])
    lambda0 = trial.suggest_float('lambda0', 0.001, 0.1, log=True)

    rmse_sum = 0
    for i in range(k):
        data = lib.get_k_fold_data(k, i, train_X, train_Y)
        net = lib.CNN_model2(dropout=0.0014)
        net = net.to(device)
        net.load_state_dict(source_dict)
        for name, value in net.named_parameters():
            if 'CNN1' in name and int(name.split('.')[1]) < num_layers:
                value.requires_grad = False
            else:
                break

        weight_decay_list = (param for name, param in net.named_parameters() if
                             name[-4:] != 'bias' and 'bn' not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
        weight_decay_list = filter(lambda p: p.requires_grad, weight_decay_list)
        no_decay_list = filter(lambda p: p.requires_grad, no_decay_list)
        parameters = [{'params': weight_decay_list},
                      {'params': no_decay_list, 'weight_decay': 0}]
        updater = torch.optim.Adam(parameters, lr, weight_decay=0)

        cost_time, best_Evals = lib.train_FT(net, *data, num_epochs, lr, batch_size, updater,
                                             device, source_param=source_dict, lambda0=lambda0, wind=None)
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

    study.optimize(optuna_objective     # 目标函数
                   , n_trials=n_trials  # 设定最大迭代次数（包括最初观测值）
                   , show_progress_bar=True  # 要不要展示进度条
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


need_BO, BO_epochs = False, 50
# # 返回一个集合，包含每个电池的整个生命周期充电阶段的电压、电流数据，以及容量（标签）
with open('dataset/train_sample.pkl', 'rb') as fp:
    sample_x, sample_y = pickle.load(fp)

random_seed = 5
cell_list = ['W5', 'W8', 'W9', 'W10']
# train_list, test_list = ['W5', 'W8', 'W9'], ['W10']

test_res = []
for g in range(4):
    if g != 1:
        continue
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
            train_X_dict[key], train_Y_dict[key] = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)  # 训练集
        elif key in test_list:
            test_X_dict[key], test_Y_dict[key] = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    train_X, test_X = my_normalize(train_X_dict, test_X_dict)
    train_Y = torch.concat([data for data in train_Y_dict.values()])
    print(len(train_X) + len(test_X))
    train_X, train_Y = train_X[::10], train_Y[::10]
    print(len(train_X))
    indices = np.arange(len(train_X))
    np.random.RandomState(seed=random_seed).shuffle(indices)
    train_X, train_Y = train_X[indices], train_Y[indices]
    test_Y = torch.concat([y for y in test_Y_dict.values()], dim=0)
    save_path = 'ev_res/CNN-FT/CNN_FT_%s'%cell_list[g]

    if need_BO:
        best_params, best_score = optimizer_optuna(BO_epochs, 'GP', optuna_objective)
        print('best_params:%s' % best_params)
        print('best_score%s' % best_score)
        with open('ev_res/CNN-FT/CNN_ev_best_hparams_%s.pkl' % cell_list[g], 'wb') as fp:
            pickle.dump(best_params, fp)
    else:
        with open('ev_res/CNN-FT/CNN_ev_best_hparams_%s.pkl' % cell_list[g], 'rb') as fp:
            best_params = pickle.load(fp)
        print(best_params)
# ------------------------------------------ model retrained according to optimal hyperparameters -----------------------------------------------
    lambda0 = best_params['lambda0']
    num_layers = best_params['num_layers']
    lr, num_epochs = best_params['lr'], best_params['num_epochs']
    batch_size = best_params['batch_size']

    net = lib.CNN_model2(0.0014)
    net = net.to(device)
    net.load_state_dict(source_dict)
    for name, value in net.named_parameters():
        if 'CNN1' in name and int(name.split('.')[1]) < num_layers:
            value.requires_grad = False
        else:
            break

    weight_decay_list = (param for name, param in net.named_parameters() if
                         name[-4:] != 'bias' and 'bn' not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
    weight_decay_list = filter(lambda p: p.requires_grad, weight_decay_list)
    no_decay_list = filter(lambda p: p.requires_grad, no_decay_list)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0}]
    updater = torch.optim.Adam(parameters, lr, weight_decay=0)

    cost_time, Evals = lib.train_FT(net, train_X, train_Y, test_X, test_Y, num_epochs, lr, batch_size, updater, device, source_dict, lambda0=lambda0, wind=None)
    torch.save(net.state_dict(), save_path)

    test_res.append([cell_list[g]] + list(Evals[1:]))
    print('测试集上:MAPE:{:.4f}, RMSE为：{:.4f}, R_2:{:.4f}'.format(Evals[1], Evals[2], Evals[3]))
print(test_res)
test_pd = pd.DataFrame(test_res, columns=['test_cell', 'MAPE', 'RMSE', 'R_2'])
test_pd.to_csv('ev_res/CNN-FT/CNN_FT_test_res.csv')

# fig, axes = plt.subplots(1, 1, figsize=(5, 4))
# # axes = axes.flat
# net.load_state_dict(torch.load(save_path))
# net.eval()
# predict_dict = {}
# scatter_X, scatter_Y = [], []
# print('--' * 20)
# with torch.no_grad():
#     for i, key in enumerate(test_X_dict.keys()):
#         x = test_X_dict[key]
#         temp = net(x.to(device)).detach().cpu()
#         scatter_X.extend(temp.numpy())
#         scatter_Y.extend(test_Y_dict[key].numpy())
#         b_mape = lib.MAPE(temp, test_Y_dict[key])
#         b_rmse = lib.RMSE(temp, test_Y_dict[key])
#         print('电池%s的容量预测的MAPE为：%.4f，RMSE为:%.4f'%(key, b_mape, b_rmse))
#         predict_dict[key] = temp.numpy()
#         cycle = np.arange(len(test_Y_dict[key]))
#         axes.plot(cycle, test_Y_dict[key], linewidth=1, zorder=1, marker='+', markersize=1, label='actual')
#         axes.plot(cycle, predict_dict[key], linewidth=1, zorder=1, marker='_', markersize=1, label='predict')
#         axes.set_title(key)
#         axes.set_ylabel('Capacity')
#         axes.set_xlabel('Cycle')
#     axes.legend()
#
# plt.figure()
# scatter_X, scatter_Y = np.array(scatter_X), np.array(scatter_Y)
# plt.scatter(scatter_X, scatter_Y, c=abs(scatter_X - scatter_Y), cmap="seismic", zorder=2, s=3)
# plt.xlim([0.7, 1])
# plt.ylim([0.7, 1])
# plt.plot([0.7, 1], [0.7, 1], '--', zorder=1)
# plt.colorbar()
# plt.show()