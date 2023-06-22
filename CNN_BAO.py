# 确定训练集和测试集，采用BO优化模型训练阶段的超参数
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import lib
import time
import visdom
import numpy as np
import pandas as pd
from tqdm import tqdm
import optuna
import pickle

# Given a set of hyperparameters, return the RMSE of the model's prediction on the validation set
def optuna_objective(trial):
    batch_size = trial.suggest_int('batch_size', 32, 512, 32)
    lr = trial.suggest_float('lr', 0.00001, 0.05, log=True)
    weight_decay=trial.suggest_float('weight_decay', 0.000001, 0.001, log=True)
    num_epochs = trial.suggest_int('num_epochs', 50, 150, 10)
    dropout = trial.suggest_float('dropout', 0.001, 0.5, log=True)

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

        cost_time, best_Evals = lib.train(net, *data, num_epochs, lr, batch_size, updater,
                                          device, wind=None)
        rmse_sum += best_Evals[2]
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


need_BO, BO_epochs = False, 50
k = 4
# Returns a collection containing voltage and current data for the entire life cycle charging phase of each battery, as well as SOH (label)
path = 'Dataset_1_NCA_battery.xlsx'
data = lib.Load_Data(path)
cells_list = [[11,12,28,29,30,31,32], list(range(1, 11)) + list(range(13, 22)), [22,23,24,25,26,27,33,34,35],
              [36,37,38], np.arange(39, 67)]
n = [1, 4, 2, 1, 6]               # stratified samples

random_seed = 5
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
save_path = 'CNN'

# 执行流程
import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

if need_BO:
    best_params, best_score = optimizer_optuna(BO_epochs, 'GP', optuna_objective)
    print('best_params:%s' % best_params)
    print('best_score%s' % best_score)
    with open('CNN_best_params.pkl', 'wb') as fp:
        pickle.dump(best_params, fp)
else:
    with open('Mult_Experiments_Results/CNN_best_params_5.pkl', 'rb') as fp:
        best_params = pickle.load(fp)
# ------------------------------------------ model retrained according to optimal hyperparameters -----------------------------------------------
wind = visdom.Visdom(env='main')
wind.line([0.], [0.], win='train', opts=dict(title='epoch_loss', legend=['train_loss']))
wind.line([[0., 0.]], [0.], win='test', opts=dict(title='error', legend=['train_RMSE', 'test_RMSE']))

dropout = best_params['dropout']
lr, num_epochs = best_params['lr'], best_params['num_epochs']
batch_size = best_params['batch_size']

res = []
for i in range(10):
    net = lib.CNN_model2(dropout)
    net = net.to(device)
    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0}]
    updater = torch.optim.Adam(parameters, lr, weight_decay=best_params['weight_decay'])
    cost_time, Evals = lib.train(net, train_X, train_Y, test_X, test_Y, num_epochs, lr, batch_size, updater, device, wind=wind)
    start = time.time()
    y = [net(i.reshape(1, 14)) for i in test_X[:1000].to(device)]    # not batch input
    end = time.time()
    res.append([cost_time, (end - start) / len(test_X[:1000])])

res = np.array(res)
print('CNN的训练时间:%s'% np.mean(res[:, 0]))
print('平均推断时间：%s' % np.mean(res[:, 1]))
# torch.save(net.state_dict(), 'model_save/%s'% (save_path))

# print('模型平均计算时间为：{:.4f} s'.format(cost_time))
# print('测试集上:MAPE:{:.4f}, RMSE为：{:.4f}, R_2:{:.4f}'.format(Evals[1], Evals[2], Evals[3]))
#
# fig, axes = plt.subplots(3, 4, figsize=(15, 9))
# axes = axes.flat
# net.load_state_dict(torch.load('model_save/%s'%save_path))
# net.eval()
# predict_dict = {}
# scatter_X, scatter_Y = [], []
# print('--' * 20)
# with torch.no_grad():
#     key_list = list(test_X_dict.keys())
#     key_list = np.random.choice(key_list, len(axes), replace=False)
#     for i, key in enumerate(key_list):
#         x = test_X_dict[key]
#         temp = net(x.to(device)).detach().cpu()
#         scatter_X.extend(temp.numpy())
#         scatter_Y.extend(test_Y_dict[key].numpy())
#         b_mape = lib.MAPE(temp, test_Y_dict[key])
#         b_rmse = lib.RMSE(temp, test_Y_dict[key])
#         print('电池%s的容量预测的MAPE为：%.4f，RMSE为:%.4f'%(key, b_mape, b_rmse))
#         predict_dict[key] = temp.numpy()
#         cycle = np.arange(len(test_Y_dict[key]))
#         axes[i].plot(cycle, test_Y_dict[key], linewidth=1, zorder=1, marker='+', markersize=1, label='actual')
#         axes[i].plot(cycle, predict_dict[key], linewidth=1, zorder=1, marker='_', markersize=1, label='predict')
#         axes[i].set_title(key)
#         if i % 4 == 0:
#             axes[i].set_ylabel('Capacity')
#         if i >= 8:
#             axes[i].set_xlabel('Cycle')
#         axes[i].legend()
#
# plt.figure()
# scatter_X, scatter_Y = np.array(scatter_X), np.array(scatter_Y)
# plt.scatter(scatter_X, scatter_Y, c=abs(scatter_X - scatter_Y), cmap="seismic", zorder=2, s=3)
# plt.xlim([0.7, 1])
# plt.ylim([0.7, 1])
# plt.plot([0.7, 1], [0.7, 1], '--', zorder=1)
# plt.colorbar()
# plt.show()