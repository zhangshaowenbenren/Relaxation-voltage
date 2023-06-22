# CNN-FT on dataset2
import pickle
import matplotlib.pyplot as plt
import optuna
import torch
import lib
import time
import visdom
import numpy as np

from optuna.integration import SkoptSampler

need_BO, BO_epochs = False, 100
save_path = 'TL_Dataset2'
random_seed = 10
need_TL, need_visual = True, True
path = 'Dataset_2_NCM_battery.xlsx'
# Returns a collection containing voltage and current data for the entire life cycle charging phase of each battery, as well as SOH (label)
data = lib.Load_Data(path)
cells_list = [np.arange(1, 24), np.arange(24, 28), np.arange(28, 56)]
n, step = [6, 1, 3], 10              # Number of stratified samples
train_list = [np.random.RandomState(seed=random_seed).choice(a, n[i], replace=False) for i, a in enumerate(cells_list)]
b = []
for i in train_list:
    b.extend(i)
train_list = b

train_X_dict, train_Y_dict, test_X_dict, test_Y_dict = lib.diviseData(data, train_list=train_list)
train_X, test_X = lib.normalize(train_X_dict, test_X_dict)
train_Y = torch.concat([data for i, data in enumerate(train_Y_dict.values())])
test_Y = torch.concat([y for y in test_Y_dict.values()], dim=0)
print('总样本:%s'%(len(train_X) + len(test_X)))

indices = np.arange(len(train_X))
np.random.RandomState(seed=random_seed).shuffle(indices)
train_X, train_Y = train_X[indices[::step]], train_Y[indices[::step]]       # Sparse sampling
print('训练样本:%s'%(len(train_X)))

indices = np.arange(len(train_X))
np.random.RandomState(seed=random_seed).shuffle(indices)
train_X, train_Y = train_X[indices], train_Y[indices]
device = lib.try_gpu()
temp = torch.load('model_save/CNN2')

# Given a set of hyperparameters, return the RMSE of the model's prediction on the validation set
def optuna_objective(trial):
    k = 4
    batch_size = trial.suggest_int('batch_size', 48, 192, step=16)
    lr = trial.suggest_float('lr', 0.0001, 0.02, log=True)
    num_epochs = trial.suggest_int('num_epochs', 40, 200, step=10)
    num_layers = trial.suggest_categorical('num_layers', [0, 3, 7, 11, 14])
    lambda0 = trial.suggest_float('lambda0', 0.00001, 0.001, log=True)
    rmse_sum = 0
    for i in range(k):
        data = lib.get_k_fold_data(k, i, train_X, train_Y)
        net = lib.CNN_model2(dropout=0.001)
        net = net.to(device)
        net.load_state_dict(temp)
        for name, value in net.named_parameters():
            if 'CNN1' in name and int(name.split('.')[1]) < num_layers:
                value.requires_grad = False
            else:break

        weight_decay_list = (param for name, param in net.named_parameters() if
                             name[-4:] != 'bias' and 'bn' not in name)
        no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
        weight_decay_list = filter(lambda p: p.requires_grad, weight_decay_list)
        no_decay_list = filter(lambda p: p.requires_grad, no_decay_list)
        parameters = [{'params': weight_decay_list},
                      {'params': no_decay_list, 'weight_decay': 0}]
        updater = torch.optim.Adam(parameters, lr, weight_decay=0)

        cost_time, best_Evals = lib.train_FT(net, *data, num_epochs, lr, batch_size, updater,
                                          device, source_param=temp, lambda0=lambda0, wind=None)
        rmse_sum += best_Evals[2]
    return rmse_sum / k


# Define the optimization process
def optimizer_optuna(n_trials, algo, optuna_objective):
    if algo == 'TPE':
        algo = optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24)
    elif algo == 'GP':
        algo = SkoptSampler(skopt_kwargs={'base_estimator': 'GP', 'n_initial_points': 10, 'acq_func': 'EI'})

    study = optuna.create_study(sampler=algo,
                                direction='minimize')

    study.optimize(optuna_objective
                   , n_trials=n_trials
                   , show_progress_bar=True
                   )
    return study.best_trial.params, study.best_trial.values


import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

if need_BO:
    best_params, best_score = optimizer_optuna(BO_epochs, 'GP', optuna_objective)
    print('best_params:%s' % best_params)
    print('best_score%s' % best_score)
    with open('TL_Dataset2_best_params.pkl', 'wb') as fp:
        pickle.dump(best_params, fp)
else:
    with open('Mult_Experiments_Results_TL_Dataset2/best_params_%s.pkl' % random_seed, 'rb') as fp:
        best_params = pickle.load(fp)
    print(best_params)
# ------------------------------------------ model retrained according to optimal hyperparameters -----------------------------------------------
wind = visdom.Visdom(env='main')
wind.line([0.], [0.], win='train', opts=dict(title='epoch_loss', legend=['train_loss']))
wind.line([[0., 0.]], [0.], win='test', opts=dict(title='error', legend=['train_RMSE', 'test_RMSE']))
net = lib.CNN_model2(dropout=0.001)
net = net.to(device)
net.load_state_dict(torch.load('model_save/CNN2'))
net.eval()
params = sum([v.numel() for k,v in net.state_dict().items()])
print('CNN的参数数量：%s'%params)

predict1 = [net(x.to(device)).detach() for x in lib.data_iter2(test_X)]
predict1 = torch.concat(predict1, dim=0).cpu()
MAPE1 = lib.MAPE(predict1, test_Y)
RMSE1 = lib.RMSE(predict1, test_Y)
R_2 = lib.R_square(predict1, test_Y)
print('ZSL, MAPE:%.4f, RMSE：%.4f, R_2:%.4f' % (MAPE1, RMSE1, R_2))

dropout = 0.001
lambda0 = best_params['lambda0']
num_layers = best_params['num_layers']
lr, num_epochs = best_params['lr'], best_params['num_epochs']
batch_size = best_params['batch_size']
# 模型微调
res = []
net = lib.CNN_model2(dropout)
net = net.to(device)
net.load_state_dict(torch.load('model_save/CNN2'))

for name, value in net.named_parameters():
    if 'CNN1' in name and int(name.split('.')[1]) < num_layers:
        value.requires_grad = False
weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
weight_decay_list = filter(lambda p: p.requires_grad, weight_decay_list)
no_decay_list = filter(lambda p: p.requires_grad, no_decay_list)
parameters = [{'params': weight_decay_list},
              {'params': no_decay_list, 'weight_decay': 0}]
updater = torch.optim.Adam(parameters, lr, weight_decay=0)

sums = 0
for i in range(1):
    cost_time, Evals = lib.train_FT(net, train_X, train_Y, test_X, test_Y, num_epochs, lr, batch_size, updater, device, temp, lambda0, wind)
    sums += cost_time
    start2 = time.time()
    y = [net(i.reshape(1, 14)) for i in test_X[:1000].to(device)]
    end = time.time()
    res.append([cost_time, (end - start2) / len(test_X[:1000])])
res = np.array(res)
print('CNN的训练时间:%s'% np.mean(res[:, 0]))
print('平均推断时间：%s' % np.mean(res[:, 1]))

torch.save(net.state_dict(), 'model_save/%s'% (save_path))

if need_visual:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flat
    net.load_state_dict(torch.load('model_save/%s' % save_path))
    net.eval()
    predict_dict = {}
    scatter_X, scatter_Y = [], []
    print('--' * 20)
    with torch.no_grad():
        key_list = list(test_X_dict.keys())
        key_list = np.random.choice(key_list, len(axes), replace=False)
        for i, key in enumerate(key_list):
            x = test_X_dict[key]
            temp = net(x.to(device)).detach().cpu()
            scatter_X.extend(temp.numpy())
            scatter_Y.extend(test_Y_dict[key].numpy())
            b_mape = lib.MAPE(temp, test_Y_dict[key])
            b_rmse = lib.RMSE(temp, test_Y_dict[key])
            print('电池%s的容量预测的MAPE为：%.4f，RMSE为:%.4f' % (key, b_mape, b_rmse))
            predict_dict[key] = temp.numpy()
            cycle = np.arange(len(test_Y_dict[key]))
            axes[i].plot(cycle, test_Y_dict[key], linewidth=1, zorder=1, marker='+', markersize=1, label='actual')
            axes[i].plot(cycle, predict_dict[key], linewidth=1, zorder=1, marker='_', markersize=1, label='predict')
            axes[i].set_title(key)
            if i % 2 == 0:
                axes[i].set_ylabel('Capacity')
            if i >= 3:
                axes[i].set_xlabel('Cycle')
            axes[i].legend()

    plt.figure()
    scatter_X, scatter_Y = np.array(scatter_X), np.array(scatter_Y)
    plt.scatter(scatter_X, scatter_Y, c=abs(scatter_X - scatter_Y), cmap="seismic", zorder=2, s=3)
    plt.xlim([0.7, 1])
    plt.ylim([0.7, 1])
    plt.plot([0.7, 1], [0.7, 1], '--', zorder=1)
    plt.colorbar()
    plt.show()