# MK-MMD for dataset2

import pickle
import optuna
import torch
from sklearn.model_selection import train_test_split
import lib
import time
import visdom
import numpy as np
from torch.utils.data import DataLoader,  TensorDataset

need_BO, BO_epochs = True, 100
random_seed = 10
path = 'Dataset_2_NCM_battery.xlsx'
source_data = lib.Load_Data('Dataset_1_NCA_battery.xlsx')
Target_data = lib.Load_Data(path)
need_visual = False

cells_list = [np.arange(1, 24), np.arange(24, 28), np.arange(28, 56)]
n = [6, 1, 3]
step = 10
train_list = [np.random.RandomState(seed=random_seed).choice(a, n[i], replace=False) for i, a in enumerate(cells_list)]
b = []
for i in train_list:
    b.extend(i)
train_list = b

train_X_dict, train_Y_dict, test_X_dict, test_Y_dict = lib.diviseData(Target_data, train_list=train_list)
train_X, test_X = lib.normalize(train_X_dict, test_X_dict)
train_Y = torch.concat([data for data in train_Y_dict.values()])
test_Y = torch.concat([y for y in test_Y_dict.values()], dim=0)
indices = np.arange(len(train_X))
np.random.RandomState(seed=random_seed).shuffle(indices)
train_X, train_Y = train_X[indices[::step]], train_Y[indices[::step]]

train0_X, val_X, train0_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.3, random_state=7)

src_X, src_Y = [], []
for key, data in source_data.items():
    step = 5 if 'D2' in key else 20
    src_X.append(torch.tensor(data['V'][::step], dtype=torch.float32))
    src_Y.append(torch.tensor(data['Q'][::step], dtype=torch.float32))
src_X, src_Y = torch.concat(src_X), torch.concat(src_Y)
src_X = (src_X - torch.tensor(4.1)) / (torch.tensor(4.2) - torch.tensor(4.1))
dataSet = TensorDataset(src_X, src_Y)
dataSet2 = TensorDataset(train0_X, train0_Y)
dataSet3 = TensorDataset(val_X, val_Y)

save_path = 'MMD_TL_Dataset2'
device = lib.try_gpu()

def optuna_objective(trial):
    batch_size = trial.suggest_int('batch_size', 32, 256, 32)
    lr = trial.suggest_float('lr', 0.001, 0.1, log=True)
    weight_decay=trial.suggest_float('weight_decay', 0.00001, 0.001, log=True)
    num_epochs = trial.suggest_int('num_epochs', 20, 100, 10)
    l2 = trial.suggest_float('LAMBDA1', 0.01, 5, log=True)
    l3 = trial.suggest_float('LAMBDA2', 0.001, 5, log=True)
    LAMBDA = [l2, l3]

    data_src = DataLoader(dataSet, batch_size=batch_size, shuffle=True, drop_last=False)
    data_tar = DataLoader(dataSet2, batch_size=batch_size, shuffle=True, drop_last=False)
    data_test = DataLoader(dataSet3, batch_size=batch_size, shuffle=True, drop_last=False)

    net = lib.DaNN(dropout=0.)
    net = net.to(device)
    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0}]
    updater = torch.optim.Adam(parameters, lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(updater, max_lr=lr, steps_per_epoch=len(data_src),
                                                    epochs=num_epochs, pct_start=0.3, div_factor=25)
    best_Evals = lib.train_mmd2(num_epochs=num_epochs, model=net, optimizer=updater, data_src=data_src, data_tar=data_tar,
                               data_test=data_test, device=device, scheduler=scheduler, l=LAMBDA)
    return best_Evals[2]


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


import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

if need_BO:
    best_params, best_score = optimizer_optuna(BO_epochs, 'GP', optuna_objective)
    print('best_params:%s' % best_params)
    print('best_score%s' % best_score)
    with open('TL_MMD_best_params.pkl', 'wb') as fp:
        pickle.dump(best_params, fp)
else:
    with open('TL_MMD_best_params.pkl', 'rb') as fp:
        best_params = pickle.load(fp)

lr, num_epochs = best_params['lr'], best_params['num_epochs']
batch_size = best_params['batch_size']
lambdas = [best_params['LAMBDA1'], best_params['LAMBDA2']]
weight_decay = best_params['weight_decay']

data_tar, data_test = lib.getDataLoder(train_X, train_Y, test_X, test_Y, batch_size)
data_src = DataLoader(dataSet, batch_size=batch_size, shuffle=True, drop_last=False)
res = []

net = lib.DaNN(dropout=0.)
net = net.to(device)

weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
parameters = [{'params': weight_decay_list},
              {'params': no_decay_list, 'weight_decay':0}]
updater = torch.optim.Adam(parameters, lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(updater, max_lr=lr, steps_per_epoch=len(data_src),
                                                    epochs=num_epochs, pct_start=0.3, div_factor=25)
start1 = time.time()
Evals = lib.train_mmd2(num_epochs=num_epochs, model=net, optimizer=updater, data_src=data_src, data_tar=data_tar, data_test=data_test,
                          device=device, scheduler=scheduler, l=lambdas)
start2 = time.time()
print(Evals)
y = [net(i.reshape(1, 14), i.reshape(1, 14)) for i in test_X[:1000].to(device)]
end = time.time()
torch.save(net.state_dict(), 'model_save/%s'% (save_path))

res.append([start2 - start1, (end - start2) / len(test_X[:1000])])
res = np.array(res)
print('MK-MMD的训练时间:%s'% np.mean(res[:, 0]))
print('平均推断时间：%s' % np.mean(res[:, 1]))