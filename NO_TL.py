# No Transfer Learning; training a model based on the training set in the target domain only,
# and then making predictions on the test set

import torch
import lib
import numpy as np

# Returns a collection containing voltage and current data for the entire life cycle charging phase of each battery, as well as SOH (label)
path = 'Dataset_2_NCM_battery.xlsx'
data = lib.Load_Data(path)                         # data type is dict, contains the cells features and label

random_seed = 10
lr, num_epochs, batch_size = 0.0001, 100, 128
cells_list = [np.arange(1, 24), np.arange(24, 28), np.arange(28, 56)]                          # Serial numbers of the three different types of batteries in dataset 1
n = [6, 1, 3]                                                                                  # Number of stratified samples
step = 10
# Randomly selected battery serial numbers for migration learning training
train_list = [np.random.RandomState(seed=random_seed).choice(a, n[i], replace=False) for i, a in enumerate(cells_list)]
b = []
for i in train_list:
    b.extend(i)
train_list = b
save_path = 'No_TL_Dataset2'
train_X_dict, train_Y_dict, test_X_dict, test_Y_dict = lib.diviseData(data, train_list=train_list)
train_X, test_X = lib.normalize(train_X_dict, test_X_dict)
train_Y = torch.concat([data for data in train_Y_dict.values()])
test_Y = torch.concat([y for y in test_Y_dict.values()], dim=0)

indices = np.arange(len(train_X))
np.random.RandomState(seed=random_seed).shuffle(indices)                  # Sparse sampling
train_X, train_Y = train_X[indices[::step]], train_Y[indices[::step]]

net = lib.CNN_model2()
device = lib.try_gpu()
net = net.to(device)
weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
parameters = [{'params': weight_decay_list},
              {'params': no_decay_list, 'weight_decay': 0}]
updater = torch.optim.Adam(parameters, lr, weight_decay=1.8e-5)

cost_time, Evals = lib.train(net, train_X, train_Y, test_X, test_Y, num_epochs, lr, batch_size, updater, device)
torch.save(net.state_dict(), 'model_save/%s'% (save_path))
print('程序运行时间为：{}'.format(cost_time))
print('Dataset2, No TL, MAPE:{:.4f}, RMSE：{:.4f}, R_2:{:.4f}'.format(Evals[1], Evals[2], Evals[3]))