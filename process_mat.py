# data preprocess for dataset 3

import scipy.io as sio
import numpy as np
import pickle
import os
import pandas as pd
import gc

cell_names = ['W4', 'W5', 'W8', 'W9', 'W10']
res = {'W4':{}, 'W5':{}, 'W8':{}, 'W9':{}, 'W10':{}}

for i in range(1, 15):
    print('cycling:%s ...'%i)
    root_file = 'D:\EV discharge\Cycling_%s\_processed_mat' % i
    for name in cell_names:
        if name + '.mat' not in os.listdir(root_file):
            continue
        print('cell name:%s'%name)
        file = name + '.mat'
        res[name]['cyclng%s'%i] = None
        paths = os.path.join(root_file, file)
        f = sio.loadmat(paths)
        lens = len(f['Step_Index_full_vec_M1_NMC25degC'])
        index = f['Step_Index_full_vec_M1_NMC25degC'].reshape(lens)
        V = f['V_full_vec_M1_NMC25degC'].reshape(lens)
        I = f['I_full_vec_M1_NMC25degC'].reshape(lens)
        t = f['t_full_vec_M1_NMC25degC'].reshape(lens)
        cha = f['t_full_vec_M1_NMC25degC'].reshape(lens)
        dis = f['dis_cap_full_vec_M1_NMC25degC'].reshape(lens)
        temp = index == 11
        start = temp.copy()
        end = temp.copy()
        for k in range(len(temp)):
            if temp[k] == True and temp[k] == temp[k - 1]:
                start[k] = False
            if temp[k] == True and temp[k] == temp[k + 1]:
                end[k] = False
        print(sum(start), sum(end))
        print()

        for k, (s, e) in enumerate(zip(np.arange(len(start))[start], np.arange(len(end))[end])):
            df = pd.DataFrame({'cycle':np.full(14, k), 't': t[s:s + 391:30] - t[s], 'V': V[s:s + 391:30], 'I': I[s:s + 391:30],
                               'charge cap': cha[s:s + 391:30], 'discharge cap': dis[s:s + 391:30]})
            if k == 0:
                res[name]['cyclng%s' % i] = df
            else:
                res[name]['cyclng%s' % i] = pd.concat([res[name]['cyclng%s' % i], df])

        del f, start, end, temp, V, I, t, cha, dis, index
        gc.collect()

name, i = 'W4', 5
paths = 'D:\EV discharge\Cycling_5\_processed_mat\W4.mat'
f = sio.loadmat(paths)

lens = len(f['Step_Index_full_vec_M1_NMC25degC'])
index = f['Step_Index_full_vec_M1_NMC25degC'].reshape(lens)
V = f['V_full_vec_M1_NMC25degC'].reshape(lens)
I = f['I_full_vec_M1_NMC25degC'].reshape(lens)
t = f['t_full_vec_M1_NMC25degC'].reshape(lens)
cha = f['t_full_vec_M1_NMC25degC'].reshape(lens)
dis = f['dis_cap_full_vec_M1_NMC25degC'].reshape(lens)

start = []

for k in range(len(V)):
    if I[k] == 0 and I[k - 1] != 0 and I[k + 10] == 0 and I[k + 100] == 0 :
        start.append(k)
print(start)

for k, s in enumerate(start):
    df = pd.DataFrame({'cycle':np.full(14, k), 't': t[s:s + 391:30] - t[s], 'V': V[s:s + 391:30], 'I': I[s:s + 391:30],
                       'charge cap': cha[s:s + 391:30], 'discharge cap': dis[s:s + 391:30]})
    if k == 0:
        res[name]['cyclng%s' % i] = df
    else:
        res[name]['cyclng%s' % i] = pd.concat([res[name]['cyclng%s' % i], df])

del f, start, V, I, t, cha, dis, index

with open('dataset/ev_cycling.pkl', 'wb') as fp:
    pickle.dump(res, fp)
