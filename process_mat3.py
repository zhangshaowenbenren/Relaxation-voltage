# PCHIP interpolation

import numpy as np
import pickle
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d, PchipInterpolator
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman', size=8)

with open('dataset/ev_cycling.pkl', 'rb') as fp:
    res = pickle.load(fp)

with open('dataset/ev_diag.pkl', 'rb') as fp:
    res2 = pickle.load(fp)

cap = sio.loadmat('dataset/capacity.mat')['processed_mat']

cell_names = ['W3',	'W4', 'W5',	'W7', 'W8',	'W9', 'W10', 'G1', 'V4', 'V5']
valid_names = ['W4', 'W5', 'W8', 'W9', 'W10']

# # Interpolation filling of the exact capacity using PCHIP interpolation
def point2traj2(x, y, new_x):
    spline = PchipInterpolator(x, y)
    t = np.arange(x[0], x[-1] + 1)     # points after interpolation
    y_pred = spline(t)
    return y_pred, spline(new_x)

sample_x, sample_y, valid_y = {}, {}, {}
fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
ev_meas_point = {}

for cell in valid_names:
    sample_x[cell] = []
    sample_y[cell] = []
    valid_y[cell] = []
    for disg in range(1, len(res2[cell])):
        sample_x[cell].append(res2[cell]['disg%s'%disg]['Voltage(V)'].values)
        sample_y[cell].append(cap[cell_names.index(cell)][disg - 1])

        temp = res[cell]['cyclng%s'%disg]
        if temp is None:
            continue
        cycle_data = temp.groupby('cycle')
        for cycle in cycle_data.groups.keys():
            sample_x[cell].append(cycle_data.get_group(cycle)['V'].values)
            sample_y[cell].append(np.nan)
    sample_x[cell].append(res2[cell]['disg%s' % (disg + 1)]['Voltage(V)'])
    sample_y[cell].append(cap[cell_names.index(cell)][disg])

    sample_x[cell] = np.stack(sample_x[cell])
    sample_y[cell] = np.stack(sample_y[cell])

    x, y = [], []
    for i in range(len(sample_y[cell])):
        if not np.isnan(sample_y[cell][i]):
            x.append(i)
            y.append(sample_y[cell][i])

    new_x = np.arange(len(sample_y[cell]))
    spline = PchipInterpolator(x, y)
    new_y = spline(new_x)
    sample_y[cell] = new_y.copy()
    ev_meas_point[cell] = [x, y]
    if cell != 'W4':
        ax.plot(new_x, new_y / 4.85, label=cell)
        ax.scatter(x, [k / 4.85 for k in  y], s=25, marker='*')

with open('dataset/train_sample.pkl', 'wb') as fp:
    pickle.dump([sample_x, sample_y], fp)

with open('dataset/ev_meas_point.pkl', 'wb') as fp:
    pickle.dump(ev_meas_point, fp)

ax.grid(linestyle='--', linewidth=1, alpha=0.3)
ax.legend()
ax.set_xlabel('Cycle Number')
ax.set_ylabel('SOH')
plt.savefig('paper_fig/ev_raw_capacity_cycle.jpg', dpi=1000, bbox_inches='tight')
plt.show()
# cycling5, W4。       源文件有问题，重新处理了一下。
# name, i = 'W4', 5
# paths = 'D:\EV discharge\Cycling_5\_processed_mat\W4.mat'
# f = sio.loadmat(paths)

# lens = len(f['Step_Index_full_vec_M1_NMC25degC'])
# index = f['Step_Index_full_vec_M1_NMC25degC'].reshape(lens)
# V = f['V_full_vec_M1_NMC25degC'].reshape(lens)
# I = f['I_full_vec_M1_NMC25degC'].reshape(lens)
# t = f['t_full_vec_M1_NMC25degC'].reshape(lens)
# cha = f['t_full_vec_M1_NMC25degC'].reshape(lens)
# dis = f['dis_cap_full_vec_M1_NMC25degC'].reshape(lens)
#
# start = []
#
# for k in range(len(V)):
#     if I[k] == 0 and I[k - 1] != 0 and I[k + 10] == 0 and I[k + 100] == 0 :
#         start.append(k)
# print(start)
#
# for k, s in enumerate(start):
#     df = pd.DataFrame({'cycle':np.full(14, k), 't': t[s:s + 391:30] - t[s], 'V': V[s:s + 391:30], 'I': I[s:s + 391:30],
#                        'charge cap': cha[s:s + 391:30], 'discharge cap': dis[s:s + 391:30]})
#     if k == 0:
#         res[name]['cyclng%s' % i] = df
#     else:
#         res[name]['cyclng%s' % i] = pd.concat([res[name]['cyclng%s' % i], df])
#
# del f, start, V, I, t, cha, dis, index
#
# with open('dataset/ev.pkl', 'wb') as fp:
#     pickle.dump(res, fp)