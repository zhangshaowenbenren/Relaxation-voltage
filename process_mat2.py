# return dataset 3 dict, key:relaxation voltage, value:SOH

import pickle
import os
import pandas as pd


cell_names = ['W4', 'W5', 'W8', 'W9', 'W10']
res = {'W4':{}, 'W5':{}, 'W8':{}, 'W9':{}, 'W10':{}}

for i in range(1, 16):
    print('Diag:%s ...'%i)
    root_file = 'D:\EV discharge\diagnostic_tests\Diag_%s\Capacity_test' % i
    for name in cell_names:
        is_find = False
        for file in os.listdir(root_file):
            if name in file:
                is_find = True
                break
        if not is_find:
            continue
        print('cell name:%s' % name)

        paths = os.path.join(root_file, file)
        df = pd.read_excel(paths, sheet_name=paths[-14:-5] + '_1')
        s = df.index[df.Step_Index == 4][0]                           # Start point of the relaxation voltage
        temp = df.loc[s:s + 391:30, ['Step_Time(s)', 'Voltage(V)', 'Current(A)']]

        res[name]['disg%s' % i] = temp
with open('dataset/ev_diag.pkl', 'wb') as fp:
    pickle.dump(res, fp)