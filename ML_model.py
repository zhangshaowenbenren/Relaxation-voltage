from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import lib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


path = 'Dataset_1_NCA_battery.csv'
data = lib.Load_Data(path)
lib.getHumanFeature(data)
test_list = [3,6,20,15,13,16,27,29,36,40,54,59,51,50]

train_X_dict, train_Y_dict, test_X_dict, test_Y_dict = lib.diviseData1(data, test_list=test_list)
train_X, test_X = lib.normalize2(train_X_dict, test_X_dict)

train_Y = np.concatenate([data for data in train_Y_dict.values()])
test_Y = np.concatenate([data for data in test_Y_dict.values()])
# ---------------------------------------------------------------------------------------------------------------SVR
svr = GridSearchCV(SVR(gamma='auto'), cv=4, param_grid={'kernel':['poly', 'rbf'], 'C':[3, 5, 8, 10, 12], 'epsilon':[0.01, 0.015, 0.02]}, n_jobs=-1)

# returns the object after the model cycle, along with the evaluation metrics and the time spent on the evaluation
svr, evals_svr, times_svr = lib.ML_fit_predict(svr, train_X, train_Y, test_X, test_Y)
print('优化后的超参数为：', svr.best_params_)
lib.myPrint('SVR', evals_svr, times_svr)
cv_result = pd.DataFrame.from_dict(svr.cv_results_)
with open('CV_res/svr_result.csv', 'w') as f:
    cv_result.to_csv(f)
# --------------------------------------------------------------------------------------------------------------- Elastic Net
Elastic_regr = GridSearchCV(ElasticNet(), param_grid={'alpha':[0.0002, 0.0005, 0.001], 'l1_ratio':[0.0002, 0.0005, 0.001]})
Elastic_regr, evals_Elastic, times_Elastic = lib.ML_fit_predict(Elastic_regr, train_X, train_Y, test_X, test_Y)
print('优化后的超参数为：', Elastic_regr.best_params_)
lib.myPrint('Elastic', evals_Elastic, times_Elastic)
cv_result = pd.DataFrame.from_dict(Elastic_regr.cv_results_)
with open('CV_res/Elastic_result.csv', 'w') as f:
    cv_result.to_csv(f)
# --------------------------------------------------------------------------------------------------------------- MLP
MLP = GridSearchCV(MLPRegressor(activation='relu', batch_size='auto'), cv=4, param_grid={
    'hidden_layer_sizes':[150, 200, 300, 400], 'learning_rate_init':[0.003, 0.01, 0.05]
})
MLP, evals_MLP, times_MLP = lib.ML_fit_predict(MLP, train_X, train_Y, test_X, test_Y)
print('优化后的超参数为：', MLP.best_params_)
lib.myPrint('MLP', evals_MLP, times_MLP)
cv_result = pd.DataFrame.from_dict(MLP.cv_results_)
with open('CV_res/MLP_result.csv', 'w') as f:
    cv_result.to_csv(f)
# --------------------------------------------------------------------------------------------------------------- XGBoost
xgb_model = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1), param_grid={
    'max_depth':[4, 5, 6], 'learning_rate':[0.01, 0.05, 0.2], 'n_estimators':[100, 150, 200]
})
xgb_model, evals_XGB, times_XGB = lib.ML_fit_predict(xgb_model, train_X, train_Y, test_X, test_Y)
print('优化后的超参数为：', xgb_model.best_params_)
lib.myPrint('XGB', evals_XGB, times_XGB)
cv_result = pd.DataFrame.from_dict(xgb_model.cv_results_)
with open('CV_res/xgb_result.csv', 'w') as f:
    cv_result.to_csv(f)
# --------------------------------------------------------------------------------------------------------------- GPR
ker = RBF(length_scale=1, length_scale_bounds='fixed')
GPR = GaussianProcessRegressor(kernel=ker, n_restarts_optimizer=2, normalize_y=False)
GPR, evals_GPR, times_GPR = lib.ML_fit_predict(GPR, train_X, train_Y, test_X, test_Y)
lib.myPrint('GPR', evals_GPR, times_GPR)
# --------------------------------------------------------------------------------------------------------------- RFR
RFR = GridSearchCV(RandomForestRegressor(n_jobs=-1), cv=4, param_grid={'max_depth':[4, 5, 6], 'n_estimators':[20, 50, 100, 200], 'min_samples_split':[2, 6, 10]})
RFR, evals_RFR, times_RFR = lib.ML_fit_predict(RFR, train_X, train_Y, test_X, test_Y)
print('优化后的超参数为：', RFR.best_params_)
lib.myPrint('RFR', evals_RFR, times_RFR)
cv_result = pd.DataFrame.from_dict(RFR.cv_results_)
with open('CV_res/RFR_result.csv', 'w') as f:
    cv_result.to_csv(f)

fig, axes = plt.subplots(4, 2, figsize=(8, 9.5))
axes = axes.flat

print('--' * 20)
sample = np.random.choice(np.arange(len(test_X_dict.keys())), len(axes) // 2, replace=False)
key_list = list(test_X_dict.keys())
plot_list = [key_list[i] for i in sample]

plt.subplots_adjust(left=0.1, bottom=0.07, right=0.95, top=0.95, wspace=0.27, hspace=0.3)
Net_labels = ['Elastic', 'svr', 'MLP', 'XGB', 'GPR']
Net_colors = ['c', 'gold', 'lime', 'b', 'k']
Net_marks = ['d', 'v', '^', 'o', 'd']

font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 9}
for i, key in enumerate(plot_list):
    cycle = np.arange(len(test_Y_dict[key]))
    axes[2 * i].plot(cycle, test_Y_dict[key], linewidth=1.5, color='r', zorder=1, label='Real')
    for j, model in enumerate([Elastic_regr, svr, MLP, xgb_model, GPR]):
        x = test_X_dict[key]
        pred1 = model.predict(x)
        error = pred1 - test_Y_dict[key]
        n = 2
        axes[2 * i].scatter(cycle[::n], pred1[::n], marker=Net_marks[j], color=Net_colors[j], s=6, label=Net_labels[j], zorder=2)
        axes[2 * i + 1].plot(cycle[::n], error[::n], marker=Net_marks[j], color=Net_colors[j], linewidth=1, label=Net_labels[j], markersize=3, zorder=2)
    axes[2 * i + 1].plot(cycle, np.zeros_like(cycle), linestyle='--', zorder=1)
    axes[2 * i].set_title('cell:' + key, fontsize=10, pad=2)
    axes[2 * i].set_ylabel('Capacity (Ah)')
    axes[2 * i + 1].set_title('cell:' + key, fontsize=10, pad=2)
    axes[2 * i + 1].set_ylabel('Error', fontsize=None, labelpad=0.2)

    if i == len(plot_list)-1:
        axes[2 * i].set_xlabel('Cycle number')
        axes[2 * i + 1].set_xlabel('Cycle number')

    axes[2 * i].legend(loc='lower left', borderpad=0.1, markerscale=2.0, labelspacing=0.2, prop=font1, fontsize=10)
    axes[2 * i + 1].legend(loc='upper left', borderpad=0.1, markerscale=2.0, labelspacing=0.2, prop=font1, fontsize=10)

plt.savefig('SOH_PK.png', dpi=1000, bbox_inches='tight')
plt.show()