import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score


def LGB_model(X, y, n_splits, n_repeats, random_state=None):
    accuracies = []
    aucs = []
    fold_results = []
    fold_indices = []

    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        fold_accuracies = []
        fold_aucs = []
        fold_indices_list = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            params = {
                'objective': 'binary',  # 任务类型为二分类
                'metric': 'binary_logloss',  # 评估指标
                'boosting_type': 'gbdt',  # 提升类型，默认 GBDT
                'num_leaves': 31,  # 树的最大叶子数
                'learning_rate': 0.05,  # 学习率
                'feature_fraction': 0.9  # 特征选择的比例
            }
            num_round = 100  # 迭代次数
            lgb_classifier = lgb.train(params, train_data, num_round, valid_sets=[test_data])


            y_pred = lgb_classifier.predict(X_test, num_iteration=lgb_classifier.best_iteration)
            accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
            auc = roc_auc_score(y_test, y_pred)

            fold_accuracies.append(accuracy)
            fold_aucs.append(auc)
            fold_indices_list.append(test_index.tolist())

        fold_results.append((fold_accuracies, fold_aucs))
        fold_indices.append(fold_indices_list)
        accuracies.append(np.mean(fold_accuracies))
        aucs.append(np.mean(fold_aucs))

    return accuracies, aucs, fold_results, fold_indices

def save_results_to_file(accuracies, aucs, fold_results, fold_indices, file_path):
    with open(file_path, 'w') as output_file:

        output_file.write(f'Mean Accuracy: {np.mean(accuracies):.2f}\n')
        output_file.write(f'Mean AUC: {np.mean(aucs):.2f}\n\n')

        for i, ((accuracies, aucs), indices) in enumerate(zip(fold_results, fold_indices)):
            output_file.write(f'Repeat {i+1}:\n')
            output_file.write(f'  Fold Accuracies: {accuracies}\n')
            output_file.write(f'  Fold AUCs: {aucs}\n')
            output_file.write(f'  Fold Indices: {indices}\n\n')

    print(f'Results have been saved to {file_path}')


def load_from_csv(X_filename='X.csv', y_filename='y.csv'):
    X = pd.read_csv(X_filename, header=None).values
    y = pd.read_csv(y_filename, header=None).values.ravel()
    return X, y

X, y = load_from_csv('/Users/xcw/Desktop/CAI_CG_Embedding/data/DC_HOMO_pDC_X.csv', '/Users/xcw/Desktop/CAI_CG_Embedding/data/DC_HOMO_pDC_Y.csv')
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

accuracies, aucs, fold_results, fold_indices = LGB_model(X, y, n_splits=10, n_repeats=100)
save_results_to_file(accuracies, aucs, fold_results, fold_indices, 'lgb_homo_DC_pDC.txt')