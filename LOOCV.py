from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import pandas as pd

def loocv_accuracy(model, X, y):
    """
    使用 Leave-One-Out 交叉验证评估模型的准确率。

    参数：
    - model: 要评估的模型,需要完成训练
    - X: 特征数据
    - y: 目标变量

    返回值：
    - 准确率
    """
    loo = LeaveOneOut()
    total_correct = 0
    total_samples = 0

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        # 预测
        y_pred = model.predict(X_test)

        # 计算准确率
        total_samples += 1
        if y_pred == y_test:
            total_correct += 1

    accuracy = total_correct / total_samples

    return accuracy

def load_from_csv(X_filename='X.csv', y_filename='y.csv'):
    X = pd.read_csv(X_filename, header=None).values
    y = pd.read_csv(y_filename, header=None).values.ravel()
    return X, y

################################################################
X, y = load_from_csv('X_homo_Ori.csv', 'y_homo.csv')
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
################################################################

################################################################
knn_classifier = KNeighborsClassifier(n_neighbors=7)
accuracy_knn = loocv_accuracy(knn_classifier, X, y)
print(f'KNN Leave-One-Out 交叉验证准确率: {accuracy_knn:.4f}')
################################################################

################################################################
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_classifier.set_params(probability=True)
accuracy_svm = loocv_accuracy(svm_classifier, X, y)
print(f'SVM Leave-One-Out 交叉验证准确率: {accuracy_svm:.4f}')
################################################################

################################################################
rf_classifier = RandomForestClassifier(random_state=42)
accuracy_rf = loocv_accuracy(rf_classifier, X, y)
print(f'Random Forest Leave-One-Out 交叉验证准确率: {accuracy_rf:.4f}')
################################################################

################################################################
xgb_classifier = XGBClassifier(objective='binary:logistic', random_state=42)
accuracy_xgb = loocv_accuracy(xgb_classifier, X, y)
print(f'XGBoost Leave-One-Out 交叉验证准确率: {accuracy_xgb:.4f}')
################################################################