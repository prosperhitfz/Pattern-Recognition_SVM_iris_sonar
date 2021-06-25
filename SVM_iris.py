import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import decomposition


from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')  # 处理报错级别问题，只显示error级别的报错


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # 定义颜色和标记符号，通过颜色列图表生成颜色示例图
    marker = ('*', '*', '*', 'x', 'p')
    colors = ('blue', 'green', 'purple', 'grey', 'black')
    edge_map = ListedColormap(colors[:len(np.unique(y))])

    # 可视化决策边界
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=edge_map)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 绘制所有的样本点
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=edge_map(idx), marker=marker[idx], s=73, label=cl)

    # 使用小方块高亮显示测试集的样本
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1,
                    edgecolors='black', marker='s', s=135)


iris = datasets.load_iris()
pca = decomposition.PCA(n_components=2)
iris1 = pca.fit_transform(iris.data)  # PCA降维至2维
X = iris1[:, [0, 1]]
y = iris.target
# print(X)
# 划分测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# 对特征值进行标准化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
# 将标准化后的训练数据和测试数据重新整合到一起
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plt.subplot(1, 3, 1)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.title('线性核函数支持向量机模型的决策区域', fontsize=19, color='w')


svm = SVC(kernel='poly', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
# 将标准化后的训练数据和测试数据重新整合到一起
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plt.subplot(1, 3, 2)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.title('多项式核函数支持向量机模型的决策区域', fontsize=19, color='w')


svm = SVC(kernel='rbf', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
# 将标准化后的训练数据和测试数据重新整合到一起
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plt.subplot(1, 3, 3)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.title('高斯核函数支持向量机模型的决策区域', fontsize=19, color='w')
plt.show()
