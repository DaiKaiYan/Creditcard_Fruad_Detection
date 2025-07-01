# 导入必要的库
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from itertools import cycle

# 设置显示选项和样式
sns.set_style('whitegrid')
pd.set_option('display.float_format', lambda x: '%.4f' % x)
warnings.filterwarnings('ignore')

# 读取数据
data_df = pd.read_csv("C:/Users/Administrator/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv")

# 数据预处理
data_df['Hour'] = data_df['Time'].apply(lambda x: divmod(x, 3600)[0])
droplist = ['V8', 'V13', 'V15', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Time']
data_df_new = data_df.drop(droplist, axis=1)

# 特征缩放
col = ['Amount', 'Hour']
sc = StandardScaler()
data_df_new[col] = sc.fit_transform(data_df_new[col])

# 构建自变量和因变量
x_feature = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'Amount', 'Hour']
x_val = data_df_new[x_feature]
y_val = data_df_new['Class']

# 特征重要性分析
clf = RandomForestClassifier(n_estimators=10, random_state=123, max_depth=4)
clf.fit(x_val, y_val)

# 数据可视化
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 6)
importances = clf.feature_importances_
feat_names = data_df_new[x_feature].columns
indices = np.argsort(importances)[::-1]
fig = plt.figure(figsize=(20, 6))
plt.title("Feature importances by RandomTreeClassifier")
x = list(range(len(indices)))
plt.bar(x, importances[indices], color='lightblue', align="center")
plt.step(x, np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(x, feat_names[indices], rotation='vertical', fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()

# 数据采样与降维
df = data_df_new.sample(frac=1)
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]
normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
new_df = normal_distributed_df.sample(frac=1, random_state=42)

X = new_df.drop('Class', axis=1)
y = new_df['Class']

# 降维
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1 - t0))

t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print("PCA took {:.2} s".format(t1 - t0))

t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
t1 = time.time()
print("Truncated SVD took {:.2} s".format(t1 - t0))

# 可视化降维结果
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
blue_patch = mpatches.Patch(color='#4169E1', label='Normal')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

ax1.scatter(X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], c=(y == 0), cmap='coolwarm', label='Normal', linewidths=2)
ax1.scatter(X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)
ax1.grid(True)
ax1.legend(handles=[blue_patch, red_patch])

ax2.scatter(X_reduced_pca[:, 0], X_reduced_pca[:, 1], c=(y == 0), cmap='coolwarm', label='Normal', linewidths=2)
ax2.scatter(X_reduced_pca[:, 0], X_reduced_pca[:, 1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)
ax2.grid(True)
ax2.legend(handles=[blue_patch, red_patch])

ax3.scatter(X_reduced_svd[:, 0], X_reduced_svd[:, 1], c=(y == 0), cmap='coolwarm', label='Normal', linewidths=2)
ax3.scatter(X_reduced_svd[:, 0], X_reduced_svd[:, 1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('TSVD', fontsize=14)
ax3.grid(True)
ax3.legend(handles=[blue_patch, red_patch])
plt.show()

# 处理不平衡数据
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(x_val, y_val)
print('通过SMOTE方法平衡正负样本后')
n_sample = y.shape[0]
n_pos_sample = y[y == 1].shape[0]
n_neg_sample = y[y == 0].shape[0]
print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample, n_pos_sample / n_sample, n_neg_sample / n_sample))
print('特征维数：', X.shape[1])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
print(f"训练集：{len(X_train)}, 测试集：{len(X_test)}")

# 模型训练与评估
param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
clf = GridSearchCV(LogisticRegression(), param_grid, cv=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('<--------Confusion Matrix-------->\n', confusion_matrix(y_test, y_pred))
print('<--------Classification Report-------->\n', classification_report(y_test, y_pred))

# 学习曲线绘制
def plot_learning_curve(estimator, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    f, ax1 = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124", label="Training")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff", label="Validation")
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")
    return plt

estimator = LogisticRegression(penalty='l2', C=10.0, max_iter=1000)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
plot_learning_curve(estimator, X, y, (0.95, 0.97), cv=cv, n_jobs=4)

# 混淆矩阵绘制
def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

y_pred_proba = clf.predict_proba(X_test)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
plt.figure(figsize=(15, 10))
j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_proba[:, 1] > i
    plt.subplot(3, 3, j)
    j += 1
    cnf_matrix = confusion_matrix(y_test, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)
    x1 = cnf_matrix[1, 1]
    x2 = (cnf_matrix[1, 0] + cnf_matrix[1, 1])
    print("threshold:{},Recall metric in the testing dataset {}->{}->{} ".format(i, x1 / x2, x1, x2))
    class_names = [0, 1]
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=f"Threshold={i}")

colors = cycle(['#ADD8E6', '#87CEFA', '#00CED1', '#00BFFF', '#4169E1', '#4682B4', '#0000CD', '#000080', 'black'])
plt.figure(figsize=(12, 12))
j = 1
for i, color in zip(thresholds, colors):
    y_test_predictions_prob = y_pred_proba[:, 1] > i
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_predictions_prob)
    area = auc(recall, precision)
    plt.plot(recall, precision, color=color, label='Threshold= %s, AUC=%0.5f' % (i, area))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.4, 1.05])
    plt.xlim([0.0, 1])
    plt.legend(loc="lower left")
plt.show()