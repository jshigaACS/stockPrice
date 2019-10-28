
def read_csv2df():
    dirpath = '/home/ubuntu/analysis/stockPrice/'
    csv_name = 'stockPriceData.csv'

    df = pd.read_csv(dirpath+csv_name)
    return df

def setting():
    y_val = '富士通(株)：翌日比'
    nonNeededCol = ['date', 'year','month', 'day']
    nonNeededCol.append(y_val)
    #独立変数の設定
    x_val = [x for x in train_df.columns if x not in nonNeededCol]

    return y_val, nonNeededCol, x_val

"""
1. 前準備
    1-1: Load CSV
    1-2: setting
"""
import pandas as pd
#1-1
train_df = read_csv2df()
#1-2
y_val, nonNeedCol, x_val = setting()
x = train_df[x_val]
y = train_df[y_val]

"""
2. 学習と検証
    2-1: train_test_split
        RandomForestRegressor
    2-2: k-fold交差検定
        
"""
def test(y_test, y_pred):
    testUpDown = []
    for test in y_test:
        if test > 0:
            testUpDown.append(1)
        else:
            testUpDown.append(-1)
    predUpDown = []
    for pred in y_pred:
        if pred > 0:
            predUpDown.append(1)
        else:
            predUpDown.append(-1)

    prob = str(metrics.accuracy_score(testUpDown, predUpDown)*100)
    return prob

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
#2-1: train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.1,
    random_state=0 #モデルの精度向上を検証するためランダムシードは一定とする
)
model1 = RandomForestRegressor(n_estimators=1000)
"""
model1.fit(x_train,y_train)
y_pred = model1.predict(x_test)
split_probability = test(y_test, y_pred)
print(split_probability)
"""
#2-2: 交差検証（K-fold)
kf = KFold(n_splits=5, shuffle=True, random_state=1)
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    print(x[train_index])
    print(x[test_index])

#label_train = [1 if l > 0 else -1 for l in y_train]
#label_test = [1 if l > 0 else -1 for l in y_test]




"""
3. パラメータ調整
    3-1: パラメータ重要度
    3-2: グリッドサーチ
    3-3: 主成分分析
    3-4: 標準化
    3-n: 残差プロット
"""

"""
補足. 各種モデルによる検証
"""
"""
from sklearn.linear_model import LogisticRegression # ロジスティック回帰(0,1変換)
from sklearn.svm import LinearSVC # SVM
from sklearn.tree import  DecisionTreeClassifier # 決定木
from sklearn.neighbors import  KNeighborsClassifier # k-NN
from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト

models = [LogisticRegression(),LinearSVC(),DecisionTreeClassifier(),KNeighborsClassifier(n_neighbors = 6),RandomForestClassifier()]
scores = {}
for model in models:
    scores[str(model).split('(')[0]] = cross_val_score(model, x, y, cv=5)
df = pd.DataFrame(scores)
df.mean()
"""