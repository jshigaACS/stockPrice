#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2. 学習と検証
2-1: train_test_splitRandomForestRegressor
2-2: k-fold交差検定
2-n: 残差プロット        
"""

import sys
from sklearn.model_selection import train_test_split

from sklearn import datasets
import pandas as pd
import seaborn as sns

def plotter(y_train_pred, y_train, y_test_pred, y_test):

    train_resi = y_train_pred - y_train
    test_resi = y_test_pred - y_test
    plt.scatter(y_train_pred, train_resi,c='blue',marker='^',label = 'train_data')
    plt.scatter(y_test_pred, test_resi,c='red',marker='s',label='test_data')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
    plt.xlim([10,50])
    plt.show()

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

    return y_val, x_val

def exe_ml(x,y,train_df):
    from module import create_model_split, create_model_cv
    import module

    #2-1: train_test_splitによるモデル評価
    x_train, x_test, y_train, y_test = train_test_split(
        x, 
        y, 
        test_size=0.3,
        random_state=0 #モデルの精度向上を検証するためランダムシードは一定とする
    )

    #train_test_split
    cms = create_model_split(x_train,x_test,y_train,y_test)
    model = cms.model
    y_test_pred = cms.y_test_pred
    split_probability, _ = module.accuracy(y_test, y_test_pred)
    y_train_pred = cms.y_train_pred

    plotter(y_train_pred,y_train,y_test_pred,y_test)

    """
    r2_test,r2_train = cms._get_r2_score()
    rmse_test,rmse_train = cms._get_rmse()

    #2-2: 交差検証（K-fold)によるモデル評価
    cmCV = create_model_cv(x,y)
    test_R2_scores,test_RMSE_Scores = cmCV.get_test_res()
    train_R2_scores,train_RMSE_Scores = cmCV.get_train_res()
    #print(test_R2_scores)
    #print(train_R2_scores)
    """
    return model, split_probability

def view(feature_imp):

    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.figure(figsize=(30,50))
    plt.show()


if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = (15.0, 15.0)
    train_df = read_csv2df()
    #1-2
    y_val, x_val = setting()
    x = train_df[x_val]
    y = train_df[y_val]

    model, accuracy = exe_ml(x, y, train_df)

    feature_imp = pd.Series(model.feature_importances_, index=x_val).sort_values(ascending=False)
    #print(accuracy)
    #print(feature_imp)

    #view(feature_imp)
    

"""
3. パラメータ調整
    3-1: パラメータ重要度
    3-2: グリッドサーチ
        n_estimater, 
    3-3: 標準化,正規化
    3-4: 正解ラベル（1:上昇, 2:中間, 3:下落）
    3-5: PCA適用
    3-n: 残差プロット

補足. 各種モデルによる検証
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
