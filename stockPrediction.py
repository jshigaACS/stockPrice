import matplotlib.pyplot as plt
import seaborn as sns
import module
from module import create_model_split,create_model_cv
import pandas as pd
import numpy as np

plt.rcParams['figure.figsize'] = (15.0, 15.0)


def read_csv2df():
    import pandas as pd

    dirpath = '/home/ubuntu/analysis/stockPrice/'
    csv_name = 'stockPriceData.csv'

    df = pd.read_csv(dirpath+csv_name)
    return df

def setting(train_df):
    y_val = '富士通(株)：翌日比'
    nonNeededCol = ['date', 'year','month', 'day']
    nonNeededCol.append(y_val)
    #独立変数の設定
    x_val = [x for x in train_df.columns if x not in nonNeededCol]

    return y_val, nonNeededCol, x_val


def exe_prediction():
    import subprocess as sp

    cmd = 'python3'
    program = 'prediction.py'
    #_dir = '/home/ubuntu/analysis/stockPrice/'
    command = [cmd,program]
    #print(command)
    sp.check_call(command)

def plot(train_df, y):
    plt.plot(train_df['date'],y, marker='o')
    plt.plot(train_df['date'],y, marker='X')
    plt.xlabel('date')
    plt.ylabel('stockVal_ratio')
    plt.tight_layout()
    plt.show()
    plt.figure()


#変数間の相関行列およびヒートマップ
def heatMap(df):
    corr_mat = df.corr(method='pearson')
    sns.heatmap(
        corr_mat, 
        vmax=1.0, 
        vmin=-1.0,
        center=0, 
        fmt='.1ft',    
    )

#実数と予測の比較グラフ
def compare_view(y_test_df):
    pred = y_test_df['y_test_pred']
    test = y_test_df['y_test']

    plt.plot(pred, label='prediction')
    plt.plot(test, label='True')
    plt.legend()

def feature_view(feature_imp):

    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.figure(figsize=(30,50))
    plt.show()

def compare_scaler(x,y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import RobustScaler

    SS = StandardScaler()
    MMS = MinMaxScaler()
    RS = RobustScaler()
    scalers = [SS,MMS,RS]
    scores = {}
    for scaler in scalers:
        #y = scaler.fit_transform(np.array(y).reshape(-1,1))
        #y = np.reshape(y,(-1))
        x = scaler.fit_transform(x)
        val = module.exe_ml(x, y)
        score = {
            str(scaler):val[1]
        }
        scores.update(score)
        #print(str(scaler)+': '+score[1])
    return scores




"""
1. 前準備
    1-1: Load CSV
    1-2: setting
"""
#1-1
train_df = read_csv2df()#train_df.hist()

#1-2
y_val, nonNeedCol, x_val = setting(train_df)

x = train_df[x_val]
y = train_df[y_val]

"""
2. 事前調査
    2-1: Yの時系列変化
        →乱高下が激しい→正規化標準化必要?
    2-2: histgram
        →正規分布、、、
    2-2: 相関行列(heatMAp)
        →1329 iシェアーズ・コア 日経225ETF
"""
#Yの時系列変化 plot(train_df, y)

#変数間のヒートマップheatMap(train_df)


"""
3. 基本モデルの作成
    3-1: ランダムフォレスト回帰
    3-2: 予測と実数のグラフ描画
    3-4: 効果的なパラメータ確認
    3-5: 正確性

#回帰モデル作成
model, probability, test_df, train_df, x_test, x_train = module.exe_ml(x, y)

#実数と予測データの比較グラフ
#compare_view(test_df)

#効果的なパラメータ描画
#feature_imp = pd.Series(model.feature_importances_, index=x_val).sort_values(ascending=False)
#feature_view(feature_imp)

#正確性
pred = test_df['y_test_pred']
test = test_df['y_test']
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
accuracy = module.accuracy(test,pred)
#print(accuracy)
#print(r2_score(test,pred))
#print(mean_squared_error(test,pred))
"""
"""
4. パラメータ調整
    4-1: 標準化: StandardScaler
    4-2: 正規化: MinMAxScaler
    4-3: 標準化（外れ値に強い）: RobustScaler
    　→ランダムフォレスト：Scalerの効果なし
    　→SVR回帰：効果あり
    4-3: grid-search パラメータ最適

"""
#scalers_score = compare_scaler(x,y)
#print(scalers_score)

"""
5. 主成分分析
    5-1: PCAにより寄与率の高い主成分のみを説明変数に使う
    →第4第5主成分までで90％
    5-2: 第4主成分までだけで学習
"""
"""
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
feature = pca.fit_transform(x)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

model = SVR(kernel='rbf', C=1e3, gamma=0.001)

pca_pipline = Pipeline(
    [
        ('decomposition', PCA(n_components=5)),
        ('sc', StandardScaler()),
        #('model',RandomForestRegressor(n_estimators=1000,max_depth=5))
        ('model', model)
    ]
)


pca_pipline.fit(x_train,train_df['y_train'])#standardScaler
y_pred_pi = pca_pipline.predict(x_test)

#accuracy = module.accuracy(test_df['y_test'],y_pred_pi)
#print(accuracy)
#print(r2_score(test_df['y_test'],y_pred_pi))
#print(mean_squared_error(test_df['y_test'],y_pred_pi))
"""

"""
6. gridserch
    6-2: standard scaler
    6-3: pca_n
    6-4: 交差検証
    6-4: SVR

最適なパラメーター = {'svm__kernel': 'rbf', 'svm__gamma': 0.001, 'svm__C': 1000, 'pca__n_components': 7}
精度 = 0.262912424494155
Pipeline(memory=None,
         steps=[('pca',
                 PCA(copy=True, iterated_power='auto', n_components=7,
                     random_state=None, svd_solver='randomized', tol=0.0,
                     whiten=False)),
                ('ss',
                 StandardScaler(copy=True, with_mean=True, with_std=True)),
                ('svm',
                 SVR(C=1000, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
                     gamma=0.001, kernel='rbf', max_iter=-1, shrinking=True,
                     tol=0.001, verbose=False))],
         verbose=False)

最適なパラメーター = {'rf__max_depth': 50, 'rf__max_features': 2, 'rf__n_estimators': 100}
精度 = 0.158470457088444
Pipeline(memory=None,
         steps=[('pca',
                 PCA(copy=True, iterated_power='auto', n_components=7,
                     random_state=None, svd_solver='auto', tol=0.0,
                     whiten=False)),
                ('rf',
                 RandomForestRegressor(bootstrap=True, criterion='mse',
                                       max_depth=50, max_features=2,
                                       max_leaf_nodes=None,
                                       min_impurity_decrease=0.0,
                                       min_impurity_split=None,
                                       min_samples_leaf=1, min_samples_split=2,
                                       min_weight_fraction_leaf=0.0,
                                       n_estimators=100, n_jobs=None,
                                       oob_score=False, random_state=None,
                                       verbose=0, warm_start=False))],
         verbose=False)
"""
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as rfr

x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.2,
    random_state=0 #モデルの精度向上を検証するためランダムシードは一定とする
)

def select_model(model):

    pca = PCA(n_components=5)
    svr = SVR()
    ss = StandardScaler()
    rf = rfr()

    if model == 'svr':
        pip_svm = Pipeline(
            [
                ('pca',pca),
                ('ss',ss),
                #('rf',rf)
                ('svm',svr),
            ]
        )
        params_svr = {
            "pca__n_components":[i for i in range(1, 6)],
            "svm__kernel":['rbf'],
            "svm__gamma":[10**i for i in range(-4,0)],
            "svm__C":[10**i for i in range(1,4)]
        }

        return pip_svm, params_svr

    else:
        pip_rf = Pipeline(
            [
                ('pca',pca),
                ('ss',ss),
                ('rf',rf)
                #('svm',svr),
            ]
        )
        params_rf = {
            #"pca__n_components":[i for i in range(1, len(x.columns))],
            'rf__n_estimators'      : [50, 100, 300],
            'rf__max_features'      : [i for i in range(1,6)],
            'rf__random_state'      : [0],
            'rf__n_jobs'            : [-1],
            #'rf__min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
            'rf__max_depth'         : [30, 40, 50, 100]
        }

        return pip_rf, params_rf


#main
models = ['svr', 'rf']
for model in models:
    pip, params = select_model(model)

    gridS = GridSearchCV(
        pip, param_grid=params,
        scoring="r2",
        return_train_score=False,
        cv=2
    )

    gridS.fit(x_train,y_train)
    print("model_name: "+model)
    print("最適なパラメーター =", gridS.best_params_)
    print("精度 =", gridS.best_score_)
    #print(gridS.best_estimator_)
    #print(gridS.cv_results_)

"""
7.optuna
"""
#GridSearchCV()

"""
8.ベイズ最適化
"""


