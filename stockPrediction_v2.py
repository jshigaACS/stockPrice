import matplotlib.pyplot as plt
import seaborn as sns
import module
from module import create_model_split,create_model_cv
import pandas as pd
import numpy as np

"""
変更点：目的変数を「偏差値」,
        説明変数に「曜日」を追加
"""

plt.rcParams['figure.figsize'] = (15.0, 15.0)



def read_csv2df():
    import pandas as pd

    dirpath = '/home/ubuntu/analysis/stockPrice/'
    csv_name = 'stockPriceData_dev_val.csv'

    df = pd.read_csv(dirpath+csv_name)
    return df

def setting_dummy(train_df):

    d = pd.get_dummies(train_df['week_d'])
    train_df['price_mon']=d['月']
    train_df['price_tue']=d['火']
    train_df['price_wed']=d['水']
    train_df['price_thu']=d['木']
    train_df['price_fri']=d['金']
    train_df=train_df.drop(columns='week_d')

    #print(train_df)

    return train_df

def setting(train_df):
    #y_val = '富士通(株)：翌日比'
    y_val = '富士通(株)：翌日比偏差値'
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

def feature_view(feature_imp,col):

    sns.barplot(x=feature_imp, y=col)
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
train_df = setting_dummy(train_df)


#1-2
y_val, nonNeedCol, x_val = setting(train_df)

x = train_df[x_val]
y = train_df[y_val]

#print(x.columns)

"""
2. 事前調査
    単純なGridSearch
"""
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
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
rf = rfr()
rf.fit(x_train,y_train)
feature_imp = rf.feature_importances_ 
col = x.columns
feature_view(feature_imp, col)

fi_df = pd.DataFrame(
    {
        'feature': list(col),
        'feature importance': feature_imp[:]
    }
).sort_values('feature importance', ascending = False)

#特徴量の少ない変数を除外
XX=0.01
use_features = fi_df.loc[fi_df['feature importance'] >= XX,'feature']
x2 = x[use_features]

"""PCA=6
pca = PCA()
svr = SVR()
scaler = StandardScaler()
pip_svm = Pipeline(
    [
        ('pca',pca),
        ('scaler',scaler),
        ('svr',svr),
    ]
)
params_svr = {
    "pca__n_components":[i for i in range(1, len(x2.columns))],
}

gridS = GridSearchCV(
    pip_svm,
    param_grid=params_svr,
    scoring="r2",
    #scoring=my_scorer,
    return_train_score=False,
    cv=3
)

gridS.fit(x_train,y_train)
#print("model_name: "+str())
print("最適なパラメーター =", gridS.best_params_)
print("精度 =", gridS.best_score_)
"""




"""
6. gridserch
    6-2: standard scaler
    6-3: pca_n
    6-4: 交差検証
    6-4: SVR

"""
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as rfr

x_train, x_test, y_train, y_test = train_test_split(
    x2, 
    y, 
    test_size=0.2,
    random_state=0 #モデルの精度向上を検証するためランダムシードは一定とする
)

def select_model(model,scaler):

    pca = PCA()
    svr = SVR()
    rf = rfr()

    if model == 'svr':
        
        pip_svm = Pipeline(
            [
                ('pca',pca),
                #('rf',rf)
                ('scaler',scaler),
                ('svm',svr),
            ]
        )
        params_svr = {
            "pca__n_components":[i for i in range(1, 7)],
            "svm__kernel":['rbf','poly','sigmoid'],
            "svm__gamma":[10**i for i in range(-4,0)],
            "svm__C":[10**i for i in range(1,4)],
        }

        return pip_svm, params_svr
    
    else:
        pip_rf = Pipeline(
            [
                ('pca',pca),
                #('scaler',scaler),
                ('rf',rf)
                #('svm',svr),
            ]
        )
        params_rf = {
            "pca__n_components":[7],
            'rf__n_estimators'      : [50],# 100],#, 300],,
            'rf__max_features'      : [i for i in range(1,7)],
            'rf__random_state'      : [0],
            'rf__n_jobs'            : [-1],
            #'rf__min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
            'rf__max_depth'         : [30]#[30, 40, 50]#, 100]
        }
        return pip_rf, params_rf

def calc_score(y_test, y_pred):
    from sklearn import metrics

    testUpDown = []
    for test in y_test:
        if test > 50:
            testUpDown.append(1)
        else:
            testUpDown.append(-1)
    predUpDown = []
    for pred in y_pred:
        if pred > 50:
            predUpDown.append(1)
        else:
            predUpDown.append(-1)

    str_prob = metrics.accuracy_score(testUpDown, predUpDown)*100
    prob = metrics.accuracy_score(testUpDown, predUpDown)
    
    return str_prob

# 誤差関数の定義
def root_mean_squared_error(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_true, y_pred))

def median_absolute_error_rate(y_true, y_pred):
    return np.median(np.absolute(y_true - y_pred) / y_true)


#main処理
models = ['rf']#['svr']#['rf']#['svr','rf']

from sklearn.metrics.scorer import make_scorer

for model in models:
    if model == 'svr':
        scalers = ['ss','mm','rs']
        for scaler in scalers:
            if scaler == 'ss':
                s = StandardScaler()
            elif scaler == 'mm':
                s = MinMaxScaler()
            else:
                s = RobustScaler()

            pip, params = select_model(model, s)
            gridS = GridSearchCV(
            pip, param_grid=params,
            scoring="r2",
            #scoring=score,
            #scoring=my_scorer,
            return_train_score=False,
            cv=2
            )

            gridS.fit(x_train,y_train)
            print("model_name: "+model)
            print("最適なパラメーター =", gridS.best_params_)
            print("精度 =", gridS.best_score_)

    else:
        pip, params = select_model(model,scaler=None)
        #scores = ['calc_score','root_mean_squared_error','median_absolute_error_rate']
        #for score in scores:
            
        #my_scorer = make_scorer(score, greater_is_better=True)

        gridS = GridSearchCV(
            pip, param_grid=params,
            scoring="r2",
            #scoring=score,
            #scoring=my_scorer,
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


