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
"""
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
print(accuracy)
print(r2_score(test,pred))
print(mean_squared_error(test,pred))
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
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
pca = PCA()
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

#plt.figure()
pca_pipline.fit(x_train,train_df['y_train'])
y_pred_pi = pca_pipline.predict(x_test)
#print(pca_pipline.score(x_train,train_df['y_train']))
accuracy = module.accuracy(test_df['y_test'],y_pred_pi)
print(accuracy)
print(r2_score(test_df['y_test'],y_pred_pi))
print(mean_squared_error(test_df['y_test'],y_pred_pi))
#plt.plot(test_df['y_test'])
#plt.plot(y_pred_pi)

"""
6. gridserch
    6-1: importance_feature
    6-2: standard scaler
    6-3: pca
    6-4: 交差検証
    6-4: svr
"""


