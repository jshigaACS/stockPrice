import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (15.0, 15.0)


def read_csv2df():
    import pandas as pd

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


def exe_prediction():
    import subprocess as sp

    cmd = 'python3'
    program = 'prediction.py'
    #_dir = '/home/ubuntu/analysis/stockPrice/'
    command = [cmd,program]
    #print(command)
    sp.check_call(command)

def plot(train_df):
    plt.plot(train_df['date'],y, marker='o')
    plt.plot(train_df['date'],y, marker='X')
    plt.xlabel('date')
    plt.ylabel('stockVal_ratio')
    plt.tight_layout()
    plt.show()
    plt.figure()

def hist(df):
    df.hist(figsize=(12,12))

def heatMap(df):
    corr_mat = df.corr(method='pearson')
    sns.heatmap(
        corr_mat, 
        vmax=1.0, 
        vmin=-1.0,
        center=0, 
        fmt='.1ft',    
    )

"""
1. 前準備
    1-1: Load CSV
    1-2: setting
"""
#1-1
train_df = read_csv2df()

#1-2
y_val, nonNeedCol, x_val = setting()

x = train_df[x_val]
y = train_df[y_val]

"""
2. 事前調査
    2-1: Yの時系列変化
        →乱高下が激しい、、、
    2-2: histgram
        →正規分布、、、
    2-2: 相関行列(heatMAp)
        →1329 iシェアーズ・コア 日経225ETF
"""
#Yの時系列変化 plot(train_df)

#hist
hist(train_df)

#独立変数間のヒートマップ
heatMap(train_df)

#exe_prediction()




