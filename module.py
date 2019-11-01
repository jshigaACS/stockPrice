from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import svm

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

import pandas as pd

class create_model_split():

    def __init__(self, x_train,x_test,y_train,y_test):
        self.y_train = y_train
        self.y_test = y_test
        model = RandomForestRegressor(n_estimators=100)        
        y_test_pred, y_train_pred, model = self._create_model_split(x_train,x_test,y_train,y_test, model)

        self.y_test_pred = y_test_pred
        self.y_train_pred = y_train_pred
        self.model = model

    def _create_model_split(self, x_train,x_test,y_train,y_test, model):
        model1 = model
        model1.fit(x_train,y_train)
        y_test_pred = model1.predict(x_test)
        #y_test_ = y_test.values.tolist()
        y_train_pred = model1.predict(x_train)
        #y_train_ = y_train.values.tolist()

        return y_test_pred, y_train_pred, model1

    def _get_r2_score(self):
        return r2_score(self.y_test,self.y_test_pred), r2_score(self.y_train,self.y_train_pred)
        

    def _get_rmse(self):
        return np.sqrt(mean_squared_error(self.y_test,self.y_test_pred)), np.sqrt(mean_squared_error(self.y_train,self.y_train_pred))


class create_model_cv():

    def __init__(self, x,y):
        model = RandomForestRegressor(n_estimators=1000, max_depth=100)        
        #model = svm.SVR(kernel='rbf',C=1)
        kf = KFold(n_splits=3, shuffle=True, random_state=1)

        model2 = model
        self.test_R2_scores = []
        self.test_RMSE_Scores = []

        self.train_R2_scores = []
        self.train_RMSE_Scores = []

        for train_index, test_index in kf.split(x):
            #x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            x_train, x_test = x[train_index], x[test_index]
            #y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model2 = model2.fit(x_train,y_train)
            #test
            y_test_pred = model2.predict(x_test)

            self.test_R2_scores.append(r2_score(y_test,y_test_pred))
            self.test_RMSE_Scores.append(np.sqrt(mean_squared_error(y_test,y_test_pred)))

            #train
            y_train_pred = model2.predict(x_train)

            self.train_R2_scores.append(r2_score(y_train,y_train_pred))
            self.train_RMSE_Scores.append(np.sqrt(mean_squared_error(y_train,y_train_pred)))


    def get_train_res(self):
        return np.mean(self.train_R2_scores), np.mean(self.train_RMSE_Scores)

    def get_test_res(self):
        return np.mean(self.test_R2_scores), np.mean(self.test_RMSE_Scores)



def accuracy(y_test, y_pred):
    from sklearn import metrics

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

    str_prob = str(metrics.accuracy_score(testUpDown, predUpDown)*100)
    prob = metrics.accuracy_score(testUpDown, predUpDown)
    return str_prob, prob

def exe_ml(x,y):

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
    split_probability, _ = accuracy(y_test, y_test_pred)
    y_train_pred = cms.y_train_pred
    
    test_df = pd.DataFrame(
        {
            'y_test_pred': y_test_pred,
            'y_test': y_test.tolist(),     
        }
    )
    
    train_df = pd.DataFrame(
        {
            'y_train_pred': y_train_pred,
            'y_train': y_train.tolist(),
            #'x_train': x_train.values,     
        }
        #index=y_train.index
    )

    #return model, split_probability,y_test_df, y_train_df
    return model, split_probability,test_df,train_df, x_test, x_train
    
    #plotter(y_train_pred,y_train,y_test_pred,y_test)


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
    
