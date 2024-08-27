import numpy as np
import pandas as pd
import os

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score
from sklearn.model_selection import train_test_split

import argparse

# Reading the data:
def get_data():
    try:
        df = pd.read_csv('//Users//shrish//Downloads//wine+quality//winequality-white.csv',sep=';')
        return df
    except Exception as e :
        raise e

def evaluate(y_true,y_pred):
    '''
    mse = mean_squared_error(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    r2 = r2_score(y_true,y_pred)
    return mse, mae, rmse, r2
    '''
    acc_sc = accuracy_score(y_true,y_pred)
    return acc_sc




def main(n_estimators , max_depth):
    df = get_data()

    # Dara=Distribution

    train , test = train_test_split(df)
    x_train = train.drop(['quality'],axis=1)
    x_test = test.drop(['quality'],axis=1)

    y_train = train[['quality']]
    y_test = test[['quality']]

    # Model-Implementation

    '''

    lr = ElasticNet()
    lr.fit(x_train,y_train)
    pred = lr.predict(x_test)

    '''
    with mlflow.start_run():

        rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
        rf.fit(x_train,y_train)
        pred = rf.predict(x_test)



        # Evaluation:
        #mse , mae , rmse , r2 = evaluate(y_test,pred)
        acc_sc = evaluate(y_test,pred)

        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("max_depth",max_depth)

        mlflow.log_metric("acc_sc",acc_sc)


        #print(f"Mean Squared Error : {mse} , Mean Absolute Error : {mae} , Root Mean Squared Error : {rmse} , R^2 Error : {r2}")
        print(f"Accuracy Score : {acc_sc}")




if __name__ == "__main__" :
    args = argparse.ArgumentParser()
    args.add_argument("--n_estimators",'-n',default=100,type=int)
    args.add_argument("--max_depth",'-m',default=10,type=int)
    parse_args = args.parse_args()
    try:
        main(n_estimators=parse_args.n_estimators,max_depth=parse_args.max_depth)
    except Exception as e :
        raise e
    

