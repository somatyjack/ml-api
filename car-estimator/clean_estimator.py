import pandas as pd
import numpy as np
import pickle
import os
import warnings
import sys
import pymysql

import mlflow
import mlflow.sklearn

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb


mlflow.set_tracking_uri('http://0.0.0.0:5000')
mlflow.set_experiment('CAR_EST')

def show_evaluation(y_tst,predicts):   
    print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_tst, predicts),2))
    print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_tst, predicts),2))
    print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_tst, predicts)),2))
    print('R2 Score:',round(r2_score(y_tst,predicts)*100,4))

# create connection
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='Navigator16@',
                             db='app_core')

# Create cursor
my_cursor = connection.cursor()

my_cursor.execute("SELECT * from car_make")
car_makes = my_cursor.fetchall()

my_cursor.execute("SELECT * from car_model")
car_models = my_cursor.fetchall()


# Close the connection
connection.close()

make_map = {}
model_map = {}
for car in car_makes:
    make_map[car[2]] = car[0]
for car in car_models:
    model_map[car[3]] = car[0]
    
fuel_map = {}    
gearbox_map = {}    


if __name__ == "__main__":

    dataset=sys.argv[1]

    columns = ['make','model','descr','year','engine_size','odo','posted_time','city','price']
    new_df = pd.read_csv(dataset,sep=" ",names=columns)

    df=pd.read_csv('data/clean/set.csv')

    # convert categories to ids
    fuel_arr = df['fuel'].unique()
    fuel_id = 1
    for type in fuel_arr:
        fuel_map[type] = fuel_id
        fuel_id += 1

    gear_arr = df['transmission'].unique()
    gear_id = 1
    for type in gear_arr:
        gearbox_map[type] = gear_id
        gear_id += 1

    new_df['make'] = new_df['make'].apply(lambda make: make.lower())
    new_df['make'] = new_df['make'].apply(lambda make: make.replace(" ",'_'))
    new_df['make'] = new_df['make'].apply(lambda make: make.replace("-",'_'))
    new_df['model'] = new_df['model'].apply(lambda make: make.lower())
    new_df['model'] = new_df['model'].apply(lambda make: make.replace(" ",'_'))
    new_df['model'] = new_df['model'].apply(lambda make: make.replace("-",'_'))

    # temp, until make model db data is cleaned
    new_df.loc[new_df[new_df['make'] == 'mercedes_benz'].index,'make'] = 'mercedes'
    new_df.loc[new_df[new_df['make'] == 'ssangyong'].index,'make'] = 'ssang_yong'

    new_df['make'] = new_df['make'].apply(lambda make: int(make_map[make]))
    new_df['model'] = new_df['model'].apply(lambda model: int(model_map[model]))
    new_df['transmission'] = new_df['transmission'].apply(lambda gearbox: int(gearbox_map[gearbox]))
    new_df['fuel'] = new_df['fuel'].apply(lambda fuel: int(fuel_map[fuel]))

    # do cross validation
    X = df.drop(['price'],axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    #normalize values
    sc = MinMaxScaler()   
    X_train= sc.fit_transform(X_train)
    X_test= sc.transform(X_test)

    # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
    with mlflow.start_run(run_name="Run_1"):

        forest = RandomForestRegressor()
        forest.fit(X_train,y_train)
        predictions = forest.predict(X_test)

        mae=metrics.mean_absolute_error(y_test, predictions)
        mse = metrics.mean_squared_error(y_test, predictions)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

        print('Mean Absolute Error (MAE):', mae)
        print('Mean Squared Error (MSE):', mse)
        print('Root Mean Squared Error (RMSE):', rmse)
        mape = np.mean(np.abs((y_test - predictions) / np.abs(y_test)))
        print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
        print('Accuracy:', round(100*(1 - mape), 2))
                
        # The default path where the MLflow autologging function stores the Keras model
        model_name = "cat-price-estimator-model" # Replace this with the name of your registered model, if necessary.
        artifact_path = "model"
        run_id = mlflow.active_run().info.run_id
        print(f"runId:{run_id}")
        
        #model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
        #model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

        #log metrics
        mlflow.log_metric("MAE",mae)
        mlflow.log_metric("MSE",mse)
        mlflow.log_metric("RMSE",rmse)

        # Publish the model
        mlflow.sklearn.log_model(forest, "model")
        #modelpath = "/home/jack/mlruns/"+run_id
        #mlflow.sklearn.save_model(forest, modelpath)      

        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    mlflow.end_run()

def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True