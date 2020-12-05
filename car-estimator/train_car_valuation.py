import pandas as pd
import numpy as np
import pickle
import os
import warnings
import sys

# Custom
import db_conn

# ML
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb

#mlflow.set_tracking_uri(uri)
mlflow.set_tracking_uri('http://0.0.0.0:5000')
mlflow.set_experiment('CAR_EST')

def show_evaluation(y_tst,predicts):   
    mae = round(metrics.mean_absolute_error(y_tst, predicts),2)
    mse = round(metrics.mean_squared_error(y_tst, predicts),2)
    rmse = round(np.sqrt(metrics.mean_squared_error(y_tst, predicts)),2)
    r_squared = round(metrics.r2_score(y_tst,predicts)*100,4)

    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse)
    print('R2 Score:',r_squared)

    return [mae,mse,rmse,r_squared]


# fetch necessary records
db = db_conn.Connection(
    '127.0.01',
    '3306',
    'root',
    'root',
    'app_core')

make_map = db.getMakeMap()
model_map = db.getModelMap()
fuel_map = {}    
gearbox_map = {}  
del db

if __name__ == "__main__":

    dataset=sys.argv[1]

    df = pd.read_csv(dataset)

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

    def model_mapper(make,model):
        return int(model_map[make + ' ' + model])

    new_model_ids = []
    for index, row in df.iterrows():
        new_model_ids.append(model_mapper(row['make'],row['model']))
    df['model'] = new_model_ids

    df['make'] = df['make'].apply(lambda make: int(make_map[make]))

    df['transmission'] = df['transmission'].apply(lambda gearbox: int(gearbox_map[gearbox]))
    df['fuel'] = df['fuel'].apply(lambda fuel: int(fuel_map[fuel]))
    
    # do cross validation
    X = df.drop(['price'],axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    #normalize values
    scaler = MinMaxScaler()   
    X_train= scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)

    # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
    with mlflow.start_run(run_name="Run_1"):

        #model implementation and fitting data
        xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.3,
                        max_depth = 24, alpha = 3, n_estimators = 200)
        xg_reg.fit(X_train,y_train)
        pred = xg_reg.predict(X_test)

        scores = show_evaluation(y_test,pred)

        # The default path where the MLflow autologging function stores the Keras model
        model_name = "cat-price-estimator-model" # Replace this with the name of your registered model, if necessary.
        artifact_path = "model"
        run_id = mlflow.active_run().info.run_id
        print(f"runId:{run_id}")
        
        #model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
        #model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

        #log metrics
        mlflow.log_metric("MAE",scores[0])
        mlflow.log_metric("MSE",scores[1])
        mlflow.log_metric("RMSE",scores[2])
        mlflow.log_metric("R2",scores[3])

        # Publish the model
        mlflow.sklearn.log_model(xg_reg, "main_model")
        mlflow.sklearn.log_model(scaler, "scaler_model")
        
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    mlflow.end_run()