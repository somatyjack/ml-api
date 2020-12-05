import flask
import pickle
import os
import numpy as np

# Custom
import db_conn

# instantiate flask 
app = flask.Flask(__name__)

MODELS_DIR = os.environ["MODELS_DIR"]
MAIN_MODEL_FILE = os.environ["MAIN_MODEL_FILE"]
SCALER_MODEL_FILE = os.environ["SCALER_MODEL_FILE"]

PORT = os.environ["PORT"]

DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]
DB_USER = os.environ["DB_USER"]
DB_PWD = os.environ["DB_PWD"]
DB_NAME = os.environ["DB_NAME"]

"""
MODELS_DIR = '/home/jack/mlruns/1/a29ec0b98f5a4cfd8eff3e0675053780/artifacts'
MAIN_MODEL_FILE = 'main_model/model.pkl'
SCALER_MODEL_FILE = 'scaler_model/model.pkl'
PORT = '5050'
DB_HOST = '127.0.0.1'
DB_PORT = 3306
DB_USER = 'root'
DB_PWD = 'root'
DB_NAME = 'app_core'
"""

MAIN_MODEL_PATH = os.path.join(MODELS_DIR, MAIN_MODEL_FILE)
SCALER_MODEL_PATH = os.path.join(MODELS_DIR, SCALER_MODEL_FILE)


# Load models from file
with open(MAIN_MODEL_PATH, 'rb') as file:
    main_model = pickle.load(file)

with open(SCALER_MODEL_PATH, 'rb') as file:
    scaler_model = pickle.load(file)

# fetch necessary records

db = db_conn.Connection(
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PWD,
    DB_NAME)

car_makes_map = db.getMakeMap()
car_models_map = db.getModelMap()
engine_types_map = db.getEngineTypeMap()
engine_sizes_map = db.getEngineSizeMap()
gearboxes_map = {
    'manual':1,
    'automatic':2,
    'semi_auto':3
}
del db


@app.route("/car-estimator/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x = params['values']
        # remap contegorical values
        try:
            make_id = car_makes_map[x[0]]
            model_id = car_models_map[x[0] + ' ' + x[1]]
            engine_type_id = engine_types_map[x[4]] 
            
            engine_size = int(x[5])
            engine_size_id = engine_sizes_map[engine_size] 
            
            car_year = int(x[2])
            car_odo = float(x[3])
            
            gear_type = (x[6]).lower().replace(' ','_')
            gearbox_id = gearboxes_map[gear_type]
        except Exception as e:
            print(e)
            data = {"success": False,'error':"invalid parameters"}
            return flask.jsonify(data)   

        x = [[
            make_id,
            model_id,
            car_year,
            car_odo,
            engine_type_id,
            engine_size_id,
            gearbox_id
        ]]
        
        x = scaler_model.transform(x)

        x = np.array(x)
        data["prediction"] = str(main_model.predict(x)[0])
        data["success"] = True

    # return a response in json format 
    return flask.jsonify(data)        

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(PORT))