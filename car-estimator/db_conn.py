
import pymysql

class Connection:
    def __init__(self,host,port,user,pwd,db_name):
        # create connection
        self.connection = pymysql.connect(host=host,
                                         port=int(port),
                                         user=user,
                                         password=pwd,
                                         db=db_name)
        self.db_cursor = self.connection.cursor()
        
    def __del__(self):
        # Close the connection
        self.connection.close()

    def getCarMakes(self):
        self.db_cursor.execute("SELECT * from car_make")
        car_makes = self.db_cursor.fetchall()
        return car_makes

    def getCarModels(self):
        self.db_cursor.execute("SELECT model.id,make.code as 'make_code',model.name,model.code from app_core.car_model model INNER JOIN app_core.car_make make ON (model.make_id = make.id)")
        car_models = self.db_cursor.fetchall()
        return car_models

    def getCarEngineTypes(self):
        self.db_cursor.execute("SELECT * FROM app_core.car_engine_type")
        car_engine_types = self.db_cursor.fetchall()
        return car_engine_types

    def getCarEngineSizes(self):
        self.db_cursor.execute("SELECT * FROM app_core.car_engine_size")
        car_engine_sizes = self.db_cursor.fetchall()
        return car_engine_sizes

    """
    ------------------------------
    Mappings
    ------------------------------
    """

    def getMakeMap(self):
        self.db_cursor.execute("SELECT * from car_make")
        car_makes = self.db_cursor.fetchall()
        
        make_map = {}
        for make in car_makes:
            make_map[make[2]] = make[0]

        return make_map
  
    def getModelMap(self):
        self.db_cursor.execute("SELECT model.id,make.code as 'make_code',model.name,model.code from app_core.car_model model INNER JOIN app_core.car_make make ON (model.make_id = make.id)")
        car_models = self.db_cursor.fetchall()
        
        model_map = {}

        for model in car_models:
            model_map[model[1] + ' ' + model[3]] = model[0] 

        return model_map

    def getEngineTypeMap(self):
        self.db_cursor.execute("SELECT * FROM app_core.car_engine_type")
        car_engine_types = self.db_cursor.fetchall()
        
        engine_type_map = {}
        for engine_type in car_engine_types:
            engine_type_map[engine_type[1]] = engine_type[0]
            
        return engine_type_map

    def getEngineSizeMap(self):
        self.db_cursor.execute("SELECT * FROM app_core.car_engine_size")
        car_engine_sizes = self.db_cursor.fetchall()
        
        engine_size_map = {}
        for engine_size in car_engine_sizes:
            engine_size_map[engine_size[1]] = engine_size[0]

        return engine_size_map

    def getTransmissionMap(self):
        self.db_cursor.execute("SELECT * FROM app_core.car_transmission")
        car_trans = self.db_cursor.fetchall()
        
        transmission_map = {}
        for trans in car_trans:
            transmission_map[trans[1]] = trans[0]

        return transmission_map        
