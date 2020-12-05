class Config(object):
    DEBUG = False
    PROD = False

class ProductionConfig(Config):
    DB_USER='root'
    DB_PASS='root'
    DB_HOST='localhost'
    DB_PORT="3306"
    DB_NAME='mlflow_tracking'
    PROD=True

class StagingConfig(Config):
    DB_USER='root'
    DB_PASS='root'
    DB_HOST='localhost'
    DB_PORT="3306"
    DB_NAME='mlflow_tracking'