#this file will contain functions that will/can be used in the entire project
import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_object:
            pickle.dump(obj, file_object)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test, y_test, models, params):
    try:
        report ={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]

            # model.fit(X_train, y_train) # Train model
            gs = GridSearchCV(model,param,cv=5)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate Train and Test dataset
            model_train_r2 = r2_score(y_train,y_train_pred)

            model_test_r2 = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = model_test_r2

            return report
        
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    