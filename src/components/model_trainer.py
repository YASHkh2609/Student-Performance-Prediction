import sys
import os
from dataclasses import dataclass

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig: #specify poth for trained model to be saved
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:

            #we will first separate our input and target features
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("separated input and target features")

            #define the models

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }
            #we will create a report for all the mdoels that we have
            #evaluate model is a function present in the utils.py file
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                            models = models)
            #get the best model score
            best_model_score = max(sorted(model_report.values()))

            #get the model with the best score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("None of the models did good. Try something else")
            logging.info("Best model found for train and test data")

            #we will save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            #let us also get the r2 score for the best model
            predicted = best_model.predict(X_test)
            rSquared = r2_score(y_test, predicted)

            return (best_model_name, rSquared)
    
        except Exception as e:
            raise CustomException(e, sys)
        