import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #used to create pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    #path for saving the transformed data model
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
            This function is for data tranformation
        '''
        try:
            #defining the features
            numerical_features = ["reading score", "writing score"]
            categorical_features = ["gender",
                                    "race/ethnicity",
                                    "parental level of education",
                                    "lunch",
                                    "test preparation course"]
            
            #pipeline for transformations on numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            #pipeline for transformations on categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"categorical features: {categorical_features}")
            logging.info(f"categorical features: {numerical_features}")

            #combines the 2 pipelines for different types of features
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")

            preprocessing_obj = self.get_data_transformer_object()
            logging.info('Obtained the preprocessing object')

            target_feature = "math score"

            input_feature_train_df = train_df.drop(columns=[target_feature], axis=1) 
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop(columns=[target_feature], axis=1) 
            target_feature_test_df = test_df[target_feature]

            logging.info("Separated input and target features for train and test dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            #np.c_ is used  to combine each set of inputs to respective outputs
            # np.c_[np.array([1,2,3]), np.array([4,5,6])]
            #     array([[1, 4],
            #            [2, 5],
            #           [3, 6]])

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] 
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(#this function is function is used to save models and is written in utils.py
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("saved model successfully")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
