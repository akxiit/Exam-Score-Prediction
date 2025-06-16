import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')



# class TotalScoreAdder(BaseEstimator, TransformerMixin):

#     """
#         This class is responsible for adding a new column 'total_score'
#         which is the sum of the scores in the specified columns.
    
#     """


#     def __init__(self, columns):
#         self.columns = columns

#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         X = X.copy()
#         X['total_score'] = X[self.columns].sum(axis=1) / 3
#         return np.round(X, 2) 



class DataTransformation:

    """
        This function is responsible for transforming the data
        It will handle the numerical and categorical columns
    
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_cols = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_cols = ['math_score', 'reading_score', 'writing_score']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                # ('total_score_adder', TotalScoreAdder(columns=num_cols)),
                ('scaler', StandardScaler())
            ])
            logging.info('Done with numerical column and also add new Total_score column.')

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))

            ])

            logging.info('Categorical Column Encoded')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, categorical_cols)
                ]
            )

            logging.info('Preprocessor object created')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')


            numerical_columns = ['math_score', 'reading_score', 'writing_score']

            train_df['total_score'] = np.round((train_df[numerical_columns].sum(axis=1) / 3) , 2)
            test_df['total_score'] = np.round((test_df[numerical_columns].sum(axis=1) / 3) , 2)

            logging.info('Added total_score column to train and test dataframes')
            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = 'total_score'


            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on training and testing dataframes')

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object.')

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj

            )

            logging.info('Preprocessor object saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)