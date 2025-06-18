import sys
import pandas as pd 

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object

class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            model_path = 'artifacts/model.pkl'

            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)

            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
        

class Customdata:

    """Responsible for getting the input data from the user and converting it into a pandas dataframe"""

    def __init__(self,
                 gender: str,
                 race_enthicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 ):
        
        self.gender = gender

        self.race_enthicity =  race_enthicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {

                'gender': [self.gender],
                'race_enthicity': [self.race_enthicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course]

            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
