import os
import sys

import pandas as pd
import numpy as np

from src.exception import CustomException
import pickle


def save_object(obj, file_path):
    """
    Save the object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

            
    except Exception as e:
        raise CustomException(e, sys)