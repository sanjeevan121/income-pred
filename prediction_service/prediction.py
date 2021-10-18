import yaml
import os
import json
import joblib
import numpy as np
from src.categorical_encoding import encode_categories
from src.numerical_scaling import scale_data
import pandas as pd
import numpy as np
import category_encoders as ce
import scipy.stats as stat
from src.get_data import read_params


params_path = "config/params.yaml"
schema_num_path = os.path.join("prediction_service", "schema_num.json")
schema_cat_path = os.path.join("prediction_service", "schema_cat.json")
schema_val_num_range_path=os.path.join("prediction_service", "schema_in.json")

class NotInRange(Exception):
    def __init__(self, message="Values entered are not in expected range"):
        self.message = message
        super().__init__(self.message)

class NotInCols(Exception):
    def __init__(self, message="Not in cols"):
        self.message = message
        super().__init__(self.message)



def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def prepare_data(config_path=params_path):
    encoder_file=config_path['transformers']['base_n_encoder']
    cat=encode_categories(config_path)
    num=scale_data(config_path)
    data_cat=np.array(cat)
    data_num=np.array(num)
    clean_data=np.c_[data_num,data_cat]
    df=pd.DataFrame(clean_data)
    for i in range(5,27):
        df[i]=df[i].apply(lambda x: int(x))
    return df
    
    
    
def predict(data):
    config = read_params(params_path)
    data=prepare_data(data)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data)
    

def get_schema(schema_path=schema_val_num_range_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema

def validate_input(dict_request):
    def _validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInCols

    def _validate_values(col, val):
        schema = get_schema()

        if not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]) :
            raise NotInRange

    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col, val)
    
    return True

#webapp response
def form_response(dict_request):
    if validate_input(dict_request):
        data = dict_request.values()
        for i in range(len(data)):
            data_num = [list(map(float,data))]
        response = predict(data)
        return response

#api response eg postman
def api_response(dict_request):
    try:
        if validate_input(dict_request):
            data = np.array([list(dict_request.values())])
            response = predict(data)
            response = {"response": response}
            return response
            
    except NotInRange as e:
        response = {"the_exected_range": get_schema(), "response": str(e) }
        return response

    except NotInCols as e:
        response = {"the_exected_cols": get_schema().keys(), "response": str(e) }
        return response


    except Exception as e:
        response = {"response": str(e) }
        return response