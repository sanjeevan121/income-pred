import json
import logging
import os
import joblib
import pytest
import prediction_service
from prediction_service.prediction import form_response, api_response
import category_encoders as ce
import scipy.stats as stat
import category_encoders as ce
import scipy.stats as stat
from src import categorical_encoding
from src import numerical_scaling

"""errors
FAILED tests/test_config.py::test_form_response_correct_range - AttributeError: 'NoneType' object has no attr...
FAILED tests/test_config.py::test_api_response_correct_range - AssertionError: assert ({'response': 'Not in c...
FAILED tests/test_config.py::test_api_response_incorrect_col - AssertionError: assert 'Values enter...xpected...

"""


input_data = {"num_incorrect_range":
        {"age":14,
        "fnlwgt":10000,
        "education-num":20,
        "capital-gain":150000,
        "capital-loss":6000,
        "hours-per-week":120},

    "num_correct_range":
    {"age":20,
    "fnlwgt":20000,
    "education-num":10,
    "capital-gain":50000,
    "capital-loss":2000,
    "hours-per-week":45},

    "cat_correct_val":
    {
    "workclass": " Private",
    "education": " HS-grad",
    "marital-status": " Never-married",
    "occupation": " Prof-specialty ",
    "relationship": " Husband",
    "race": " White",
    "sex": " Male",
    "country": " United-States"
    },

    "incorrect_col":
    {'age': 5,
    'fnlwgt': 5,
    'education num': 5,
    'capital gain': 5,
    'capital loss': 5,
    'hours-per-week': 5,
    'workclass': "some_string",
    'education': "some_string",
    'marital status': "some_string",
    'occupation': "some_string",
    'relationship': "some_string",
    'race': "some_string",
    'sex': "some_string",
    'country': "some_string"
}
}

TARGET_value = {
    "min": 0,
    "max": 1
}


# e
def test_form_response_correct_range(data=input_data["num_correct_range"].update(input_data['cat_correct_val'])):
    res = form_response(data)
    assert (res==TARGET_value["min"] or res==TARGET_value["max"])

# e
def test_api_response_correct_range(data=input_data["num_correct_range"]):
    res = api_response(data)
    assert (res==TARGET_value["min"] or res==TARGET_value["max"])

def test_form_response_incorrect_range(data=input_data["num_incorrect_range"]):
    with pytest.raises(prediction_service.prediction.NotInRange):
        res=form_response(data)

def test_api_response_incorrect_range(data=input_data["num_incorrect_range"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInRange().message
# e
def test_api_response_incorrect_col(data=input_data["incorrect_col"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInCols().message

