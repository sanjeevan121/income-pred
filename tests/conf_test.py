import pytest
import yaml
import os
import json
from src.categorical_encoding import encode_categories
from src.numerical_scaling import scale_data
import category_encoders as ce
import scipy.stats as stat

@pytest.fixture
def config(config_path="config/params.yaml"):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

@pytest.fixture
def schema_in(schema_path="schema_in.json"):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema