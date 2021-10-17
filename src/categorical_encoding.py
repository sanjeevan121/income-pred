import os
import pandas as pd
import numpy as mp
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder,StandardScaler
from get_data import read_params
import pickle 
import argparse

def map_workclass(df):
    workclass_mapper = {
    ' State-gov': "other",
    ' Self-emp-not-inc': "other",
    ' Federal-gov': "other",
    ' Local-gov': "other",
    ' ?': "other",
    ' Self-emp-inc': "other",
    ' Without-pay': "other",
    ' Never-worked': "other"
    }
    df["workclass"] = df.workclass.map(workclass_mapper).fillna(df["workclass"])
    return df
    
def map_education(df):
    education_mapper ={' 11th': 'other',
    ' Masters': 'other',
    ' 9th': 'other',
    ' Assoc-acdm': 'other',
    ' Assoc-voc': 'other',
    ' 7th-8th': 'other',
    ' Doctorate': 'other',
    ' Prof-school': 'other',
    ' 5th-6th': 'other',
    ' 10th': 'other',
    ' 1st-4th': 'other',
    ' Preschool': 'other',
    ' 12th': 'other'}

    df['education'] = df['education'].map(education_mapper).fillna(df['education'])
    return df
    
def map_marital_status(df):
    marital_status_mapper = {' Divorced': 'other',
   ' Married-spouse-absent': 'other',
   ' Separated': 'other',
   ' Married-AF-spouse': 'other',
   ' Widowed': 'other'}

    df['marital-status'] = df['marital-status'].map(marital_status_mapper).fillna(df['marital-status'])
    return df

def map_occupation(df):
    occupation_mapper = {
    ' ?': 'Prof-specialty',
    ' Protective-serv': 'other',
    ' Armed-Forces': 'other',
    ' Priv-house-serv': 'other',
    ' Tech-support': 'other',
    ' Farming-fishing': 'other',
    ' Handlers-cleaners': 'other'
    }

    df['occupation'] = df['occupation'].map(occupation_mapper).fillna(df["occupation"])
    return df


def encode_categories(config_path):
    try:
        config=read_params(config_path)
        raw_data_path=config['load_data']['raw_dataset_csv']
        encoder_file=config['transformers']['base_n_encoder']
        df=pd.read_csv(raw_data_path,sep=',')
        
        map_workclass(df)
        map_education(df)
        map_marital_status(df)
        map_occupation(df)
        data_categorical=df[['workclass','education','marital-status','occupation','relationship','race','sex','country']]
    
       
        base_n=loaded_model = pickle.load(open(encoder_file, 'rb'))
        data_categorical=pd.DataFrame(base_n.transform(data_categorical))
        
        return data_categorical
       
    
    except Exception as e:
        print(e)


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    default_config=os.path.join('config','params.yaml')
    args.add_argument('--config',default=default_config)
    args.add_argument('--datasource',default=None)
    parsed_args = args.parse_args()
    encode_categories(config_path=parsed_args.config)


