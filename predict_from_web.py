import numpy as np
import scipy.stats as stat
import joblib
import pickle
import pandas as pd

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


def encode_categories(data):
   
    encoder_file="transformers/baseN_encoder.pkl"
    df=pd.DataFrame(data)

    map_workclass(df)
    map_education(df)
    map_marital_status(df)
    map_occupation(df)
    data_categorical=df[['workclass','education','marital-status','occupation','relationship','race','sex','country']]


    base_n=loaded_model = pickle.load(open(encoder_file, 'rb'))
    data_categorical=pd.DataFrame(base_n.transform(data_categorical))

    return data_categorical



def scale_data(df):
    
    scaler_file="transformers/standard_scaler.pkl"
    data_numerical=df[['age','education-num','capital-gain','capital-loss','hours-per-week']]
    sc=pickle.load(open(scaler_file, 'rb'))
    data_numerical=pd.DataFrame(sc.transform(data_numerical))

    return data_numerical

def predict(dict_req):
    for k,v in dict_req.items():
        dict_req[k] = [v]
    dict_req=(dict_req)
    df=pd.DataFrame(dict_req)
    intcols=['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    df[intcols] = df[intcols].apply(pd.to_numeric)
    data_cat=encode_categories(df)
    data_num=scale_data(df)
    clean_data=np.c_[data_num,data_cat]
    loaded_model = joblib.load("saved_models/model.joblib")
    if loaded_model.predict(clean_data)[0]==0:
        return 'income less than $50000'
    else:
        return 'income more than $50000'

