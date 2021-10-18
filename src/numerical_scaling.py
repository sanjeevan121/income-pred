import scipy.stats as stat
from matplotlib import pylab
import argparse
import os
import pandas as pd
import numpy as np
from src import get_data
from src.get_data import read_params
import pickle

def scale_data(config_path):
    try:
        config=read_params(config_path)
        raw_data_path=config['load_data']['raw_dataset_csv']
        scaler_file=config['transformers']['standard_scaler']
        df=pd.read_csv(raw_data_path,sep=',')
        data_numerical=df[['age','education-num','capital-gain','capital-loss','hours-per-week']]
        data_numerical['age']=np.log(data_numerical['age'])
        data_numerical['education-num'],_=stat.boxcox(data_numerical['education-num'])
        data_numerical['hours-per-week'],_=stat.boxcox(data_numerical['hours-per-week'])
        sc=pickle.load(open(scaler_file, 'rb'))
        data_numerical=pd.DataFrame(sc.transform(data_numerical))

        return data_numerical

    except Exception as e:
        print(e)


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    default_config=os.path.join('config','params.yaml')
    args.add_argument('--config',default=default_config)
    args.add_argument('--datasource',default=None)
    parsed_args = args.parse_args()
    scale_data(config_path=parsed_args.config)


