import os
import pandas as pd
import numpy as mp
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder,StandardScaler
from get_data import read_params
import pickle 
import argparse
import scipy.stats as stat
from categorical_encoding import encode_categories
from numerical_scaling import scale_data
import numpy as np

def preprocessor(config_path):
    try:
        config=read_params(config_path)
        raw_data_path=config['load_data']['raw_dataset_csv']
        interim_path = config['data_preprocessing']['preprocessed_data_dir']

        df=pd.read_csv(raw_data_path,sep=',')
        df['target']=df.salary.map({" <=50K":0," >50K":1})
        y=np.array(df['target'])
        y=y.reshape(32561,1)
        df=df.drop(labels='salary',axis=1)
        cat=encode_categories(config_path)
        num=scale_data(config_path)
        data_cat=np.array(cat)
        data_num=np.array(num)
        clean_data=np.c_[data_num,data_cat,y]
        df=pd.DataFrame(clean_data)
        for i in range(5,27):
            df[i]=df[i].apply(lambda x: int(x))

        df.to_csv(interim_path,index=False,sep=',',encoding='utf-8')

    except Exception as e:
        print(e)

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    default_config=os.path.join('config','params.yaml')
    args.add_argument('--config',default=default_config)
    args.add_argument('--datasource',default=None)
    parsed_args = args.parse_args()
    preprocessor(config_path=parsed_args.config)

