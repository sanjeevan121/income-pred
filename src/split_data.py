from sklearn.model_selection import train_test_split
import os
import argparse
from get_data import read_params
import pandas as pd


def split_and_save_data(config_path):
    config=read_params(config_path)
    train_data_path=config['split_data']['train_path']
    test_data_path=config['split_data']['test_path']
    raw_data_path=config['preprocess']['preprocess_data_dir']
    split_ratio=config['split_data']['test_size']
    
    random_state=config['base']['random_state']
    
    df=pd.read_csv(raw_data_path,sep=',')
    
    train,test=train_test_split(df,test_size=split_ratio,random_state=random_state,stratify=df['Output'])
    
    train.to_csv(train_data_path,sep=',',index=False,encoding='utf-8')
    test.to_csv(test_data_path,sep=',',index=False,encoding='utf-8')

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    default_config=os.path.join('config','params.yaml')
    args.add_argument('--config',default=default_config)
    args.add_argument('--datasource',default=None)
    parsed_args = args.parse_args()
    split_and_save_data(config_path=parsed_args.config)