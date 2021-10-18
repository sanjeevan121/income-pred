import os
import yaml
import pandas as pd
import argparse


def read_params(config_path):
    with open(config_path) as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config=read_params(config_path)
    data_path=config["data_source"]["s3_source"]
    df=pd.read_csv(data_path,sep=',')
    return df

if __name__=='__main__':
    args=argparse.ArgumentParser()
    default_config=os.path.join('config','params.yaml')
    args.add_argument('--config',default=default_config)
    args.add_argument('--datasource',default=None)
    parsed_args = args.parse_args()
    data= get_data(config_path=parsed_args.config)