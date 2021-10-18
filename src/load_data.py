#read data from datasource
#save it in data/rae for further process


import os
import yaml
import argparse
from get_data import read_params,get_data


def load_and_save(config_path):
    config=read_params(config_path)
    df=get_data(config_path)
    raw_data_path=config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path,index=False)



if __name__=='__main__':
    args=argparse.ArgumentParser()
    default_config=os.path.join('config','params.yaml')
    args.add_argument('--config',default=default_config)
    args.add_argument('--datasource',default=None)
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)