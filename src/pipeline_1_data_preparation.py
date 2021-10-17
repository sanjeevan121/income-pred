import os
import argparse 
import yaml
import logging
from src.get_data import read_params
  
def main(config_path,datasource):
    config=read_params(config_path)
    #datasource=read_params(datasource)
    print(config)


if __name__=="__main__":
    args=argparse.ArgumentParser()
    default_config=os.path.join('config','params.yaml')
    args.add_argument('--config',default=default_config)
    args.add_argument('--datasource',default=None)
    parsed_args = args.parse_args()
    main(config_path=parsed_args.config,datasource=parsed_args.datasource)



