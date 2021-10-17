import os
import  numpy as np
import argparse
import pandas as pd
import sys
from get_data import read_params
import joblib
import json
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,f1_score,precision_score,recall_score

def eval_metrics(test_y,pred_y):
    precision=precision_score(test_y,pred_y)
    recall=recall_score(test_y,pred_y)
    f1=f1_score(test_y,pred_y)
    roc_auc=roc_auc_score(test_y,pred_y)
    return precision,recall,f1,roc_auc


def train_and_evaluate(config_path):
    config=read_params(config_path)
    test_data_path=config['split_data']['test_path']
    train_data_path=config['split_data']['train_path']
    random_state=config['base']['random_state']
    model_dir=config['model_dir']

    c=config['training']['SVC']['params']['C']
    kernel=config['training']['SVC']['params']['kernel']
    degree=config['training']['SVC']['params']['degree']
    gamma=config['training']['SVC']['params']['gamma']
    tol=config['training']['SVC']['params']['tol']
    break_ties=config['training']['SVC']['params']['break_ties']

    target=config['base']['target_col']
    

    train=pd.read_csv(train_data_path)
    test=pd.read_csv(test_data_path)

    train_y=train[target]
    test_y=test[target]

    train_x=train.drop(target,axis=1)
    test_x=test.drop(target,axis=1)

    svc_clf=SVC(C=c,
                kernel=kernel,
                degree=degree,
                gamma=gamma,
                tol=tol,
                verbose=3,
                break_ties=break_ties)
                
    svc_clf.fit(train_x,train_y)
        
    preds=svc_clf.predict(test_x)
    (precision,recall,f1,roc_auc)=eval_metrics(test_y,preds)

    scores_file=config['reports']['scores']
    params_file=config['reports']['params']


    with open(scores_file,'w') as f:
        scores={
            'precison':precision,
            'recall':recall,
            'f1':f1,
            'roc_auc':roc_auc
        }
        json.dump(scores, f,indent=2)
    
    with open(params_file,'w') as f:
        params={
            'C':c,
            'kernel':kernel,
            'degree':degree,
            'gamma':gamma,
            'tol':tol
        }
        json.dump(params, f,indent=2)



if __name__ == '__main__':
    args=argparse.ArgumentParser()
    default_config=os.path.join('config','params.yaml')
    args.add_argument('--config',default=default_config)
    args.add_argument('--datasource',default=None)
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)

