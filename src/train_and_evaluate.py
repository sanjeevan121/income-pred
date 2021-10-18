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
import joblib

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

    c=config['estimators']['SVC']['params']['C']
    kernel=config['estimators']['SVC']['params']['kernel']
    degree=config['estimators']['SVC']['params']['degree']
    gamma=config['estimators']['SVC']['params']['gamma']
    tol=config['estimators']['SVC']['params']['tol']
   
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
                verbose=1,
                random_state=random_state,
                )
                
    svc_clf.fit(train_x,train_y)
        
    preds=svc_clf.predict(test_x)
    (precision,recall,f1,roc_auc)=eval_metrics(test_y,preds)

    print("Support Vector Classifier model (C={}, kernel={},degree={},gamma={},tol={}):".format(c,kernel,degree,gamma,tol))
    print("  precision: %s" % precision)
    print("  recall: %s" % recall)
    print("  f1_score: %s" % f1)
    print("  roc_auc: %s" % roc_auc)

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
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(svc_clf, model_path)



if __name__ == '__main__':
    args=argparse.ArgumentParser()
    default_config=os.path.join('config','params.yaml')
    args.add_argument('--config',default=default_config)
    args.add_argument('--datasource',default=None)
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)

