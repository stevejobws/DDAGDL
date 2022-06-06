import pandas as pd
import numpy as np
from numpy import *
import torch
from src.utils import *
from src.model import *
from src.train import *

from sklearn.model_selection import  StratifiedKFold,KFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description="DDAGDL")
    parser.add_argument("-d", "--dataset_name", default="B-dataset", type=str,
                        choices=["B-dataset", "C-dataset", "F-dataset"])
    parser.add_argument("-n", "--n_folds", default=10, type=int, choices=[5, 10, -1], help="cross valid fold num")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--epsilon", default=0.03, type=float, help="the value of delta")
    parser.add_argument("--embedding_dim", default=64, type=int)
    parser.add_argument("--layer_num", default=3, type=int)
    parser.add_argument("--lr", default=1e-1, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)
    args = parser.parse_args()
    return args

def main(config):
    path = './data/'+config["dataset_name"]
    # Negative Generate
    Positive = pd.read_csv( path+'/DrDiNum.csv',header=None)
    Adj = incidence_matrix(Positive)
    Negative = np.transpose(np.where(Adj.toarray() == 0))
    Negative = pd.DataFrame(Negative)
    Negative[1] = Negative[1]+max(Positive[0])+1
    Negative = Negative.sample(n=18416, random_state=18416)
    Embedding = train(config)
    Positive[2] = Positive.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
    Negative[2] = Negative.apply(lambda x: 0 if x[0] < 0 else 0, axis=1)
    result = pd.concat([Positive,Negative]).reset_index(drop=True)
    X = pd.concat([Embedding.loc[result[0].values.tolist()].reset_index(drop=True),Embedding.loc[result[1].values.tolist()].reset_index(drop=True)],axis=1)
    Y = result[2]
    # DDA prediction
    i=0
    aucs=[]
    skf = StratifiedKFold(n_splits=config["n_folds"],random_state=0, shuffle=True)
    for train_index, test_index in skf.split(X,Y):     
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        model = XGBClassifier(n_estimators=999,n_jobs=-1)
        model.fit(np.array(X_train), np.array(Y_train))
        y_score0 = model.predict(np.array(X_test))
        y_score_RandomF = model.predict_proba(np.array(X_test))
        fpr,tpr,thresholds=roc_curve(Y_test,y_score_RandomF[:,1])
        roc_auc=auc(fpr,tpr)
        aucs.append(roc_auc)
        print("---------------------------------------------")
        print("fold = ", i)
        print('AUC:', roc_auc)
        print("---------------------------------------------\n")
        i+=1
    print('Mean:',mean(aucs))
if __name__=="__main__":
    args = get_parser()
    config = vars(args)
    print(config)
    main(config)
