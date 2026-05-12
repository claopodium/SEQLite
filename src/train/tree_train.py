from data_process.onehot_encoding import one_hot_encode, to_oh_tensor
from data_process.VHSE_encoding import to_VHSE_tensor, VHSE_encoding, to_VHSE_vector
from data_process.data_preprocess import extract
from data_process.normalize import np_Normalize
from model.Xgboost import XGBoost
from model.Random_Forest import RF
from vision.loss_plot import loss_plot

import pandas as pd
import torch
import numpy as np
from Bio import SeqIO



def tree_train(df, model_name, encode, estimators, epoch, batchsize, max_depth, lr, task):

    save_path = f"./weight/trial.pth"

    df = pd.DataFrame(extract(df))

    if encode == "VHSE":
        encode = to_VHSE_vector
        map = VHSE_encoding
    elif encode == "ONEHOT":
        encode = to_oh_tensor
        map = one_hot_encode

    if model_name == "Random Forest":
        model = RF
    elif model_name == "XGBoost":
        model = XGBoost
    
    X,y = encode(df)

    X = X.reshape(X.shape[0], -1)
    y = np_Normalize(y)

    model = model(estimators=estimators, max_depth=max_depth, lr=lr, task=task)
    
    print("Training begin")
    model.fit(X, y)
    print("Training finished")

    return model