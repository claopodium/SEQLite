from data_process.data_preprocess import continuous_read, extract
from data_process.format_process import get
from data_process.VHSE_encoding import VHSE_encoding
from data_process.onehot_encoding import one_hot_encode
from data_process.seq_process import gap_process

from eval.iter_eval import sensitive_analysis
from model.SLP import SLP
from model.MLP import MLP
from model.ResNet import ResNet
from model.CNN import CNN
from model.Transformer import Trans
from model.Random_Forest import RF
from model.Xgboost import XGBoost
from vision.loss_plot import loss_plot
from vision.heatmap import heatmap

from train.tree_train import tree_train
from train.nn_train import nn_train

import pandas as pd
import numpy as np
import torch
from Bio import SeqIO
import os
import time

def data_input(path):
    df = get(path)
    return pd.DataFrame(extract(df))

def rid_star(seq):
    ter = len(seq)
    if seq[ter-1] == "*":
        seq = seq[:len(seq)-1]
    return seq

def pipeline(path, model_name, lr, encode, epoch, batchsize, estimator, max_depth, task, wt_seq):

    if encode == "VHSE":
        map = VHSE_encoding
    elif encode == "ONEHOT":
        map = one_hot_encode

    df = data_input(path)
    wt_seq = rid_star(wt_seq)
    df = gap_process(df, wt_seq)

    if model_name == "XGBoost" or model_name == "RF":
        train_strategy = tree_train
    else:
        train_strategy = nn_train

    model = train_strategy(df, model_name, encode, estimator, epoch, batchsize, max_depth, lr, task)
    print(type(model))
    if model_name == "XGBoost" or model_name == "RF":
        wt_val = model.predict(np.stack(map(wt_seq)).reshape(1,-1))
        istree = True
    else:
        model.eval()
        wt_val = model(torch.tensor(map(wt_seq), dtype=torch.float32).unsqueeze(0))
        wt_val = float(
            wt_val.detach().cpu().item()
        )
        istree = False
    print("Prediction done!")

    result = sensitive_analysis(model, wt_seq, map, istree)
    print("Analysis done!")
    heatmap(result, wt_seq=wt_seq, wt_value=wt_val)
    print("Done")