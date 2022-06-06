import numpy as np
import pandas as pd
import torch
import argparse

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
tf.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)

from src.utils import *
from src.model import *


def attention(hops, adj, feature_list, alpha=0.15):
    input_feature = []
    hop_max = hops.max().int().item()
    for i in range(adj.shape[0]):
        hop = hops[i].int().item()
        if hop == 0:
            fea = feature_list[0][i].unsqueeze(0)
        else:
            fea = 0
            allfeatures = []
            for j in range(hop_max):
                if j < hop:                  
                    fea += (1-alpha)*feature_list[j][i].unsqueeze(0) + alpha*feature_list[0][i].unsqueeze(0)
                    allfeatures.append(fea[0])
                else:                  
                    fea += feature_list[0][i].unsqueeze(0)
                    allfeatures.append(fea[0])                
            allf = fea_conver(torch.stack(allfeatures,0).detach().numpy().tolist())          
            input_feature.append(allf)
    input_feature = torch.stack(input_feature,0)
    att = Attention_layer()
    output_feature = att(input_feature.detach().numpy())
    print(output_feature.shape) 
    return output_feature

def propagate(features, k, adj_norm):
    feature_list = []
    feature_list.append(features)
    for i in range(1, k):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))
    return feature_list


def cal_hops(adj, feature_list, norm_fea_inf, k, epsilon=0.02):
    hops = torch.Tensor([0]*(adj.shape[0]))
    mask_before = torch.Tensor([False]*(adj.shape[0])).bool()

    for i in range(k):
        dist = (feature_list[i] - norm_fea_inf).norm(2, 1)
        mask = (dist<epsilon).masked_fill_(mask_before, False)
        mask_before.masked_fill_(mask, True)
        hops.masked_fill_(mask, i)
    mask_final = torch.Tensor([True]*(adj.shape[0])).bool()
    mask_final.masked_fill_(mask_before, False)
    hops.masked_fill_(mask_final, k-1)
    return hops


def train(config):

    path = './data/'+config["dataset_name"]
    print(path)
    AllNode = pd.read_csv( path + '/Allnode.csv',names=[0,1],skiprows=1)
    Alledge = pd.read_csv( path + '/Alledge.csv',header=None)
    features = pd.read_csv( path + '/AllNodeAttribute.csv', header = None)
    features = features.iloc[:,1:]

    adj, features  = load_data(Alledge,features)

    node_sum = adj.shape[0]
    edge_sum = adj.sum()/2
    row_sum = (adj.sum(1) + 1)
    norm_a_inf = row_sum/ (2*edge_sum+node_sum)

    adj_norm = sparse_mx_to_torch_sparse_tensor(aug_random_walk(adj))

    features = F.normalize(features, p=1)
    feature_list = []
    feature_list.append(features)
    for i in range(1, config['epochs']):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))

    norm_a_inf = torch.Tensor(norm_a_inf).view(-1, node_sum)
    norm_fea_inf = torch.mm(norm_a_inf, features)

    hops = torch.Tensor([0]*(adj.shape[0]))
    mask_before = torch.Tensor([False]*(adj.shape[0])).bool()

    for i in range(config['epochs']):
        dist = (feature_list[i] - norm_fea_inf).norm(2, 1)
        mask = (dist<config['epsilon']).masked_fill_(mask_before, False)
        mask_before.masked_fill_(mask, True)
        hops.masked_fill_(mask, i)
    mask_final = torch.Tensor([True]*(adj.shape[0])).bool()
    mask_final.masked_fill_(mask_before, False)
    hops.masked_fill_(mask_final, config['epochs']-1)
    print("Local Smoothing Iteration calculation is done.")

    input_feature = attention(hops, adj, feature_list)
    Emdebding_input_feature = pd.DataFrame(input_feature.numpy())
    # Emdebding_input_feature.to_csv('./data/18416/input_feature.csv', header=None,index=False)
    # pd.DataFrame(weight[0].numpy()).to_csv('./data/18416/weight.csv', header=None,index=False)
    print("Local Smoothing is done.")
    return Emdebding_input_feature