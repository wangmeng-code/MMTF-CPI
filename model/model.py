# -*- coding: utf-8 -*-
import copy
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from utils.Utils import *
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout

class MLP(nn.Module):
    def __init__(self, input_dim, n_hidden_3, n_hidden_4, n_classes):
        super(MLP, self).__init__()

        self.MLP_linear1 = LinearLayer(input_dim, n_hidden_3, use_bias=True)
        self.MLP_linear2 = LinearLayer(n_hidden_3, n_hidden_4, use_bias=True)
        self.MLP_linear3 = LinearLayer(n_hidden_4, n_classes, use_bias=True)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)

    def forward(self, v):
        v = self.MLP_linear1(v)
        v = self.relu(v)
        v = self.MLP_linear2(v)
        v = self.relu(v)
        output_v = self.MLP_linear3(v)
        return output_v


class DiagWeightLayer(nn.Module):
    def __init__(self, in_features):
        super(DiagWeightLayer, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features,))))

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.weight, mean=0, std=0.1)

    def forward(self, x):
        x = torch.matmul(x, torch.diag_embed(self.weight))

        return x

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(LinearLayer, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features, out_features))))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(out_features,))))
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.weight, mean=0, std=0.1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x += self.bias

        return x

class Spectral_GCN(nn.Module):
    def __init__(self, device, n_features, n_hidden_2, n_embbding):
        super(Spectral_GCN, self).__init__()
        self.device = device
        self.linear1 = DiagWeightLayer(n_features)
        self.linear2 = LinearLayer(n_features, n_hidden_2, use_bias=True)
        self.linear_out = LinearLayer(n_hidden_2, n_embbding, use_bias=True)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)
        self.U, self.U_T = decomposition()
        
        
    def forward(self, x):
        u_w = self.linear1(self.U.to(self.device))
        Z = torch.matmul(u_w, self.U_T.to(self.device))
        layer_1 = self.relu(torch.matmul(x, Z.T))
        layer_1 = self.dropout(layer_1)

        layer_2 = self.linear2(layer_1)
        layer_2 = self.relu(layer_2)
        layer_2 = self.dropout(layer_2)

        layer_out = self.linear_out(layer_2)

        return layer_out

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class AttentiveFP(nn.Module):
    """ drug structure features """
    def __init__(self, device, node_feat_size, edge_feat_size, num_layers = 2, num_timesteps = 2, graph_feat_size = 200, predictor_dim=None):
        super(AttentiveFP, self).__init__()
        self.device = device
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps)
        self.transform = nn.Linear(graph_feat_size, predictor_dim)
    
    def forward(self, bg):
        bg = bg.to(self.device)
        node_feats = bg.ndata.pop('h')
        edge_feats = bg.edata.pop('e')
        node_feats = self.gnn(bg, node_feats, edge_feats)
        graph_feats = self.readout(bg, node_feats, False)
        return self.transform(graph_feats)


class Parallel_CNN(nn.Module):
    """ protein structure features """
    def __init__(self, embed_dim, hid_dim, kernels=[3, 5, 7], dropout_rate=0.5):
        super(Parallel_CNN, self).__init__()
        padding1 = (kernels[0] - 1) // 2
        padding2 = (kernels[1] - 1) // 2
        padding3 = (kernels[2] - 1) // 2
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )
        
        self.conv = nn.Sequential(
            nn.Linear(hid_dim*len(kernels), hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
        )
    def forward(self, protein):
        protein = protein.permute([0, 2, 1])  #[bs, hid_dim, seq_len]
        features1 = self.conv1(protein)
        features2 = self.conv2(protein)
        features3 = self.conv3(protein)
        
        features = torch.cat((features1, features2, features3), 1)
        features = features.max(dim=-1)[0]  #[bs, hid_dim*3]
        return self.conv(features)


class Tensor_Fusion(nn.Module):
    """ Tensor Fusion Module"""
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, gate3=1, dim1=128, dim2=128, dim3=128, scale_dim1=1, scale_dim2=1, scale_dim3=1, mmhid=256, dropout_rate=0.25):
        super(Tensor_Fusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og, dim1, dim2, dim3 = dim1, dim2, dim3, dim1//scale_dim1, dim2//scale_dim2, dim3//scale_dim3
        skip_dim = dim1_og+dim2_og+dim3_og if skip else 0


        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())   
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og+dim3_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og+dim3_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og+dim3_og, dim3))
        self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1)*(dim3+1), 256), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU())


    def forward(self, vec1, vec2, vec3):
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(torch.cat((vec1, vec2,vec3), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(torch.cat((vec1, vec2,vec3), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)
        
        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = self.linear_z3(torch.cat((vec1, vec2, vec3), dim=1))
            o3 = self.linear_o3(nn.Sigmoid()(z3)*h3)
        else:
            h3 = self.linear_h2(vec3)
            o3 = self.linear_o2(h3)
        

        # Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o3 = torch.cat((o3, torch.cuda.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        
        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)

        if self.skip: out = torch.cat((out, vec1, vec2,vec3), 1)
        out = self.encoder2(out)
        return out



class MMTF_CPI(nn.Sequential):
    def __init__(self, device):
        super(MMTF_CPI, self).__init__()
        self.device = device
        
        self.Str_module_drug = AttentiveFP(self.device,
                                            node_feat_size = 39, 
                                            edge_feat_size = 11,  
                                            num_layers = 3, 
                                            num_timesteps = 2, 
                                            graph_feat_size = 64, 
                                            predictor_dim = 256)
        
        
        self.Str_module_protein = Parallel_CNN(100, hid_dim=64)
        
        self.Trans_module_drug = Spectral_GCN(self.device,
                                  n_features = 978, 
                                  n_hidden_2 = 2048,
                                  n_embbding = 100)
        
        self.Trans_module_protein = Spectral_GCN(self.device,
                                  n_features = 978, 
                                  n_hidden_2 = 2048,
                                  n_embbding = 100)
       
        self.mm = Tensor_Fusion(dim1=320, dim2=128, dim3=200, scale_dim1=8, gate1=1, scale_dim2=8, gate2=1,scale_dim3=8, gate3=1, skip=True, mmhid=256)
        
        self.classifier = nn.Linear(256, 1)
        

    def forward(self, data):
        # Structure modality
        d_s = self.Str_module_drug(data[0])
        p_s = self.Str_module_protein(data[1])
        
        # Heterogeneous network modality
        d_h = data[2]
        p_h = data[3]

        # Transcriptional profiling modality
        d_t = self.Trans_module_drug(data[4])
        p_t = self.Trans_module_protein(data[5])
        
        v_s = torch.cat((d_s, p_s),dim=1)
        v_h = torch.cat((d_h, p_h), dim=1)
        v_t = torch.cat((d_t, p_t),dim=1)
        
        # Tensor Fusion
        fusion = self.mm(v_s, v_h, v_t)
        
        # CPI Prediction
        pred = self.classifier(fusion)

        return pred


