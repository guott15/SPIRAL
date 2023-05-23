#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python



import os
import numpy as np
import scipy.sparse as sp
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset,DataLoader
from torch.optim import Adam


import scanpy as sc
import pandas as pd
from operator import itemgetter
import random
import time

from pytorch_revgrad import RevGrad
from .layers import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator,PoolAggregator



def build_mlp(layers, activation=nn.ReLU(), bn=False, dropout=0,bias=True):
    net = nn.Sequential()
    for i in range(1, len(layers)):
        net1=[]
        net1.append(nn.Linear(layers[i-1], layers[i],bias=bias))
        if bn:
            net1.append(nn.BatchNorm1d(layers[i]))
        if activation is not None:
            net1.append(activation)
        if dropout > 0:
            net1.append(nn.Dropout(dropout))
        net1=nn.Sequential(*net1)
        net.add_module('layer'+str(i),net1)
    return net

class Encoder(nn.Module):
    def __init__(self,dims):
        super(Encoder, self).__init__()
        [x_dim,h_dim1,z_dim]=dims
        self.n_hdim=len(h_dim1)
        if self.n_hdim>0:
            self.hidden1=build_mlp([x_dim]+h_dim1,activation=nn.ReLU(), bn=True, dropout=0,bias=True)
            h_dim1=h_dim1[-1]
        else:
            self.hidden1=nn.Identity()
            h_dim1=x_dim
        self.z_layer=nn.Linear(h_dim1, z_dim)
    def forward(self, x):
        en_h=[]
        a=x
        if self.n_hdim>0:
            for i in np.arange(self.n_hdim):
                a=self.hidden1[i](a)
                en_h.append(a)
        if len(en_h)>0:
            z=self.z_layer(en_h[-1])
        else:
            z=self.z_layer(x)
        return en_h,z
    
class Decoder(nn.Module):
    def __init__(self,dimsR):
        super(Decoder, self).__init__()
        [z_dim,h_dim1,x_dim]=dimsR
        self.n_hdim=len(h_dim1)
        if self.n_hdim>0:
            self.hidden1=build_mlp([z_dim]+h_dim1,activation=nn.ReLU(), bn=True, dropout=0,bias=True)
            h_dim1=h_dim1[-1]
        else:
            self.hidden1=nn.Identity()
            h_dim1=z_dim
        self.x_layer = nn.Linear(h_dim1, x_dim)
    def forward(self, z,act=nn.Sigmoid()):
        de_h=[]
        a=z
        if self.n_hdim>0:
            for i in np.arange(self.n_hdim):
                a=self.hidden1[i](a)
                de_h.append(a)
        if len(de_h)>0:
            x_bar=self.x_layer(de_h[-1])
        else:
            x_bar=self.x_layer(z)
        if act is not None:
            return de_h,act(x_bar)
        else:
            return de_h,x_bar
        
class AE(nn.Module):
    def __init__(self, dims,dimsR):
        super(AE, self).__init__()
        self.en=Encoder(dims)
        self.de=Decoder(dimsR)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x,de_act=None):
        enc_h,z=self.en(x)
        dec_h,x_bar=self.de(z,de_act)
        return enc_h,z,dec_h,x_bar



class Classifier(nn.Module):
    def __init__(self,dims):
        super(Classifier, self).__init__()
        [z_dim,h_dim1,out_dim]=dims
        self.n_hdim=len(h_dim1)
        if self.n_hdim>0:
            self.hidden1=build_mlp([z_dim]+h_dim1,activation=nn.ReLU(), bn=True, dropout=0,bias=True)
            h_dim1=h_dim1[-1]
        else:
            self.hidden1=nn.Identity()
            h_dim1=z_dim
        self.out=nn.Linear(h_dim1, out_dim)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, z,act):
        h=self.hidden1(z)
        out=self.out(h)
        if act is not None:
            return act(out)
        else:
            return out


class Discriminator(nn.Module):
    def __init__(self, dims):
        super(Discriminator, self).__init__()
        [z_dim,h_dim1,out_dim]=dims
        self.n_hdim=len(h_dim1)
        if self.n_hdim>0:
            self.hidden1=build_mlp([z_dim]+h_dim1,activation=nn.ReLU(), bn=True, dropout=0,bias=True)
            h_dim1=h_dim1[-1]
        else:
            self.hidden1=nn.Identity()
            h_dim1=z_dim
        self.out=nn.Linear(h_dim1, out_dim)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, z,act):
        h=self.hidden1(z)
        out=self.out(h)
        if act is not None:
            return act(out)
        else:
            return out







class GraphSAGE(nn.Module):

    def __init__(self, input_dim, hidden_dims,
                 agg_class=MaxPoolAggregator, dropout=0.0, num_samples=25,BN=False):
        """
        num_samples : int
            Number of neighbors to sample while aggregating. Default: 25.
        """
        super(GraphSAGE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.agg_class = agg_class
        self.num_samples = num_samples
        self.num_layers = len(hidden_dims)
        self.aggregators = nn.ModuleList([agg_class(input_dim, input_dim)])
        self.aggregators.extend([agg_class(dim, dim) for dim in hidden_dims])
        self.BN=BN


        c = 3 if agg_class == LSTMAggregator else 2
        self.fcs = nn.ModuleList([nn.Linear(c*input_dim, hidden_dims[0])])
        self.fcs.extend([nn.Linear(c*hidden_dims[i-1], hidden_dims[i]) for i in range(1, len(hidden_dims))])
        
        if self.BN:
            self.bns=nn.ModuleList([nn.BatchNorm1d(hidden_dim) for hidden_dim in hidden_dims])
        else:
            self.bns=None
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, features, node_layers, mappings, rows):
        out = features
        for k in range(self.num_layers):
            nodes = node_layers[k+1]
            mapping = mappings[k]
            init_mapped_nodes = np.array([mappings[0][v] for v in nodes], dtype=np.int64)
            cur_rows = rows[init_mapped_nodes]
            aggregate = self.aggregators[k](out, nodes, mapping, cur_rows,self.num_samples).cuda()                                        
            cur_mapped_nodes = np.array([mapping[v] for v in nodes], dtype=np.int64)
            out = torch.cat((out[cur_mapped_nodes, :], aggregate), dim=1)
            out = self.fcs[k](out)
            if k+1 < self.num_layers:
                if self.BN:
                    out=self.bns[k](out)
                out=self.relu(out)
                out=self.dropout(out)
#                 out = out.div(out.norm(dim=1, keepdim=True)+1e-6)
        return out


class A_G_Combination(nn.Module):
    def __init__(self, AEdims, AEdimsR,GSdims,agg_class,num_samples,zdim,beta):
        super(A_G_Combination, self).__init__()
        AEzdim=AEdims[-1]
        GSzdim=GSdims[-1]
        self.ae=AE(AEdims, AEdimsR)
        self.gs=GraphSAGE(AEdims[0],GSdims,agg_class,num_samples=num_samples)
        self.beta=beta
#         self.combine_layer=nn.Linear(AEzdim+GSzdim,zdim)
    def forward(self,x,node_layers, mappings,rows,de_act):
        x1=x[[mappings[0][i] for i in node_layers[-1]],:]
        _,ae_z=self.ae.en(x1)
        gs_z=self.gs(x,node_layers, mappings,rows)
#         z=self.combine_layer(torch.cat((ae_z,gs_z),dim=1))
        z=(1-self.beta)*ae_z+self.beta*gs_z
        _,x_bar=self.ae.de(z,de_act)
        final_z=[ae_z,gs_z,z]
        return final_z,x_bar,x1
    


class A_G_Combination_DA(nn.Module):
    def __init__(self, AEdims, AEdimsR,GSdims,agg_class,num_samples,zdim,znoise_dim,beta,CLdims,DIdims):
        super(A_G_Combination_DA, self).__init__()
        self.znoise_dim=znoise_dim
        self.agc=A_G_Combination(AEdims, AEdimsR,GSdims,agg_class,num_samples,zdim,beta)
        self.clas=Classifier(CLdims)
        self.disc=Discriminator(DIdims)
    def forward(self,x,node_layers, mappings,rows,lamda,de_act,cl_act):
        self.revgrad=RevGrad(lamda)
        final_z,x_bar,x1=self.agc(x,node_layers, mappings,rows,de_act)
        ae_z,gs_z,z=final_z
        znoise=z[:,:self.znoise_dim]
        zbio=z[:,self.znoise_dim:]
        clas_out=self.clas(znoise,act=cl_act)
        disc_out=self.disc(self.revgrad(zbio),act=cl_act)
        ae_out=[x_bar,x1]
        return final_z,ae_out,clas_out,disc_out

class AE_DA(nn.Module):
    def __init__(self, AEdims, AEdimsR,znoise_dim,CLdims,DIdims):
        super(AE_DA, self).__init__()
        self.znoise_dim=znoise_dim
        self.ae=AE(AEdims, AEdimsR)
        self.clas=Classifier(CLdims)
        self.disc=Discriminator(DIdims)
    def forward(self,x,lamda,de_act,cl_act):
        self.revgrad=RevGrad(lamda)
        _,ae_z,_,x_bar=self.ae(x,de_act)
        znoise=ae_z[:,:self.znoise_dim]
        zbio=ae_z[:,self.znoise_dim:]
        clas_out=self.clas(znoise,act=cl_act)
        disc_out=self.disc(self.revgrad(zbio),act=cl_act)
        ae_out=[x_bar,x]
        final_z=[ae_z]
        return final_z,ae_out,clas_out,disc_out
    

