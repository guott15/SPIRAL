#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np


import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt
import umap.umap_ as umap
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset,DataLoader
from torch.optim import Adam

from pytorch_revgrad import RevGrad
from .model import *
from .utils import *
from .layers import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator,PoolAggregator
import sys
from tqdm import tqdm

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cuda=torch.cuda.is_available()
# device=np.arange(torch.cuda.device_count())
# if cuda:
#     torch.cuda.set_device(device)




# In[17]:

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class SPIRAL_integration:
    def __init__(self,params,feat_file,edge_file,meta_file):
        super(SPIRAL_integration, self).__init__()

        self.params = params
        setup_seed(self.params.seed)
        self.model=A_G_Combination_DA(self.params.AEdims, self.params.AEdimsR,self.params.GSdims,self.params.agg_class,self.params.num_samples,
                                      self.params.zdim,self.params.znoise_dim,self.params.beta,self.params.CLdims,self.params.DIdims).cuda()
        self.optim=Adam(self.model.parameters(),lr=self.params.lr,weight_decay=self.params.weight_decay)
        self.epochs= self.params.epochs
        self.BS=self.params.batch_size
        self.dataset,self.Y,self.adj,self.dist,self.feat,self.meta=self.prepare_data(feat_file,edge_file,meta_file)
        self.de_act=nn.Sigmoid() 
        self.sample_num=len(np.unique(self.Y))
        if self.sample_num==2:
            self.cl_act=nn.Sigmoid()
        else:
            self.cl_act=nn.Softmax(dim=1)

        self.data_loader=DataLoader(dataset=self.dataset, batch_size=self.BS,shuffle=True,num_workers=8,drop_last=True)
        self.unsupervised_loss=UnsupervisedLoss(self.adj.tolil().rows, self.dist, self.params.Q,self.params.N_WALKS,self.params.WALK_LEN,self.params.N_WALK_LEN,self.params.NUM_NEG)
        self.feat1=torch.Tensor(self.feat.values).float().cuda()
        self.Y1=torch.Tensor(self.Y).cuda()                                  
    
    def train(self):
        self.model.train()
        print('--------------------------------')
        print('Training.')
        # with tqdm(total=self.epochs, file=sys.stdout) as pbar:
        for epoch in np.arange(0,self.epochs):
            total_loss=0.0;AE_loss=0.0;GS_loss=0.0;CLAS_loss=0;DISC_loss=0
            t=time.time()
            IDX=[]
            for (batch_idx, target_idx) in enumerate(self.data_loader):
                if len(np.unique(np.array(IDX)))==self.feat.shape[0]:
                    break
                target_idx=target_idx[0]
                all_idx=np.asarray(list(self.unsupervised_loss.extend_nodes(target_idx.tolist())))
                IDX=IDX+all_idx.tolist()
                all_layer,all_mapping=layer_map(all_idx.tolist(),self.adj,len(self.params.GSdims))
                all_rows=self.adj.tolil().rows[all_layer[0]]
                all_feature=self.feat1[all_layer[0],:]
                all_embed,ae_out,clas_out,disc_out=self.model(all_feature,all_layer,all_mapping,all_rows,self.params.lamda,self.de_act,self.cl_act)
                [ae_embed,gs_embed,embed]=all_embed
                [x_bar,x]=ae_out
                gs_loss = self.unsupervised_loss.get_loss_xent(embed, all_idx)
                ae_loss=nn.BCELoss()(x_bar,x)
                if self.sample_num==2:
                    true_batch=self.Y1[all_layer[-1]]
                    clas_loss=nn.BCELoss()(clas_out,true_batch.reshape(-1,1))
                    disc_loss=nn.BCELoss()(disc_out,true_batch.reshape(-1,1))
                else:
                    true_batch=self.Y1[all_layer[-1]].long()
                    clas_loss=nn.CrossEntropyLoss()(clas_out,true_batch)
                    disc_loss=nn.CrossEntropyLoss()(disc_out,true_batch)
                loss=ae_loss*self.params.alpha1+gs_loss*self.params.alpha2+clas_loss*self.params.alpha3+disc_loss*self.params.alpha4

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss+=loss.item()
                AE_loss+=ae_loss.item()
                GS_loss+=gs_loss.item()
                CLAS_loss+=clas_loss.item()
                DISC_loss+=disc_loss.item()

            aa=(batch_idx+1)
            print('total_loss/AE_loss/GS_loss/clas_loss/disc_loss=','%.5f/%.5f/%.5f/%.5f/%.5f'%(total_loss/aa,AE_loss/aa,GS_loss/aa,
                                                                                   CLAS_loss/aa,DISC_loss/aa))    
                # pbar.set_description('processed: %d' % (1 + epoch))
                # pbar.set_postfix(total_loss=total_loss/aa,GS_loss=GS_loss/aa,CLAS_loss=CLAS_loss/aa,DISC_loss=DISC_loss/aa)
                # pbar.update(1)    
    def save_model(self):
        torch.save(self.model.state_dict(),self.params.model_file)
        print('Saving model to %s' % self.params.model_file)
        
    def load_model(self):
        saved_state_dict = torch.load(self.params.model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % self.params.model_file)
        
    def prepare_data(self,feat_file,edge_file,meta_file,SEP=','):
        dataset,feat,adj,dist=load_data(feat_file,edge_file,SEP)
        x=minmax_scale(feat.values,axis=1)
        feat=pd.DataFrame(x,index=feat.index,columns=feat.columns)
        meta=pd.read_csv(meta_file[0],header=0,index_col=0)
        for i in np.arange(1,len(meta_file)):
            meta=pd.concat((meta,pd.read_csv(meta_file[i],header=0,index_col=0)),axis=0)
        ub=np.unique(meta.loc[:,'batch'])
        Y=np.zeros(meta.shape[0])
        for i in range(len(ub)):
            Y[np.where(meta.loc[:,'batch']==ub[i])[0]]=i
        return dataset,Y,adj,dist,feat,meta




