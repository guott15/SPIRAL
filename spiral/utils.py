#!/usr/bin/env python
# coding: utf-8

# In[84]:


import os
import numpy as np
import scipy.sparse as sp
import networkx as nx

import scanpy as sc
import pandas as pd
from sklearn.neighbors import kneighbors_graph as knn_g
from sklearn.preprocessing import minmax_scale
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from operator import itemgetter
import random
from numpy.random import choice
import matplotlib.pyplot as plt
import umap.umap_ as umap
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset,DataLoader


# In[126]:


# ###construct spacial graph####
# knn_neigbhor=10
# ####postive:random walk###
# N_WALKS=10
# WALK_LEN=2
# ###negative:the range of neighbor###
# N_WALK_LEN=2
# NUM_NEG=20


# In[128]:

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def refine(sample_id, pred, dis, num_nbs=8):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<(num_nbs+1)/2) and (np.max(v_c)>(num_nbs+1)/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
    return refined_pred


def process_adj(edges,n,loop=False,normalize_adj=False):
    m=edges.shape[0]
    u,v=edges[:,0],edges[:,1]
    adj=sp.coo_matrix((np.ones(m),(u,v)),shape=(n,n),dtype=np.float32)
    adj+=adj.T.multiply(adj.T>adj)-adj.multiply(adj.T>adj)
    if loop:
        adj += sp.eye(n)
    if normalize_adj:
        degrees=np.power(np.array(np.sum(adj, axis=1)),-0.5).flatten()
        degrees=sp.diags(degrees)
        adj=(degrees.dot(adj.dot(degrees)))
    return adj

def run_random_walks(G, N_WALKS,WALK_LEN,weight=None):
    nodes=G.nodes()
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(N_WALKS):
            curr_node = node
            for j in range(WALK_LEN):
                w=weight[curr_node]
                nei=[n for n in G.neighbors(curr_node)]
                w=itemgetter(*nei)(w)
                w=w/np.sum(w)
                if len(nei)==1:
                    w=np.array([w])
                next_node = random.choices(nei,weights=w)[0]
#                 next_node = random.choice(G.neighbors(curr_node))
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

def generate_pos_pair(adj,node,N_WALKS,WALK_LEN):
    G1=nx.convert_matrix.from_numpy_matrix(adj)
    mapping={v: k for v, k in enumerate(node)}
    G1=nx.relabel_nodes(G1, mapping)
    weights=[(i,j,1) for i,j in G1.edges()]
    G1.add_weighted_edges_from(weights)
    weight=dict()
    for node in G1.nodes():
        a=dict()
        for nei_node in G1.neighbors(node):
            a[nei_node]=G1[node][nei_node]['weight']
        weight[node]=a
    pos_pair=run_random_walks(G1,N_WALKS,WALK_LEN,weight)
    return pos_pair

def layer_map(idx,adj,num_layers):
    adj=adj.tolil()
    rows=adj.rows
    if type(idx) is int:
        node_layers = [np.array([idx], dtype=np.int64)]
    elif type(idx) is list:
        node_layers = [np.array(idx, dtype=np.int64)]
    for _ in range(num_layers):
        prev = node_layers[-1]
        arr = [node for node in prev]
        arr.extend([v for node in arr for v in rows[node]])
        arr = np.array(list(set(arr)), dtype=np.int64)
        node_layers.append(arr)
    node_layers.reverse()
    mappings = [{j : i for (i,j) in enumerate(arr)} for arr in node_layers]
    return node_layers,mappings


def get_negtive_nodes(nodes,dist,adj,N_WALK_LEN,NUM_NEG,dist_aware):
    adj_lists=adj.tolil()
    adj_lists=adj_lists.rows
    negtive_pairs=[]
    for node in nodes:
        neighbors = set([node])
        frontier = set([node])
        for i in range(N_WALK_LEN):
            current = set()
            for outer in frontier:
                current |= set(adj_lists[int(outer)])
            frontier = current - neighbors
            neighbors |= current
        a=dist[node]    
        all_node=set(list(a.keys()))
        far_nodes=all_node-neighbors
        if dist_aware:
            val=np.array([a[i] for i in far_nodes])
        else:
            val=np.array([1 for i in far_nodes])
        val=val/np.sum(val)
        neg_samples=choice(a=list(far_nodes), size=NUM_NEG,p=val,replace=False) if NUM_NEG < len(far_nodes) else far_nodes
        negtive_pairs.extend([[node, neg_node] for neg_node in neg_samples])
    return negtive_pairs

def load_data(feat_file,edge_file,SEP,dist_aware=True,hvg_file=None):
    feat=pd.read_csv(feat_file[0],header=0,index_col=0,sep=SEP)
    edge=np.loadtxt(edge_file[0],dtype=str)
    batch={}
    for i in np.arange(feat.shape[0]):
        batch[i]=np.arange(feat.shape[0])
    if len(edge.shape)<2:
        edge=np.array([[i.split(":")[0],i.split(":")[1]] for i in edge])
    if len(feat_file)>1:
        for k in np.arange(1,len(feat_file)):
            a=pd.read_csv(feat_file[k],header=0,index_col=0,sep=SEP)
            for i in np.arange(feat.shape[0],a.shape[0]+feat.shape[0]):
                batch[i]= np.arange(feat.shape[0],a.shape[0]+feat.shape[0])
            if hvg_file is not None:
                a=a.loc[:,np.loadtxt(hvg_file,dtype=str)]
            feat=pd.concat((feat,a),axis=0)
            a=np.loadtxt(edge_file[k],dtype=str)
            if len(a.shape)<2:
                a=np.array([[i.split(":")[0],i.split(":")[1]] for i in a])
            edge=np.vstack((edge,a))
    node_mapping=[{j:i for(i,j) in enumerate(feat.index)}]
    node_mapping=node_mapping[0]
    edge=np.array([[node_mapping[i[0]],node_mapping[i[1]]] for i in edge])
    adj=process_adj(edge,feat.shape[0])
#     pos_pairs=generate_pos_pair(adj.todense(),feat.index,N_WALKS,WALK_LEN)
#     pos_pairs=np.array([[node_mapping[i[0]],node_mapping[i[1]]] for i in pos_pairs])
    nodes=np.arange(feat.shape[0])
#     neg_pairs=np.array(get_negtive_nodes(nodes,dist,adj,N_WALK_LEN,NUM_NEG,dist_aware))
#     dataset=TensorDataset(torch.Tensor(list(node_mapping.values())).int(),torch.cat((torch.zeros(num1),torch.ones(num2))).int())
    dataset=TensorDataset(torch.Tensor(list(nodes)).int())
    return dataset,feat,adj,batch

class UnsupervisedLoss(object):
    """docstring for UnsupervisedLoss"""
    def __init__(self, adj_lists, batch, Q=10,N_WALKS=6,WALK_LEN=1,N_WALK_LEN=5,num_neg=6,MARGIN=3):
        super(UnsupervisedLoss, self).__init__()
        self.Q = Q
        self.N_WALKS = N_WALKS
        self.WALK_LEN = WALK_LEN
        self.N_WALK_LEN = N_WALK_LEN
        self.MARGIN = MARGIN
        self.num_neg = num_neg
        self.adj_lists = adj_lists
        self.batch = batch

        self.target_nodes = None
        self.positive_pairs = []
        self.negtive_pairs = []
        self.node_positive_pairs = {}
        self.node_negtive_pairs = {}
        self.unique_nodes_batch = []

    def get_loss_sage(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            # Q * Exception(negative score)
            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score = self.Q*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
            #print(neg_score)

            # multiple positive score
            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score = torch.log(torch.sigmoid(pos_score))
            #print(pos_score)

            nodes_score.append(torch.mean(- pos_score - neg_score).view(1,-1))

        loss = torch.mean(torch.cat(nodes_score, 0))

        return loss

    def get_loss_margin(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)

            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)

            nodes_score.append(torch.max(torch.tensor(0.0), neg_score-pos_score+self.MARGIN).view(1,-1))
            # nodes_score.append((-pos_score - neg_score).view(1,-1))

        loss = torch.mean(torch.cat(nodes_score, 0),0)

        # loss = -torch.log(torch.sigmoid(pos_score))-4*torch.log(torch.sigmoid(-neg_score))

        return loss


    def extend_nodes(self, nodes):
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negtive_pairs = []
        self.node_negtive_pairs = {}

        self.target_nodes = nodes
        self.get_positive_nodes(nodes)
        # print(self.positive_pairs)
        self.get_negtive_nodes(nodes)
        # print(self.negtive_pairs)
        self.unique_nodes_batch = list(set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x]))
        assert set(self.target_nodes) < set(self.unique_nodes_batch)
        return self.unique_nodes_batch

    def get_positive_nodes(self, nodes):
        return self._run_random_walks(nodes)

    def get_negtive_nodes(self, nodes):
        for node in nodes:
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.N_WALK_LEN):
                current = set()
                for outer in frontier:
                    current |= set(self.adj_lists[int(outer)])
                frontier = current - neighbors
                neighbors |= current
            a=self.batch[node]
            train_nodes = set(list(a))
            far_nodes = train_nodes - neighbors
            neg_samples = random.sample(far_nodes, self.num_neg) if self.num_neg < len(far_nodes) else far_nodes
            self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
            self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
        return self.negtive_pairs

    def _run_random_walks(self, nodes):
        for node in nodes:
            if len(self.adj_lists[int(node)]) == 0:
                continue
            cur_pairs = []
            for i in range(self.N_WALKS):
                curr_node = node
                for j in range(self.WALK_LEN):
                    neighs = self.adj_lists[int(curr_node)]
                    next_node = random.choice(list(neighs))
                    # self co-occurrences are useless
                    if next_node != node:
                        self.positive_pairs.append((node,next_node))
                        cur_pairs.append((node,next_node))
                    curr_node = next_node

            self.node_positive_pairs[node] = cur_pairs
        return self.positive_pairs
    
    def get_loss_xent(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

        pos_score = torch.Tensor([]).cuda()
        neg_score = torch.Tensor([]).cuda()
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score=torch.cat((neg_score,torch.sum(embeddings[node_indexs]*embeddings[neighb_indexs],dim=1)))

            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score=torch.cat((pos_score,torch.sum(embeddings[node_indexs]*embeddings[neighb_indexs],dim=1)))
        pos_score=nn.Sigmoid()(pos_score)
        neg_score=nn.Sigmoid()(neg_score)
        loss=nn.BCELoss()(pos_score,torch.ones(pos_score.shape[0]).cuda())+nn.BCELoss()(neg_score,torch.zeros(neg_score.shape[0]).cuda())
        return loss