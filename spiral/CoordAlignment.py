#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import ot
import os


# In[2]:


def kl_divergence_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.
    
    Takes advantage of POT backend to speed up computation.
    
    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)
    
    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)
    
    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X,log_Y.T)
    return nx.to_numpy(D)


# In[37]:


class CoordAlignment:
    def __init__(self,input_file,output_dirs,ub,flags,clust_cate,R_dirs,alpha=0.5,dissimilarity='euc',norm=False,loss_fun='square_loss',
                  nx=ot.backend.NumpyBackend(),numItermax=200,types="weighted_mean"):
        super(CoordAlignment, self).__init__()
        
        meta_file,coord_file,embed_file,cluster_file=input_file
        self.output_dirs=output_dirs
        self.alpha=alpha
        self.numItermax=numItermax
        self.types=types
        self.norm=norm
        self.dissimilarity=dissimilarity
        self.flags=flags
        self.clust_cate=clust_cate
        self.meta,self.Coord,self.clusters,self.embed,self.ub=self.inputs(meta_file,coord_file,embed_file,cluster_file,ub)
        self.cc,self.ref_id=self.AlignClust(self.embed,self.Coord,self.ub)
        self.AlignPairs(self.Coord.iloc[:,2:],ub,nx)
        self.AlignCoordShared(self.embed,self.Coord,self.cc,ub,self.types)
        self.AlignCoordSpecific(self.Coord,ub,R_dirs)
        self.New_Coord.to_csv(self.output_dirs+"gtt_new_coordinate"+self.flags+'_'+self.clust_cate+".csv")

    def inputs(self,meta_file,coord_file,embed_file,cluster_file,ub,znoise_dim=4):
        embed=pd.read_csv(embed_file,header=0,index_col=0,sep=',')
        embed=embed.iloc[:,znoise_dim:]
        meta=pd.read_csv(meta_file[0],header=0,index_col=0,sep=',')
        for i in np.arange(1,len(meta_file)):
            meta=pd.concat((meta,pd.read_csv(meta_file[i],header=0,index_col=0,sep=',')))  
        
        clusters=pd.read_csv(cluster_file,header=0,index_col=0,sep=',')
        clusters.columns=['clusters']
        Coord=pd.read_csv(coord_file[0],header=0,index_col=0,sep=',')
        for i in np.arange(1,len(coord_file)):
            Coord=pd.concat((Coord,pd.read_csv(coord_file[i],header=0,index_col=0,sep=',')))
        Coord=Coord.loc[:,['x','y']]
        Coord.loc[:,'clusters']=clusters.loc[Coord.index,'clusters']
        Coord.loc[:,'batch']=meta.loc[Coord.index,'batch']
        return meta,Coord,clusters,embed,ub
    
    def AlignClust(self,embed,Coord,ub):
        cc=[[] for i in np.arange(len(ub))]
        maxc=0
        for i in np.arange(len(ub)):
            x=len(np.unique(Coord.loc[Coord.loc[:,'batch']==ub[i],'clusters'].values))
            if x>maxc:
                ref_id=i
                maxc=x
        c1=np.unique(Coord.loc[Coord.loc[:,'batch']==ub[ref_id],'clusters'].values)
        for i in np.arange(len(ub)):
            cc[i]=np.intersect1d(c1,np.unique(Coord.loc[Coord.loc[:,'batch']==ub[i],'clusters'].values))
        coord1=Coord.loc[Coord.loc[:,'batch']==ub[ref_id],:]
        for bb in np.setdiff1d(np.arange(len(ub)),ref_id):
            uc=cc[bb]
            coord2=Coord.loc[Coord.loc[:,"batch"]==ub[bb],]
            for clust in uc:
                coord1_=coord1.loc[coord1.loc[:,'clusters']==clust,:]
                coord2_=coord2.loc[coord2.loc[:,'clusters']==clust,:]
                embed1_=embed.loc[coord1_.index,:]
                embed2_=embed.loc[coord2_.index,:]
                embed1_.to_csv(self.output_dirs+"embed_"+str(ub[ref_id])+"_"+str(clust)+self.flags+".csv")
                embed2_.to_csv(self.output_dirs+"embed_"+str(ub[bb])+"_"+str(clust)+self.flags+".csv")
                coord1_.to_csv(self.output_dirs+"coord_"+str(ub[ref_id])+"_"+str(clust)+self.flags+".csv")
                coord2_.to_csv(self.output_dirs+"coord_"+str(ub[bb])+"_"+str(clust)+self.flags+".csv")
        Coord.iloc[:,2:].to_csv(self.output_dirs+"gtt_clusters"+self.flags+".csv")
        return cc,ref_id
    
    def gwd(self,M, C1, C2, p, q, G_init = None, loss_fun='square_loss', alpha=0.5, armijo=False, log=False,numItermax=200):
        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)
        def f(G):
            return ot.gromov.gwloss(constC, hC1, hC2, G)
        def df(G):
            return ot.gromov.gwggrad(constC, hC1, hC2, G)
        res=ot.gromov.cg(p, q, (1 - alpha) * M, alpha, f, df, G_init, armijo=armijo, C1=C1, C2=C2, constC=constC)
        return res
    
    def AlignPairs(self,clusters,ub,nx):
        for i in np.setdiff1d(np.arange(len(ub)),self.ref_id):
            uc=np.intersect1d(clusters['clusters'][clusters['batch']==ub[self.ref_id]],
                             clusters['clusters'][clusters['batch']==ub[i]])
            for clust in uc:
                embed1=pd.read_csv(self.output_dirs+"embed_"+str(ub[self.ref_id])+"_"+str(clust)+self.flags+".csv",header=0,index_col=0,sep=",")
                embed2=pd.read_csv(self.output_dirs+"embed_"+str(ub[i])+"_"+str(clust)+self.flags+".csv",header=0,index_col=0,sep=",")
                coord1=pd.read_csv(self.output_dirs+"coord_"+str(ub[self.ref_id])+"_"+str(clust)+self.flags+".csv",header=0,index_col=0,sep=",")
                coord2=pd.read_csv(self.output_dirs+"coord_"+str(ub[i])+"_"+str(clust)+self.flags+".csv",header=0,index_col=0,sep=",")
                coord1=coord1.loc[embed1.index,:]
                coord2=coord2.loc[embed2.index,:]
                ###每个batch内部spot的空间距离####
                a=np.float64(nx.from_numpy(coord1.values[:,:2]))
                b=np.float64(nx.from_numpy(coord2.values[:,:2]))
                D1=ot.dist(a,a, metric='euclidean')
                D2=ot.dist(b,b, metric='euclidean')
                if self.norm:
                    D1 /= nx.min(D1[D1>0])
                    D2 /= nx.min(D2[D2>0])
                ####两个batch spot的低维表示的距离#####
                X1,X2 = nx.from_numpy(embed1.values), nx.from_numpy(embed2.values)
                if self.dissimilarity.lower()=='euclidean' or self.dissimilarity.lower()=='euc':
                    M = ot.dist(X1,X2)
                else:
                    s1 = X1 + 0.01
                    s2 = X2 + 0.01
                    M = kl_divergence_backend(s1, s2)
                    M = nx.from_numpy(M)
                ####每个batch的spot的分布#####
                d1 = nx.ones((embed1.shape[0],))/embed1.shape[0]
                d2 = nx.ones((embed2.shape[0],))/embed2.shape[0]
                ####计算mapping#####
                G0 = d1[:, None] * d2[None, :]
                res=self.gwd(M,D1,D2,d1,d2,G_init=G0,alpha=self.alpha)
                pi=pd.DataFrame(res,index=embed1.index,columns=embed2.index)
                pi.to_csv(self.output_dirs+"gwd_pi_"+str(ub[self.ref_id])+"_"+str(ub[i])+"_"+str(clust)+self.flags+".csv")
                
    def AlignCoordShared(self,embed,Coord,cc,ub,types):
        coord1=Coord.loc[Coord.loc[:,'batch']==ub[self.ref_id],:]
        self.New_Coord=coord1
        for bb in np.setdiff1d(np.arange(len(ub)),self.ref_id):

            uc=cc[bb]
            coord2=Coord.loc[Coord.loc[:,'batch']==ub[bb],:]
            for clust in uc:

                coord1_=coord1.loc[coord1.loc[:,'clusters']==clust,:]
                coord2_=coord2.loc[coord2.loc[:,'clusters']==clust,:]
                embed1_=embed.loc[coord1_.index,:]
                embed2_=embed.loc[coord2_.index,:]
                pi=pd.read_csv(self.output_dirs+"gwd_pi_"+str(ub[self.ref_id])+"_"+str(ub[bb])+"_"+str(clust)+self.flags+".csv",index_col=0,header=0,sep=',')

                pi=pi.loc[coord1_.index,coord2_.index]
                new_coord2=coord2_
                for i in np.arange(coord2_.shape[0]):
                    if types=='weighted_mean':
                        aa1=np.array(pi.iloc[:,i]).reshape(1,pi.shape[0])
                        aa1=aa1/np.sum(aa1)
                        aa2=coord1_.iloc[:,:2].values
                        new_coord2.iloc[i,:2]=np.matmul(aa1,aa2)[0,:]
                    if types=='mean':
                        idx=np.where(pi.iloc[:,i]!=0)[0]
                        new_coord2.iloc[i,:2]=coord1_.iloc[idx,:2].mean(axis=0)
                self.New_Coord=pd.concat((self.New_Coord,new_coord2))
    
    def AlignCoordSpecific(self,Coord,ub,R_dirs):
        os.environ['R_HOME']=R_dirs
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr
        import rpy2.robjects.numpy2ri
        robjects.r.library("vegan")
        rpy2.robjects.numpy2ri.activate()
        base = importr('base')
        for i in np.setdiff1d(np.arange(len(ub)),self.ref_id):
            sc=np.setdiff1d(np.unique(Coord.loc[Coord.loc[:,'batch']==ub[i],'clusters']),self.cc[i])                
            new=self.New_Coord.loc[self.New_Coord.loc[:,'batch']==ub[i],:]
            old=Coord.loc[new.index,:]
            Coord1=Coord.iloc[np.where(Coord.loc[:,'batch']==ub[i])[0],:]
            if len(sc)>0:
                PROCRUSTES=robjects.r['procrustes'](base.as_matrix(new.iloc[:,:2].values),base.as_matrix(old.iloc[:,:2].values),scale=False)
                idx=[]
                for j in sc:
                    idx=idx+np.where(Coord1.loc[:,'clusters']==j)[0].tolist()
                x1=Coord1.iloc[idx,:]
                x1.iloc[:,:2]=robjects.r['predict'](PROCRUSTES, base.as_matrix(x1.iloc[:,:2].values))
                self.New_Coord=pd.concat((self.New_Coord,x1))
