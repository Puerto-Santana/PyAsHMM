# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:53:14 2022
@author: Carlos Puerto-Santana at KTH royal institute of technology, Stockholm, Sweden
@Licence: CC BY 4.0
"""
import numpy as np
import matplotlib.pyplot as plt 
import os
import warnings
warnings.filterwarnings("ignore")
path = os.getcwd()
models = os.path.dirname(path)
os.chdir(models)
from KDE_AsHMM import KDE_AsHMM
from AR_ASLG_HMM import AR_ASLG_HMM as hmm
#%% Functions to generate synthetic data

def dag_v(G):
    """
    Executes the Kahn topological sorting  algorithm. 

    Parameters
    ----------
    G : TYPE numpy array size KxK
        DESCRIPTION. a graph with K variables

    Returns
    -------
    list
        DESCRIPTION. A bool indicating if it is a acyclic graph and a topological sort in L. 

    """
    G2 = np.copy(G)
    L = []
    booli = False
    ss = np.sum(G2,axis=1)
    s = np.where(ss==0)[0]
    while len(s)>0:
        si = s[0]
        s = s[1:]
        L.append(si)
        indexi = np.where(G2.T[si]==1)[0]
        for x in indexi:
            G2[x][si] = 0
            if np.sum(G2[x])==0:
                s =np.concatenate([s,[x]])
    if np.sum(G2)==0:
        booli = True
    return [booli,np.array(L)]

# Helpful functions

def square(x):
    return x**2

def sin(x):
    return np.sin(2*np.pi*2*x)

def ide(x):
    return x

def log_gaussian_kernel(x):
    return -0.5*(x**2+np.log(2*np.pi))

def gaussian_kernel(x):
    return np.exp(-0.5*x**2)/np.sqrt(2*np.pi)

# Generate multidimensional data

def gen_nl_random(ns,seq,G,L,P,p,M,means,sigma,k,f1,f2):
    ncols = len(means[0])
    x = np.zeros([P,ncols])
    for l in seq:
        xl = gen_nl_random_l(ncols, ns[l], G[l], L[l], P, p[l], M[l], means[l], sigma[l], f1,f2,k[l],x)
        x = np.concatenate([x,xl],axis=0)
    return x[P:]


def gen_nl_random_l(ncols,nrows,G,L,P,p,M,means,sigma,f1,f2,k,y):
    x = y[-P:]
    for t in range(nrows):
        xt = gen_nl_random_1sample(G, L, P, p, M, means, sigma, f1,f2,k, x[-P:])
        x = np.concatenate([x,xt],axis=0)
    return x[P:]


def gen_nl_random_1sample(G,L,P,p,M,means,sigma,f1,f2,k,seed ):
    K = len(G)
    xt = np.ones([1,K])
    for i in L:
        if np.sum(M[i]) ==0 :
            xt[0][i] = np.random.normal(means[i],sigma[i],1)
        else:
            mean  = means[i]
            for w in range(K):
                mean += M[i][w]*f1(means[w]-xt[0][w])+k[i]
            for l in range(P):
                f2m = M[i][K+l]*f2((seed[-l-1][i]))
                mean += f2m
            xt[0][i] = np.random.normal(mean,sigma[i],1)
    return xt
            
#%% Generating data
K       = 7
N       = 3
nses    = [100,100,100]
seqss   = [0,1,2,1,0,2,0]

G = np.zeros([3,7,7])
G[1][1][0] = 1; G[1][2][0] = 1; G[1][4][3] = 1
G[2][1][0] = 1; G[2][2][0] = 1; G[2][4][3] = 1;  G[2][3][0]  =1; G[2][4][0]  =1

k = np.zeros([N,K])

L          = []
for i in range(3):
    lel = dag_v(G[i])
    print(lel[0])
    L.append(lel[1])

L            = np.array(L)
P            = 2
p            = np.array([[0,0,0,0,0,0,0],
                         [0,1,0,1,0,0,0],
                         [1,2,1,2,1,0,0]])

means_g = np.array([ [    0,    -10,  20,     0,    8, 1.0, 2.0],
                     [    2,    -1,     3,     2,    2, 1.0, 2.0], 
                     [   -2,     2,    -3,    -2,   -2, 1.0, 2.0]])

sigmas_g = np.array( [[0.1,   0.3,  0.5,  0.2,  1.8, 0.5, 0.6],
                      [1,     0.7,  0.2,  1.2,  1.4, 0.5, 0.6], 
                      [2,     0.5,  0.6,  1.3,  0.2, 0.5, 0.6]])

MT            = np.zeros([N,K,K+P])
MT[1][1][0] = 1.5; MT[1][2][0] =  -0.9; MT[1][4][3] = 2 
MT[2][1][0] = -0.9; MT[2][2][0] = 1.5; MT[2][4][3] = -2 

MT[1][1][7]   =  0.5; MT[1][3][7] = 0.6 
MT[2][0][7]   =  0.4; MT[2][1][7] = 0.4; MT[2][1][8] = 0.4; MT[2][2][7] =  0.4 ; MT[2][3][7] = -0.5 ; MT[2][3][8] = -0.3;  MT[2][4][7]= 0.6

#%% Training models
data_gen     = gen_nl_random(nses,seqss,G,L,P,p,MT,means_g,sigmas_g,k,square,ide)
lengths_gen  = np.array([len(data_gen)])

model1 = KDE_AsHMM(data_gen, 3,P=P)
model1.EM()

model2 = KDE_AsHMM(data_gen, 3,P=P)
model2.copy(model1)
model2.SEM()

model21 = KDE_AsHMM(data_gen, 3,P=P,struc=False)
model21.copy(model1)
model21.SEM()

model22 = KDE_AsHMM(data_gen, 3,P=P,lags=False)
model22.copy(model1)
model22.SEM()

model3 = hmm(data_gen,lengths_gen, 3,P=P)
model3.EM()

model4 = hmm(data_gen, lengths_gen, 3,P=P)
model4.SEM()

model5 = KDE_AsHMM(data_gen,  3,p=p,G=G,P=P)
model5.EM()
#%% Save models
model1.save(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models",name="synt_mod1")
model2.save(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models",name="synt_mod2")
model21.save(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models",name="synt_mod21")
model22.save(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models",name="synt_mod22")
model3.save(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models",name="synt_mod3")
model4.save(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models",name="synt_mod4")
model5.save(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models",name="synt_mod5")
#%% Load models
# model1.load(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models\synt_mod1.kdehmm")
# model2.load(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models\synt_mod2.kdehmm")
# model21.load(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models\synt_mod21.kdehmm")
# model22.load(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models\synt_mod22.kdehmm")
# model3.load(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models\synt_mod3.kdehmm")
# model4.load(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models\synt_mod4.kdehmm")
# model5.load(r"C:\Users\fox_e\Documents\PyAsHMM\models\KDE_AsHMM testing\models\synt_mod5.kdehmm")
#%% Testing
pruebas =5
ll1  = []
ll2  = []
ll21 = []
ll22 = []
ll3  = []
ll4  = []
ll5  = []
for t in range(pruebas):
    data_gen_test = gen_nl_random(nses,seqss,G,L,P,p,MT,means_g,sigmas_g,k,square,ide)
    ll1.append(model1.log_likelihood(data_gen_test ))
    ll2.append(model2.log_likelihood(data_gen_test ))
    ll21.append(model21.log_likelihood(data_gen_test ))
    ll22.append(model22.log_likelihood(data_gen_test ))
    ll3.append(model3.log_likelihood(data_gen_test ))
    ll4.append(model4.log_likelihood(data_gen_test ))
    ll5.append(model5.log_likelihood(data_gen_test ))
    print(str(round(100*(t+1)/(pruebas*1.),3))+"%")
        
    
ll1  = np.array(ll1)
ll1  = ll1[~np.isnan(ll1)]
ll2  = np.array(ll2)
ll2  = ll2[~np.isnan(ll2)]
ll21 = np.array(ll21)
ll21 = ll21[~np.isnan(ll21)]
ll22 = np.array(ll22)
ll22 = ll22[~np.isnan(ll22)]
ll3  = np.array(ll3)
ll3  = ll3[~np.isnan(ll3)]
ll4  = np.array(ll4)
ll4  = ll4[~np.isnan(ll4)]
ll5  = np.array(ll5)
ll5  = ll5[~np.isnan(ll5)]

print("Likelihood KDE-HMM:               "+ str(np.mean(ll1)))
print("Likelihood KDE-AsHMM:             "+ str(np.mean(ll2)))
print("Likelihood KDE-AsHMM no BN opt:   "+ str(np.mean(ll21)))
print("Likelihood KDE-AsHMM no AR opt:   "+ str(np.mean(ll22)))
print("Likelihood HMM:                   "+ str(np.mean(ll3)))
print("Likelihood AR-AsLG-HMM:           "+ str(np.mean(ll4)))
print("Likelihood KDE-HMM with known BN: "+ str(np.mean(ll5)))