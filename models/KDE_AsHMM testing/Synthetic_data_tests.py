# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:53:14 2022
@author: Carlos Puerto-Santana at KTH royal institute of technology, Stockholm, Sweden
@Licence: CC BY 4.0
"""
import numpy as np
# import os
# path = os.getcwd()
# models = os.path.dirname(path)
# os.chdir(models)
from KDE_AsHMM import KDE_AsHMM
from AR_ASLG_HMM import AR_ASLG_HMM as hmm
#%%Functions to generate data

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
    return np.sign(x)*np.abs(x)**2

def sin(x):
    return np.sin(2*np.pi*1.5*x)

def ide(x):
    return x


def log_gaussian_kernel(x):
    return -0.5*(x**2+np.log(2*np.pi))

def gaussian_kernel(x):
    return np.exp(-0.5*x**2)/np.sqrt(2*np.pi)

# Generate multidimensional data

def gen_nl_random(ns,seq,G,L,P,p,M,means,sigma,k,f1,f2):
    ncols = len(means[0])
    x = np.zeros([0,ncols])
    for l in seq:
        xl = gen_nl_random_l(ncols, ns[l], G[l], L[l], P, p[l], M[l], means[l], sigma[l], f1,f2,k)
        x = np.concatenate([x,xl],axis=0)
    return x


def gen_nl_random_l(ncols,nrows,G,L,P,p,M,means,sigma,f1,f2,k):
    K = ncols 
    x = np.ones([P,K])*means[np.newaxis]
    for t in range(nrows):
        xt = gen_nl_random_1sample(G, L, P, p, M, means, sigma, f1,f2,k, x[-P:])
        x = np.concatenate([x,xt],axis=0)
    return x


def gen_nl_random_1sample(G,L,P,p,M,means,sigma,f1,f2,k,seed ):
    K = len(G)
    xt = np.ones([1,K])
    for i in L:
        if np.sum(M[i]) ==0 :
            xt[0][i] = np.random.normal(means[i],sigma[i],1)
        else:
            mean  = means[i]
            for w in range(K):
                mean += M[i][w]*f1(xt[0][w]-means[i])
                # mean += M[i][w]*f1(xt[0][w])
            for l in range(P):
                ki = k[i]
                f2m = f2(M[i][K+l]*(seed[-l-1][i]))
                mean += ki*f2m
            xt[0][i] = np.random.normal(mean,sigma[i],1)
    return xt
            



     

#%% TEST
K = 7
N=3
means_g = np.array([ [ 2.3,  4.2,  3.5,  2.5,  1.2, 1.0, 2.0],
                     [-2.1,  3.4, -2.4,  3.2, -4.1, 1.0, 2.0], 
                     [-3.5, -1.3, -4.4, -0.5, -2.2, 1.0, 2.0]])

sigmas_g = np.array( [[0.3,  0.7,  0.8,  0.6,  0.7, 0.5, 0.6],
                      [0.9,  0.8,  0.3,  0.9,  1.2, 0.5, 0.6], 
                      [0.8,  0.5,  0.2,  0.4,  0.8, 0.5, 0.6]])

k = np.array([ 1, 1, 1, 1, 1, 1 ,1])

G = np.zeros([3,7,7])
G[1][1][0] = 1; G[1][3][2] = 1; G[1][4][1] = 1; 
G[2][1][0] = 1; G[2][3][0] = 1; G[2][4][3] = 1; G[2][5][3] = 1; G[2][5][6] = 1 ; G[2][6][2] =1 ; G[2][2][0]  =1

L          = []
for i in range(3):
    lel = dag_v(G[i])
    L.append(lel[1])

L            = np.array(L)
P            = 2
p            = np.array([[0,0,0,0,0,0,0],
                         [0,1,0,1,0,0,0],
                         [1,2,1,2,1,0,0]])

MT            = np.zeros([N,K,K+P])
MT[1][1][0] = 1; MT[1][3][2] = 1; MT[1][4][1] = 1; 
MT[2][1][0] = 1; MT[2][3][0] = 1; MT[2][4][3] = 1; MT[2][5][3] = 1; MT[2][5][6] = 1 ; MT[2][6][2] =1 ; G[2][2][0]  =1

MT[1][1][7]   =  0.5; MT[1][3][7] = 0.6 
MT[2][0][7]   =  0.4; MT[2][1][7] = 0.4; MT[2][1][8] = 0.5; MT[2][2][7] =  0.4 ; MT[2][3][7] = -0.5 ; MT[2][3][8] = -0.3;  MT[2][4][7]= 0.80
nses         = [100,100,100]
seqss        = [0,1,2,1,0,2,0]
data_gen     = gen_nl_random(nses,seqss,G,L,P,p,MT,means_g,sigmas_g,k,square,ide)
lengths_gen  = np.array([len(data_gen)])


model1 = KDE_AsHMM(data_gen, 3,P=P)
model1.EM()
v =model1.v


model2 = KDE_AsHMM(data_gen, 3,P=P)
# model2.copy(model1)
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

model5 = KDE_AsHMM(data_gen,  3,p=p,G=G,P=P,v=v)
model5.EM()

model6 = KDE_AsHMM(data_gen, 3,P=P)

mod2 = KDE_AsHMM(data_gen, 3,P=P)
mod2.copy(model2)

data_gen_test = gen_nl_random(nses,seqss,G,L,P,p,MT,means_g,sigmas_g,k,square,ide)
ll1  = model1.log_likelihood(data_gen_test )
ll2  = model2.log_likelihood(data_gen_test )
ll21 = model21.log_likelihood(data_gen_test )
ll22 = model22.log_likelihood(data_gen_test )
ll3  = model3.log_likelihood(data_gen_test )
ll4  = model4.log_likelihood(data_gen_test )
ll5  = model5.log_likelihood(data_gen_test )
ll6  = model6.log_likelihood(data_gen_test )
print("Likelihood KDE-HMM:               "+ str(ll1))
print("Likelihood KDE-AsHMM:             "+ str(ll2))
print("Likelihood KDE-AsHMM no BN opt:   "+ str(ll21))
print("Likelihood KDE-AsHMM no AR opt:   "+ str(ll22))
print("Likelihood HMM:                   "+ str(ll3))
print("Likelihood AR-AsLG-HMM:           "+ str(ll4))
print("Likelihood KDE-HMM with known BN: "+ str(ll5))
print("Likelihood KDE-HMM no train:      "+ str(ll6))
# Toco cmabiar el penalty, pero ya se obtienen grafos más razonables, se cambio ln(T) -> T
# Si se le entrega una inicialización de v aleatoria, puede llevar a resultados muy buenos o muy malos
# debe buscarse una forma de iniciar el vector v tal que ayude a descubrir patrones lineales y no lineales, y pueda ser replicable...
#%% TEST 2

ar_data = np.concatenate([data_gen[:,2][np.newaxis],np.roll(data_gen[:,2],1)[np.newaxis],np.roll(data_gen[:,2],2)[np.newaxis]],axis=0)
ar_data = ar_data.T[2:]
ar_model = KDE_AsHMM(ar_data, 3,P=P)
ar_model.EM()
