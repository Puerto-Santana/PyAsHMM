# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 13:05:11 2022
@author: Carlos Puerto-Santana at KTH royal institute of technology, Stockholm, Sweden
@Licence: CC BY 4.0
"""
import matplotlib.pyplot as plt
import numpy as np
import os
path = os.getcwd()
models = os.path.dirname(path)
os.chdir(models)
import pandas as pd
from KDE_AsHMM import KDE_AsHMM 
from AR_ASLG_HMM import AR_ASLG_HMM as hmm
#%% Dataset 
root_train = r"C:\Users\fox_e\OneDrive\Documentos\datasets\BDD_IMS_100ms"
n_ar = 1
ind = np.concatenate([np.arange(n_ar),np.arange(10,n_ar+10),np.arange(20,n_ar+20),np.arange(20,n_ar+20)])

rootb =r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kde_fig\synt"
try:
    os.mkdir(rootb)
except:
    pass
rootdata = r"C:\Users\fox_e\OneDrive\Documentos\datasets\BDD_IMS_100ms"
b1_data = pd.read_csv(rootdata+r"\test1_Col0_peak.txt",header=None) 
b2_data = pd.read_csv(rootdata+r"\test1_Col2_peak.txt",header=None) 
b3_data = pd.read_csv(rootdata+r"\test1_Col4_peak.txt",header=None) 
b4_data = pd.read_csv(rootdata+r"\test1_Col6_peak.txt",header=None) 
indbe = np.concatenate([np.arange(n_ar)*2+3,np.arange(10,n_ar+10)*2+3,np.arange(20,n_ar+20)*2+3,np.arange(30,n_ar+30)*2+3])
# indbe = [3,23,43,63] #BPFO, BPFI, BSF, FTF
Ob1  = np.array(b1_data[indbe])
Ob2  = np.array(b2_data[indbe])
Ob3  = np.array(b3_data[indbe])
Ob4  = np.array(b4_data[indbe])
O1 = []
O2 = []
O3 = []
O4 = []
for i in range(4*n_ar):
    O1i = np.reshape(Ob1.T[i],[int(len(Ob1)/10),10])
    O2i = np.reshape(Ob2.T[i],[int(len(Ob2)/10),10])
    O3i = np.reshape(Ob3.T[i],[int(len(Ob3)/10),10])
    O4i = np.reshape(Ob4.T[i],[int(len(Ob4)/10),10])
    O1.append(np.sqrt(np.mean(O1i**2,axis=1)))
    O2.append(np.sqrt(np.mean(O2i**2,axis=1)))
    O3.append(np.sqrt(np.mean(O3i**2,axis=1)))
    O4.append(np.sqrt(np.mean(O4i**2,axis=1)))
train1 = (np.array(O1).T)[:-10]
train2 = (np.array(O2).T)[:-10]
train3 = (np.array(O3).T)[:-10]
train4 = (np.array(O4).T)[:-10]

mu1 = np.mean(train1,axis=0)
mu2 = np.mean(train2,axis=0)
mu3 = np.mean(train3,axis=0)
mu4 = np.mean(train4,axis=0)

nu1 = np.var(train1,axis=0)**0.5;kappa1 = np.mean(train1,axis=0)/nu1
nu2 = np.var(train2,axis=0)**0.5;kappa2 = np.mean(train2,axis=0)/nu2
nu3 = np.var(train3,axis=0)**0.5;kappa3 = np.mean(train3,axis=0)/nu3
nu4 = np.var(train4,axis=0)**0.5;kappa4 = np.mean(train4,axis=0)/nu4


rootb =r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kde_fig\synt"
rootdata =  r"C:\Users\fox_e\OneDrive\Documentos\datasets\BDD_IMS_100ms"
b1_test = pd.read_csv(rootdata+r"\test3_Col0_peak.txt",header=None) 
b2_test = pd.read_csv(rootdata+r"\test3_Col1_peak.txt",header=None) 
b3_test = pd.read_csv(rootdata+r"\test3_Col2_peak.txt",header=None) 
b4_test = pd.read_csv(rootdata+r"\test3_Col3_peak.txt",header=None) 
indbe = np.concatenate([np.arange(n_ar)*2+3,np.arange(10,n_ar+10)*2+3,np.arange(20,n_ar+20)*2+3,np.arange(30,n_ar+30)*2+3])
Obt1  = np.array(b1_test[indbe])
Obt2  = np.array(b2_test[indbe])
Obt3  = np.array(b3_test[indbe])
Obt4  = np.array(b4_test[indbe])
O1t = []
O2t = []
O3t = []
O4t = []
for i in range(4*n_ar):
    O1i = np.reshape(Obt1.T[i],[int(len(Obt1)/10),10])
    O2i = np.reshape(Obt2.T[i],[int(len(Obt2)/10),10])
    O3i = np.reshape(Obt3.T[i],[int(len(Obt3)/10),10])
    O4i = np.reshape(Obt4.T[i],[int(len(Obt4)/10),10])
    O1t.append(np.sqrt(np.mean(O1i**2,axis=1)))
    O2t.append(np.sqrt(np.mean(O2i**2,axis=1)))
    O3t.append(np.sqrt(np.mean(O3i**2,axis=1)))
    O4t.append(np.sqrt(np.mean(O4i**2,axis=1)))
test11=((np.array(O1t).T))[:6200]
test12 =((np.array(O2t).T))[:6200]
test13 =((np.array(O3t).T))[:6200]
test14 =((np.array(O4t).T))[:6200]


test11 = (test11-np.min(train1,axis=0))/(np.max(train1,axis=0)-np.min(train1,axis=0))*1000
test12 = (test12-np.min(train2,axis=0))/(np.max(train2,axis=0)-np.min(train2,axis=0))*1000
test13 = (test13-np.min(train3,axis=0))/(np.max(train3,axis=0)-np.min(train3,axis=0))*1000
test14 = (test14-np.min(train4,axis=0))/(np.max(train4,axis=0)-np.min(train4,axis=0))*1000

train1 = (train1-np.min(train1,axis=0))/(np.max(train1,axis=0)-np.min(train1,axis=0))*1000
train2 = (train2-np.min(train2,axis=0))/(np.max(train2,axis=0)-np.min(train2,axis=0))*1000
train3 = (train3-np.min(train3,axis=0))/(np.max(train3,axis=0)-np.min(train3,axis=0))*1000
train4 = (train4-np.min(train4,axis=0))/(np.max(train4,axis=0)-np.min(train4,axis=0))*1000
#%% Training
N= 4
model1 = KDE_AsHMM(train1, N,P=1)
model1.EM()

model2 = KDE_AsHMM(train1, N,P=1)
model2.copy(model1)
model2.SEM()

model21 = KDE_AsHMM(train1, N,P=1,struc=False)
model21.copy(model1)
model21.SEM()

model22 = KDE_AsHMM(train1, N,P=1,lags=False)
model22.copy(model1)
model22.SEM()

model3 = hmm(train1,np.array([len(train1)]), N,P=1)
model3.EM()

model4 = hmm(train1, np.array([len(train1)]), N,P=1)
model4.SEM()
#%% Testing
ll1 =[ model1.log_likelihood(test11),
      model2.log_likelihood(test11),
      model21.log_likelihood(test11),
      model22.log_likelihood(test11),
      model3.log_likelihood(test11)]





