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
from KDE_AsHMM import KDE_AsHMM as kde
from AR_ASLG_HMM import AR_ASLG_HMM as hmm
#%% Import data
# Download the files from the dataset https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill

# rootf = "PATH OF THE FILE WERE YOU UNZIPPED THE FILES"
rootf = r"C:\Users\fox_e\OneDrive\Documentos\datasets\CNC milling machine\CSV"
files = os.listdir(rootf)
datasets = []
verdad = []
P_u=1
for f in files:
    dataf = pd.read_csv(rootf+"\\"+f)
    key = 'Machining_Process'
    xdif = np.array(dataf["X1_ActualPosition"])[np.newaxis]
    ydif = np.array(dataf["Y1_ActualPosition"])[np.newaxis]
    zdif = np.array(dataf["Z1_ActualPosition"])[np.newaxis]
    sdif = np.array(dataf["S1_ActualPosition"])[np.newaxis]

    datasetf = np.concatenate([xdif, ydif, zdif, sdif],axis=0).T
    datasetf = datasetf+np.random.normal(0,0.5,datasetf.shape)
    datasets.append(datasetf)
labels_cnc= ["X-ActualPosition","Y-ActualPosition","Z-ActualPosition","Spindle-ActualPosition"]

worn  =  [3, 4, 5, 6, 7,  8,  9,  15]
unworn = [0, 1, 2, 10, 11, 12, 13, 14, 16,17]
train = True

n_states = 9 #Numbe rof hidden states for all models
# Folds definition
fold1 = [0,12] 
fold2 = [1,13]
fold3 = [2,14]
fold4 = [10,16]
fold5 = [11,17]
folds = [fold1,fold2,fold3,fold4,fold5]
#%% Data visualization
model = kde(datasets[10],n_states)
model.plot_data_scatter()
#%% Training 
# root_accepted = "PATH WHERE YOU PUT WANT TO SAVE YOUR RESULTS"
root_accepted = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\accepted_models"
try:
    os.mkdir(root_accepted)
except:
    pass
models_accepted = []
for i in range(len(folds)):
    dataset_i = np.zeros([0,datasetf.shape[1]])
    for j in folds[i]:
        dataset_i = np.concatenate([dataset_i,datasets[j]])
    modeli = []
    model1 = kde(dataset_i,n_states,P=P_u)
    model2 = kde(dataset_i,n_states,P=P_u)
    model3 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=P_u)
    model4 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=P_u)
    if train == True:
        model1.EM()
        model1.save(root_accepted, name= "KDE_HMM_accepted"+str(i))
        model2.SEM()
        model2.save(root_accepted, name= "KDE_AsHMM_accepted" +str(i))
        model3.EM()
        model3.save(root_accepted, name= "HMM_accepted" +str(i))
        try:
            model4.SEM()
            model4.save(root_accepted, name= "AR-AsLG-HMM_accepted"+str(i))
        except:
            model4 = hmm(datasets[i],np.array([len(datasets[i])]),n_states,P=P_u)
            model4.EM()
            model4.save(root_accepted, name= "AR-AsLG-HMM_accepted" +str(i))
        modeli.append([i,model1,model2,model3,model4])
    else:
        model1.load(root_accepted+"\\"+"KDE_HMM_accepted"+str(i)+".kdehmm")
        model2.load(root_accepted+"\\"+"KDE_AsHMM_accepted"+str(i)+".kdehmm")
        model3.load(root_accepted+"\\"+"HMM_accepted"+str(i)+".ashmm")
        model4.load(root_accepted+"\\"+"AR-AsLG-HMM_accepted"+str(i)+".ashmm")
        modeli.append([i,model1,model2,model3,model4])
    models_accepted.append(modeli)
    
#%% Testing
ll_accepted = []
for l in range(len(folds)):
    llacceptedi = []
    models_unworni = models_accepted[l]
    actual_i = models_unworni[0][0]
    mod1 = models_unworni[0][1]
    mod2 = models_unworni[0][2]
    mod3 = models_unworni[0][3]
    mod4 = models_unworni[0][4]
    for j in range(len(folds)):
        if j!= l:
            for k in folds[j]:
                ll1ij = mod1.log_likelihood(datasets[k],xunit=True)
                ll2ij = mod2.log_likelihood(datasets[k],xunit=True)
                ll3ij = mod3.log_likelihood(datasets[k],xunit=True)
                ll4ij = mod4.log_likelihood(datasets[k],xunit=True)
                llacceptedi.append([ll1ij,ll2ij,ll3ij,ll4ij])
    ll_accepted.append(llacceptedi)
ll_accepted = np.array(ll_accepted)
np.save(root_accepted+"\\"+"ll_accepted",ll_accepted)

ll_accepted2non = []
for l in range(len(folds)):
    llacceptednoni = []
    models_unworni = models_accepted[l]
    actual_i = models_unworni[0][0]
    mod1 = models_unworni[0][1]
    mod2 = models_unworni[0][2]
    mod3 = models_unworni[0][3]
    mod4 = models_unworni[0][4]
    for j in worn:
        ll1ij = mod1.log_likelihood(datasets[j],xunit=True)
        ll2ij = mod2.log_likelihood(datasets[j],xunit=True)
        ll3ij = mod3.log_likelihood(datasets[j],xunit=True)
        ll4ij = mod4.log_likelihood(datasets[j],xunit=True)
        llacceptednoni.append([ll1ij,ll2ij,ll3ij,ll4ij])
    ll_accepted2non.append(llacceptednoni)
ll_accepted2non = np.array(ll_accepted2non)
np.save(root_accepted+"\\"+"ll_accepted2non",ll_accepted2non)


#%% Plots y tablas a mostrar
ll_accepted2non = np.load(root_accepted+"\\"+"ll_accepted2non.npy")
ll_accepted     = np.load(root_accepted+"\\"+"ll_accepted.npy")

#mean log likelihoods per fold
mtll_accepted = np.mean(ll_accepted,axis=1)
mtll_accepted2non = np.mean(ll_accepted2non,axis=1)

#mean log-likelihoods in average
fm_accepted     = np.median(mtll_accepted,axis=0)   
fm_accepted2non = np.median(mtll_accepted2non,axis=0)     
#%% Para borrar
root_accepted = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\unworn_models"
try:
    os.mkdir(root_accepted)
except:
    pass
models_accepted = []
for i in range(len(folds)):
    dataset_i = np.zeros([0,datasetf.shape[1]])
    for j in folds[i]:
        dataset_i = np.concatenate([dataset_i,datasets[j]])
    modeli = []
    model1 = kde(dataset_i,n_states,P=P_u)
    model2 = kde(dataset_i,n_states,P=P_u)
    model3 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=P_u)
    model4 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=P_u)
    model1.load(root_accepted+"\\"+"KDE_HMM_unworn"+str(i)+".kdehmm")
    model2.load(root_accepted+"\\"+"KDE_AsHMM_unworn"+str(i)+".kdehmm")
    model3.load(root_accepted+"\\"+"HMM_unworn"+str(i)+".ashmm")
    model4.load(root_accepted+"\\"+"AR-AsLG-HMM_unworn"+str(i)+".ashmm")
    modeli.append([i,model1,model2,model3,model4])
    models_accepted.append(modeli)
    

            
            
            

            
            



