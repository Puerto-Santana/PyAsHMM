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
rootf = r"C:\Users\fox_e\OneDrive\Documentos\datasets\CNC milling machine\CSV"
files = os.listdir(rootf)
datasets = []
for f in files:
    dataf = pd.read_csv(rootf+"\\"+f)
    # ac_ps = np.array(np.sqrt(dataf["X1_ActualPosition"]**2+dataf["Y1_ActualPosition"]**2+dataf["Z1_ActualPosition"]**2))[np.newaxis]
    # ac_sp = np.array(np.sqrt(dataf["X1_ActualVelocity"]**2+dataf["Y1_ActualVelocity"]**2+dataf["Z1_ActualVelocity"]**2))[np.newaxis]
    # ac_ac = np.array(np.sqrt(dataf["X1_ActualAcceleration"]**2+dataf["Y1_ActualAcceleration"]**2+dataf["Z1_ActualAcceleration"]**2))[np.newaxis]
    # s_ps  = np.array(dataf['S1_CommandPosition'])[np.newaxis]
    # s_sp  = np.array(dataf['S1_ActualVelocity'])[np.newaxis]
    # s_ac  = np.array(dataf['S1_ActualAcceleration'])[np.newaxis]
    # ac_v =  np.array(np.sqrt(dataf["X1_DCBusVoltage"]**2+dataf["Y1_DCBusVoltage"]**2+dataf["Z1_DCBusVoltage"]**2))[np.newaxis]
    # ac_i =  np.array(np.sqrt(dataf["X1_OutputCurrent"]**2+dataf["Y1_OutputCurrent"]**2+dataf["Z1_OutputCurrent"]**2))[np.newaxis]
    # ac_ov = np.array(np.sqrt(dataf["X1_OutputVoltage"]**2+dataf["Y1_OutputVoltage"]**2+dataf["Z1_OutputVoltage"]**2))[np.newaxis]
    # ac_p =  np.array(np.sqrt(dataf["X1_OutputPower"]**2+dataf["Y1_OutputPower"]**2))[np.newaxis]
    # datasetf = np.concatenate([ac_ps,ac_sp,ac_ac,ac_v,ac_i,ac_ov,ac_p],axis=0).T
    # datasetf = np.concatenate([ac_ps, ac_ac, s_ps, s_ac],axis=0).T
    # datasetf = datasetf+np.random.normal(0,1,datasetf.shape)
    # datasets.append(datasetf)
    features  = ['X1_ActualPosition',
                  'Y1_ActualPosition', 
                  'Z1_ActualPosition', 
                  'S1_ActualPosition']
    key = 'Machining_Process'
    dataf = dataf[dataf[key] != "Prep"]
    dataf = dataf[dataf[key] != "Starting"]
    dataf = dataf[dataf[key] != "End"]
    dataf = dataf[dataf[key] != "end"]
    datasets.append(np.array(dataf[features])+np.random.normal(0,0.5,[len(dataf),4]))  
    
#%% training
worn   = [5, 6, 7, 8, 9, 12 ,13, 14, 15, 17]
unworn = [0, 1, 2, 3, 4, 10, 11, 16]
fold1  = [3, 11]
fold2  = [4, 16]
fold3  = [0, 10]
fold4  = [1,  2]
folds  = [fold1,fold2,fold3,fold4]

fold5 = [5,9]
fold6  = [7,13]
fold7 = [6,12]
fold8 = [8,14]
fold9  = [15,17]
folds2 = [fold5,fold6,fold7,fold8,fold9]
train = False
n_states = 7
 
try:
    os.mkdir(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\unworn_models")
except:
    pass
root_unworn = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\unworn_models"
models_unworn = []
for i ,l in enumerate(folds):
    dataset_i = np.zeros([0,4])
    for j in l:
        dataset_i = np.concatenate([dataset_i,datasets[j]])
    modeli = []
    model1 = kde(dataset_i,n_states,P=1)
    model2 = kde(dataset_i,n_states,P=1)
    model3 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=1)
    model4 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=1)
    if train == True:
        model1.EM()
        model1.save(root_unworn, name= "KDE_HMM_unworn"+str(i))
        model2.SEM()
        model2.save(root_unworn, name= "KDE_AsHMM_unworn" +str(i))
        model3.EM()
        model3.save(root_unworn, name= "HMM_unworn" +str(i))
        try:
            model4.SEM()
            model4.save(root_unworn, name= "AR-AsLG-HMM_unworn"+str(i))
        except:
            model4 = hmm(datasets[i],np.array([len(datasets[i])]),n_states,P=1)
            model4.EM()
            model4.save(root_unworn, name= "AR-AsLG-HMM_unworn" +str(i))
        modeli.append([i,model1,model2,model3,model4])
    else:
        model1.load(root_unworn+"\\"+"KDE_HMM_unworn"+str(i)+".kdehmm")
        model2.load(root_unworn+"\\"+"KDE_AsHMM_unworn"+str(i)+".kdehmm")
        model3.load(root_unworn+"\\"+"HMM_unworn"+str(i)+".ashmm")
        model4.load(root_unworn+"\\"+"AR-AsLG-HMM_unworn"+str(i)+".ashmm")
        modeli.append([i,model1,model2,model3,model4])
    models_unworn.append(modeli)
    
try:
    os.mkdir(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\worn_models")
except:
    pass
root_worn = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\worn_models"
models_worn = []
for i ,l in enumerate(folds2):
    dataset_i = np.zeros([0,4])
    for j in l:
        dataset_i = np.concatenate([dataset_i,datasets[j]])
    modeli = []
    model1 = kde(dataset_i,n_states,P=1)
    model2 = kde(dataset_i,n_states,P=1)
    model3 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=1)
    model4 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=1)
    if train == True:
        model1.EM()
        model1.save(root_worn, name= "KDE_HMM_worn"+str(i))
        model2.SEM()
        model2.save(root_worn, name= "KDE_AsHMM_worn" +str(i))
        model3.EM()
        model3.save(root_worn, name= "HMM_worn" +str(i))
        try:
            model4.SEM()
            model4.save(root_worn, name= "AR-AsLG-HMM_worn"+str(i))
        except:
            model4 = hmm(datasets[i],np.array([len(datasets[i])]),n_states,P=1)
            model4.EM()
            model4.save(root_worn, name= "AR-AsLG-HMM_worn" +str(i))
        modeli.append([i,model1,model2,model3,model4])
    else:
        model1.load(root_worn+"\\"+"KDE_HMM_worn"+str(i)+".kdehmm")
        model2.load(root_worn+"\\"+"KDE_AsHMM_worn"+str(i)+".kdehmm")
        model3.load(root_worn+"\\"+"HMM_worn"+str(i)+".ashmm")
        model4.load(root_worn+"\\"+"AR-AsLG-HMM_worn"+str(i)+".ashmm")
        modeli.append([i,model1,model2,model3,model4])
    models_worn.append(modeli)

#%% Testing
ll_unworn = []
for l in range(len(folds)):
    llunworni = []
    models_unworni = models_unworn[l]
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
                llunworni.append([ll1ij,ll2ij,ll3ij,ll4ij])
    ll_unworn.append(llunworni)
ll_unworn = np.array(ll_unworn)
np.save(root_unworn+"\\"+"ll_unworn",ll_unworn)

ll_un2worn = []
for l in range(len(folds)):
    llworni = []
    models_unworni = models_unworn[l]
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
        llworni.append([ll1ij,ll2ij,ll3ij,ll4ij])
    ll_un2worn.append(llworni)
ll_un2worn = np.array(ll_un2worn)
np.save(root_unworn+"\\"+"ll_un2worn",ll_un2worn)
#%% Plots y tablas a mostrar
ll_un2worn = np.load(root_unworn+"\\"+"ll_un2worn.npy")
ll_unworn  = np.load(root_unworn+"\\"+"ll_unworn.npy")

#log likelihoods per training dataset
mtll_unworn  = np.mean(ll_unworn,axis=1)
mtll_un2worn = np.mean(ll_un2worn,axis=1)
   
fm_unworn = np.mean(mtll_unworn,axis=0)   
fm_un2worn = np.mean(mtll_un2worn,axis=0)            

            
            
            

            
            



