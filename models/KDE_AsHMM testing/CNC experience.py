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
                  'S1_CommandPosition']
    datasets.append(np.array(dataf[features])+np.random.normal(0,0.25,[len(dataf),4]))  
#%% training
worn   = [5,6,7,8,9,12,13,14,15,17]
unworn = [0,1,2,3,4,10,11,16]
train = False
# Worn models 
models_worn =[]
try:
    os.mkdir(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\worn_models")
except:
    pass
root_worn = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\worn_models"
for i in worn:
    modeli = []
    model1 = kde(datasets[i],7,P=1)
    model2 = kde(datasets[i],7,P=1)
    model3 = hmm(datasets[i],np.array([len(datasets[i])]),7,P=1)
    model4 = hmm(datasets[i],np.array([len(datasets[i])]),7,P=1)
    if train == True:
        model1.EM()
        model1.save(root_worn, name= "KDE_HMM_worn"+str(i) )
        model2.SEM()
        model1.save(root_worn, name= "KDE_AsHMM_worn"+str(i) )
        model3.EM()
        model1.save(root_worn, name= "HMM_worn"+str(i) )
        try:
            model4.SEM()
            model1.save(root_worn, name= "AR-AsLG-HMM_worn" +str(i))
        except:
            model4 = hmm(datasets[i],np.array([len(datasets[i])]),7,P=1)
            model4.EM()
            model1.save(root_worn, name= "AR-AsLG-HMM_worn"+str(i) )
        modeli = [model1,model2,model3,model4]
    else:
        model1.load(root_worn+"\\"+"KDE_HMM_worn"+str(i)+".kdehmm")
        model2.load(root_worn+"\\"+"KDE_AsHMM_worn"+str(i)+".kdehmm")
        model3.load(root_worn+"\\"+"HMM_worn"+str(i)+".ashmm")
        model4.load(root_worn+"\\"+"AR-AsLG-HMM_worn"+str(i)+".ashmm")
        modeli = [i,model1,model2,model3,model4]
    models_worn.append(modeli)
        
        
# unworn models 
try:
    os.mkdir(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\unworn_models")
except:
    pass
root_unworn = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\unworn_models"
models_unworn = []
for i in unworn:
    modlei = []
    model1 = kde(datasets[i],7,P=1)
    model2 = kde(datasets[i],7,P=1)
    model3 = hmm(datasets[i],np.array([len(datasets[i])]),7,P=1)
    model4 = hmm(datasets[i],np.array([len(datasets[i])]),7,P=1)
    if train == True:
        model1.EM()
        model1.save(root_unworn, name= "KDE_HMM_unworn"+str(i))
        model2.SEM()
        model1.save(root_unworn, name= "KDE_AsHMM_unworn" +str(i))
        model3.EM()
        model1.save(root_unworn, name= "HMM_unworn" +str(i))
        try:
            model4.SEM()
            model1.save(root_unworn, name= "AR-AsLG-HMM_unworn"+str(i))
        except:
            model4 = hmm(datasets[i],np.array([len(datasets[i])]),7,P=1)
            model4.SEM()
            model1.save(root_unworn, name= "AR-AsLG-HMM_unworn" +str(i))
        modeli.append([i,model1,model2,model3,model4])
    else:
        model1.load(root_unworn+"\\"+"KDE_HMM_unworn"+str(i)+".kdehmm")
        model2.load(root_unworn+"\\"+"KDE_AsHMM_unworn"+str(i)+".kdehmm")
        model3.load(root_unworn+"\\"+"HMM_unworn"+str(i)+".ashmm")
        model4.load(root_unworn+"\\"+"AR-AsLG-HMM_unworn"+str(i)+".ashmm")
        modeli.append([i,model1,model2,model3,model4])
#%% Testing
# worn
ll_worn = []
for i in worn:
    llworni = []
    models_worni = models_worn[i]
    actual_i = models_worni[0]
    mod1 = models_worni[1]
    mod2 = models_worni[2]
    mod3 = models_worni[3]
    mod4 = models_worni[4]
    for j in worn:
        if j!= i:
            ll1ij = mod1.log_likelihood(datasets[j],xunit=True)
            ll2ij = mod2.log_likelihood(datasets[j],xunit=True)
            ll3ij = mod3.log_likelihood(datasets[j],xunit=True)
            ll4ij = mod4.log_likelihood(datasets[j],xunit=True)
            llworni.append([ll1ij,ll2ij,ll3ij,ll4ij])
    ll_worn.append(llworni)
ll_worn = np.array(ll_worn)
np.save(root_worn+"\\"+"ll_worn",ll_worn)

ll_unworn = []
for i in unworn:
    llunworni = []
    models_unworni = models_unworn[i]
    actual_i = models_unworni[0]
    mod1 = models_unworni[1]
    mod2 = models_unworni[2]
    mod3 = models_unworni[3]
    mod4 = models_unworni[4]
    for j in unworn:
        if j!= i:
            ll1ij = mod1.log_likelihood(datasets[j],xunit=True)
            ll2ij = mod2.log_likelihood(datasets[j],xunit=True)
            ll3ij = mod3.log_likelihood(datasets[j],xunit=True)
            ll4ij = mod4.log_likelihood(datasets[j],xunit=True)
            llunworni.append([ll1ij,ll2ij,ll3ij,ll4ij])
    ll_unworn.append(llunworni)
ll_unworn = np.array(ll_unworn)
np.save(root_unworn+"\\"+"ll_unworn",ll_unworn)

ll_un2worn = []
for i in unworn:
    llworni = []
    models_unworni = models_unworn[i]
    actual_i = models_unworni[0]
    mod1 = models_unworni[1]
    mod2 = models_unworni[2]
    mod3 = models_unworni[3]
    mod4 = models_unworni[4]
    for j in worn:
        ll1ij = mod1.log_likelihood(datasets[j],xunit=True)
        ll2ij = mod2.log_likelihood(datasets[j],xunit=True)
        ll3ij = mod3.log_likelihood(datasets[j],xunit=True)
        ll4ij = mod4.log_likelihood(datasets[j],xunit=True)
        llworni.append([ll1ij,ll2ij,ll3ij,ll4ij])
    ll_un2worn.append(llworni)
ll_un2worn = np.array(ll_un2worn)
np.save(root_unworn+"\\"+"ll_un2worn",ll_un2worn)
        



        
            
            

            
            
            

            
            



