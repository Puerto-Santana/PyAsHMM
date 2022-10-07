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


def omega(data,P=0):
    states = list(set(data['Machining_Process']))
    index = {}
    for i in range(len(states)):
        index[states[i]] = i
    
    v = np.ones([len(data)-P,len(index)])*1e-6
    for t in range(P,len(data)):
        tind = data['Machining_Process'][t]
        v[t-P][index[tind]] = 1
    v = v/np.sum(v,axis=0)
    
    return v

#%% Import data
rootf = r"C:\Users\fox_e\OneDrive\Documentos\datasets\CNC milling machine\CSV"
files = os.listdir(rootf)
datasets = []
verdad = []
n_states_d  = []
P_u=1
for f in files:
    dataf = pd.read_csv(rootf+"\\"+f)
    key = 'Machining_Process'
    # dataf = dataf[dataf[key] != "Prep"]
    # dataf = dataf[dataf[key] != "Starting"]
    # dataf = dataf[dataf[key] != "End"]
    # dataf = dataf[dataf[key] != "end"]
    xdif = np.array(dataf["X1_ActualPosition"])[np.newaxis]
    ydif = np.array(dataf["Y1_ActualPosition"])[np.newaxis]
    zdif = np.array(dataf["Z1_ActualPosition"])[np.newaxis]
    sdif = np.array(dataf["S1_ActualPosition"])[np.newaxis]

    datasetf = np.concatenate([xdif, ydif, zdif, sdif],axis=0).T
    datasetf = datasetf+np.random.normal(0,0.5,datasetf.shape)
    verdad.append(omega(dataf,P =P_u))
    datasets.append(datasetf)
    n_states_d.append([len(verdad[-1][0])])
labels_cnc= ["X1-ActualPosition","Y1-ActualPosition","Z1-ActualPosition","S1-ActualPosition"]

    
worn  =  [3, 4, 5, 6, 7,  8,  9,  15]
unworn = [0, 1, 2, 10, 11, 12, 13, 14, 16,17]
train = True

n_states = len(verdad[0][0])
fold1 = [0,10]
fold2 = [1,13]
fold3 = [2,14]
fold4 = [10,16]
fold5 = [11,17]
folds = [fold1,fold2,fold3,fold4,fold5]

# model = kde(datasets[i],int(len(verdad[i][0])),P=P_u, v = verdad[i])
# model.EM()
# model.plot_all_pairs_scatter(name=str(i))
# malos, 4, 15,3
# model2 = kde(datasets[4],4)
# model2.EM()
# model2.plot_all_pairs_scatter(labels= labels_cnc,name="kde")

#%% training con folds
n_states = 5
try:
    os.mkdir(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\unworn_models")
except:
    pass
root_unworn = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\unworn_models"
models_unworn = []
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
        model1.save(root_unworn, name= "KDE_HMM_unworn"+str(i))
        model2.SEM()
        model2.save(root_unworn, name= "KDE_AsHMM_unworn" +str(i))
        model3.EM()
        model3.save(root_unworn, name= "HMM_unworn" +str(i))
        try:
            model4.SEM()
            model4.save(root_unworn, name= "AR-AsLG-HMM_unworn"+str(i))
        except:
            model4 = hmm(datasets[i],np.array([len(datasets[i])]),n_states,P=P_u)
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
    
#%% Training Sin folds
try:
    os.mkdir(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\unworn_models_verdad")
except:
    pass
root_unworn = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\unworn_models_verdad"
models_unworn = []
for i in range(len(unworn)):
    dataset_i = datasets[unworn[i]]
    modeli = []
    n_states = n_states_d[unworn[i]][0]
    verdad_i = verdad[unworn[i]]
    model1 = kde(dataset_i,n_states,P=P_u,v=verdad_i)
    model2 = kde(dataset_i,n_states,P=P_u,v= verdad_i)
    model3 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=P_u)
    model4 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=P_u)
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
            model4 = hmm(datasets[i],np.array([len(datasets[i])]),n_states,P=P_u)
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
#%% 
# try:
#     os.mkdir(r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\worn_models")
# except:
#     pass
# root_worn = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\CNCmachine\worn_models"
# models_worn = []
# for i in range(len(worn)):
#     dataset_i = np.zeros([0,datasetf.shape[1]])
#     # for j in range(len(worn)):
#     #     if j != i:
#     dataset_i = np.concatenate([dataset_i,datasets[worn[i]]])
#     modeli = []
#     model1 = kde(dataset_i,n_states,P=0)
#     model2 = kde(dataset_i,n_states,P=0)
#     model3 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=0)
#     model4 = hmm(dataset_i,np.array([len(dataset_i)]),n_states,P=0)
#     if train == True:
#         model1.EM()
#         model1.save(root_worn, name= "KDE_HMM_worn"+str(i))
#         model2.SEM()
#         model2.save(root_worn, name= "KDE_AsHMM_worn" +str(i))
#         model3.EM()
#         model3.save(root_worn, name= "HMM_worn" +str(i))
#         try:
#             model4.SEM()
#             model4.save(root_worn, name= "AR-AsLG-HMM_worn"+str(i))
#         except:
#             model4 = hmm(datasets[i],np.array([len(datasets[i])]),n_states,P=1)
#             model4.EM()
#             model4.save(root_worn, name= "AR-AsLG-HMM_worn" +str(i))
#         modeli.append([i,model1,model2,model3,model4])
#     else:
#         model1.load(root_worn+"\\"+"KDE_HMM_worn"+str(i)+".kdehmm")
#         model2.load(root_worn+"\\"+"KDE_AsHMM_worn"+str(i)+".kdehmm")
#         model3.load(root_worn+"\\"+"HMM_worn"+str(i)+".ashmm")
#         model4.load(root_worn+"\\"+"AR-AsLG-HMM_worn"+str(i)+".ashmm")
#         modeli.append([i,model1,model2,model3,model4])
#     models_worn.append(modeli)

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


# ll_worn = []
# for l in range(len(folds2)):
#     llworni = []
#     models_worni = models_worn[l]
#     actual_i = models_worni[0][0]
#     mod1 = models_worni[0][1]
#     mod2 = models_worni[0][2]
#     mod3 = models_worni[0][3]
#     mod4 = models_worni[0][4]
#     for j in range(len(folds2)):
#         if j!= l:
#             for k in folds2[j]:
#                 ll1ij = mod1.log_likelihood(datasets[k],xunit=True)
#                 ll2ij = mod2.log_likelihood(datasets[k],xunit=True)
#                 ll3ij = mod3.log_likelihood(datasets[k],xunit=True)
#                 ll4ij = mod4.log_likelihood(datasets[k],xunit=True)
#                 llworni.append([ll1ij,ll2ij,ll3ij,ll4ij])
#     ll_worn.append(llworni)
# ll_worn = np.array(ll_worn)
# np.save(root_worn+"\\"+"ll_worn",ll_worn)

#%% Plots y tablas a mostrar
ll_un2worn = np.load(root_unworn+"\\"+"ll_un2worn.npy")
ll_unworn  = np.load(root_unworn+"\\"+"ll_unworn.npy")
# ll_worn    = np.load(root_worn+"\\"+"ll_worn.npy")
#log likelihoods per training dataset
mtll_unworn  = np.mean(ll_unworn,axis=1)
mtll_un2worn = np.mean(ll_un2worn,axis=1)
# mtll_worn    = np.mean(ll_worn,axis=1)
   
fm_unworn  = np.median(mtll_unworn,axis=0)   
fm_un2worn = np.median(mtll_un2worn,axis=0)    
# fm_worn    = np.mean(mtll_worn,axis=0)        

            
            
            

            
            



