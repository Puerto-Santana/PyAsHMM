# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:10:22 2022
@author: Carlos Puerto-Santana at KTH royal institute of technology, Stockholm, Sweden
@Licence: CC BY 4.0
"""
import pandas as pd
from scipy.io import wavfile
# from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import numpy as np
import matplotlib.pyplot as plt
from playsound import playsound 
import os
import warnings
warnings.filterwarnings("ignore")
path = os.getcwd()
models = os.path.dirname(path)
os.chdir(models)
from KDE_AsHMM import KDE_AsHMM as kde
from AR_ASLG_HMM import AR_ASLG_HMM as hmm
from librosa.feature import mfcc
#%% Functions

def cut_mfcc(features):
    """
    

    Parameters
    ----------
    features : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    bf = True
    gf = True
    lcut = 0
    hcut = -1
    
    for i in range(len(features)):
        if np.sum(features[i]) == 0 and bf == True:
            lcut = i
        else:
            bf = False
        if np.sum(features[-i]) ==0  and gf == True:
            hcut = -i
        else:
            gf = False
    
    return features[lcut:hcut]
    
def extract_data(atlas,fold,category,root):
    """
    

    Parameters
    ----------
    atlas : TYPE
        DESCRIPTION.
    fold : TYPE
        DESCRIPTION.
    category : TYPE
        DESCRIPTION.
    root : TYPE
        DESCRIPTION.

    Returns
    -------
    train : TYPE
        DESCRIPTION.
    test : TYPE
        DESCRIPTION.
    leng : TYPE
        DESCRIPTION.

    """
    n_mf = 5
    test = []   
    files = atlas[fold][category]
    for f in files:
        samplerate, sig = wavfile.read(root+"\\"+f)
        # playsound(root+"\\"+f)
        sig = sig.astype(float)
        # mfcc_feat = cut_mfcc(mfcc(sig,sr=samplerate,n_mfcc = n_mf,win_length =1280, hop_length= 640  ).T)
        mfcc_feat = cut_mfcc(mfcc(sig,sr=samplerate,n_mfcc = n_mf,win_length =1600, hop_length= 800  ).T)
        mfcc_feat = mfcc_feat+np.random.normal(0,0.1,[len(mfcc_feat),n_mf])
        if np.sum(mfcc_feat) != 0:
            # test = np.concatenate([test,mfcc_feat],axis=0)
            test.append(mfcc_feat)
    
    leng = []
    train = np.zeros([0,n_mf])  
    for i in range(5):
        if i != fold:
            traini = np.zeros([0,n_mf])    
            files = atlas[i][category]
            for f in files:
                
                samplerate, sig = wavfile.read(root+"\\"+f)  
                # samplerate, sig = wavfile.read(root+"\\"+files[0])
                sig = sig.astype(float)
                # mfcc_feat = cut_mfcc(mfcc(sig,sr=samplerate,n_mfcc = n_mf,win_length =1280, hop_length= 640).T)
                mfcc_feat = cut_mfcc(mfcc(sig,sr=samplerate,n_mfcc = n_mf,win_length =1600, hop_length= 800  ).T)
                if np.sum(mfcc_feat) != 0:
                    traini = np.concatenate([traini,mfcc_feat],axis=0)
                traini = traini+np.random.normal(0,0.1,[len(traini),n_mf])
                
                
            train = np.concatenate([train,traini],axis=0)
            leng.append(len(traini))
    train = np.array(train)
    leng = np.array(leng)

    return train , test, leng
    
def train_model(data, lengths, n_hidden_states,P, type_model,sroot,fold,key,train=True,force=False):
    """
    Trains or loads model depending on the type_model, key and fold

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    lengths : TYPE
        DESCRIPTION.
    n_hidden_states : TYPE
        DESCRIPTION.
    type_model : TYPE, optional
        DESCRIPTION. The default is "HMM".

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    root = sroot +"\\"+type_model+"-fold-"+str(fold)
    try:
        os.mkdir(root)
    except:
        pass
    if type_model == "HMM":
        model = hmm(data,lengths,n_hidden_states,P=P)
        if force==False:
            try:
                model.load(root+"\\"+type_model+"-"+key+".ashmm")
            except:    
                if train==True:
                    model.EM()
                    model.save(root,name=type_model+"-"+key)
                else:
                    model.load(root+"\\"+type_model+"-"+key+".ashmm")
        else:
            model.EM()
            model.save(root,name=type_model+"-"+key)
            
            
    if type_model == "AR-AsLG-HMM":
        model = hmm(data,lengths,n_hidden_states,P=P)
        if force== False:
            try:
                model.load(root+"\\"+type_model+"-"+key+".ashmm")
            except:
                if train==True:
                    try:
                        model.SEM()
                    except:
                        model = hmm(data,lengths,n_hidden_states,P=P,struc=False,lags=False)
                        model.SEM()
                    model.save(root,name=type_model+"-"+key)
                else:
                    model.load(root+"\\"+type_model+"-"+key+".ashmm")
        else:
            try:
                model.SEM()
            except:
                model = hmm(data,lengths,n_hidden_states,P=P,struc=False,lags=False)
                model.EM()
            model.save(root,name=type_model+"-"+key)
            
            
    if type_model == "KDE-HMM":
        model = kde(data,n_hidden_states,P=P)
        if force== False:
            try:
                model.load(root+"\\"+type_model+"-"+key+".kdehmm")
            except:
                if train==True:
                    model.EM()
                    model.save(root,name=type_model+"-"+key)
                else:
                    model.load(root+"\\"+type_model+"-"+key+".kdehmm")
        else:
            model.EM()
            model.save(root,name=type_model+"-"+key)
            
            
    if type_model == "KDE-AsHMM":
        model = kde(data,n_hidden_states,P=P)
        if force== False:
            try:
                model.load(root+"\\"+type_model+"-"+key+".kdehmm")
            except:
                if train==True:
                    model.SEM()
                    model.save(root,name=type_model+"-"+key)
                else:
                    model.load(root+"\\"+type_model+"-"+key+".kdehmm")
        else:
            model.SEM()
            model.save(root,name=type_model+"-"+key)
                
    if type_model == "KDE-BNHMM":
        model = kde(data,n_hidden_states,lags=False,P=P)
        if force== False:
            try:
                model.load(root+"\\"+type_model+"-"+key+".kdehmm")
            except:
                if train==True:
                    model.SEM()
                    model.save(root,name=type_model+"-"+key)
                else:
                    model.load(root+"\\"+type_model+"-"+key+".kdehmm")
        else:
            model.SEM()
            model.save(root,name=type_model+"-"+key)
            
            
    if type_model == "KDE-ARHMM":
        model = kde(data,n_hidden_states,struc=False,P=P)
        if force==False:
            try:
                model.load(root+"\\"+type_model+"-"+key+".kdehmm")
            except:        
                if train==True:
                    model.SEM()
                    model.save(root,name=type_model+"-"+key)
                else:
                    model.load(root+"\\"+type_model+"-"+key+".kdehmm")
        else:
            model.SEM()
            model.save(root,name=type_model+"-"+key)
            
    return model


def one_fold_model_training(atlas,categories,n_hidden_states, fold, root,sroot,type_model = "HMM",P=4,force= False):
    """
    

    Parameters
    ----------
    atlas : TYPE
        DESCRIPTION.
    categories : TYPE
        DESCRIPTION.
    n_hidden_states : TYPE
        DESCRIPTION.
    fold : TYPE
        DESCRIPTION.
    root : TYPE
        DESCRIPTION.
    type_model : TYPE, optional
        DESCRIPTION. The default is "HMM".

    Returns
    -------
    count_matrix : TYPE
        DESCRIPTION.

    """
    models =  {}
    testin = {}
    it =0
    for key_id in categories:
        train , test, leng = extract_data(atlas,fold,key_id,root)
        model_key = train_model(train,leng, n_hidden_states,P,type_model,sroot,fold,key_id,force=force)
        models[key_id] = model_key
        testin[key_id] = test
        print(str(it)+key_id)
        it=it+1

def one_fold_model_loading_testing(atlas,categories,n_hidden_states, fold, root,sroot,type_model,P=4):
    """
    

    Parameters
    ----------
    atlas : TYPE
        DESCRIPTION.
    categories : TYPE
        DESCRIPTION.
    n_hidden_states : TYPE
        DESCRIPTION.
    fold : TYPE
        DESCRIPTION.
    root : TYPE
        DESCRIPTION.
    type_model : TYPE, optional
        DESCRIPTION. The default is "HMM".

    Returns
    -------
    count_matrix : TYPE
        DESCRIPTION.

    """
    models =  {}
    testin = {}
    for key_id in categories:
        train , test, leng = extract_data(atlas,fold,key_id,root)
        model_key = train_model(train,leng, n_hidden_states,P,type_model,sroot,fold,key_id,train=False)
        models[key_id] = model_key
        testin[key_id] = test
    
    
    count_matrix = np.zeros( [len(categories),len(categories)])
    for i, key_id in enumerate(categories):
        print("Test : "+key_id+" ("+str(i)+")")
        test_id = testin[key_id]
        for test_signal in test_id:
            scores = []
            for k in categories:
                scores.append(models[k].log_likelihood(test_signal,xunit=True))
            prediction = categories[np.argmax(scores)]
            pred_index = np.argwhere(np.array(categories) == prediction).T[0][0]
            count_matrix[i][pred_index] = count_matrix[i][pred_index]+1
    return [models, count_matrix]


def all_fold_training(atlas,categories,n_hidden_states,root,sroot,type_model="HMM",P=4):
    """
    

    Parameters
    ----------
    atlas : TYPE
        DESCRIPTION.
    categories : TYPE
        DESCRIPTION.
    n_hidden_states : TYPE
        DESCRIPTION.
    root : TYPE
        DESCRIPTION.
    sroot : TYPE
        DESCRIPTION.
    type_model : TYPE, optional
        DESCRIPTION. The default is "HMM".
    P : TYPE, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    None.

    """
    one_fold_model_training(atlas,categories,n_hidden_states, 0, root,sroot,type_model,P=P)
    print("Fold 0 completed")
    one_fold_model_training(atlas,categories,n_hidden_states, 1, root,sroot,type_model,P=P)
    print("Fold 1 completed")
    one_fold_model_training(atlas,categories,n_hidden_states, 2, root,sroot,type_model,P=P)
    print("Fold 2 completed")
    one_fold_model_training(atlas,categories,n_hidden_states, 3, root,sroot,type_model,P=P)
    print("Fold 3 completed")
    one_fold_model_training(atlas,categories,n_hidden_states, 4, root,sroot,type_model,P=P)
    print("Fold 4 completed")
            

def aggregated_confusion(atlas,categories, n_hidden_states, root, sroot,type_model, P=4 ):
    """
    

    Parameters
    ----------
    atlas : TYPE
        DESCRIPTION.
    categories : TYPE
        DESCRIPTION.
    n_hidden_states : TYPE
        DESCRIPTION.
    root : TYPE
        DESCRIPTION.
    type_model : TYPE, optional
        DESCRIPTION. The default is "HMM".
    P : TYPE, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    confusion : TYPE
        DESCRIPTION.

    """
    
    confusion_0   = one_fold_model_loading_testing(atlas, categories,n_hidden_states, 0, root, sroot, type_model,P=P)
    print("Fold 0 completed")
    np.save(sroot+"\\"+type_model+"validation_0", confusion_0)
    
    confusion_1   = one_fold_model_loading_testing(atlas, categories,n_hidden_states, 1, root, sroot, type_model,P=P)
    print("Fold 1 completed")
    np.save(sroot+"\\"+type_model+"validation_1", confusion_1)
    
    confusion_2   = one_fold_model_loading_testing(atlas, categories,n_hidden_states, 2, root, sroot, type_model,P=P)
    print("Fold 2 completed")
    np.save(sroot+"\\"+type_model+"validation_2", confusion_2)
    
    confusion_3   = one_fold_model_loading_testing(atlas, categories,n_hidden_states, 3, root, sroot, type_model,P=P)
    print("Fold 3 completed")
    np.save(sroot+"\\"+type_model+"validation_3", confusion_3)
    
    confusion_4   = one_fold_model_loading_testing(atlas, categories,n_hidden_states, 4, root, sroot, type_model,P=P)
    print("Fold 4 completed")
    np.save(sroot+"\\"+type_model+"validation_4", confusion_4)
    
    confusion = confusion_0[1]+confusion_1[1]+confusion_2[1]+confusion_3[1]+confusion_4[1]

    return confusion

def accuracy(conf_matrix):
    """
    

    Parameters
    ----------
    conf_matrix : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.sum(np.diag(conf_matrix))/np.sum(conf_matrix)

def load_model(sroot, type_model,fold,key):
    """
    

    Parameters
    ----------
    sroot : TYPE
        DESCRIPTION.
    type_model : TYPE
        DESCRIPTION.
    fold : TYPE
        DESCRIPTION.
    key : TYPE
        DESCRIPTION.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    root = sroot +"\\"+type_model+"-fold-"+str(fold)

    if type_model == "HMM":
        model = hmm(np.random.normal(1,1,[100,2]),np.array([100]),1)
        model.load(root+"\\"+type_model+"-"+key+".ashmm")
            
    if type_model == "AR-AsLG-HMM":
        model = hmm(np.random.normal(1,1,[100,2]),np.array([100]),1)
        model.load(root+"\\"+type_model+"-"+key+".ashmm")
            
    if type_model == "KDE-HMM":
        model = kde(np.random.normal(1,1,[100,2]),1)
        model.load(root+"\\"+type_model+"-"+key+".kdehmm")
            
    if type_model == "KDE-AsHMM":
        model = kde(np.random.normal(1,1,[100,2]),1)
        model.load(root+"\\"+type_model+"-"+key+".kdehmm")
            
    if type_model == "KDE-BNHMM":
        model = kde(np.random.normal(1,1,[100,2]),1)
        model.load(root+"\\"+type_model+"-"+key+".kdehmm")
            
    if type_model == "KDE-ARHMM":
        model = kde(np.random.normal(1,1,[100,2]),1)
        model.load(root+"\\"+type_model+"-"+key+".kdehmm")
    return model

#%% Crear atlas
root = r"C:\Users\fox_e\OneDrive\Documentos\datasets\Voice_recognition\audio\audio\16000"
catalog = pd.read_csv(r"C:\Users\fox_e\OneDrive\Documentos\datasets\Voice_recognition\esc50.csv")
categories =  list(set(catalog["category"]))
atlas = []
for i in range(5):
    dic = {}
    for j in categories:
        dic[j] = []
    for l in range(len(catalog)):
        cate = catalog["category"][l]
        if int(catalog["fold"][l]) == i+1:
            dic[cate].append(catalog["filename"][l])
    atlas.append(dic)
sroot = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\Audio"
#%% Aprendiendo HMM para todas las etiquetas
force = True
for i in range(5):
    one_fold_model_training(atlas,categories,3, i, root,sroot,"HMM",P=1,force=force)
    
for i in range(5):
    one_fold_model_training(atlas,categories,3, i, root,sroot,"AR-AsLG-HMM",P=1,force=force)

for i in range(5):
    one_fold_model_training(atlas,categories,3, i, root,sroot,"KDE-AsHMM",P=1,force=force)
    
for i in range(5):
    one_fold_model_training(atlas,categories,3, i, root,sroot,"KDE-HMM",P=1,force=force)
    


#%% Cross-validation
aggregated_confusion(atlas, categories, 3, root, sroot,"HMM")
aggregated_confusion(atlas, categories, 3, root, sroot,"AR-AsLG-HMM")
aggregated_confusion(atlas, categories, 3, root, sroot,"KDE-HMM")
aggregated_confusion(atlas, categories, 3, root, sroot,"KDE-AsHMM")
#%% Resultados
loc = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\Audio"
# modelo = "HMM"
# modelo = "AR-AsLG-HMM"
# modelo = "KDE-HMM"
# modelo = "KDE-AsHMM"
modelos  =["HMM","AR-AsLG-HMM","KDE-HMM","KDE-AsHMM"]
matrices = []
presiciones = []
matrix_sum = []
accuracies = []
for j in modelos:
    accuri = []
    matrix_sumi = np.zeros([50,50])
    matrizi = []
    for i in range(5):
       file = loc+ "\\" + j+"validation_"+str(i)+".npy"
       mati = np.load(file, allow_pickle=True)[-1]
       matrizi.append(mati)
       accuri.append(accuracy(mati))
       matrix_sumi += mati
    matrix_sum.append(matrix_sumi)
    matrices.append(matrizi)
    accuracies.append(accuri)
    

# pres = accuracy(matrix_sum)
# print("Presicion de "+modelo +"\n"+ str(presiciones) +"\n" +"Presicion acumulada: " +str(pres))
# print("Presicion media: " +str(np.mean(presiciones)) )



