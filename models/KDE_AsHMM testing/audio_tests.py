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
    test = []   
    files = atlas[fold][category]
    for f in files:
        samplerate, sig = wavfile.read(root+"\\"+f)
        # playsound(root+"\\"+f)
        sig = sig.astype(float)
        mfcc_feat = cut_mfcc(mfcc(sig,sr=samplerate,n_mfcc = 13).T[:,1:])
        if np.sum(mfcc_feat) != 0:
            # test = np.concatenate([test,mfcc_feat],axis=0)
            test.append(mfcc_feat)
    
    leng = []
    train = np.zeros([0,12])  
    for i in range(5):
        if i != fold:
            traini = np.zeros([0,12])    
            files = atlas[i][category]
            for f in files:
                samplerate, sig = wavfile.read(root+"\\"+f)
                sig = sig.astype(float)
                mfcc_feat = cut_mfcc(mfcc(sig,sr=samplerate,n_mfcc = 13).T[:,1:])
                if np.sum(mfcc_feat) != 0:
                    traini = np.concatenate([traini,mfcc_feat],axis=0)
                traini = traini+np.random.normal(0,1,[len(traini),12])
            train = np.concatenate([train,traini],axis=0)
            leng.append(len(traini))
    train = np.array(train)
    leng = np.array(leng)
    return train , test, leng
    
def train_model(data, lengths, n_hidden_states, type_model = "HMM"):
    """
    

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
    if type_model == "HMM":
        model = hmm(data,lengths,n_hidden_states)
        model.EM()
    if type_model == "AR-AsLG-HMM":
        model = hmm(data,lengths,n_hidden_states)
        model.SEM()
    if type_model == "KDE-HMM":
        model = kde(data,n_hidden_states)
        model.EM()
    if type_model == "KDE-AsHMM":
        model = kde(data,n_hidden_states)
        model.SEM()
    if type_model == "KDE-AsHMM_no_AR":
        model = kde(data,n_hidden_states,lags=False)
        model.SEM()
    if type_model == "KDE-AsHMM_no_BN":
        model = kde(data,n_hidden_states,struc=False)
        model.SEM()
    return model



def one_fold_model_training_testing(atlas,categories,n_hidden_states, fold, root,type_model = "HMM"):
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
        model_key = train_model(train,leng, n_hidden_states,type_model)
        models[key_id] = model_key
        testin[key_id] = test
        print(key_id)
    
    count_matrix = np.zeros( [len(categories),len(categories)])
    for i, key_id in enumerate(categories):
        test_id = testin[key_id]
        for test_signal in test_id:
            scores = []
            for k in categories:
                scores.append(models[k].log_likelihood(test_signal))
            prediction = categories[np.argmax(scores)]
            pred_index = np.argwhere(np.array(categories) == prediction).T[0][0]
            count_matrix[i][pred_index] = count_matrix[i][pred_index]+1
    return count_matrix
            
    
        
    

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
#%% crear dataset de entrenamiento y prueba:
categories = ["dog","cat"]
confusion_0_hmm   = one_fold_model_training_testing(atlas, categories,3, 0, root,type_model = "HMM")
confusion_0_ashmm = one_fold_model_training_testing(atlas, categories,3, 0, root,type_model="AR-AsLG-HMM")
