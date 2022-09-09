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
        mfcc_feat = cut_mfcc(mfcc(sig,sr=samplerate,n_mfcc = 13).T)
        if np.sum(mfcc_feat) != 0:
            # test = np.concatenate([test,mfcc_feat],axis=0)
            test.append(mfcc_feat)
    
    leng = []
    train = np.zeros([0,13])  
    for i in range(5):
        if i != fold:
            traini = np.zeros([0,13])    
            files = atlas[i][category]
            for f in files:
                samplerate, sig = wavfile.read(root+"\\"+f)
                sig = sig.astype(float)
                mfcc_feat = cut_mfcc(mfcc(sig,sr=samplerate,n_mfcc = 13).T)
                if np.sum(mfcc_feat) != 0:
                    traini = np.concatenate([traini,mfcc_feat],axis=0)
                traini = traini+np.random.normal(0,0.1,[len(traini),13])
            train = np.concatenate([train,traini],axis=0)
            leng.append(len(traini))
    train = np.array(train)
    leng = np.array(leng)
    return train , test, leng
    
def train_model(data, lengths, n_hidden_states,P, type_model = "HMM"):
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
        model = hmm(data,lengths,n_hidden_states,P=P)
        model.EM()
    if type_model == "AR-AsLG-HMM":
        model = hmm(data,lengths,n_hidden_states,P=P)
        model.SEM()
    if type_model == "KDE-HMM":
        model = kde(data,n_hidden_states,P=P)
        model.EM()
    if type_model == "KDE-AsHMM":
        model = kde(data,n_hidden_states,P=P)
        model.SEM()
    if type_model == "KDE-AsHMM_no_AR":
        model = kde(data,n_hidden_states,lags=False,P=P)
        model.SEM()
    if type_model == "KDE-AsHMM_no_BN":
        model = kde(data,n_hidden_states,struc=False,P=P)
        model.SEM()
    return model



def one_fold_model_training_testing(atlas,categories,n_hidden_states, fold, root,type_model = "HMM",P=4):
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
        model_key = train_model(train,leng, n_hidden_states,P,type_model)
        models[key_id] = model_key
        testin[key_id] = test
        print(key_id)
    
    count_matrix = np.zeros( [len(categories),len(categories)])
    for i, key_id in enumerate(categories):
        test_id = testin[key_id]
        for test_signal in test_id:
            scores = []
            for k in categories:
                scores.append(models[k].log_likelihood(test_signal,xunit=True))
            prediction = categories[np.argmax(scores)]
            pred_index = np.argwhere(np.array(categories) == prediction).T[0][0]
            count_matrix[i][pred_index] = count_matrix[i][pred_index]+1
    return [models, count_matrix]
            

def aggregated_confusion(atlas,categories, n_hidden_states, root,type_model="HMM", P=4, root_models= None ):
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
    
    confusion_0   = one_fold_model_training_testing(atlas, categories,n_hidden_states, 0, root,type_model,P=P)
    if root_models is not None:
        rooti = root_models +"\\"+type_model+"-fold-"+str(0)
        try:
            os.mkdir(rooti)
        except:
            pass
        for j, o in enumerate(categories):
            confusion_0[0][o].save(rooti,name= type_model+"-"+categories[j])
            
    confusion_1   = one_fold_model_training_testing(atlas, categories,n_hidden_states, 1, root,type_model,P=P)
    if root_models is not None:
        rooti = root_models +"\\"+type_model+"-fold-"+str(1)
        try:
            os.mkdir(rooti)
        except:
            pass
        for j, o in enumerate(categories):
            confusion_1[0][o].save(rooti,name= type_model+"-"+categories[j])
                
    confusion_2   = one_fold_model_training_testing(atlas, categories,n_hidden_states, 2, root,type_model,P=P)
    if root_models is not None:
        rooti = root_models +"\\"+type_model+"-fold-"+str(2)
        try:
            os.mkdir(rooti)
        except:
            pass
        for j, o in enumerate(categories):
            confusion_2[0][o].save(rooti,name= type_model+"-"+categories[j])
                
    confusion_3   = one_fold_model_training_testing(atlas, categories,n_hidden_states, 3, root,type_model,P=P)
    if root_models is not None:
        rooti = root_models +"\\"+type_model+"-fold-"+str(3)
        try:
            os.mkdir(rooti)
        except:
            pass
        for j, o in enumerate(categories):
            confusion_3[0][o].save(rooti,name= type_model+"-"+categories[j])
                
    confusion_4   = one_fold_model_training_testing(atlas, categories,n_hidden_states, 4, root,type_model,P=P)
    if root_models is not None:
        rooti = root_models +"\\"+type_model+"-fold-"+str(4)
        try:
            os.mkdir(rooti)
        except:
            pass
        for j, o in enumerate(categories):
            confusion_4[0][o].save(rooti,name= type_model+"-"+categories[j])
    
    confusion = confusion_0[1]+confusion_1[1]+confusion_2[1]+confusion_3[1]+confusion_4[1]

    return confusion

def accuracy(conf_matrix):
    return np.sum(np.diag(conf_matrix))/np.sum(conf_matrix)

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
#%% Prueba computo matrices de confusion para cada fold.
root_models = r"C:\Users\fox_e\Dropbox\Doctorado\Tentative papers\Kernel HMM\KDE-JMM elsevier\kdefig\Audio"
categories = ["dog","cat","hen"]
confusion_hmm     = aggregated_confusion(atlas, categories,3, root,type_model = "HMM",P=1,root_models=root_models)
confusion_ashmm   = aggregated_confusion(atlas, categories,3, root,type_model = "AR-AsLG-HMM",P=1,root_models=root_models)
confusion_kde     = aggregated_confusion(atlas, categories,3, root,type_model = "KDE-HMM",P=1,root_models=root_models)
confusion_askde   = aggregated_confusion(atlas, categories,3, root,type_model = "KDE-AsHMM",P=1,root_models=root_models)
