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
from KDE_AsHMM import KDE_AsHMM
from AR_ASLG_HMM import AR_ASLG_HMM as hmm
from librosa.feature import mfcc
#%% Functions
def cut_mfcc(features):
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
    train = np.zeros([0,12])    
    files = atlas[fold][category]
    for f in files:
        samplerate, sig = wavfile.read(root+"\\"+f)
        # playsound(root+"\\"+f)
        sig = sig.astype(float)
        mfcc_feat = cut_mfcc(mfcc(sig,sr=samplerate,n_mfcc = 13).T[:,1:])
        if np.sum(mfcc_feat) != 0:
            train = np.concatenate([train,mfcc_feat],axis=0)
    train = train+np.random.normal(0,1,[len(train),12])
    test = []
    for i in range(5):
        if i != fold:
            testi = np.zeros([0,12])    
            files = atlas[i][category]
            for f in files:
                samplerate, sig = wavfile.read(root+"\\"+f)
                sig = sig.astype(float)
                mfcc_feat = cut_mfcc(mfcc(sig,sr=samplerate,n_mfcc = 13).T[:,1:])
                if np.sum(mfcc_feat) != 0:
                    testi = np.concatenate([testi,mfcc_feat],axis=0)
                testi = testi+np.random.normal(0,1,[len(testi),12])
            test.append(testi)
    return train , test
    

#%% Crear atlas
root = r"C:\Users\fox_e\OneDrive\Documentos\datasets\Voice_recognition\audio\audio\16000"
catalog = pd.read_csv(r"C:\Users\fox_e\OneDrive\Documentos\datasets\Voice_recognition\esc50.csv")
categories =  set(catalog["category"])
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
train_dog, test_dog = extract_data(atlas,0,"dog",root)    
train_cat, test_cat = extract_data(atlas,0,"cat",root)    
#%%
model_perros = KDE_AsHMM(train_dog,3)
model_perros.EM()

model_gatos = KDE_AsHMM(train_cat,3)
model_gatos.EM()

print("perros-perros")
print(model_perros.log_likelihood(test_dog[0], xunit=True))
print(model_perros.log_likelihood(test_dog[1], xunit=True))
print(model_perros.log_likelihood(test_dog[2], xunit=True))
print(model_perros.log_likelihood(test_dog[3], xunit=True))
print("perros-gatos")
print(model_perros.log_likelihood(test_cat[0], xunit=True))
print(model_perros.log_likelihood(test_cat[1], xunit=True))
print(model_perros.log_likelihood(test_cat[2], xunit=True))
print(model_perros.log_likelihood(test_cat[3], xunit=True))


print("gatos-perros")
print(model_gatos.log_likelihood(test_dog[0], xunit=True))
print(model_gatos.log_likelihood(test_dog[1], xunit=True))
print(model_gatos.log_likelihood(test_dog[2], xunit=True))
print(model_gatos.log_likelihood(test_dog[3], xunit=True))
print("gatos-gatos")
print(model_gatos.log_likelihood(test_cat[0], xunit=True))
print(model_gatos.log_likelihood(test_cat[1], xunit=True))
print(model_gatos.log_likelihood(test_cat[2], xunit=True))
print(model_gatos.log_likelihood(test_cat[3], xunit=True))
