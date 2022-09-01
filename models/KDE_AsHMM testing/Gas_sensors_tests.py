# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:53:32 2022
@author: Carlos Puerto-Santana at KTH royal institute of technology, Stockholm, Sweden
@Licence: CC BY 4.0
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import Orange
import os
path = os.getcwd()
models = os.path.dirname(path)
os.chdir(models)
#%% Load data
data_gas = pd.read_csv(r"C:\Users\fox_e\OneDrive\Documentos\datasets\Gas sensors\HT_Sensor_dataset.dat",header=0)
