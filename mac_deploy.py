#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:13:01 2022

@author: anne
"""

from mac_module import EDA,ModelCreation,ModelEvaluation
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
import numpy as np
import datetime
import warnings
import pickle
import json
import os

#%%
warnings.filterwarnings("ignore") # supress warning
'''
This deployment module will categorised your text  into these 5 categories:
['tech' 'business' 'sport' 'entertainment' 'politics']
Super fun! Try it out
'''

#%% Constant

MODEL_PATH = os.path.join(os.getcwd(),'saved_models','model.h5')
TOKEN_PATH = os.path.join(os.getcwd(),'saved_models','tokenizer_data.json')
OHE_PATH = os.path.join(os.getcwd(), 'saved_models', "ohe.pkl")

#%% Path loading

# Model loading
model = load_model(MODEL_PATH)

# Token loading
with open(TOKEN_PATH, "r") as json_file:
    token = json.load(json_file)
# Encoder loading
with open(OHE_PATH, "rb") as r:
    ohe = pickle.load(r)
    
#%% New Input

new_text = [input("Input your text here: ")]

#%% Clean the data
eda = EDA()
clean_text = eda.split(new_text)

#%% Data preprocessing

loaded_token = tokenizer_from_json(token)
clean_text = loaded_token.texts_to_sequences(clean_text)
clean_text = eda.text_pad_sequence(clean_text)

#%% Model prediction
pred = model.predict(np.expand_dims(clean_text, axis=-1))
pred = ohe.inverse_transform(pred)
print(f"The prediction for the input text is {pred}")



