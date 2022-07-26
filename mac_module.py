#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 09:38:57 2022

@author: anne
"""

import os
import re
import json
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#%%
URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
MODEL_PATH = os.path.join(os.getcwd(),'saved_models','model.h5')
TOKEN_PATH = os.path.join(os.getcwd(),'saved_models','tokenizer_data.json')
OHE_PATH = os.path.join(os.getcwd(), 'saved_models', "ohe.pkl")
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
#%%

class EDA():
    
    def split(self,data):
        '''
        This function converts all letters: split into list.
        Also filters numerical data
        '''

        for index, text in enumerate(data):
            data[index] = re.sub('[^a-zA-Z]', ' ', text).split()
            
        return data
        
    def category_tokenizer(self,data,token_save_path,
                            num_words=10000,oov_token='<OOV>',prt=False):
    
        # Tokenizer to vectorize the words
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)
    
        # To save the tokenizer for deployment purpose
        token_json = tokenizer.to_json()
        with open(TOKEN_PATH, 'w') as json_file:
            json.dump(token_json, json_file)
        
        # to observe the number of words
        word_index = tokenizer.word_index
    
        if prt == True:
            # to view the tokenized words
            # to print(word_index)
            print(dict(list(word_index.items())[0:10]))
        
        # To vectorize the sequences of text 
        data = tokenizer.texts_to_sequences(data)
    
        return data

    def text_pad_sequence(self,data):
        return pad_sequences(data, maxlen=335,padding='post',
                             truncating='post')
        
class ModelCreation():
    
    def lstm_layer(self,num_words, nb_categories, 
                   embedding_output=64, nodes=32, dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))  # added the embedding layer
        model.add(Bidirectional(LSTM(nodes,return_sequences=True))) #added bidirectional
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes,return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))
        model.summary()
        
        return model

class ModelEvaluation():
    
    def cr(self, y_true, y_pred):
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))
    
    def plot_hist_graph(self,hist):
        plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.xlabel('epoch')
        plt.legend(['Training Loss','Validation Loss'])
        plt.show()

        plt.figure()
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.xlabel('epoch')
        plt.legend(['Training Acc','Validation Acc'])
        plt.show()




