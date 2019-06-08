import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

#input pandas dataframes of the lemmatized headlines and labels
class implement_model():
    def __init__(self,headlines,labels,vocab_size):
        test_proportion= 0.3
        #pad everything to just 10 words
        max_len = 10
        layer_size = 16
        headlines = pad_sequences(headlines,padding='post',maxlen=max_len)

        training_data,test_data = self.split_data(headlines,labels,test_proportion)

        model = self.build_model(vocab_size,layer_size)

        no_epochs = 10
        batch_size = 100
        self.train_model(model,training_data,no_epochs,batch_size)

    #splits into training & test sets
    def split_data(self,headlines,labels,test_prop):
        l = len(headlines) -1

        indices = np.random.permutation(headlines.shape[0])
        training_indices, test_indices = indices[round(test_prop*l):], indices[:round(test_prop*l)]

        training_data,test_data = headlines[training_indices,:], headlines[test_indices,:]
        training_labels, test_labels = labels[training_indices], labels[test_indices]
        print(len(training_labels),len(test_labels))
        return (training_data, training_labels),(test_data,test_labels)

    def build_model(self,vocab_size,layer_size):
        model=keras.Sequential()
        model.add(keras.layers.Embedding(vocab_size,layer_size))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(layer_size,activation=tf.nn.relu))
        model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
        return model

    def train_model(self,model,data,no_epochs,batch_size):
        data


        history = model.fit(data[0],
                data[1],
                epochs=no_epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=2
        )
        return history
