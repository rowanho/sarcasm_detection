import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# the model implementation using an embedding layer, and padded numerical
# representations of each word for the
class implement_model():
    def __init__(self,headlines,labels,vocab_size):
        test_proportion= 0.3
        #pad everything to just 10 words
        max_len = 10
        layer_size = 32
        headlines = pad_sequences(headlines,padding='post',maxlen=max_len)
        training_data,test_data = self.split_data(headlines,labels,test_proportion)

        model = self.build_model(vocab_size,layer_size)

        no_epochs = 10
        batch_size = 1000

        #validation
        self.history  = self.validate_model(model,training_data,no_epochs,batch_size)

        #final training
        new_model = self.build_model(vocab_size,layer_size)
        self.train_final_model(new_model,training_data,no_epochs,batch_size)
        self.final_res = new_model.evaluate(x=test_data[0],y=test_data[1])
        new_model.save("saved_model.h5")

    # splits into training & test sets
    def split_data(self,headlines,labels,test_prop):
        l = len(headlines) -1

        indices = np.random.permutation(headlines.shape[0])
        training_indices, test_indices = indices[round(test_prop*l):], indices[:round(test_prop*l)]

        training_data,test_data = headlines[training_indices,:], headlines[test_indices,:]
        training_labels, test_labels = labels[training_indices], labels[test_indices]
        return (training_data, training_labels),(test_data,test_labels)

    def build_model(self,vocab_size,layer_size):
        model=keras.Sequential()
        model.add(keras.layers.Embedding(vocab_size,layer_size))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(layer_size*2,activation=tf.nn.relu))
        model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
        model.compile(optimizer='RMSprop',loss='binary_crossentropy',metrics=['acc'])
        return model

    def plot_hist(self):
        history = self.history
        plt.figure(figsize=(21,11))
        plt.plot(history.epoch,history.history['val_acc'],label="validation data accuracy")
        plt.plot(history.epoch,history.history['acc'],label="training data accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.xlim([0,max(history.epoch)])
        plt.title("Validation vs training accuracy")
        plt.show()

    def validate_model(self,model,data,no_epochs,batch_size):
        history = model.fit(data[0],
                data[1],
                epochs=no_epochs,
                batch_size=batch_size,
                validation_split=0.3,
                verbose=2
        )
        return history

    def train_final_model(self,model,data,no_epochs,batch_size):
        history = model.fit(data[0],
            data[1],
            epochs=no_epochs,
            batch_size=batch_size,
            verbose=2
        )
