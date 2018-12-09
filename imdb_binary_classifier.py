# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:05:42 2018

@author: karan.verma
"""

import numpy as np
from keras import models
from keras import layers
from keras.datasets import imdb
from keras import optimizers
import matplotlib.pyplot as plt


# using all word freqeuncies
#(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

# using all word freqeuncies = 10k
(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words = 10000)

#convert data to tensor using one hot encoding, essentially putting 1 where there is a value and a 0 otherwise
#split data only after converting otherwise it wont be compatible
x_train = np.zeros([len(train_data),10000])
for number, sequence in enumerate(train_data):
    x_train[number, sequence] = 1

#in the book it is done as a function...but like right now.....fuck it
x_test = np.zeros([len(test_data),10000])
for number, sequence in enumerate(test_data):
    x_test[number, sequence] = 1

#vectorize labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# HAVE NOT RANDOMIZED THE DATA!!!! TRY AND DO THAT LATER
#get validation split, 60% training, 40% validation --> 15000 TRAINING, 10000 VALIDATION
x_train_partial, validation_x_train = x_train[:15000], x_train[15000:]
y_train_partial, validation_y_train = y_train[:15000], y_train[15000:]

#model iterations
activations = ['relu', 'tanh']
loss_func = ['mse', 'binary_crossentropy']
hidden_layer_units = [16, 32, 64]
hidden_layers = [1,2,3]

for activation in activations:
    for function in loss_func:
        for units in hidden_layer_units:
            for lyrs in hidden_layers:
                
            
                #write model
                model = models.Sequential()
                model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
                
                for i in range(0,lyrs):
                    model.add(layers.Dense(units, activation = activation))    
                    
                model.add(layers.Dense(1, activation = 'sigmoid'))
                
                #optimzers, loss etc.
                model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
                              loss = function,
                              metrics = ['accuracy'])
                
                history = model.fit(x_train_partial,
                                    y_train_partial,
                                    epochs = 10,
                                    batch_size = 512,
                                    validation_data=(validation_x_train, validation_y_train))
                
                #for graphs
                hist_dict = history.history
                validation_loss_values = hist_dict['val_loss']
                validation_acc_values = hist_dict['val_acc']
                training_loss_values = hist_dict['loss']
                training_acc_values = hist_dict['acc']
                
                epochs = np.arange(1,11)
                
                plt.plot(epochs, validation_loss_values, 'b', label = 'Validation Loss')
                plt.plot(epochs, training_loss_values, 'r', label='Training Loss')
                plt.title('Loss with ACTIVATION %s, LOSS FUNCTION %s, %d HIDDEN LAYERS & %d HIDDEN UNITS' % (activation, function, lyrs, units))
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()                
                plt.clf()
                
                plt.plot(epochs, validation_acc_values, 'b', label = 'Validation Accuracy')
                plt.plot(epochs, training_acc_values, 'r', label='Training Accuracy')
                plt.title('Accuracy with ACTIVATION %s, LOSS FUNCTION %s, %d HIDDEN LAYERS & %d HIDDEN UNITS' % (activation, function, lyrs, units))
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()                
                plt.clf()


    
