import tensorflow
from keras import layers, models, optimizers
import keras
import tensorflow as tf

def sequence_model(sequence_length, lat, lon, nvar):
     model = models.Sequential([
         layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=(sequence_length, lat, lon, nvar),
                           padding='same', return_sequences=True),
         layers.BatchNormalization(),
         layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True),
         layers.BatchNormalization(),
         layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False),  
         layers.BatchNormalization(),
         layers.Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same') 
     ])
     return model


#def sequence_model(sequence_length, lat, lon, nvar, dropout_rate=0.2):
#    model = models.Sequential([
#        layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=(sequence_length, lat, lon, nvar),
#                          padding='same', return_sequences=True),
#        layers.BatchNormalization(),
#        layers.Dropout(dropout_rate), 
#        
#        layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True),
#        layers.BatchNormalization(),
#        layers.Dropout(dropout_rate), 
#        
#        layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False),
#        layers.BatchNormalization(),
#        layers.Dropout(dropout_rate),  
#        
#        layers.Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same') 
#    ])
#    return model

