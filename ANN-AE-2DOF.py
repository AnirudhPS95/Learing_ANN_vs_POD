# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 22:56:29 2021

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 22:42:36 2020

@author: HP
"""




import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Importing tensorflow and keras 
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers

from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split





#%%Reshpaing and plotting data
X=np.loadtxt("2D_LidDriven.dat")

#plt.figure(1)
#plt.scatter(X[:,0], X[:,1])



#%%Data preprocessing
from sklearn.preprocessing import StandardScaler
stn=StandardScaler()
X_norm=stn.fit_transform(X)

#%Encoder_Decoder

#%%Encoder
x=tensorflow.keras.layers.Input(shape=(3), name="encoder_input")

encoder_dense_layer1=tensorflow.keras.layers.Dense(units=100, name="encoder_dense_1")(x)
encoder_activ_layer1=tensorflow.keras.layers.LeakyReLU(name="encoder_leakyrelu_1")(encoder_dense_layer1)

encoder_dense_layer2=tensorflow.keras.layers.Dense(units=50, name="encoder_dense_2")(encoder_activ_layer1)
encoder_activ_layer2=tensorflow.keras.layers.LeakyReLU(name="encoder_leakyrelu_2")(encoder_dense_layer2)

encoder_dense_layer3=tensorflow.keras.layers.Dense(units=2, name="encoder_dense_3")(encoder_activ_layer2)
encoder_output=tensorflow.keras.layers.LeakyReLU(name="encoder_output")(encoder_dense_layer3)

encoder = tensorflow.keras.models.Model(x, encoder_output, name="encoder_model")
encoder.summary()
#%%Decoder
decoder_input = tensorflow.keras.layers.Input(shape=(2), name="decoder_input")

decoder_dense_layer1 = tensorflow.keras.layers.Dense(units=50, name="decoder_dense_1")(decoder_input)
decoder_activ_layer1 = tensorflow.keras.layers.LeakyReLU(name="decoder_leakyrelu_1")(decoder_dense_layer1)

decoder_dense_layer2 = tensorflow.keras.layers.Dense(units=100, name="decoder_dense_2")(decoder_activ_layer1)
decoder_activ_layer2 = tensorflow.keras.layers.LeakyReLU(name="decoder_leakyrelu_2")(decoder_dense_layer2)

decoder_dense_layer3 = tensorflow.keras.layers.Dense(units=3, name="decoder_dense_3")(decoder_activ_layer2)
decoder_output = tensorflow.keras.layers.LeakyReLU(name="decoder_output")(decoder_dense_layer3)

decoder = tensorflow.keras.models.Model(decoder_input, decoder_output, name="decoder_model")
decoder.summary()


#%%Autoencoder
ae_input = tensorflow.keras.layers.Input(shape=(3), name="AE_input")
ae_encoder_output = encoder(ae_input)
ae_decoder_output = decoder(ae_encoder_output)

ae = tensorflow.keras.models.Model(ae_input, ae_decoder_output, name="AE")

ae.summary()


ae.compile(loss="mse", optimizer=tensorflow.keras.optimizers.Adam(lr=0.001))
fp='/Experiment/dontknow.hdf5'
model_cb=ModelCheckpoint(filepath=fp, monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=50,verbose=1)
cb = [model_cb, early_cb]

X_train,X_test,y_train,y_test=train_test_split(X_norm,X_norm,test_size=0.1,random_state=1)

history=ae.fit(X_train, y_train,
                epochs=1000,
                batch_size=10,
                shuffle=True,
                validation_data=(X_test, y_test),
                 )
#%%
#history=ae.fit(X_norm, X_norm, epochs=2000)

#%%
encoded_data=encoder.predict(X_norm)
decoded_data=decoder.predict(encoded_data)

#%%
plt.scatter(X_norm[:,0], X_norm[:,1],label='Real Data',color='blue')
plt.scatter(decoded_data[:,0], decoded_data[:,1], label='Reconstructed Data',color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

#%%
plt.plot(history.history['loss'],label='MSE(loss) v/s epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()

