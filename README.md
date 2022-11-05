# Learing_ANN
Mtech Project part dedicated to learning ANN and comparing it with conventional tool used for mode decomposition i.e. POD


## Description 

This part of Mtech project was to learn about ANN and showcase its powers over conventional tool used for mode decomposition being POD. Then some hyper parameter tuning 
was explored. The hyperparameters taken into consideration were number of nodes in each layer and the number of layers. 


## Comparing POD with linear AE and non-linear AE for 1-DOF data

First a simple Artificial neural network of autoencoder structure was used on 1 DOF data. The inspiration for this was taken from towardsdatascience. Some datasets were
created for the 1DOF datafrom linearly verying to completely non-linear data. For all both the ANN and POD were used and thus comparison was drawn( this also one can 
find in the paper related to this topic(will update the name of the paper)). 
  The input dataset is a vector of size containing all the datapoints.
  The linear autoencoder uses linear activation function or in essence it activates the ANN node with value as is.
  
Following observations could be made from the results obtained:
  1. A linear AE(autoencoder) is same as POD. For that matter POD is superior to ANN with POD being a one step solver and ANN being an iterative solver
  2. The non-linear-AE is where ANN pulls aways from POD


## Comparing POD with non-linear AE for 2-DOF data

For this still a simple ANN-AE was used. The data were 3-D equation and 2 fluid flow velocity contour plots. One of the contour plots was of simple Lid-Driven cavity 
flow. The code for the same is also given along with rest of the materials. 


## PS
The complete work can be read in the dissertation report shared herewith. The hypertuning of parameters is also given in detail
