# Convolutional-Neural-Network

## 1.1 DeepCrack
This a convolutional Neural Network just as U-Net to segment pavement crack from background. This code is taken from source of DeepCrack (https://github.com/qinnzou/DeepCrack). We are planning to fine tune DeepCrack 
on our pavement distress datasets. We are going through several test runs on CRACKTRE260 and EdmCrack600 and other published pavement distress datasets. 

Before using this model to your research cite from Zou Q, Zhang Z, Li Q, Qi X, Wang Q and Wang S, DeepCrack: Learning Hierarchical Convolutional Features for Crack Detection, IEEE Transactions on Image Processing, vol. 28, no. 3, pp. 1498-1512, 2019.

## VGG 19 

Applied VGG 19 to predict the class of cracks in EdmCrack 600. I categorizied all of the 600 cracks into Longitudinal, Alligator, Transverse and Delaminations . Then tuned the VGG19 with different hyperparameters to predict the pavemenent distress types. 

Ref: Q. Mei, M. Gül, and M.R. Azim, Densely connected deep neural network considering connectivity of pixels for automatic crack detection. Automation in Construction, 2020. 110: p. 103018. 
