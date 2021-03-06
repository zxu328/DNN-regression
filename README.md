# DNN-regression
This is a repo of DNN state level gun-ownership regression software by Python and Pytorch 1.3.1.

This software is equiped with CUDA GPU parallel training.

The source code is included in code directory 

This code includes:

(1) CVDNN.py: Main file to train DNN with k-fold cross-validation. The DNN is tuned by training data, and the testing errors are supervised in training process to prevent overfitting.

(2) DNNClass.py: The DNN class file includes fit and predict functions. Fit function is used to train the DNN and predict function is used to predict the label of fut=ure features by thrained DNN.

(3) model.py: 3 hidden layers DNN model file.

(4) utilities.py: utilities functions. Including loadData and calRSquare functions. loadData is used to load .xlsx training and testing data, and calRSquare is used to calculate R2 coefficient of label y and prediction y_hat.

(5) TrainingData.xlsx: training dataset.

(6) TestingData.xlsx: testing dataset 1 (validation #1).

(7) testingData.xlsx: testing dataset 2 (validation #2).

(8) DNNResidualTrain.dat: DNN training residual results.

(9) DNNResidualTest.dat: DNN testing dataset1 residual results.

(10) DNNResidualTest2.dat: DNN testing dataset2 residula results.

(11) savedModel.pt: A pre-saved DNN model trained by the developers. It has testing R2 of 96.63%. (This 96.63% value may be varied for different machine and machine precision. The model should be retrained if this machine error happens by seting 'reload = 0' in CVDNN.)



Users can download our source codes and run their own training and testing on python and pythorch platform.

Run 'python CVDNN.py' on terminal under correct directory path can run our program.


If the users want use their own data instead of our provided testing data, they should do the following steps:


1. Collect their data and format it as our 'testingData.xlsx' file.

2. Name the data file as 'testingData.xlsx' to replace the original one.

3. Run 'python CVDNN.py' on the termial under 'code' directory.

4. Get model prediction residuals on the users' own testing data in 'DNNResidualTest.dat'.

