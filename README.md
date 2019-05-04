# Intelligent Systems Final Assignment

At the end of this assignment, the students will

- Have worked in a team to develop a basic Hand-Written Digit Categorization system using various learning algorithms including:
  - Artifcial Neural Networks (ANNs)
  - Support Vector Machines (SVMs)
  - k-Nearest Neigbhors (kNNs)
  - Naive-Bayesian Model (NB)
- Have transformed a scanned bitmap dataset into features for training and testing;
- Have incorporated proper training/testing strategies, in particular cross-validation and simple statis-
tical analysis of the results;
- Have communicated their results through a detailed written report.

## classify.py

Responsible for selecting the classifier, training, k-folding, creating results and saving models

## digit_categorization.py

Responsible for reading in DataSet/ and selecting whether to use one specific classifier or to use all the classifiers

## driver.py

Main point of entry to run the program

## DataSet/

This directory contains scanned handwritten digits.
The process took a set of digits written in individual cells, cropped them and resized them to fit a 32x32 bitmap.
The files are stored as input_N_D_X.type where

- N represents the specific "user" number.
- D represents the specific "digit" 0-9
- X represents one specific instance (should be 10 instances per user per digit)

The bmp is an actual bitmap image that you can open in a graphical viewer but it requires a bit more processing to read in (though it isn't hard).
The data is more human readable and easily read with a simple program.
   The format consists of one line WIDTH HEIGHT giving the dimensions of the image (e.g. 32x32)
   This is then followed by HEIGHT rows of WIDTH gray-scale values.
     The values range from 0 to 255 with 0 meaning BLACK and 255 meaning WHITE
The json is a JSON data dump of the bitmap and is more machine readable.
   When loaded it should just be a two-dimensional array.

E.g. input_3_2_4.data is the human readable scan of the 4th occurrence of digit 2 for user 3.

## CompetionDataSet/

This directory contains scanned handwritten digits different from the ones in DataSet. This was used to make predictions using a specific
classifier('our team used svcOvO') different models for Intelligent Systems(CSC 350) final assignment.

## csv/

This directory contains the results of different classifiers during the testing phase of developing.
The results contains mcc score, accuracy, precision, recall for each digit in each fold and a overall accuracy and mcc score for each fold.

## models/

This directory contains the models created from each classifier when training on the entire data in DataSet.

## competion.csv

This contains the predictions and file number for each input in CompetionDataSet.
