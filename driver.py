import csv
import glob
import re

import numpy as np
from keras import models
from matplotlib import image
from sklearn.externals import joblib

import digit_recognition


def runCompetition(folderName, classifierName):
    loaded_images = []
    file_number_list = []
    for filename in glob.glob(f'{folderName}/*.bmp'):
        # Load image
        img_data = image.imread(filename)

        # Convert each image to a 1D array
        img_data = np.asarray(img_data)
        img_data = img_data.flatten()

        # Append filename to file_number_list
        file_number = re.findall(r'\d+', filename)[0]
        file_number_list.append(file_number)

        # Store loaded image
        loaded_images.append(img_data)

    # Convert list to numpy array
    loaded_images = np.asarray(loaded_images, dtype=np.int32)

    # Split into input (X) and output (Y) variables
    X = loaded_images

    # Normalize
    X = X / 255

    # Attempt to load model a max of 3 times
    for i in range(0, 3):
        while True:
            try:
                if(classifierName == 'nn'):
                    model = models.load_model('models/nn_model.h5')
                else:
                    model = joblib.load(f'models/{classifierName}_model.pkl')
            except (OSError, FileNotFoundError):
                # Create and save model using classifierName
                digit_recognition.classifyOne(
                    False, classifierName=classifierName)
                continue
            break

    # Make predictions
    predictions = model.predict(X)

    # Output csv file
    with open('competition.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for file_number, prediction in zip(file_number_list, predictions):
            csv_writer.writerow([file_number, prediction])


# Three ways to run the program
# digit_recognition.classifyOne(False, classifierName='svcOvR')
digit_recognition.classifyAll(False)
# runCompetition('SOMEFOLDER', 'svcOvO')
exit()
