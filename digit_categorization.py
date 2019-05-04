# Imports
import glob

import numpy as np
from matplotlib import image

import classify

# Random seed for reproducibility
seed = 7
np.random.seed(seed)


# Load all images in a directory
def load_images():
    loaded_images = []
    expected_values = []
    filename_list = []
    for filename in glob.glob('DataSet/*.bmp'):
        # Load image
        img_data = image.imread(filename)

        # Convert each image to a 1D array
        img_data = np.asarray(img_data)
        img_data = img_data.flatten()

        # Append expected value
        expected_value = int(filename.split('_')[2])

        # Append filename to filename_list
        filename_list.append(filename)

        # Store loaded image
        loaded_images.append(img_data)
        # Store expected values
        expected_values.append(expected_value)

    # Convert list to numpy array
    loaded_images = np.asarray(loaded_images, dtype=np.int32)
    expected_values = np.asarray(expected_values, dtype=np.int32)

    # Split into input (X) and output (Y) variables
    X = loaded_images
    Y = expected_values

    # Normalize
    X = X / 255
    return (X, Y)


# Classify all
def classifyAll(useKFold):
    X, Y = load_images()
    for classifierName in classify.classifiers:
        classify.classify(X, Y, classifierName, useKFold=useKFold)


# Classify one
def classifyOne(useKFold, classifierName='nn'):
    X, Y = load_images()
    classify.classify(X, Y, classifierName, useKFold=useKFold)
