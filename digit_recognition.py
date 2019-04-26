# Imports
import glob

import numpy as np
from matplotlib import image

import models

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load all images in a directory
loaded_images = []
expected_values = []
for filename in glob.glob('D:/Users/DaronPC/Downloads/DataSet/*.bmp'):
    # Load image
    img_data = image.imread(filename)

    # Convert each image to a 1D array
    img_data = np.asarray(img_data)
    img_data = img_data.flatten()

    # Append expected value
    expected_value = int(filename.split('_')[2])
    arr1 = np.zeros(10, dtype=int)
    arr1[expected_value] = 1

    # Store loaded image
    loaded_images.append(img_data)
    # Store expected values
    expected_values.append(arr1)

# Convert list to numpy array
loaded_images = np.asarray(loaded_images, dtype=np.int16)
expected_values = np.asarray(expected_values, dtype=np.int16)

# Split into input (X) and output (Y) variables
X = loaded_images
Y = expected_values
