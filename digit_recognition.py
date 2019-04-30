# Imports
import glob
import numpy as np
import models
from matplotlib import image

# Random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load all images in a directory
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

# Classify all
# for classifier in models.classifiers:
#     models.classify(X, Y, model=classifier)

# Classify one
models.classify(X, Y, model='bNB')
