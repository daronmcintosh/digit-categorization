# Imports
import glob

import numpy as np
from matplotlib import image
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import models

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load all images in a directory
loaded_images = []
expected_values = []
filename_list = []
for filename in glob.glob('C:/Users/DaronLaptop/Downloads/DataSet/*.bmp'):
    # Load image
    img_data = image.imread(filename)

    # Convert each image to a 1D array
    img_data = np.asarray(img_data)
    img_data = img_data.flatten()

    # Append expected value
    expected_value = int(filename.split('_')[2])
    arr1 = np.zeros(10, dtype=int)
    arr1[expected_value] = 1

    # Append filename to filename_list
    filename_list.append(filename)

    # Store loaded image
    loaded_images.append(img_data)
    # Store expected values
    # expected_values.append(arr1)
    expected_values.append(expected_value)

# Convert list to numpy array
loaded_images = np.asarray(loaded_images, dtype=np.float32)
expected_values = np.asarray(expected_values, dtype=np.float32)

# Split into input (X) and output (Y) variables
X = loaded_images
Y = expected_values
percent = int(len(loaded_images)*0.9)
X_train = X[:percent]
Y_train = Y[:percent]
X_test = X[percent:]
Y_test = Y[percent:]

# models.main(X, Y, model='nn')
# print(models(X, Y, model="svcOvR"))

# clf = GaussianNB()
# clf.fit(X_train, Y_train)
# score = clf.score(X_test, Y_test)
# predictions = clf.predict(X_test)
# print(score)

neigh = KNeighborsClassifier()
neigh.fit(X_train, Y_train)
score = neigh.score(X_test, Y_test)
print(score)

# clf = OneVsOneClassifier(
#     LinearSVC(random_state=seed))
# clf.fit(X_train, Y_train)
# score = clf.score(X_test, Y_test)
# print(score)

# # Create model
# model = Sequential()
# model.add(Dense(1024, input_dim=1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# # Compile model
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam', metrics=['accuracy'])

# # Fit the model
# model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)

# # Evaluate the model
# scores = model.evaluate(X_test, Y_test, verbose=0)
# print(scores[1] * 100)

# # Get Predictions
# predictions = model.predict(X_test)
# print()
