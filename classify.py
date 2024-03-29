import csv
import pathlib
from math import sqrt

import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

tf.logging.set_verbosity(tf.logging.ERROR)  # Removes all the warnings.


seed = 7


# Oneshot target variables
def oneShotY(Y):
    new_Y = []
    for digit in Y:
        one_shot = np.zeros(10, dtype=int)
        one_shot[digit] = 1
        new_Y.append(one_shot)
    return np.asarray(new_Y)


# Unoneshot target variables
def unOneShotY(Y):
    new_Y = []
    for arr in Y:
        digit = np.argmax(arr)
        new_Y.append(digit)
    return np.asarray(new_Y)


# Get input and output based on the amount of parameters passed
def get_input_output(X, Y, *args):
    X_test = X
    Y_test = Y
    X_train = args[0] if len(args) == 2 else X
    Y_train = args[1] if len(args) == 2 else Y
    return (X_test, Y_test, X_train, Y_train)


def nn(X, Y, *args):
    X_test, Y_test, X_train, Y_train = get_input_output(X, Y, *args)
    Y_test = oneShotY(Y_test)
    Y_train = oneShotY(Y_train)

    # Create model
    model = Sequential()
    model.add(Dense(1024, input_dim=1024, activation='relu',
                    kernel_initializer='normal'))
    # model.add(Dense(128, activation='relu'))  # With 85.4%, Without 86?
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=0)

    # Evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    score = scores[1]
    predictions = model.predict(X_test)

    predictions = unOneShotY(predictions)

    return (predictions, score, model)


def svcOvR(X, Y, *args):
    X_test, Y_test, X_train, Y_train = get_input_output(X, Y, *args)
    # Create classifier
    model = OneVsRestClassifier(
        SVC(random_state=seed, gamma='scale'))
    model.fit(X_train, Y_train)

    # Evaluate classifier
    score = model.score(X_test, Y_test)
    predictions = model.predict(X_test)

    return (predictions, score, model)


def svcOvO(X, Y, *args):
    X_test, Y_test, X_train, Y_train = get_input_output(X, Y, *args)
    # Create classifier
    model = OneVsOneClassifier(
        SVC(random_state=seed, gamma='scale'))
    model.fit(X_train, Y_train)

    # Evaluate classifier
    score = model.score(X_test, Y_test)
    predictions = model.predict(X_test)

    return (predictions, score, model)


def kNN(X, Y, *args):
    X_test, Y_test, X_train, Y_train = get_input_output(X, Y, *args)
    # Create classifier
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)

    # Evaluate classifier
    score = model.score(X_test, Y_test)
    predictions = model.predict(X_test)

    return (predictions, score, model)


def gNB(X, Y, *args):
    X_test, Y_test, X_train, Y_train = get_input_output(X, Y, *args)
    # Create classifier
    model = GaussianNB()
    model.fit(X_train, Y_train)

    # Evaluate classifier
    score = model.score(X_test, Y_test)
    predictions = model.predict(X_test)

    return (predictions, score, model)


classifiers = {
    'nn': nn,
    'svcOvR': svcOvR,
    'svcOvO': svcOvO,
    'kNN': kNN,
    'gNB': gNB
}


def classify(X, Y, classifierName, useKFold=True):
    # Select classifier
    classifier = classifiers[classifierName]

    # Train and Predict on entire data then save the model
    if(not useKFold):
        result = classifier(X, Y)
        model = result[2]
        print(classifierName, end=" ")
        print(f'Accuracy: {result[1]:.2f}', end=" ")
        print(f'MCC score: {metrics.matthews_corrcoef(Y, result[0]):0.2f}')

        # Create directory if it does not exists
        pathlib.Path('models/').mkdir(exist_ok=True)

        if(classifierName == 'nn'):
            model.save('models/nn_model.h5')
        else:
            joblib.dump(model, f'models/{classifierName}_model.pkl')
        return

    # Create k-fold
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    # Create directory if it does not exists
    pathlib.Path('csv/').mkdir(exist_ok=True)

    # Create filename using the classifier choosen
    filename = f'csv/{classifierName}.csv'
    # Create a file using the model choosen
    with open(filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Split X,Y in 10 folds of train and test
        loop_number = 0
        for train, test in kfold.split(X, Y):
            # Keep track of fold
            loop_number += 1
            fold_str = f'Fold {loop_number}'

            # Create column headers
            csv_writer.writerow([fold_str, 'Digit', 'Precision', 'Recall',
                                 'F1-Score', 'MCC Score', 'Accuracy', 'TP',
                                 'TN', 'FP', 'FN', '', 'Confusion Matrix'])

            # Get accuracy and predictions from classifer
            result = classifier(X[test], Y[test], X[train], Y[train])

            # Get prediction, score from results
            predictions, score = result[0], result[1]

            # Generate confusion matrix
            confusion_matrix = metrics.confusion_matrix(Y[test], predictions)

            # Get TP, TN, FP, FN values
            FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
            FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            TP = np.diag(confusion_matrix)
            TN = confusion_matrix.sum() - (FP + FN + TP)

            digit = 0
            for tp, tn, fp, fn in zip(TP, TN, FP, FN):
                # P = TP/(TP + FP)
                dividend = tp
                divisor = (tp+fp)
                precision = 1 if divisor == 0 else dividend / divisor
                # R = TP/(TP + FN)
                recall = tp/(tp+fn)
                # F1 = (2*P*R)/(P+R)
                f1_score = (2*precision*recall)/(precision+recall)
                # MCC = ((TP * TN) - (FP*FN)) divided by
                #       (sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)))
                dividend = (tp * tn) - (fp*fn)
                divisor = sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
                mcc_score = 1 if divisor == 0 else dividend / divisor
                # ACC = (TP+TN)/(TP+TN+FP+FN)
                accuracy = (tp+tn)/(tp+tn+fp+fn)
                # Round values
                precision = np.round(precision, decimals=2)
                recall = np.round(recall, decimals=2)
                f1_score = np.round(f1_score, decimals=2)
                mcc_score = np.round(mcc_score, decimals=2)
                accuracy = np.round(accuracy, decimals=2)

                # Unpacking confusion matrix
                c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = \
                    confusion_matrix[digit]

                # Write values to file
                csv_writer.writerow(['', digit, precision, recall, f1_score,
                                     mcc_score, accuracy, tp, tn, fp, fn, '',
                                     c0, c1, c2, c3, c4, c5, c6, c7, c8, c9
                                     ])
                digit += 1
            # Calculate and write overall mcc
            overall_mcc_score = metrics.matthews_corrcoef(Y[test], predictions)
            overall_mcc_score = np.round(overall_mcc_score, decimals=2)
            csv_writer.writerow(['Overall MCC', overall_mcc_score])

            # Calculate and write overall accuracy
            overall_accuracy = np.round(score, decimals=2)
            csv_writer.writerow(['Overall Accuracy', overall_accuracy])

            # Write a blank row to separate folds
            csv_writer.writerow([])
