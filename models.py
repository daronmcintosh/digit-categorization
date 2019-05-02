import csv
import numpy as np
from sklearn import metrics
from math import sqrt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
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


# Unonshot target variables
def unOneShotY(Y):
    new_Y = []
    for arr in Y:
        digit = np.argmax(arr)
        new_Y.append(digit)
    return np.asarray(new_Y)


def nn(X_test, Y_test, X_train, Y_train):
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
    model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=1)

    # Evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    score = scores[1]
    predictions = model.predict(X_test)

    predictions = unOneShotY(predictions)

    return (predictions, score, model)


def svcOvR(X_test, Y_test, X_train, Y_train):
    # Create classifier
    model = OneVsRestClassifier(
        SVC(random_state=seed, gamma='scale'))
    model.fit(X_train, Y_train)

    # Evaluate classifier
    score = model.score(X_test, Y_test)
    predictions = model.predict(X_test)

    return (predictions, score, model)


def svcOvO(X_test, Y_test, X_train, Y_train):
    # Create classifier
    model = OneVsOneClassifier(
        SVC(random_state=seed, gamma='scale'))
    model.fit(X_train, Y_train)

    # Evaluate classifier
    score = model.score(X_test, Y_test)
    predictions = model.predict(X_test)

    return (predictions, score, model)


def kNN(X_test, Y_test, X_train, Y_train):
    # Create classifier
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)

    # Evaluate classifier
    score = model.score(X_test, Y_test)
    predictions = model.predict(X_test)

    return (predictions, score, model)


def gNB(X_test, Y_test, X_train, Y_train):
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


def classify(X, Y, model='nn', kFold=True):
    # Create k-fold
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    # Select classifier
    classifier = classifiers[model]

    # Create a file using the model choosen
    with open(f'{model}.csv', mode='w', newline='') as output_file:
        output_writer = csv.writer(output_file)

        # Split X,Y in 10 folds of train and test
        loop_number = 0
        for train, test in kfold.split(X, Y):
            # Keep track of fold
            loop_number += 1
            fold_str = f'Fold {loop_number}'

            # Create column headers
            output_writer.writerow([fold_str, 'Digit', 'Precision', 'Recall',
                                    'F1-Score', 'MCC Score', 'Accuracy', 'TP',
                                    'TN', 'FP', 'FN'])

            # Get accuracy and predictions from classifer
            result = classifier(X[test], Y[test], X[train], Y[train])

            # Get prediction, score, model from results
            predictions, score, model = result[0], result[1], result[2]

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

                # Write values to file
                output_writer.writerow(['', digit, precision, recall, f1_score,
                                        mcc_score, accuracy, tp, tn, fp, fn])
                digit += 1

            # Calculate and write overall mcc
            overall_mcc_score = metrics.matthews_corrcoef(Y[test], predictions)
            overall_mcc_score = np.round(overall_mcc_score, decimals=2)
            output_writer.writerow(['Overall MCC', overall_mcc_score])

            # Calculate and write overall accuracy
            overall_accuracy = np.round(score, decimals=2)
            output_writer.writerow(['Overall Accuracy', overall_accuracy])

            # Write a blank row to separate folds
            output_writer.writerow([])
