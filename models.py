import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) #Removes all the warnings.

from keras.layers import Dense
from keras.models import Sequential
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import numpy as np

seed = 7


def main(X, Y, model='nn'):
    classifiers = {
        'nn': nn,
        'svcOvR': svcOvR,
        'svcOvO': svcOvO,
        'kNN': kNN,
        'gNB': gNB,
        'bNB': bNB,
    }
    # Create k-fold
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    # Select classifier
    classifier = classifiers[model]

    # Store Y_true and Y_pred to evaluate overall mcc score
    Y_true = []
    Y_pred = []

    # Split X,Y in 10 folds of train and test
    for train, test in kfold.split(X, Y):
        result = classifier(X[test], Y[test], X[train], Y[train])
        # Create 10*10 grid and iterate through it once to add the values
        # Iterate through it again to find the TP, TN, FP, FN.
        # It might be possible to iterate through it just once.
        # Time complexity will still be O(n^2) so dont sweat it

        # Append each Y_true and Y_pred
        Y_true.append(Y[train])
        # The below has to account for oneshotting
        # Maybe un-oneshot!? before sending the predictons back?
        Y_pred.append(result[0])
        print(result[1])


def nn(X_test, Y_test, X_train, Y_train):
    print("Training on "+str(len(X_train))+ " samples. Validating on "+str(len(X_test))+ " samples.")
    Y_test = oneShotY(Y_test)
    Y_train = oneShotY(Y_train)

    # #Normalize
    # X_train = X_train / 255
    # X_test = X_test / 255

    # Create model
    model = Sequential()
    model.add(Dense(1024, input_dim=1024, activation='relu', kernel_initializer='normal'))
    # model.add(Dense(128, activation='relu')) #With 85.4%, Without 86?
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=0)

    # Evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    score = scores[1] * 100
    predictions = model.predict(X_test)

    return (predictions, score)


def svcOvR(X_test, Y_test, X_train, Y_train):
    # Create classifier
    clf = OneVsRestClassifier(
        SVC(random_state=seed, gamma='scale'))
    clf.fit(X_train, Y_train)

    # Evaluate classifier
    score = clf.score(X_test, Y_test) * 100
    predictions = clf.predict(X_test)

    return (predictions, score)


def svcOvO(X_test, Y_test, X_train, Y_train):
    # Create classifier
    clf = OneVsOneClassifier(
        SVC(random_state=seed, gamma='scale'))
    clf.fit(X_train, Y_train)

    # Evaluate classifier
    score = clf.score(X_test, Y_test)
    predictions = clf.predict(X_test)

    return (predictions, score)


def kNN(X_test, Y_test, X_train, Y_train):
    # Create classifier
    neigh = KNeighborsClassifier()
    neigh.fit(X_train, Y_train)

    # Evaluate classifier
    score = neigh.score(X_test, Y_test)
    predictions = neigh.predict(X_test)

    return (predictions, score)


def gNB(X_test, Y_test, X_train, Y_train):
    # Create classifier
    clf = GaussianNB()
    clf.fit(X_train, Y_train)

    # Evaluate classifier
    score = clf.score(X_test, Y_test)
    predictions = clf.predict(X_test)

    return (predictions, score)


def bNB(X_test, Y_test, X_train, Y_train):
    Y_test = oneShotY(Y_test)
    Y_train = oneShotY(Y_train)

    # Create classifier
    clf = BernoulliNB()
    clf.fit(X_train, Y_train)

    # Evaluate classifier
    score = clf.score(X_test, Y_test)
    predictions = clf.predict(X_test)

    return (predictions, score)


# One shot target variables
def oneShotY(Y):
    new_Y = []
    for digit in Y:
        one_shot = np.zeros(10, dtype=int)
        one_shot[digit] = 1
        new_Y.append(one_shot)
    return np.asarray(new_Y)
