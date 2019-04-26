from keras.layers import Dense
from keras.models import Sequential
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

seed = 7


# TODO: neural network and bnb predicts the same thing consistently


def test():
    print('Hello')


def nn(X, Y):
    # Create model
    model = Sequential()
    model.add(Dense(1024, input_dim=1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

    # Evaluate the model
    scores = model.evaluate(X, Y, verbose=0)
    score = scores[1] * 100

    # Get Predictions
    predictions = model.predict(X)
    return (predictions, score)


def svcOvR(X, Y):
    clf = OneVsRestClassifier(
        SVC(random_state=seed, gamma='scale'))
    clf.fit(X, Y)
    score = clf.score(X, Y) * 100

    predictions = clf.predict(X)

    return (predictions, score)


def svcOvO(X, Y):
    # TODO: Convert Y to one dimensional array
    clf = OneVsOneClassifier(
        SVC(random_state=seed, gamma='scale'))
    clf.fit(X, Y)
    score = clf.score(X, Y)
    predictions = clf.predict(X)

    return (predictions, score)


def kNN(X, Y):
    neigh = KNeighborsClassifier()
    neigh.fit(X, Y)
    score = neigh.score(X, Y)
    predictions = neigh.predict(X)

    return (predictions, score)


def gNB(X, Y):
    # TODO: Convert Y to one dimensional array
    clf = GaussianNB()
    clf.fit(X, Y)
    score = clf.score(X, Y)
    predictions = clf.predict(X)

    return (predictions, score)


menu_functions = {
    'nn': nn,
    'svcOvR': svcOvR,
    'svcOvO': svcOvO,
    'kNN': kNN,
    'gNB': gNB,
}


def main(X, Y, model='nn'):
    # Split into folds and do loop here?
    func = menu_functions[model]
    func(X, Y)
    # if model == 'nn':
    #     nn(X, Y)
    # elif model == 'svcOvR':
    #     svcOvR(X, Y)
    # # elif model == 'svcOvR':
    # #     svcOvR(X, Y)
