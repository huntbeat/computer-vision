from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import time
import datetime as dt

import mnist_demo as m
import numpy as np

def main():
    number = [1000, 5000, 10000, 15000, 60000]
    for data in number:
        x_train, y_train = m.read_mnist('MNIST_data/train-images-idx3-ubyte.gz',
                                        'MNIST_data/train-labels-idx1-ubyte.gz')
        x_train = x_train[:data]
        y_train = y_train[:data]
        X_train = x_train.reshape(-1, 28*28).astype(np.float32)
        X_train = X_train * (2.0/255.0) - 1.0
        x_test, y_test = m.read_mnist('MNIST_data/t10k-images-idx3-ubyte.gz',
                                      'MNIST_data/t10k-labels-idx1-ubyte.gz')
        X_test = x_test.reshape(-1, 28*28).astype(np.float32)
        X_test = X_test * (2.0/255.0) - 1.0

        classif = OneVsRestClassifier(LinearSVC(C=100.))

        print "Started learning..."
        before1 = dt.datetime.now()
        classif.fit(X_train, y_train)
        after1 = dt.datetime.now()
        print "Done learning!"
        beforeA = dt.datetime.now()
        scoreA = classif.score(X_test, y_test)
        afterA = dt.datetime.now()
        beforeTA = dt.datetime.now()
        scoreTA = classif.score(X_train, y_train)
        afterTA = dt.datetime.now()
        print "Test data accuracy: ", scoreA
        print "Training data accuracy: ", scoreTA
        print "Time it took to train once: ", after1 - before1
        print "Time it took to verify test: ", afterA - beforeA
        print "Time it took to verify training: ", afterTA - beforeTA

        print "Learning again..."
        before2 = dt.datetime.now()
        classif.fit(X_train, y_train)
        after2 = dt.datetime.now()
        print "Done learning!"
        beforeB = dt.datetime.now()
        scoreB = classif.score(X_test, y_test)
        afterB = dt.datetime.now()
        beforeTB = dt.datetime.now()
        scoreTB = classif.score(X_train, y_train)
        afterTB = dt.datetime.now()
        print "Test data accuracy: ", scoreB
        print "Training data accuracy: ", scoreTB
        print "Time it took to train once: ", after2 - before2
        print "Time it took to verify test: ", afterB - beforeB
        print "Time it took to verify training: ", afterTB - beforeTB

main()
