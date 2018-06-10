from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import gzip
import struct
import cv2
import mnist_demo as m

def main():
    IMAGE_SIZE = 28
    number1 = input("How many training datasets should we use: ")
    number2 = input("How many classifiers: ")
    x_train, y_train = m.read_mnist('MNIST_data/train-images-idx3-ubyte.gz',
                                    'MNIST_data/train-labels-idx1-ubyte.gz')
    x_train = x_train[:number1]
    y_train = y_train[:number1]
    X_train = x_train.reshape(-1, IMAGE_SIZE*IMAGE_SIZE).astype(np.float32)
    X_train = X_train * (2.0/255.0) - 1.0
    X_train = X_train * (2.0/255.0) - 1.0
    x_test, y_test = m.read_mnist('MNIST_data/t10k-images-idx3-ubyte.gz',
                                  'MNIST_data/t10k-labels-idx1-ubyte.gz')
    X_test = x_test.reshape(-1, IMAGE_SIZE*IMAGE_SIZE).astype(np.float32)
    X_test = X_test * (2.0/255.0) - 1.0
    x_test, y_test = m.read_mnist('MNIST_data/t10k-images-idx3-ubyte.gz',
                                  'MNIST_data/t10k-labels-idx1-ubyte.gz')
    X_test = x_test.reshape(-1, IMAGE_SIZE*IMAGE_SIZE).astype(np.float32)
    X_test = X_test * (2.0/255.0) - 1.0

    bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=number2,
    learning_rate=1)
    
    print "Started learning..."

    before1 = datetime.datetime.now()
    bdt_real.fit(X_train, y_train)
    after1 = datetime.datetime.now()
    
    print "Done learning once!"    

    before2 = datetime.datetime.now()
    score = bdt_real.score(X_test, y_test)
    after2 = datetime.datetime.now()

    print "Training accuracy: ", score
    print "Time it took to train once: ", after1 - before1
    print "Time it took verify: ", after2 - before2

    print "Learning again..."

    before1 = datetime.datetime.now()
    bdt_real.fit(X_train, y_train)
    after1 = datetime.datetime.now()
    
    print "Done learning twice!"    

    before2 = datetime.datetime.now()
    score = bdt_real.score(X_test, y_test)
    after2 = datetime.datetime.now()

    print "Training accuracy: ", score
    print "Time it took to train twice: ", after1 - before1
    print "Time it took verify: ", after2 - before2

main()
