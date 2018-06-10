# MNIST handwritten digit recognition - data file loading demo
# Written by Matt Zucker, April 2017

import numpy as np
import gzip
import struct
import cv2

IMAGE_SIZE = 28

######################################################################
# Read a 32-bit int from a file or a stream

def read_int(f):
    buf = f.read(4)
    data = struct.unpack('>i', buf)
    return data[0]

######################################################################
# Open a regular file or a gzipped file to decompress on-the-fly

def open_maybe_gz(filename, mode='rb'):

    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)

######################################################################
# Read the MNIST data from an images file or a labels file. The file
# formats are documented at http://yann.lecun.com/exdb/mnist/

def read_mnist(images_file, labels_file):

    images = open_maybe_gz(images_file)

    imagic = read_int(images)
    assert(imagic == 2051)
    icount = read_int(images)
    rows = read_int(images)
    cols = read_int(images)
    assert(rows == IMAGE_SIZE and cols == IMAGE_SIZE)

    print 'reading', icount, 'images of', rows, 'rows by', cols, 'cols.'

    labels = open_maybe_gz(labels_file)

    lmagic = read_int(labels)
    assert(lmagic == 2049)
    lcount = read_int(labels)

    print 'reading', lcount, 'labels.'

    assert(icount == lcount)

    image_array = np.fromstring(images.read(icount*rows*cols),
                                dtype=np.uint8).reshape((icount,rows,cols))

    label_array = np.fromstring(labels.read(lcount),
                                dtype=np.uint8).reshape((icount))
    
    return image_array, label_array

######################################################################
# Show use of the MNIST data set:

def main():

    # Read images and labels. This is reading the 10k-element test set
    # (you can also use the other pair of filenames to get the
    # 60k-element training set).
    images, labels = read_mnist('MNIST_data/t10k-images-idx3-ubyte.gz',
                                'MNIST_data/t10k-labels-idx1-ubyte.gz')


    # This is a nice way to reshape and rescale the MNIST data
    # (e.g. to feed to PCA, Neural Net, etc.) It converts the data to
    # 32-bit floating point, and then recenters it to be in the [-1,
    # 1] range.
    classifier_input = images.reshape(-1, IMAGE_SIZE*IMAGE_SIZE).astype(np.float32)
    classifier_input = classifier_input * (2.0/255.0) - 1.0

    ##################################################
    # Now just display some stuff:
    
    print 'images has datatype {}, shape {}, and ranges from {} to {}'.format(
        images.dtype, images.shape, images.min(), images.max())

    print 'input has datatype {}, shape {}, and ranges from {} to {}'.format(
        classifier_input.dtype, classifier_input.shape,
        classifier_input.min(), classifier_input.max())
    
    for i, image in enumerate(images):
        display = cv2.resize(image, (8*IMAGE_SIZE, 8*IMAGE_SIZE),
                             interpolation=cv2.INTER_NEAREST)

        print 'image {} has label {}'.format(i, labels[i])
        cv2.imshow('win', display)
        while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass

######################################################################

if __name__ == '__main__':
    main()
