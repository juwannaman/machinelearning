'''
Created on April 29, 2016
kNN: k Nearest Neighbors

@author: pbharrin
@modified: juwannaman
'''

from numpy import *
import operator

def createDataSet():
    """
    :return: x, y
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classifier(testX, trainingX, labels, k=3):
    """
    :param testX      : input x that needed to by classified
    :param trainingX  : training data (x)
    :param labels   : labels of training data (y)
    :param k        : k we choose (default 3)
    :return:
    """

    ''' calculate distance '''
    # number of training set
    dataSetSize = trainingX.shape[0]

    # copy dataSetSize number of testX
    # numpy.tile: repeat elements of an array
    diffMat = tile(testX, (dataSetSize, 1)) - trainingX

    # Euclidean Distance
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    # Sort the distances
    # numpy.argsort: returns the indices that would sort an array
    sortedDistIndices = distances.argsort()

    # class dictionary {'class' : number of appearance}
    classCount = {}

    # choose the shortest distance of k points
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]   # the nearest k points
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1    # get(), if voteLabel does not exist, return 0

    # sort
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

