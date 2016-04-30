from numpy import *
import pandas as pd

def file2training(filename):

    # get the number of lines of the file
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)

    returnMat = zeros((numberOfLines, 3))
    classLabel = []
    index = 0

    for line in arrayLines:

        # remove all '\r'
        line = line.strip()
        # split by '\t'
        listFromLine = line.split('\t')
        # features
        returnMat[index,:] = listFromLine[0:3]
        # labels
        classLabel.append(listFromLine[-1])
        index += 1

    # classLabelDict = {'a':1, 'b':2, ...}
    classLabelSet = set(classLabel)
    classLabelDict = {}
    i = 1
    for label in classLabelSet:
        classLabelDict[label] = i
        i += 1
    classLabel = pd.Series(classLabel)
    classLabelVector = list(classLabel.map(classLabelDict))

    return returnMat, classLabelVector