import matplotlib
import matplotlib.pyplot as plt
from numpy import *
from util import file2training


""" kNN.py application """
import kNN
group, labels = kNN.createDataSet()
classed = kNN.classifier([0,0], group, labels, 3)
print "The data should be classified in group", classed

""" kNNdate.py application """
import kNNdate
datingDataMat, datingLabels = file2training.file2training('datingTestSet.txt')

# plot
fig = plt.figure()
ax = fig.add_subplot(3, 1, 1)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
ax = fig.add_subplot(3, 1, 2)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
ax = fig.add_subplot(3, 1, 3)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()
