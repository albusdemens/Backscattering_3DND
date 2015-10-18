# Alberto Cereser, 24 Feb 2014
# alcer@fysik.dtu.dk, Technical University of Denmark
# Thsi script plots the blobs collected at a certain projection, each with a different colour

import matplotlib
from pylab import *
import numpy as np

filename = ('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/Python_code/mask.txt')
matrix = np.loadtxt(filename)
plt.imshow(matrix, cmap = plt.cm.flag, interpolation='none')
plt.show()
