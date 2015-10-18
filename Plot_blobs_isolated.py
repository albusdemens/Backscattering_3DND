import matplotlib
from pylab import *
import numpy as np

number = 6
filename = ('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/subplots_blob_isolated/subplot_blob_isolated_001_%03i.txt' % (number))
matrix = np.loadtxt(filename)
plt.imshow(matrix, cmap = cm.Greys_r, interpolation='none')
plt.show()
