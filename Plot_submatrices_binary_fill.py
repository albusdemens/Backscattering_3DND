import matplotlib
from pylab import *
import numpy as np

number = 1
filename = ('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/subplots_binary_filled/subplot_binary_002_%03i.txt' % (number))
matrix = np.loadtxt(filename)
plt.imshow(matrix, cmap = cm.Greys_r, interpolation='none')
plt.show()
