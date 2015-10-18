import matplotlib
from pylab import *
import numpy as np

number = 10
filename = ('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/subplots_binary/subplot_binary_003_%03i.txt' % (number))
matrix = np.loadtxt(filename)
plt.imshow(matrix, cmap = cm.Greys_r, interpolation='none')
plt.show()
