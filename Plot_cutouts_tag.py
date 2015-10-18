import matplotlib
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

filename = ('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/cutouts_combined/cutouts_combined_tag_002.txt')
matrix = np.loadtxt(filename)
#plt.imshow(matrix, cmap = cm.Greys_r, interpolation='none')
plt.imshow(matrix, cmap = cm.Accent, interpolation='none') # Option to plot in colour
plt.show()
