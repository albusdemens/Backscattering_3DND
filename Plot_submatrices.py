import matplotlib
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

projection_number = 2
filename = ('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/subplots/subplot_001_%03i.txt' % (projection_number))
matrix = np.loadtxt(filename)
plt.imshow(matrix, cmap = cm.Greys_r, interpolation='none')
plt.show()
