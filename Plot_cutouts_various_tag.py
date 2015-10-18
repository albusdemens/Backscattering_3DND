# Alberto Cereser, 17 Mar 2014
# alcer@fysik.dtu.dk, Technical University of Denmark
# Code to plot together all the cutouts sharing the same tag

import matplotlib
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

filename = ('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/Python_code/Points_tag_various.txt')
matrix = np.loadtxt(filename)
plt.imshow(matrix, cmap = cm.Greys_r, interpolation='none')
#plt.imshow(matrix, cmap = cm.Accent, interpolation='none')
plt.show()
