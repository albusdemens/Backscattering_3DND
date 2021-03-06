# Alberto Cereser, 17 Mar 2014
# alcer@fysik.dtu.dk, Technical University of Denmark
# Code to plot together all the cutouts sharing the same tag

import matplotlib
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

image_iter = ('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/cutouts_combined/cutouts_combined_%03d.txt' % i for i in xrange(161, 180))
image_sum = np.loadtxt(next(image_iter))

for image in image_iter:
    image_sum += np.loadtxt(image)

plt.imshow(image_sum, cmap=cm.Greys_r, interpolation='none')
#plt.imshow(matrix, cmap = cm.Accent, interpolation='none')
plt.show()