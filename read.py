import scipy
import numpy
import numpy as np
from scipy import *
from numpy import *
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import guppy
from guppy import hpy
import gc
import cv2
from memory_profiler import profile
import sys

if __name__ == '__main__': 
#def the_reader(index, x, y, peak_number):
    total = len(sys.argv)
    index = int(sys.argv[1])
    x= int(sys.argv[2])
    y = int(sys.argv[3])
    peak_number = int(sys.argv[4])
    im = cv2.imread("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/Midi/Edited_images/Final_image/final_image_%03i.tif" % (index), -1)
    #image_name = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/Midi/Edited_images/Final_image/final_image_%03i.tif" % (index), -1)
    #im = numpy.zeros(shape=(150,150))
    imarray = numpy.array(im)
    peak = im[x,y] #This is the estimated center of mass. We want to check if the value is correct. To do so, we have to find what pixels are part of the diffraction spot
    size_box = 150 #The value has been chosen so that all diffraction spots can stay inside
    pixels_box = int(size_box + 1)
    x_min = int(x - (0.5*(size_box)))
    x_max = int(x + (0.5*(size_box)))
    y_min = int(y - (0.5*(size_box)))
    y_max = int(y + (0.5*(size_box)))
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > 2160: #The size is 2160x2592, but I consider 2159 and 2591 because after we deal with x_max+1 and y_max+1
    	x_max = 2160
    if y_max > 2592:
    	y_max = 2592
    submatrix = im[x_min:x_max+1, y_min:y_max+1]
    filename = ('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/subplots/subplot_%03i_%03i.txt' % (index, peak_number))
    submatrix_1 = np.array(submatrix,numpy.uint16)
    np.savetxt(filename, submatrix_1, fmt='%i')
    #ran_x = float(rank[0])
    #ran_y = float(rank[1])