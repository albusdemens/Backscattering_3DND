# Alberto Cereser, Feb 2014
# alcer@fysik.dtu.dk, Technical University of Denmark
#This code extract from the spt file the lines with spots such that average intensity per spot <100, more than tot pixles

import scipy
import numpy
import numpy as np
import scipy.misc
from scipy.misc import toimage
from scipy import *
from numpy import *
from scipy import ndimage
from PIL import Image
from wand.image import Image
from wand.display import display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import guppy
from guppy import hpy
import gc
from memory_profiler import profile
import sys

if __name__ == '__main__': 
#def Peak_analyzer_simple_neighbours_part2(index, x, y, peak_number, line_number, index_1, filename, f):
    total = len(sys.argv)
    index = int(sys.argv[1])
    x= int(sys.argv[2])
    y = int(sys.argv[3])
    size_box = 150 
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
    peak_number = int(sys.argv[4])
    line_number = int(sys.argv[5])
    index_1 = int(sys.argv[6])
    filename = str(sys.argv[7])
    output_filename = str(sys.argv[8])
    submatrix_1 = np.loadtxt(filename)
    ran = submatrix_1.shape
    #print ran
    ran_x = ran[0]
    ran_y = ran[1]
    neighbours_1 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_2 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_3 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_4 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_5 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_6 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_7 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_8 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_mask_1 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_mask_2 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_mask_3 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_mask_4 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_mask_5 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_mask_6 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_mask_7 = ndarray((ran_x-1,ran_y-1),int)
    neighbours_mask_8 = ndarray((ran_x-1,ran_y-1),int)
    for i in range(1, ran_x-2):
        for j in range(1, ran_y-2):
            #print i, j
            neighbours_1[i][j] = submatrix_1[i-1][j-1]
            neighbours_2[i][j] = submatrix_1[i-1][j]
            neighbours_3[i][j] = submatrix_1[i-1][j+1]
            neighbours_4[i][j] = submatrix_1[i][j-1]
            neighbours_5[i][j] = submatrix_1[i][j+1]
            neighbours_6[i][j] = submatrix_1[i+1][j-1]
            neighbours_7[i][j] = submatrix_1[i+1][j]                        
            neighbours_8[i][j] = submatrix_1[i+1][j+1]
    mask = ndarray((ran_x-1,ran_y-1), int)
    mask_1 = ndarray((ran_x-1,ran_y-1), int)
    threshold = 4
    number_neighbours = 5 #This is the number of neighbours above the threshold
    for n in range(1, ran_x-2):
        for o in range(1, ran_y-2):
            counter = 0
            #The treshold value should be lower than the one we use with grain spotter
            if neighbours_1[n,o] > threshold:
                counter = counter +1
            if neighbours_2[n,o] > threshold:
                counter = counter +1
            if neighbours_3[n,o] > threshold:
                counter = counter +1
            if neighbours_4[n,o] > threshold:
                counter = counter +1
            if neighbours_5[n,o] > threshold:
                counter = counter +1
            if neighbours_6[n,o] > threshold:
                counter = counter +1
            if neighbours_7[n,o] > threshold:
                counter = counter +1
            if neighbours_8[n,o] > threshold:
                counter = counter +1    
            if counter > number_neighbours:
                mask[n,o] = 1
            else:
                mask[n,o] = 0
    for i in range(0, ran_x):
        for j in range(0, ran_y):
            if 0 < i <  ran_x-2:
                if 0 < j <  ran_y-2:
                    neighbours_mask_1[i,j] = (mask[i-1,j-1])
                    neighbours_mask_2[i,j] = (mask[i-1,j])
                    neighbours_mask_3[i,j] = (mask[i-1,j+1])
                    neighbours_mask_4[i,j] = (mask[i,j-1])
                    neighbours_mask_5[i,j] = (mask[i,j+1])
                    neighbours_mask_6[i,j] = (mask[i+1,j-1])
                    neighbours_mask_7[i,j] = (mask[i+1,j])
                    neighbours_mask_8[i,j] = (mask[i+1,j+1])
    for q in range(1, ran_x-1):
        for r in range(1, ran_y-1):
            counter = 0
            if neighbours_mask_1[q,r] == 1:
                counter = counter +1
            if neighbours_mask_2[q,r] == 1:
                counter = counter +1
            if neighbours_mask_3[q,r] == 1:
                counter = counter +1
            if neighbours_mask_4[q,r] == 1:
                counter = counter +1
            if neighbours_mask_5[q,r] == 1:
                counter = counter +1
            if neighbours_mask_6[q,r] == 1:
                counter = counter +1
            if neighbours_mask_7[q,r] == 1:
                counter = counter +1
            if neighbours_mask_8[q,r] == 1:
                counter = counter +1    
            if counter >= number_neighbours:
                mask_1[q,r] = 1  
    mask_2 = np.array(mask, numpy.uint16)
    mask_2_filled = ndimage.binary_fill_holes(mask_2)
    ### Uncomment if you want to save mask_2_filled ###
    #np.savetxt(filename_filled, mask_2_filled, fmt='%i')
    # Now we want to clean the blobs, leaving only the central one. This is done using flood filling
    label_im, nb_labels = ndimage.label(mask_2_filled)   
    blob_size = ndimage.sum(mask_2_filled, label_im, range(nb_labels + 1))
    x_center = int((x_max - x_min)/2)
    y_center = int((y_max - y_min)/2)
    label_center = label_im[x_center, y_center]
    if label_center > 0:
        our_blob_area = blob_size[label_center]
        if our_blob_area > 1000: 
            our_blob_CM = ndimage.center_of_mass(mask_2_filled, label_im, label_center)
            #We need to isolate the central blob, so we can calculate the correlation between different objects
            mask_blob_isolated = np.array(mask, numpy.uint16)
            for i in range(0, ran_x-1):
                for j in range(0, ran_y-1):
                    if label_im[i,j] == label_center:
                        mask_blob_isolated[i,j] = 1
                    else:
                        mask_blob_isolated[i,j] = 0
            # We want to write the CM in global coordinates
            x_global = x + our_blob_CM[0] - 75
            y_global = y + our_blob_CM[1] - 75
            a1 = int(index)
            a2 = float(index_1)
            a3 = float(our_blob_area)
            # It's not clear why, but we switch x and y
            a4 = float(y_global) 
            a5 = float(x_global)  
            #with open(output_filename, "r+") as f:
            f = open(output_filename, "r+")
            line_number = sum(1 for line_number in f) # This is needed to print the line number 
            filename_blob_isolated = ('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/subplots_blob_isolated/subplot_blob_isolated_%03i_%03i.txt' % (index, int(line_number + 1)))
            np.savetxt(filename_blob_isolated, mask_blob_isolated, fmt='%i')
            #print mask_blob_isolated.shape, x_global, y_global
            f.write("%i %f %f %f %f %s %i\n" % (a1, a2, a3, a4, a5, str(filename_blob_isolated) , int(line_number + 1)))
            print (line_number + 1)
            f.close
            #print index, index_1, peak_number, our_blob_area, x_global, y_global, filename_blob_isolated, line_number
            # index is the projection number (1, 2, ...) and image_1 is the angle (1, 3, ...)
            #sizes = ndimage.sum(mask_2_filled, label_im, range(nb_labels + 1))
        gc.collect()