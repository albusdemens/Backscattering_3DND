# Alberto Cereser, 4 Mar 2014
# alcer@fysik.dtu.dk, Technical University of Denmark
# For each projection, this code puts together all the collected cutouts

#import scipy
#from scipy import signal
import pandas as pd
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from skimage import data
from skimage.feature import match_template
import cv2
import gc

if __name__ == '__main__':
    
	data = pd.read_csv('Fe_PSI_spt_refined.txt', sep=" ", header = None)
	data.columns = ["Angle_number", "Omega", "Intensity", "X", "Y", "Address", "ID"]
    
	# Here are the variables to change, depending on the number of projections considered
	# and on the lines of the txt file (Fe_PSI_spt_refined.txt)
	Number_of_projections = 181
	Number_of_lines_in_txt = 3491
	numrows = len(data)
    
	# The idea is, for each angle, to make a new image putting together the cutouts
	# Expressed in global coordinates
	for i in range(1, (Number_of_projections + 1)):
		new_image_simple = np.zeros((2700, 2300)) # 2592 is the number of rows, and 2160 the number of columns (RC, as in Roman Catholic)
		new_image_tag = np.zeros((2700, 2300))
		filename_new_image_simple = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/cutouts_combined/cutouts_combined_%03i.txt" % (i))
		filename_new_image_tag = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/cutouts_combined/cutouts_combined_tag_%03i.txt" % (i))
		for j in range(0, Number_of_lines_in_txt - 1): # We read each line of the refined spt file
            # Remember that the number of the first line is 0
			if (data.Angle_number[j] == i):
				x_center_blob = data.X[j]
				y_center_blob = data.Y[j]
				cutout = np.loadtxt(data.Address[j])
				array_cutout = np.array(cutout)
				scaled_array_cutout = np.multiply(array_cutout, data.ID[j])
				ran = array_cutout.shape
				ran_x_before_connectivity_search = ran[0]
				ran_y_before_connectivity_search = ran[1]
				size_box = 150
				# Remember that, for the cutouts, the center of the matrix is different from its cm!!!
				x_min_before_connectivity_search = int(x_center_blob - (0.5*(ran_x_before_connectivity_search)))
				x_max_before_connectivity_search = int(x_center_blob + (0.5*(ran_x_before_connectivity_search)))
				y_min_before_connectivity_search = int(y_center_blob - (0.5*(ran_y_before_connectivity_search)))
				y_max_before_connectivity_search = int(y_center_blob + (0.5*(ran_y_before_connectivity_search)))
				x_min = int(x_center_blob - (0.5*(size_box)))
				x_max = int(x_center_blob + (0.5*(size_box)))
				y_min = int(y_center_blob - (0.5*(size_box)))
				y_max = int(y_center_blob + (0.5*(size_box)))
				if int(x_min) < 0:
					x_min = int(0)
				if int(y_min) < 0:
					y_min = int(0)
				if int(x_max) > 2159: #The size is 2160x2592, but I consider 2159 and 2591 because after we deal with x_max+1 and y_max+1
					x_max = int(2159)
				if int(y_max) > 2591:
					y_max = int(2591)
				x_MIN = max(x_min, x_min_before_connectivity_search)
				x_MAX = min(x_max, x_max_before_connectivity_search)
				y_MIN = max(y_min, y_min_before_connectivity_search)
				y_MAX = min(y_max, y_max_before_connectivity_search)
				size_box_x = x_MAX - x_MIN + 1
				size_box_y = y_MAX - y_MIN + 1
				for k in range(0, size_box_x - 1): # Roman Catholic!
					for l in range(0, size_box_y - 1):
						print x_MIN, x_MAX, y_MIN, y_MAX, (size_box_x - 1), (size_box_y - 1), j
						# The formulas for k in l in global coordinates take into account that
						# the dimension of the cutouts is not fixed
						global_l = int(l + x_center_blob - (x_center_blob - x_MIN))
						global_k = int(k + y_center_blob - (y_center_blob - y_MIN))
						if scaled_array_cutout[k,l] != 0:
							new_image_simple[global_k, global_l] = array_cutout[k, l]
							new_image_tag[global_k, global_l] = scaled_array_cutout[k, l]
							#print filename_new_image, new_image
		np.savetxt(filename_new_image_simple, new_image_simple, fmt='%i')
		np.savetxt(filename_new_image_tag, new_image_tag, fmt='%i')
		del new_image_simple, new_image_tag
		gc.collect()