# Alberto Cereser, 4 Feb 2014
# alcer@fysik.dtu.dk, Technical University of Denmark
# For the location of the cutouts, the code follows the approach described in
# http://scikit-image.org/docs/dev/auto_examples/plot_template.html

import scipy
import pandas as pd
import numpy as np
from numpy import ndarray
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from skimage import data
from skimage.feature import match_template
import cv2

if __name__ == '__main__':
	# I start reading in the list of diffraction spots data made using Peak_analyzer_simple_neighbours.py
	data = pd.read_csv('Fe_PSI_spt_tagged_14apr.txt', sep=" ", header = None)
	data.columns = ["Angle_number", "Omega", "Intensity", "X", "Y", "Address", "Correlation", "ID", "ID_next"]#, "flag"]
	
	# Omega is the angle on the goniometer (1, 3, ...); angle_number counts which number
	# we are considering, with no gaps (1, 2, ...)
    
	# Here are the variables to change, depending on the number of projections considered
	# and on the lines of the txt file (Fe_PSI_spt_refined.txt)
	Number_of_projections = 181
	Number_of_lines_in_txt = 3491
	numrows = len(data)
	counter_array = []
	correlation_threshold_value = 0.7
    
	# I make a square mask, so to ignore the central black region
	X_mask = 1135
	Y_mask = 1251
	size_mask = 800
	radius = 200 # This is the size of the region we consider to find a spot similar to one previously recorded
	output_file = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/Python_code/Fe_PSI_spt_tagged_points_grouped.txt")
	tag = np.zeros(Number_of_lines_in_txt -1)  # This is the index we use to differentiate the various hyperbola
	tag_correlation = np.zeros(Number_of_lines_in_txt) # This is where we temporarily store the correlation values
    
	label = 0
	for i in range(0, Number_of_lines_in_txt-1):
		if data.ID_next[i] != 0:  # I want the point to be followed by another point
			if tag[data.ID_next[i]-1] == 0:
				if tag[i] == 0:        # I want to check that the point hasn't bee already assigned to an hyperbola
					label = label + 1
					tag[i] = label
					tag[data.ID_next[i]-1] = label
					tag_correlation[data.ID_next[i]-1] = data.Correlation[i]
				if tag[i] !=0:
					tag[data.ID_next[i]-1] = tag[i]
					tag_correlation[data.ID_next[i]-1] = data.Correlation[i]
			if tag[data.ID_next[i]-1] != 0:
				if data.Correlation[i] > tag_correlation[data.ID_next[i]-1]:
					if tag[i] == 0:        # I want to check that the point hasn't bee already assigned to an hyperbola
						label = label + 1
						tag[i] = label
						tag[data.ID_next[i]-1] = label
						tag_correlation[data.ID_next[i]-1] = data.Correlation[i]
					if tag[i] !=0:
						tag[data.ID_next[i]-1] = tag[i]
						tag_correlation[data.ID_next[i]-1] = data.Correlation[i]
        	#distance = math.sqrt((data.X[i] - data.X[data.ID_next[i]-1])**2 + (y_center - data.Y[j])**2)
        	f = open(output_file, "a+")
        	f.write("%i %f %f %f %f %s %f %i %i %i\n" % (data.Angle_number[i], data.Omega[i], data.Intensity[i], data.X[i], data.Y[i], data.Address[i], tag_correlation[i], data.ID[i], data.ID_next[i], int(tag[i])))
        	f.close
        	print data.ID[i], data.ID_next[i], int(tag[i])