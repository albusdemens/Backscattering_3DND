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

if __name__ == '__main__':
	# I start reading in the list of diffraction spots data made using Peak_analyzer_simple_neighbours.py
	data = pd.read_csv('Fe_PSI_spt_refined.txt', sep=" ", header = None)
	data.columns = ["Angle_number", "Omega", "Intensity", "X", "Y", "Address", "ID"]#, "flag"]
	
	# Omega is the angle on the goniometer (1, 3, ...); angle_number counts which number 
	# we are considering, with no gaps (1, 2, ...) 

	# Here are the variables to change, depending on the number of projections considered
	# and on the lines of the txt file (Fe_PSI_spt_refined.txt)
	Number_of_projections = 181
	Number_of_lines_in_txt = 3493
	numrows = len(data)
	counter_array = []
	correlation_threshold_value = 0.7

	# I make a square mask, so to ignore the central black region
	X_mask = 1135
	Y_mask = 1251
	size_mask = 800
	radius = 200 # This is the size of the region we consider to find a spot similar to one previously recorded 
	a = np.zeros(Number_of_lines_in_txt)
	output_file = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/Python_code/Fe_PSI_spt_tagged.txt")
	
	for i in range(1, (Number_of_projections + 1)): 
		filename_cutouts_combined = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/cutouts_combined/cutouts_combined_%03i.txt" % (i))
		filename_cutouts_combined_tag = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/cutouts_combined/cutouts_combined_tag_%03i.txt" % (i))
		image = np.loadtxt(filename_cutouts_combined)
		image_tagged = np.loadtxt(filename_cutouts_combined_tag)
		for j in range(0, Number_of_lines_in_txt):
			if ((data.Angle_number[j] + 1) == i):	# we map cutouts in the following image
				cutout = np.loadtxt(data.Address[j])
				index = data.Angle_number[j] #+ 1
				array_cutout = np.array(cutout)
				array_image = np.array(image)
				correlation = match_template(image, cutout)
				ij = np.unravel_index(np.argmax(correlation), correlation.shape)
				y, x = ij[::-1]
				ran = array_cutout.shape
				ran_y = ran[1]
				ran_x = ran[0]
				x_center = x + (ran_x/2)
				y_center = y + (ran_y/2)
				# insert case when spot in central square
				# We now calculate the distance between the cutout center (in Omega) and the point we found 
				# (in Omega + 1)
				distance = math.sqrt((x_center - data.X[j])**2 + (y_center - data.Y[j])**2)
				if distance < 200:  # We search that the two points are not too far away
					if (np.amax(correlation) > correlation_threshold_value): 
						# We need now to find the cutout which is closer to the point where we located 
						# The center of the cutout
						tag = image_tagged[x_center, y_center]
				else:
					tag = 0
				f = open(output_file, "a+")
				f.write("%i %f %f %f %f %s %f %i %i\n" % (data.Angle_number[j], data.Omega[j], data.Intensity[j], data.X[j], data.Y[j], data.Address[j], np.amax(correlation), data.ID[j], int(tag)))
				print j, tag, y_center, x_center, data.X[int(tag)], np.amax(correlation)
				f.close