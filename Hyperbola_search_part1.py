# Alberto Cereser, 4 Feb 2014
# alcer@fysik.dtu.dk, Technical University of Denmark
# For each point, this script looks for its successor
# For the location of the cutouts, the code follows the approach described in 
# http://scikit-image.org/docs/dev/auto_examples/plot_template.html
import sys
#import pandas as pd
import numpy as np
from numpy import ndarray
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from skimage import data
from skimage.feature import match_template
import os

if __name__ == '__main__':
	# I start reading in the list of diffraction spots data made using Peak_analyzer_simple_neighbours.py
	input_filename = "Fe_PSI_spt_refined.txt"
	input_file = open(input_filename, 'r') 
	
	# Here are the variables to change, depending on the number of projections considered
	# and on the lines of the txt file (Fe_PSI_spt_refined.txt)
	Number_of_projections = 181
	Number_of_lines_in_txt = 3493
	counter_array = []
	correlation_threshold_value = 0.7

	# I make a square mask, so to ignore the central black region
	X_mask = 1135
	Y_mask = 1251
	output_filename = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/Python_code/Fe_PSI_spt_tagged_15apr.txt")
	
	for i in range(2, (Number_of_projections + 1)):
		cmd = 'python Hyperbola_search_part2.py %i %s %s %i %i %f' % (i, input_filename, output_filename, Number_of_projections, Number_of_lines_in_txt, correlation_threshold_value)
		os.system(cmd)
	#	filename_cutouts_combined = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/cutouts_combined/cutouts_combined_%03i.txt" % (i))
	#	filename_cutouts_combined_tag = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/cutouts_combined/cutouts_combined_tag_%03i.txt" % (i))
	#	image = np.loadtxt(filename_cutouts_combined)
	#	image_tagged = np.loadtxt(filename_cutouts_combined_tag)
	#	for line in input_file:
	#		line_elements = line.split()
	#		Angle_number = int(line_elements[0])
	#		Omega = float(line_elements[1])
	#		Intensity = float(line_elements[2])
	#		X = float(line_elements[3])
	#		Y = float(line_elements[4])
	#		Address = str(line_elements[5])
	#		ID = int(line_elements[6])
	#		print i, Angle_number
			#if ((Angle_number + 1) == i):	# we map cutouts in the following 
				#cutout = np.loadtxt(Address)
				#index = Angle_number #+ 1
				#array_cutout = np.array(cutout)
				#array_image = np.array(image)
				#correlation = match_template(image, cutout)
				#ij = np.unravel_index(np.argmax(correlation), correlation.shape)
				#x, y = ij[::-1]
				#ran = array_cutout.shape
				#ran_y = ran[1]
				#ran_x = ran[0]
				#x_center = x + (ran_x/2)
				#y_center = y + (ran_y/2)
				#print x_center, y_center
				# To do: insert case when spot in central square
				# We now calculate the distance between the cutout center (in Omega) and the point we found 
				# (in Omega + 1)
				#distance = math.sqrt((x_center - X)**2 + (y_center - Y)**2)
				#print i
				#if distance < 200:  # We search that the two points are not too far away
				#	if (np.amax(correlation) > correlation_threshold_value): 
				#		# We need now to find the cutout which is closer to the point where we located 
				#		# The center of the cutout
				#		tag = image_tagged[y_center, x_center]
				#	else:
				#		tag = 0
				#	print distance, tag
				#f = open(output_file, "a+")
				#f.write("%i %f %f %f %f %s %f %i %i\n" % (Angle_number, Omega, Intensity, X, Y, Address, np.amax(correlation), ID, int(tag)))
				#f.close
	input_file.close()