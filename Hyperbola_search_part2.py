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

if __name__ == '__main__':
	total = len(sys.argv)
	i = int(sys.argv[1])
	input_filename = str(sys.argv[2])
	output_filename = str(sys.argv[3])
	Number_of_projections = int(sys.argv[4])
	Number_of_lines_in_txt = int(sys.argv[5])
	correlation_threshold_value = float(sys.argv[6])
	input_file = open(input_filename, 'r') 
	filename_cutouts_combined = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/cutouts_combined/cutouts_combined_%03i.txt" % (i))
	filename_cutouts_combined_tag = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/cutouts_combined/cutouts_combined_tag_%03i.txt" % (i))
	image = np.loadtxt(filename_cutouts_combined)
	image_tagged = np.loadtxt(filename_cutouts_combined_tag)
	for line in input_file:
		line_elements = line.split()
		Angle_number = int(line_elements[0])
		Omega = float(line_elements[1])
		Intensity = float(line_elements[2])
		X = float(line_elements[3])
		Y = float(line_elements[4])
		Address = str(line_elements[5])
		ID = int(line_elements[6])
		print i, Angle_number
		if ((Angle_number + 1) == i):	# we map cutouts in the following 
			cutout = np.loadtxt(Address)
			index = Angle_number #+ 1
			array_cutout = np.array(cutout)
			array_image = np.array(image)
			correlation = match_template(image, cutout)
			ij = np.unravel_index(np.argmax(correlation), correlation.shape)
			x, y = ij[::-1]
			ran = array_cutout.shape
			ran_y = ran[1]
			ran_x = ran[0]
			x_center = x + (ran_x/2)
			y_center = y + (ran_y/2)
			# We now calculate the distance between the cutout center (in Omega) and the point we found 
			# (in Omega + 1)
			distance = math.sqrt((x_center - X)**2 + (y_center - Y)**2)
			#print i
			if distance < 200:  # We search that the two points are not too far away
				if (np.amax(correlation) > correlation_threshold_value): 
			#		# We need now to find the cutout which is closer to the point where we located 
			#		# The center of the cutout
					tag = image_tagged[y_center, x_center]
				else:
					tag = 0
			else:
				tag = 0
			f = open(output_filename, "a+")
			f.write("%i %f %f %f %f %s %f %i %i\n" % (Angle_number, Omega, Intensity, X, Y, Address, np.amax(correlation), ID, int(tag)))
			f.close
		# Now we take into account the central mask
		else:
			if X in range(785, 1485) and Y in range(950, 1550):
				# I mirror the X coordinate with respect to the center
				X1 = (2*1135) - X
				cutout = np.loadtxt(Address)
				index = Angle_number #+ 1
				array_cutout = np.array(cutout)
				array_image = np.array(image)
				correlation = match_template(image, cutout)
				ij = np.unravel_index(np.argmax(correlation), correlation.shape)
				x, y = ij[::-1]
				ran = array_cutout.shape
				ran_y = ran[1]
				ran_x = ran[0]
				x_center = x + (ran_x/2)
				y_center = y + (ran_y/2)
				distance = math.sqrt((x_center - X1)**2 + (y_center - Y)**2)
				if distance < 200:  # We look for a point simmetric to the one we have in the subplot
					if (np.amax(correlation) > correlation_threshold_value): 
				#		# We need now to find the cutout which is closer to the point where we located 
				#		# The center of the cutout
						tag = image_tagged[y_center, x_center]
					else:
						tag = 0
				else:
					tag = 0
				f = open(output_filename, "a+")
				f.write("%i %f %f %f %f %s %f %i %i\n" % (Angle_number, Omega, Intensity, X, Y, Address, np.amax(correlation), ID, int(tag)))
				f.close
	input_file.close()