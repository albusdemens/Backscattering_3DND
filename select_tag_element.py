# Alberto Cereser, 17 Mar 2014
# alcer@fysik.dtu.dk, Technical University of Denmark
# For a certain tag, we print the corresponding points

import scipy
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

	data = pd.read_csv("Fe_PSI_spt_tagged_points_grouped.txt", sep=" ", header = None)
	data.columns = ["Angle_number", "Omega", "Intensity", "X", "Y", "Address", "Correlation", "ID", "ID_next", "Tag"]
	output_file = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/Python_code/Points_tag_128.txt")

	Number_of_lines_in_txt = 3642
	Number_of_tags = 544
	counter = 0
	for i in range(0, Number_of_lines_in_txt):
		if (data.Tag[i] == 128):
			counter = counter + 1
            #print counter
			f = open(output_file, "a+")
			f.write("%i %f %f %f %f %s %f %i %i\n" % (data.Angle_number[i], data.Omega[i], data.Intensity[i], data.X[i], data.Y[i], data.Address[i], data.Correlation[i], data.ID[i], data.Tag[i]))
			f.close

