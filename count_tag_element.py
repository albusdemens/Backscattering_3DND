# Alberto Cereser, 17 Mar 2014
# alcer@fysik.dtu.dk, Technical University of Denmark
# For each tag, we count the number of elements

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
	output_file = ("/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/Python_code/Number_points_per_tag.txt")

	Number_of_lines_in_txt = 3462
	Number_of_tags = 544
	
	counter = np.zeros(Number_of_tags)
	for j in range(1, Number_of_tags):
		for i in range(0, Number_of_lines_in_txt):
			if data.Tag[i] == j:
				counter[j] = counter[j] + 1	
				#print data.Tag[i], i, j#, counter[j]
		f = open(output_file, "a+")
		f.write("%i %i\n" % (j, counter[j]))
		f.close

