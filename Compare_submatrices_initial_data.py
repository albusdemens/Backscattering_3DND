# Alberto Cereser, 24 Nov 2013
# alcer@fysik.dtu.dk, Technical University of Denmark
# This code checks that when we make the cutouts we preserve the intensity

import matplotlib
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Image
import cv, cv2

number = 6
filename = ('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/subplots/subplot_001_%03i.txt' % (number))
matrix = loadtxt(filename)
ran = matrix.shape
print ran
ran_x = ran[0]
ran_y = ran[1]
#im = cv2.imread('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/Midi/Edited_images/Final_image/final_image_001.tif',-1)
#im_1 = np.array(im,np.uint16)
filename_1 = ('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/Midi/Edited_images/Final_image/final_image_001.txt')
#np.savetxt(filename_1, im_1, fmt='%i')
data = loadtxt(filename_1)
ran_1 = data.shape
ran_1_x = ran[0]
ran_1_y = ran[1]
x_im = 829	
y_im = 876
size_square = 150
x_min = int(x_im - size_square/2)
y_min = int(y_im - size_square/2)
x_max = int(x_im + size_square/2)
y_max = int(y_im + size_square/2)
print x_min, x_max
#Commands to print the final_image we are considering (we extract our submatrix from there)
data_1 = data[x_min:x_max+1, y_min:y_max+1]
print data_1.shape
data_2 = ndarray((ran_x,ran_y), int)
for i in range(0, ran_x):
   for j in range(0, ran_y):
		if data_1[i,j] > 20:
   			data_2[i,j] = 20
   		else:
   			data_2[i,j] = data_1[i,j]
plt.imshow(data_2, cmap = cm.Greys_r, interpolation='none')
plt.show()

#This is the center of the diffraction spot, as calculated by peaksearch
x_im_0 = x_im - 75
y_im_0 = y_im - 75
#submatrix = data[x_im-75:x_im+76, y_im-75:y_im+76]

# for i in range(0, 150):
# 	for j in range(0, 150):
# 		print matrix[i,j], submatrix[i,j] 

print matrix[1,1], data[878,829]