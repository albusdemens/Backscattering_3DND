# Alberto Cereser, Feb 2014
# alcer@fysik.dtu.dk, Technical University of Denmark
# This code extract from the spt file the lines with spots such that average intensity per spot <100, more than tot pixles

import scipy
import numpy
import numpy as np
import scipy.misc
from scipy import *
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import gc
#from memory_profiler import profile
import os
import time

#@profile
#def Peak_analyzer_part1():
if __name__ == '__main__': 
    # Open the file for reading.

    rd_file_name = "peaks_t10_cleaned.spt"
    rd_file = open(rd_file_name, 'r')  
    #Open the file for writing
    #f = open("output.txt", "w")                                          
    output_filename = "Fe_PSI_spt_refined.txt"
    header_flag=False
    index = 0
    line_number = 1
    index_1 = -1
    peak_number = 1
    # Read through the header                                                    
    for line in rd_file: 
        gc.collect
        line_number = line_number + 1
        line_1 = line.rstrip("\n")                                            
        # Decide what to do based on the content in the line.                    
        if "#" in line.lower():                                             
            header_flag=True
            # Don't print the header                                             
            pass
        elif line.strip() == "":                                                                                       
            pass
        else:                                                                    
            # We print the body.  Lines end with a carriage return, so we don't  
            # add another one.                                                   
            if (header_flag==True):
                #index = 1
                index = index + 1
                index_1 = index_1 + 2
                header_flag=False
            line_elements = line.split()
            x = int(float(line_elements[2]))
            y = int(float(line_elements[3]))
            pixel_number = int(line_elements[0])
            peak_number = peak_number + 1
            cmd = 'python read.py %i %i %i %i' % (index, x, y, peak_number)
            os.system(cmd)
            #time.sleep(10)
            filename = ('/Users/Alberto/Documents/Data_analysis/ICON_Aug2013/Data_analysis/subplots/subplot_%03i_%03i.txt' % (index, peak_number))
            cmd_1 = 'python Peak_analyzer_simple_neighbours_part2.py %i %i %i %i %i %i %s %s' % (index, x, y, peak_number, line_number, index_1, filename, output_filename)
            os.system(cmd_1)
            print x, y
            #Peak_analyzer_part2(index, x, y, peak_number, line_number, index_1, filename, f)
            #del filename
            gc.collect()
    # Clean up
    rd_file.close()

#if __name__ == '__main__':
#    Peak_analyzer_part1()    
