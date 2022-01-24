# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 18:51:22 2021

@author: P303007
"""

#---------------------Monochromator (using Region of interest setting)

import glob
from numpy import *
from matplotlib import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import sem
from scipy.signal import find_peaks

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig1,ax1 = plt.subplots(figsize =  [8,8])
fig2,ax2 = plt.subplots(figsize = 1,2,[8,8])


def offset_correction(x,m,c):  #baseline correcion
        return m*x+c

def function(WL):  #function what i want to plot on secondary x axis
        return 10**4/WL   #converting to  10^3 cm-1

def inverse(WL):  #inverse of the above defined function 
        return 10**4/WL  

def convert_ROI(file) :#, specified_lines) :  #converting data to numpy (Region of interest)
    
    line_0 = []  # gives number of pixels
    line_1 = []  # has word "Strip"
    line_2 =[]  # lines 2 and 3 have intensity values  (perpendicular fibre)
    line_3 = []  # lines 2 and 3 have intensity values  (parallel fibre)
    
    line_all = []  #appending all lines
    
    line_all.append(line_0)
    line_all.append(line_1)
    line_all.append(line_2)
    line_all.append(line_3)
  
    #print (file_cal[0])
    
    lines = [ 0 , 1 , 2 , 3] 
  
    for pos,value in enumerate(file) :   #storing each line separately
           
           count = 0  #starting index to store the values from string
           if pos in lines :  #chosing the required line
               
                for space,k in enumerate(value) :   #checking where spcae is present
                
                  if k == '\t' or  space == len(str(value)) - 1 :  #storing values by checking where '\t' is present
                       
                       line_all[pos].append(value[count  : space    : 1])
                       count = space + 1 
                       
    for l in range (0, 4 ) :   #converting the values in the line to ndarray, type = 'float'
         
        if l == 0 :    # to get number of pixels - 1 to 512
                    
            line_all[l] = str(line_all[l])
            
            line_all[l]  = line_all[l].replace("'", "")  #removing additional quotes
            line_all[l] = line_all[l].split(",")    #separting strings by comma
            line_all[l] = line_all[l] [1 : 513]
            
            pixels = zeros(len(line_all[l])) 
            
            for u in range(0, len(pixels)) :
                
                pixels[u] = line_all[l][u]
            
         #   line_all[l] = np.array(line_all[l], dtype = float)  #converting to float
          
        if l == 2 or l == 3 : #line 1 has words "strip", so omitting line 1
            
         
            line_all[l] = str(line_all[l])
            line_all[l]  = line_all[l].replace("'", "")   #removing additional quotes
            line_all[l] = line_all[l].split(",")  #separting strings by comma
           
            line_all[l] = line_all[l] [1 : 513]   
                      
            if l == 3 :  #line 3 - fibre placed parallel
                
                m_parallel = zeros(len(line_all[l]))
                
                for u in range(0, len(m_parallel)) :
                    
                    if u == 511 :   # removing  ' ]' from last element
                        
                        last_element = str(line_all[l][u])
                        
                        line_all[l][u] = last_element[0 : len(last_element) - 1]
                        
                        line_all[l][u] = float(line_all[l][u])
                        
                        m_parallel[u] = line_all[l][u]
                    
                    else :
                        m_parallel[u] = line_all[l][u]
            
            if l == 2 :  #line 3 - fibre placed perpendicular
                 
                      m_perpend = zeros(len(line_all[l]))
                
                      for u in range(0, len(m_perpend)) :
                          
                           if u == 511 :   # removing  ' ]' from last element
                        
                                last_element = str(line_all[l][u])
                        
                                line_all[l][u] = last_element[0 : len(last_element) - 1]
                        
                                line_all[l][u] = float(line_all[l][u])
                                
                                m_perpend[u] = line_all[l][u]
                        
                           else :
            
                                m_perpend[u] = line_all[l][u]
                    
                    
       
    return (pixels, m_perpend,  m_parallel)
     
                 
    
def offset_correction(x,m,c):  #baseline correcion funtion
        return m*x+c

def offset_function (position_offset, value_offset, position, value) :    #offset correction 
    
    correction, co = curve_fit(offset_correction, position_offset , value_offset )  #taking central value to describe the method used  (for paper)
            
    fit_offset = offset_correction(position_offset , *correction)

    value = value - mean(fit_offset)  #subracting baseline 
    
    return (value,fit_offset)

def linear_dichroism (parallel, perpendicular) :  # linear dichrosim
    
    dicrohism = zeros (len (parallel)) #calculating the difference of the spectra
    total = zeros (len (parallel))   #denominator of reduced LDR
    ldr =  zeros (len (parallel))    #defining LDR array
    
    for z in range (0, len(dicrohism)) :
        
        dicrohism[z] = parallel[z] - perpendicular[z]
        total [z] =  (parallel[z] + 2*perpendicular[z])/3
        
        ldr[z] =  dicrohism[z]/total [z]   #ldr
    
    return (ldr)

#%% Calibration  #calibraing pixels to WL (nm)

calibration_files = glob.glob("Calibration\\*.txt")  #reading calibration files 
#print (calibration_files)


for i in range(0, 1 ) :#len(calibration_files)) :
    
     
    file_cal = open(calibration_files[i+2])
    
    pixels,   m_perpend, m_parallel  = convert_ROI(file_cal)
    
#    print (pixels[255], len(pixels), pixels[511])
    
    peaks_positive = find_peaks(m_perpend, height= 4000)  #finding peaks
    
    pixel_peaks = zeros( (len(peaks_positive[0])))
    
    for m in range (0, len(pixel_peaks)) :
        
        #print (peaks_positive[0][m])
        pixel_peaks[m] = peaks_positive[0][m] + 1
  
  #  print (peaks_positive, pixel_peaks, len(pixel_peaks))
   # pixel_peaks = array(pixel_peaks)

    WL = zeros(len(pixels)) 
    
    WL
    for t in range(0, len(WL)) :
        
        
        WL[234] = 546.08
        
        deltaWL = 0.493
        
        WL[t] = WL[234] - (234 - t) *deltaWL
       
     
    
    if 1 : #i == 0  :
   #     ax2[i].plot (pixels, m_parallel , color = 'blue' ,label ="parallel" ) #m_parallel, 'o')
        ax2.plot (pixels, m_perpend, color = 'red' , label = "perpendicular" ) #m_parallel, 'o')
        
     #   ax3.plot (WL, m_perpend)
        
        for t in range (0, len(pixel_peaks)) :  # marking peaks in the calibration plot
               
            ax2.axvline ( x = pixel_peaks[t], color = 'k', linestyle = '--')
        
       
        ax2.set_xlabel ("Pixels")
        ax2.set_ylabel("Intensity")
        #ax2.legend()


#%% DWNTs(ROI)

files_all = glob.glob("*.txt")  #reading data files path
print (files_all)


for i in range(0, len(files_all) ) : 
    
      
    files_ref = open(files_all[0])  # ref_file
     
    files = open(files_all[i+2])     #data file
    
    pixels_ref,   m_perpend_ref, m_parallel_ref  = convert_ROI(files_ref)
    
    pixels,   m_perpend, m_parallel  = convert_ROI(files)
    
  
    spectra_parallel  = m_parallel
    spectra_perpend = m_perpend

    ref_parallel  = m_parallel_ref
    ref_perpend = m_perpend_ref
    
    absorb_parall_raw =  -log(spectra_parallel/abs(ref_parallel))  #reading data from one detector
    
    absorb_parall,y = offset_function (pixels [370 : 430], absorb_parall_raw[370 : 430], pixels, absorb_parall_raw)  #offset correction
     
    absorb_perpend_raw =  -log(spectra_perpend/abs(ref_perpend)) #reading data from second detector
    absorb_perpend,z = offset_function (pixels [370 : 430], absorb_perpend_raw[370 : 430], pixels, absorb_perpend_raw)  #offset correction
    
    WL = zeros(len(pixels))  #defining wavelength
    
    for t in range(0, len(WL)) :  #pixels to WL
        
        
        WL[234] = 546.08
        
        deltaWL = 0.493     # 1 pixel = 0.493 nm
        
        WL[t] = WL[234] - (234 - t) *deltaWL
    
    ax2[i].plot (WL, absorb_perpend ,  label ="parallel" )  
     
    ax2[i].plot (WL, absorb_parall ,  label ="perpendicular" )
      
    ax2[i].set_title ("Flow-rate  (microreactor) " + str(i+1))
       
    ax2[i].set_ylabel ("Absorbance")
    ax2[i].set_xlabel ("Wavelength (nm)")
    
    ax2[i].legend()

        
        
 