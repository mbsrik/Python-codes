# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 18:51:22 2021

@author: P303007
"""


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

#fig2,ax2 = plt.subplots(figsize = [8,8])
##fig3,ax3 = plt.subplots(figsize = [8,8])
#fig4,ax4 = plt.subplots(figsize = [8,8])

#fig5,ax5 = plt.subplots(figsize = [8,8])



def offset_correction(x,m,c):  #baseline correcion
        return m*x+c

def function(WL):  #function what i want to plot on secondary x axis
        return 10**4/WL   #converting to  10^3 cm-1

def inverse(WL):  #inverse of the above defined function 
        return 10**4/WL  

def offset_correction(x,m,c):  #baseline correcion
        return m*x+c

def offset_function (position_offset, value_offset, position, value) :
    
    correction, co = curve_fit(offset_correction, position_offset , value_offset )  #taking central value to describe the method used  (for paper)
            
    fit_offset = offset_correction(position_offset , *correction)

    value = value - (mean(fit_offset))  #subracting baseline 
    
    return (value,fit_offset)

def linear_dichroism (parallel, perpendicular) :
    
    dicrohism = zeros (len (parallel)) #calculating the difference of the spectra
    total = zeros (len (parallel))   #denominator of reduced LDR
    ldr =  zeros (len (parallel))    #defining LDR array
    
    for z in range (0, len(dicrohism)) :
        
        dicrohism[z] = parallel[z] - perpendicular[z]
        total [z] =  (parallel[z] + 2*perpendicular[z])/3
        
        ldr[z] =  dicrohism[z]/total [z]   #ldr
    
    return (ldr)#dicrohism, total)



#%%  DWNTs 

fig1,ax1 = plt.subplots(figsize = [8,8])

def lambda_abs(file_name) :   #reading data from lambda_absorption setup
    
    count = 0
    
    wl =zeros(171)    #wavelength  (fixing for 480 nm to 650 nm)
    intensity = zeros(171)  #intensity

    for pos,value in enumerate(file_name) :   #storing each line separately
         #  print (pos,value)
          
            if pos > 85 and pos < 257:  #chosing lines where data is present  (480 - 650 nm)
                
                wl[count] = value [0 : 5]    #WL 
                intensity[count] = value [11 : 19]  #intensity
                
                count = count + 1
        
    intensity_corrected,y = offset_function (wl [0 : 30], intensity[0 : 30], wl, intensity)   #baseline correction
     
    return (wl, intensity_corrected)
    
        
files_all = glob.glob("*.SP")

print (files_all)


for i in range(0, len(files_all)) :
    
    file = open(files_all[i])
    WL,data = lambda_abs(file)  #WL, absorbance
    
    ax1.plot(WL, data, label = [i])  
   
ax1.set_xlabel ("Wavelength (nm)")
ax1.set_ylabel ("Absorbance")    
ax1.set_xkinm (480 , 650)    
    
    