# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 17:48:24 2021

@author: P303007
"""

#HAAS setup, small spectrometer
import glob
from numpy import *
from matplotlib import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import sem

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
import matplotlib as mpl


plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams['axes.spines.right'] = True
#fig1,ax1 = plt.subplots(1,2,figsize = [8,8])
fig2,ax2 = plt.subplots(figsize = [8,8])

fig4,ax4 = plt.subplots(figsize = [8,8])
fig5,ax5 = plt.subplots(figsize = [8,8])

def offset_correction(x,m,c):  #baseline correcion
        return m*x+c

def function(WL):  #function what i want to plot on secondary x axis
        return 10**4/WL  

def inverse(WL):  #inverse of the above defined function 
        return WL  

def inner_peak_WL (WL, average) :
    
    index_low = where (WL > 594)  #defining lower limit
    index_low = index_low[0][0]
    
    inner_peak = where (average[index_low : -1 ] == max (average[index_low : -1]))
    inner_peak = inner_peak[0][0] + index_low
    
    return (inner_peak)

def main(x,  h,c,w,  h1,c1,w1,  h2,c2,w2,  h3,c3,w3,  h4,c4,w4,    h5,c5,w5,    h6,c6,w6, h7,c7,w7 ) :
    
     
    return h*w**2/((x-c)**2+w**2) + h1*w1**2/((x-c1)**2+w1**2) + h2*w2**2/((x-c2)**2+w2**2) + h3*w3**2/((x-c3)**2+w3**2) +  h4*w4**2/((x-c4)**2+w4**2) + h5*w5**2/((x-c5)**2+w5**2)  + h6*w6**2/((x-c6)**2+w6**2) + h7*w7**2/((x-c7)**2+w7**2) #+  h6*exp((-(x-c6)**2)/(2*w6**2)) + h7*exp((-(x-c7)**2)/(2*w7**2))

def lorr(x,h,c,w) :
    
    return h*w**2/((x-c)**2+w**2)

def main_fd(x,  h1,c1,w1,  h2,c2,w2,  h3,c3,w3,  h4,c4,w4,    h5,c5,w5,    h6,c6,w6 ) : #, h7,c7,w7 ) :
    
     
    return  h1*w1**2/((x-c1)**2+w1**2) + h2*w2**2/((x-c2)**2+w2**2) + h3*w3**2/((x-c3)**2+w3**2) +  h4*w4**2/((x-c4)**2+w4**2) + h5*w5**2/((x-c5)**2+w5**2) + h6*w6**2/((x-c6)**2+w6**2) #+ h7*w7**2/((x-c7)**2+w7**2)  #+  h6*exp((-(x-c6)**2)/(2*w6**2)) + h7*exp((-(x-c7)**2)/(2*w7**2))

    
def guass(x,h,c,w) : #,h2,c2,w2):
     
    return h*exp((-(x-c)**2)/(2*w**2))


def convert(file, specified_lines) :  #converting data to numpy
    
      #creating integer number of lines between start and end line (only the data file)
    lines = np.linspace(specified_lines[0],specified_lines[-1], specified_lines[-1] - specified_lines[0] + 1) 
    
    WL = []  #wavelength
    data = []  #OD values
    
    for pos,value in enumerate(file) : 
        
        if pos in lines :   #selecting only the data needed (without the description)
            
            value  = value.replace(',','.')   #replaceing ',' with '.'
            
            for space,k in enumerate(value) :   #checking where spcae is present
             
                 if  k == '\t' :
                   
                     WL.append(value[0 : space])   #copying WL
                  
                     data.append(value [space + 1 : -1])  #copying value (after space, one more character present. so, space + 1 )
                       
    data = np.asarray(data, dtype = float)   #converting to ndarray   
    WL = np.asarray(WL, dtype = float)    #converting to ndarray  

#fixing range between 480 nm to 650 nm
    
    ind_high = where(WL > 650)
    ind_high = ind_high[0][0]

    ind_low = where(WL > 480)
    ind_low = ind_low[0][0]

    WL = WL[ind_low : ind_high]
    data = data[ind_low : ind_high]
    
    #baseline correction (620 nm to 650 nm range considered)
    
    index_offset = where (WL > 620)[0][0]
    WL_offset = WL[index_offset : -1]
    data_offset = data[index_offset : -1 ]
                
    correction, co = curve_fit(offset_correction, WL_offset , data_offset)
    fit_offset = offset_correction(WL_offset , *correction)
    
    fit_offset = np.average(fit_offset)
        
    data =    data -  fit_offset  #offset correction 
  
    return (WL,data)
                     

files = glob.glob("*.txt")  

print (files)


#Seconmdary axis
f = lambda x: 10**4/x
g = lambda x: 1/x

#%% pure NTs

for i in range (0,len(files)) :

    file = open(files[i])
    
    specified_lines = [17,3664 ]  #start and end line of data
    
   # print (i , "0" , sem(WL))
    
    if i == 2:  #chosing required file
        
        WL, data = convert(file,specified_lines)
       
        inner_peak_dw = inner_peak_WL(WL, data)
        
         ax2.plot(10**4/WL,data/data[inner_peak_dw], color = 'blue',  label =  "Double wall nanotubes") #
       

secax = ax2.secondary_xaxis('top',functions=(f,g)) #secondary axis

secax.set_xlabel("Wavelength (nm)")


ax2.set_xlabel (r'Wavenumber (10 $^3$  cm$^{-1}$)', size = MEDIUM_SIZE)

ax2.set_ylabel ("Absorbance (normalised)")

ax2.set_ylim(-0.05,1.6)
ax2.set_xlim(10**4/650, 10**4/480)

