# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 17:48:24 2021

@author: P303007
"""


import glob
from numpy import *
from matplotlib import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import csv
from scipy.stats import sem
from sklearn.preprocessing import StandardScaler
import statistics
from scipy import interpolate
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import collections
import scipy.stats as stats
from scipy.signal import find_peaks

from sympy import Symbol, lambdify
import sympy



SMALL_SIZE = 18
MEDIUM_SIZE = 16
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize= BIGGER_SIZE)  # fontsize of the figure title




fig1,ax1 = plt.subplots(1,6,figsize = [8,8])
fig2,ax2 = plt.subplots(1,6,figsize = [8,8])
fig4,ax4 = plt.subplots(1,6,figsize = [8,8])
fig5,ax5 = plt.subplots(1,6,figsize = [8,8])

fig7,ax7 = plt.subplots(1,6,figsize = [8,8])

fig3,ax3 = plt.subplots(figsize = [8,8])

fig9,ax9 = plt.subplots(figsize = [8,8])
fig6,ax6 = plt.subplots(1,2,figsize = [8,8])

fig8,ax8 = plt.subplots(figsize = [8,8])

fig10,ax10 = plt.subplots(2,2,figsize = [8,8])

def guass(x, h,c,w, h1,c1,w1,  b):  #2 Gaussain for individual profiles
     
    return (h*exp((-(x-c)**2)/(2*w**2)) + h1*exp((-(x-c1)**2)/(2*w1**2)))  + b


def guass_single(x, h,c,w):
     
    return h*exp((-(x-c)**2)/(2*w**2)) 


def offset_correction(x,m,c):  #baseline correcion
        return m*x+c
    
def guass_average(x, h,c,w, h1,c1,w1, h2,c2,w2,  h3,c3,w3 ) : #,  h4,c4,w4,  h5,c5,w5 ) :   #4 Gaussian fits to model the averaged TEM profile
     
    return h*exp((-(x-c)**2)/(2*w**2)) + h1*exp((-(x-c1)**2)/(2*w1**2)) +  h2*exp((-(x-c2)**2)/(2*w2**2)) + h3*exp((-(x-c3)**2)/(2*w3**2))

    

def centering(position,value) :  #aligining data

    if (len(position %2) == 0 ) :
               middle = int(len(position)/2)               
    
    else :
       middle = int((len(position) + 1)/2)
   
    ind_high = where (value[0 : middle]  == max(value[0 : middle]))
    ind_high = ind_high[0][0]
        
    index_1 = where (value  == min(value [0 : middle]) ) #second mimima
    index_1 = index_1[0][0]
    
    index_2 = where (value  == min(value [middle : middle + 25]) ) #third mimima  
    index_2 = index_2[0][0]
        
    constant =  0*mean(value [0 : 40]) 
   
    limit_h_low = [  value[index_1] - 10, value[index_2]  - 10 ]
    limit_h_high = [  value[index_1] + 10, value[index_2] + 10 ]
      
    limit_x_low = [ position[index_1 - 3], position[index_2 - 3]]
    limit_x_high = [ position[index_1 + 3], position[index_2 + 3]]
      
    limit_low =  [ limit_h_low[0],limit_x_low[0],0.5,        limit_h_low[1],limit_x_low[1],0.5,    -5 ]
    limit_high = [ limit_h_high[0],limit_x_high[0],3,        limit_h_high[1],limit_x_high[1],3,     5 ]
    
    fit_g,co = curve_fit(guass, position, value, bounds = (limit_low,limit_high)) #Gaussian fits
    
    center = 0*(abs(fit_g[1]) + abs(fit_g[4]))/2
    
    aligned_x = zeros(len(position))
    
    for i in range (0, len(position)) :
        
         aligned_x[i] = position[i] - center  # aligns the data, where the averaged center distance is set to 0, and the two minima are at equal distance from the averaged center value
    
    #shifting center
    fit_g[1] = fit_g[1] - center
    fit_g[4] = fit_g[4] - center

   
    return (fit_g, co )



def model_guass(x,gray):  #Modelling averaged TEM profiles
    
   peaks_positive = find_peaks(average, height= 80)

   peaks_negative = find_peaks (- average, height = 170)
    
   bounds_lower = [30, x[peaks_positive[0][0]]-0.3 ,0.1,  -500,x[peaks_negative[0][0]]-0.3,0.1,  -500,x[peaks_negative[0][1]]-0.3,0.1,  30, x[peaks_positive[0][1]]-0.3,0.1] #,  -500,x[peaks_negative[0][3]]-0.3,0.1,    30, x[peaks_positive[0][1]]-0.3 ,0.1]          
   bounds_upper = [300, x[peaks_positive[0][0]]+0.3 ,3,     -50,x[peaks_negative[0][0]]+0.3,3,    -50,x[peaks_negative[0][1]]+0.3,3,    300, x[peaks_positive[0][1]]+0.3,3] #  -50,x[peaks_negative[0][3]]+0.3,3,    300, x[peaks_positive[0][1]]+0.3 ,3]          

   average_model, co_fit = curve_fit(guass_average, x, average , bounds =(bounds_lower,bounds_upper) , maxfev = 5000)
   
   
#   average_derivative =  average_fit.diff(x)
   
   return (average_model, co_fit)

                       
    
def finding_boundary (position, value, *fit) :
     
   
     low_left  = where (position > fit[1])     #left outer dip
     low_left = low_left[0][0]                                
    
     high_left = where (value [0 : low_left ] > 0 )  #finding first point > 0 LHS
     high_left = high_left[0][-1]  #- 4
     
     low_right  = where (position > fit[4])
     low_right = low_right[0][0]                    #right outer dip
     
     high_right = where (value [low_right : -1 ] > 0 )   #finding first point > 0 RHS
     high_right = high_right[0][0] + 1 +  low_right
   
     limits = [high_left,low_left,  low_right , high_right ]  

     x_req_0 = zeros(2)  #boundary points on left and right
     
     for m in range (0 , 2 ) :  #left and right boundary distances
          
        #  m = 1
          x_interpolate = position [limits[m*2] : limits[m*2+1]]
          y_interpolate = value [limits[m*2] : limits[m*2+1]]
       
          
          if m == 0 :  #defining range of contrast values between LHS maxima and left outer dip
              y_req = array (range ( int(y_interpolate [-1]), int(y_interpolate [0]),  1))   #finding the crossing point
                
          else : #defining range of contrast values between right outer dip and RHS maxima
              y_req = array (range ( int(y_interpolate [0]), int(y_interpolate [-1]),  1))    #finding the crossing point
         
          x_req  = interp1d(y_interpolate, x_interpolate)(y_req)  #splined gray value
          
          y_req_0 = where (y_req == 0)
          y_req_0 = abs(y_req[0])
          
          #print ("yes" , y_req_0, x_req[y_req_0])
          x_req_0[m] = x_req[y_req_0]   
     
     return ( x_req_0[0]  , x_req_0[1]   )  #- x_req_0[0]  + x_req_0[1]) 
   
   
#%% Averaged and integrated data


files_all =  glob.glob("SW\\***.csv")
print (files_all)


dip_dip = zeros(len(files_all))

boundary = zeros (len(files_all))

std_err = zeros(len(files_all)) #stnadard error on averageing

boundary_each = []   #boundary distance
dip_dip_each = []   #dip to dip distance

fwhm_left = []   #fwhm left dip
fwhm_right = []  #fwhm left dip

fit_amplitude = zeros(2)  #storing amplitude of 4 gaussians

fit_max = []  # to store max amplitude

error_inner = []  #error in dip to dip distance

fwhm_left_error = []  #error in fwhm left dip
fwhm_right_error = [] #error in fwhm right dip

SD_inner_left = []   #Sd of modelled left tdip
SD_inner_right = []  #SD of modelled right dip

std_mean = [] #stanadard error of mean of each profile

#%%
if 1 :
    
    for i in range (0, len(files_all)) :
                 
            
      
            position = []   # in nm
            value = []      # grayvalue
                  
            
            file = open(files_all[i])
         #   print (files[i])
            
            data = csv.reader(file, delimiter=',')
            
            for distance,gray in data :
            
                   value.append(gray)   #contrast values
                   position.append(distance) #distance (nm)
        
            value = value [1 : len(value)]   
            position = position [1 : len(position)]
            
            value = np.asarray(value, dtype = float)   #converting to ndarray   
            position = np.asarray(position, dtype = float)    #converting to ndarray 
            
            position_offset = []
            position_offset.extend(position[0 : 45])
            position_offset.extend(position[-45  : -1])
            
            value_offset = []
            value_offset.extend(value[0 : 45])
            value_offset.extend(value[-45 : -1])
            
            value_offset = array(value_offset, object )
            position_offset = array(position_offset, object)
            
            correction, co = curve_fit(offset_correction, position_offset , value_offset )  #taking central value to describe the method used  (for paper)
            
            fit_offset = offset_correction(position , *correction)
            
            if i == 22 :  #DATA PROCESSING - PAPER
            
                ax10[0][0].plot(position, value,  '-o', label = "Data")  #DATA PROCESSING - PAPER
             #   ax10[0][0].axhline(y = 0, color = 'k')
                
                ax10[0][0].plot (position, fit_offset, label = "Baseline")
            
            value = value - fit_offset  #subracting baseline 
            
            if i == 1 :  #DATA PROCESSING - PAPER
                
                ax10[0][1].plot(position, value, '-o',  label = "Data")
                ax10[0][1].axhline(y = 0)
            
            fit, co   = centering(position, value )  #modeling profiles
            
            
            if i == 22 :  #DATA PROCESSING - PAPER
                
                ax10[1][0].plot(position, value, '-o', label = "Data")
                ax10[1][0].plot(position, guass(position, *fit) - fit[-1], label = "Fit")
                ax10[1][0].axhline(y = 0, color = 'k')
            
            co = np.sqrt(np.diag(co))
            
            fwhm_left.append(2.35*fit[2])
            fwhm_right.append(2.35*fit[5])
            
            fwhm_left_error.append(2.35*co[2])
            fwhm_right_error.append(2.35*co[5])
            
            value = value - fit[-1]  #subtracting base line
            
            x_left, x_right= finding_boundary(position, value,  *fit)  #finding boundary distance
            boundary_each.append (x_right - x_left)   #BOUNDARY DISTANCE
                          
            center = (x_left + x_right)/2   #Centering the data between two boudnary points
            
            x_align = zeros(len(position))
    
            for k in range (0, len(position)) :
                    
                     x_align[k] = position[k] - center  # aligns the data, where the averaged center distance is set to 0, and the two minima are at equal distance from the averaged center value
              #  
            for l in range (0, 2) :   #aligning the modelled data
                    
                     fit[3*l + 1] = fit[3*l + 1] - center
            
            std_mean.append(sem(x_align))   #SEM 
 
            #plotting individual profiles
            
            if 1 :
               
                if i < 6 :
           
                    ax1[i].plot(x_align, value   , marker='o' )
                    ax1[i].plot (x_align , guass(x_align, *fit) - fit[-1] )
                    ax1[i].axhline (y = 0 , color = 'k')
                    ax1[i].set_xlim(-10,10)
                
                if i > 5 and i < 12 : 
          
                    ax2[i - 6].plot(x_align, value    , marker='o' )
                    ax2[i - 6].plot (x_align , guass(x_align, *fit) - fit[-1] )
                    ax2[i - 6].axhline (y = 0 , color = 'k')
                
                if i > 11 and i < 18 :
                   
               
                    ax4[i - 12].plot(x_align, value   , marker='o' )
                    ax4[i - 12].plot (x_align , guass(x_align, *fit)- fit[-1]  )
                    ax4[i - 12].axhline (y = 0 , color = 'k')
                
                if i > 17 and i < 24 :
                 
                    ax5[i - 18].plot(x_align, value   , marker='o' )
                    ax5[i - 18].plot (x_align , guass(x_align, *fit) - fit[-1] )
                    ax5[i - 18].axhline (y = 0 , color = 'k')
                
                if i > 23 :
                   
                  
                    ax7[i - 24].plot(x_align, value   , marker='o' )
                    ax7[i - 24].plot (x_align , guass(x_align, *fit) - fit[-1] )
                    ax7[i - 24].axhline (y = 0 , color = 'k')
                    
          
    
            if (len(position %2) == 0 ) :
                   middle = int(len(position)/2)               
        
            else :
                   middle = int((len(position) + 1)/2)
            
  
            if i == 22 :  #DATA PROCESSING - PAPER
                
                ax10[1][1].plot(x_align, value  , '-o', label = "Data")
                ax10[1][1].plot(x_align, guass(x_align, *fit) - fit[-1], label = "Fit")
                ax10[1][1].axhline(y = 0, color = 'k')
                
                ax10[1][1].axvline (x = x_left - center , color = 'k' ,  linestyle = '--')#(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                ax10[1][1].axvline (x = x_right - center , color = 'k',  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                    
           
    #%% Plotting individual profiles
            if True : 
                if i < 6 :
             #       ax1[i].axvline (x = x_align [ind_one + ind_max_left] , color = 'k' ,  linestyle = '--')#(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
              #      ax1[i].axvline (x =x_align [ind_two + index_first_min] , color = 'k',  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                    
                     ax1[i].axvline (x = x_left , color = 'k' ,  linestyle = '--')#(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                     ax1[i].axvline (x = x_right , color = 'k',  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                    
                     ax1[i].set_xlabel ("Distance (nm)")
                     ax1[0].set_ylabel ("Gray value")
                     ax1[i].set_title ("Profile " + str(i+1))
                    
                    
                if i > 5 and i < 12  :
                  
                   # ax2[i - 6].axvline (x = x_align [ind_one + ind_max_left] , color = 'k',  linestyle = '--')#(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                   # ax2[i - 6].axvline (x = x_align [ind_two + index_first_min] , color = 'k' ,  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                
                    ax2[i - 6].axvline (x = x_left , color = 'k' ,  linestyle = '--') #(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                    ax2[i - 6].axvline (x =x_right , color = 'k',  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                
                    ax2[i - 6].set_xlabel ("Distance (nm)")
                    ax2[0].set_ylabel ("Gray value")
                    ax2[i - 6].set_title ("Profile " + str(i+1))
                
                
                if i > 11 and i < 18 :
                    
                  #  ax4[i - 12].axvline (x = x_align [ind_one + ind_max_left] , color = 'k',  linestyle = '--')#(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                   # ax4[i - 12].axvline (x = x_align [ind_two + index_first_min] , color = 'k' ,  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                
                    ax4[i - 12].axvline (x = x_left , color = 'k' ,  linestyle = '--') #(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                    ax4[i - 12].axvline (x = x_right , color = 'k',  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                
                    ax4[i - 12].set_xlabel ("Distance (nm)")
                    ax4[0].set_ylabel ("Gray value")
                    ax4[i - 12].set_title ("Profile " + str(i+1))
                    
                if i > 17 and i < 24 :
                    
                   # ax5[i - 18].axvline (x = x_align [ind_one + ind_max_left] , color = 'k',  linestyle = '--')#(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                   # ax5[i - 18].axvline (x = x_align [ind_two + index_first_min] , color = 'k' ,  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                    
                    ax5[i - 18].axvline (x = x_left , color = 'k',  linestyle = '--')#(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                    ax5[i - 18].axvline (x = x_right , color = 'k' ,  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                
                    ax5[i - 18].set_xlabel ("Distance (nm)")
                    ax5[0].set_ylabel ("Gray value")
                    ax5[i - 18].set_title ("Profile " + str(i+1))
                
                if i > 23 :
                    
                  #  ax7[i - 24].axvline (x = x_align [ind_one + ind_max_left] , color = 'k',  linestyle = '--')#(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                   # ax7[i - 24].axvline (x = x_align [ind_two + index_first_min] , color = 'k' ,  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                
                    ax7[i - 24].axvline (x = x_left , color = 'k',  linestyle = '--')#(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                    ax7[i - 24].axvline (x =x_right , color = 'k' ,  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                
                    ax7[i - 24].set_xlabel ("Distance (nm)")
                    ax7[0].set_ylabel ("Gray value")
                    ax7[i - 24].set_title ("Profile " + str(i+1))
                
    
           # boundary_each.append (x_align [ind_two + index_first_min] - x_align [ind_one + ind_max_left] )
            
          
            print (i ,- x_left + x_right, fit[4] - fit[1] )
    
            dip_dip_each.append(fit[4] - fit[1])  #DIP to DIP distance
            
            error_inner.append( np.sqrt(co[1]**2 + co[4]**2))  #error in  dip to dip
            
            SD_inner_left.append(fit[2])  #left inner dip
            SD_inner_right.append(fit[5]) #right inner dip
            
            fit_max.append(min(fit_amplitude))  #due to base line subtraction, values are already negative (so, minima chosen)
                
    #%% Averaging 

            if 1 :#i!= 3 :
               # print (x_align[0], x_align[-1])
                index_lower = where (x_align > -12.5  )
                index_lower = index_lower[0][0]
            
                index_higher = where (x_align > 12)
                index_higher = index_higher[0][0]
                
                x_align = x_align[index_lower : index_higher]
                value = value [index_lower : index_higher]
                
               
       #         print (files[i], x_align[0], x_align[-1], len(x_align))
                
                x_new = linspace(-11.5 , 11.5 , 150)
                
                value_new  = interp1d(x_align, value)(x_new)  #splined gray value
                
            
                if i == 0 :
                    
                    integrate = zeros(len(x_new))
                    integrate = value_new
                
                else :
                    
                    integrate = integrate + value_new
                
                average = integrate / (len(files_all)-1)
                

#for u in range(0, len(average)) :
 #  average_SW.write (str(x_new[u]) + "\t" + str(average[u]) + "\n" )  # left and right inner dip
       
   
#print (boundary_each) 
ax9.errorbar (x_new, average, yerr = sem(average) , marker = 'o', label = "Data" )
ax9.axhline ( y = 0  , color = 'k')
   
#ax3.set_title ("Averaged profile_SWNTS")
ax9.set_xlabel ("Distance (nm)")
ax9.set_ylabel ("Normlaised Line profile contrast (arb. units)")
ax9.legend()

  
#print (fit_params)
for y in range(0, 2 ):  #Labeling plot (data processing -PAPER)
    
    for p in range (0, 2) :
        ax10[y][p].set_xlabel("Distance (nm)")
        ax10[y][p].set_ylabel("Line profile contrast (arb. units)")
        ax10[y][p].legend()
                

#%% ----Histograms

boundary_each = np.sort(boundary_each)
boundary_each = np.round(boundary_each, 1)

dip_dip_each = np.sort(dip_dip_each)
dip_dip_each = np.round(dip_dip_each, 1)

elements_count_boundary = collections.Counter(boundary_each)  #counts frequency of each diameter value


bound_freq =  elements_count_boundary.keys() #frequency of each boudary values
bound_freq = list(bound_freq)
bound_values =  elements_count_boundary.values()  #boundary values
bound_values = list(bound_values)  

elements_count_dip = collections.Counter(dip_dip_each)

dip_freq =  elements_count_dip.keys() #frequency of each dip dip values
dip_freq = list(dip_freq)
dip_values =  elements_count_dip.values()  #dip dip values
dip_values = list(dip_values)  


#Copying boundary distance and dip to dip distance values and frequency

#for f in range (0 , len(bound_values)) :  #writing averaged_profiles from each tube to txt file. For comparison with DW NTs
        
     # boundary_all.write (str(bound_freq[f]) + '\t'  + str(bound_values[f]) + "\n" )

#for f in range (0 , len(dip_values)) :  #writing averaged_profiles from each tube to txt file. For comparison with DW NTs
        
     # dip_dip_all.write (str(dip_freq[f]) + '\t'  + str(dip_values[f]) + "\n" )

#----------------------------------------

ax6[0].bar(elements_count_boundary.keys(), elements_count_boundary.values(), align='center', width=0.2,facecolor="blue", edgecolor='k' , alpha = 0.5,  lw = 2,  label = "Boundary_distance" )  
ax6[1].bar(elements_count_dip.keys(), elements_count_dip.values(), align='center', width=0.2,facecolor="blue", edgecolor='k' , alpha = 0.5,  lw = 2,  label = "Dip-to-dip_distance" )  

ax8.bar(elements_count_boundary.keys(), elements_count_boundary.values(), align='center', width=0.1,facecolor="brown", edgecolor='k' , alpha = 0.5,  lw = 2,  label = "Boundary distance" )  
ax8.bar(elements_count_dip.keys(), elements_count_dip.values(), align='center', width=0.1,facecolor="green", edgecolor='k' , alpha = 0.5,  lw = 2,  label = "Inner dip-to-dip distance" )  

  
ax6[0].set_title ("Boundary distance")
ax6[0].set_xlim(9.5 ,14.5)

ax6[1].set_title("Dip to dip distance")
ax6[1].set_xlim(5.5 , 8.5)

ax8.set_xlabel ("Distance (nm)")
ax8.set_ylabel("Number of line profiles")


ax8.set_xlim(5.5, 14.5)
ax8.legend()


print ("Boundary distance " , round(mean(boundary_each),1) , " ± " , round(std(boundary_each),1))

print ("Inner dip " , round(mean(dip_dip_each),1) , " ± " , round(std(dip_dip_each),1))

print ("SWNTs left SD dip " , round(mean(SD_inner_left),1) ," ± " , round(std(SD_inner_left),1))
print ("SWNTs right SD dip " , round(mean(SD_inner_right),1) ," ± " , round(std(SD_inner_right),1))

for k in range (0, 2 ) :
    
    for j in range(0, 2) :
    
        ax10[k][j].xaxis.label.set_size(15) 
        ax10[k][j].yaxis.label.set_size(15) 
        
        
        ax10[k][j].tick_params(axis='x', labelsize= 15 )
        ax10[k][j].tick_params(axis='y', labelsize= 15 )
