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
from scipy.stats import norm
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import collections
import matplotlib as mpl
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

mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False



fig1,ax1 = plt.subplots(1,7,figsize = [8,8])
fig2,ax2 = plt.subplots(1,7,figsize = [8,8])
fig11,ax11 = plt.subplots(1,8,figsize = [8,8])
fig12,ax12 = plt.subplots(1,8,figsize = [8,8])

fig3,ax3 = plt.subplots(figsize = [8,8])

#fig4,ax4 = plt.subplots(1,3,figsize = [8,8])
#fig5,ax5 = plt.subplots(1,3,figsize = [8,8])

#fig6,ax6 = plt.subplots(2,1 ,figsize = [8,8] , sharex = True)
fig7,ax7 = plt.subplots(1,2,figsize = [8,8])
fig8,ax8 = plt.subplots(2,1,figsize = [8,8], sharex = True)

fig9,ax9 = plt.subplots(2,1,figsize = [8,8])

fig10,ax10 = plt.subplots(2,2,figsize = [8,8])

def guass(x, h,c,w, h1,c1,w1, h2,c2,w2,  h3,c3,w3 , b):
     
    return (h*exp((-(x-c)**2)/(2*w**2)) + h1*exp((-(x-c1)**2)/(2*w1**2)) +  h2*exp((-(x-c2)**2)/(2*w2**2)) + h3*exp((-(x-c3)**2)/(2*w3**2))) + 1*b

def guass_average(x, h,c,w, h1,c1,w1, h2,c2,w2,  h3,c3,w3,  h4,c4,w4,  h5,c5,w5 ) : 
     
    return h*exp((-(x-c)**2)/(2*w**2)) + h1*exp((-(x-c1)**2)/(2*w1**2)) +  h2*exp((-(x-c2)**2)/(2*w2**2)) + h3*exp((-(x-c3)**2)/(2*w3**2)) + h4*exp((-(x-c4)**2)/(2*w4**2)) +  h5*exp((-(x-c5)**2)/(2*w5**2)) # - h6*exp((-(x-c6)**2)/(2*w6**2)) 


def offset_correction(x,m,c):  #baseline correcion
        return m*x+c
    
def guass_single(x, h,c,w):
     
    return h*exp((-(x-c)**2)/(2*w**2)) 

def centering(j , position,value) :  #aligining data

    if (len(position %2) == 0 ) :
              middle = int(len(position)/2)               
   
    else :
              middle = int((len(position) + 1)/2)
   
    ind_high = where (value[0 : middle]  == max(value[0 : middle]))
    ind_high = ind_high[0][0]
        
    index_2 = where (value  == min(value [middle : middle + 24]) ) #third mimima  
    index_2 = index_2[0][0]
    
    if j == 2  :
        
          index_1 = where (value  [middle - 14 : middle]  == min(value [middle - 14 : middle]) ) #second mimima
          index_1 = index_1[0][0]  +  middle - 14
          
          index_0 = where( value == min(value[ind_high  : (index_1 - 3)]))     #first mimima   
          index_0 = index_0[0][0]
    
           
    else :
        index_0 = where( value == min(value[ind_high  : (ind_high + 10)]))     #first mimima   
        index_0 = index_0[0][0]
    
        index_1 = where (value  [index_0 + 10 : middle]  == min(value [index_0  + 10 : middle]) ) #second mimima
        index_1 = index_1[0][0]  +  index_0 + 10
     
    index_3 = where (value == min(value[index_2 + 6 : index_2 + 45 ]))   # fourth minima
    index_3 = index_3[0][0]
         
    constant =  0*mean(value [0 : 40]) 
   
    limit_h_low = [  value[index_0] - 20 , -500 , -500, -500 ] #value[index_1] -20, value[index_2] - 20, value[index_3]  - 20 ]
    limit_h_high = [  value[index_0] + 20 , -100 , -100,-100 ]# value[index_1] + 20 , value[index_2] + 20, value[index_3]  + 20]
      
    limit_x_low = [position[index_0 - 6],  position[index_1 - 6], position[index_2 - 6],  position[index_3 - 6]]
    limit_x_high = [position[index_0 + 6], position[index_1 + 6], position[index_2 + 6],  position[index_3  + 6]]
     
    
    limit_low =  [ limit_h_low[0],limit_x_low[0], 0.3,      limit_h_low[1],limit_x_low[1],0.1,        limit_h_low[2],limit_x_low[2],0.1,     limit_h_low[3], limit_x_low[3],0.1,   -5 ]
    limit_high = [ limit_h_high[0],limit_x_high[0], 3,      limit_h_high[1],limit_x_high[1],3,        limit_h_high[2],limit_x_high[2],3,     limit_h_high[3], limit_x_high[3],3,  5 ]

#limit_low =  [ limit_h_low[0],limit_x_low[0], 0.3,      limit_h_low[1],limit_x_low[1],0.1,        limit_h_low[2],limit_x_low[2],0.1,     limit_h_low[3], limit_x_low[3],0.1,   -5 ]
 #   limit_high = [ limit_h_high[0],limit_x_high[0], 3,      limit_h_high[1],limit_x_high[1],3,        limit_h_high[2],limit_x_high[2],3,     limit_h_high[3], limit_x_high[3],3,  5 ]
       
    fit_g,co = curve_fit(guass, position, value, bounds = (limit_low,limit_high), maxfev = 5000) #Gaussian fits
    center = 0*(abs(fit_g[4]) + abs(fit_g[7]))/2
    
   # aligned_x = zeros(len(position))
    
    #for i in range (0, len(position)) :
        
        # aligned_x[i] = position[i] - center  # aligns the data, where the averaged center distance is set to 0, and the two minima are at equal distance from the averaged center value
    
    #shifting center
    fit_g[1] = fit_g[1] - center
    fit_g[4] = fit_g[4] - center
    fit_g[7] = fit_g[7] - center
    fit_g[10] = fit_g[10] - center
    
    #print (fit_g[1], fit_g[10])
    
  #  fit_g[0] = fit_g[0] - fit_g[-1]
  #  fit_g[3] = fit_g[1] - fit_g[-1]
  #  fit_g[6] = fit_g[2] - fit_g[-1]
  #  fit_g[9] = fit_g[3] - fit_g[-1]
      
    return ( fit_g,  co )

def model_guass(x,gray):  #Modelling averaged TEM profiles
    
   peaks_positive = find_peaks(average, height= 50)

   peaks_negative = find_peaks (- average, height = 70)
    
   bounds_lower = [30, x[peaks_positive[0][0]]-0.3 ,0.1,  -500,x[peaks_negative[0][0]]-0.3,0.1,  -500,x[peaks_negative[0][1]]-0.3,0.1,  -500,3,0.1,  -500,5,0.1,    30, x[peaks_positive[0][1]]-0.3 ,0.1]          
   bounds_upper = [300, x[peaks_positive[0][0]]+0.3 ,3,  -50,x[peaks_negative[0][0]]+0.3,3,  -50,x[peaks_negative[0][1]]+0.3,3,          -100,4,3,   -50,6,3,    300, x[peaks_positive[0][1]]+0.3 ,3]          

   average_model, co_fit = curve_fit(guass_average, x, average , bounds =(bounds_lower,bounds_upper) , maxfev = 5000)
   
   
#   average_derivative =  average_fit.diff(x)
   
   return (average_model, co_fit)
        
    
def derivative (*fit_params):  #finding derivative of Gaussain function 
    
    #storing values of variables of Gaussian function
    
    counter = 0
    h = zeros (6)
    c = zeros (6)
    w = zeros (6)
    
    for i in range (0, 6) :
        
        h[i] = fit_params [0 + counter]
        c[i] = fit_params [1 + counter]
        w[i] = fit_params [2 + counter]
        
        #print (counter)
        
        counter = counter + 3
        
    x = Symbol('x')
    
    #Definig the Gaussian function
    
    req_fun = h[0]*sympy.exp((-(x-c[0])**2)/(2*w[0]**2)) + h[1]*sympy.exp((-(x-c[1])**2)/(2*w[1]**2)) +  h[2]*sympy.exp((-(x-c[2])**2)/(2*w[2]**2)) + h[3]*sympy.exp((-(x-c[3])**2)/(2*w[3]**2)) + h[4]*sympy.exp((-(x-c[4])**2)/(2*w[4]**2)) +  h[5]*sympy.exp((-(x-c[5])**2)/(2*w[5]**2)) 
    
    #differentaiing the function
    f_der = req_fun.diff(x)  #only the algebraic expression
    
    req_fun = lambdify(x, req_fun)
    
    f_der = lambdify(x, f_der)  #can give values
    
    return (f_der)
      

   
   
#%% Reading data from DWNTs TEM profile and averaging

files_all = glob.glob("DW\\***.csv")  #reading files
print (files_all)

#files_all = files_all[6 : 29]
#print (files_all)

dip_dip = zeros(len(files_all))      #inner dip to dip distance

boundary = zeros (len(files_all)) #boundary distance
dip_dip_outer =  zeros (len(files_all)) #outer dip to dip distance

std_err = zeros(len(files_all)) #stnadard error on averageing

boundary_each = []    #storing boundary distance
dip_dip_each = []  #storing inner dip dip distance
dip_dip_each_outer = []  #storing outer dip dip distance

fwhm_left_inner = []  #fwhm left inner dip
fwhm_right_inner = []  #fwhm right inner dip

fwhm_left_outer = []  #fwhm left outer dip
fwhm_right_outer = []  #fwhm left outer dip

fit_amplitude = zeros(4)  #storing amplitude of 4 gaussians

fit_max = []  # to store max amplitude
error_inner = [] #stores error in dip to dip (inner)

#--------stroring fwhm values
fwhm_left_inner_error = []   
fwhm_right_inner_error = []
fwhm_left_outer_error = [] 
fwhm_right_outer_error = []

std_mean = [] #stanadard error of mean of each profile

#standard deviation in FWHM 
SD_inner_left = []
SD_inner_right = []
SD_outer_left = []
SD_outer_right = []

length = [] #to know the average number of data points in each data set               

def finding_boundary (position, value,count, *fit) :  #finding crossing points
     
   
     
     low_left  = where (position > fit[1])     #left outer dip
     low_left = low_left[0][0] - 1       
   
    #---------------different conditions used based on individual profiles
     if count == 9 : #Profile 27
         
         high_left = where (value [0 : low_left ] > 0 )  #finding first point > 0 LHS
         high_left = high_left[0][-1]  - 4
        
     if count == 7 :  #Profile 24
         
         high_left = where (value [0 : low_left ] > 0 )  #finding first point > 0 LHS
         high_left = high_left[0][-1]  - 14
        
     if count !=9 and count != 7 :
         high_left = where (value [0 : low_left ] > 0 )  #finding first point > 0 LHS
         high_left = high_left[0][-1]  - 1
        
     low_right  = where (position > fit[10])
     low_right = low_right[0][0]                    #right outer dip
     
     high_right = where (value [low_right : -1 ] > 0 )   #finding first point > 0 RHS
     high_right = high_right[0][0] + 2 +  low_right
   
     limits = [high_left,low_left,  low_right , high_right ]  
     
  
     x_req_0 = zeros(2)  #boundary points on left and right
     
     for m in range (0 , 2 ) :  #left and right boundary distances
         
          x_interpolate = position [limits[m*2] : limits[m*2+1]]   
          y_interpolate = value [limits[m*2] : limits[m*2+1]]
          
         # print (x_interpolate , y_interpolate)
          
          if m == 0 :  #defining range of contrast values between LHS maxima and left outer dip
              y_req = array (range ( int(y_interpolate [-1]), int(y_interpolate [0]),  1))   #finding the crossing point
              
          else : #defining range of contrast values between right outer dip and RHS maxima
              y_req = array (range ( int(y_interpolate [0]), int(y_interpolate [-1]),  1))    #finding the crossing point
          
         
          x_req  = interp1d(y_interpolate, x_interpolate)(y_req)  #splined gray value (1d interpolation)
          
          y_req_0 = where (y_req == 0)
          y_req_0 = abs(y_req[0])
          
          #print ("yes" , y_req_0, x_req[y_req_0])
          x_req_0[m] = x_req[y_req_0]   
     
     return ( x_req_0[0], x_req_0[1])

#%%
if 1 :
    
    
    for i in range (0, len(files_all)) :#len(files) ) :#len(files)) :
            
         
            position = []   # in nm
            value = []      # grayvalue
                  
            
            file = open(files_all[i])
            
            data = csv.reader(file, delimiter=',')
       
            for distance,gray in data :
            
                   value.append(gray) 
                   position.append(distance)
        
            value = value [1 : len(value)]      #contrast arb. units
            position = position [1 : len(position)]   #distance in nm
                    
            value = np.asarray(value, dtype = float)   #converting to ndarray   
            position = np.asarray(position, dtype = float)    #converting to ndarray 
            
            position_offset = []
            position_offset.extend(position[0 : 30])
            position_offset.extend(position[-20 : -1])
            
            value_offset = []
            value_offset.extend(value[0 : 30])
            value_offset.extend(value[-20 : -1])
            
            value_offset = numpy.asarray(value_offset, dtype = float)
            position_offset = numpy.asarray(position_offset , dtype = float)
            
            correction, co = curve_fit(offset_correction, position_offset , value_offset )  # (baseline correction)
            
            fit_offset = offset_correction(position , *correction)

            if i == 22 :  #DATA PROCESSING - PAPER
            
                ax10[0][0].plot(position, value,  '-o', label = "Data")  #DATA PROCESSING - PAPER
                             
                ax10[0][0].plot (position, fit_offset, label = "Baseline") 
                
            value = value - mean(fit_offset)  #subracting baseline 
            
            if i == 1 :  #DATA PROCESSING - PAPER
                
                ax10[0][1].plot(position, value,  '-o' , label = "Data")
                ax10[0][1].axhline(y = 0)
            
            if i == 6 or i == 14 or i== 9 or i == 5 :   #defining limits differently for aligning profiles (depeds on each file)
                j = 2 
               
            
            else :
                j = 0
           
            
            fit, co  = centering(j ,position, value )  #modeling individual profiles
            
            if i == 22 :  #DATA PROCESSING - PAPER
                
                ax10[1][0].plot(position, value, '-o', label = "Data")
                ax10[1][0].plot(position, guass(position, *fit) - fit[-1], label = "Fit")
                ax10[1][0].axhline(y = 0, color = 'k')
            
            co = np.sqrt(np.diag(co))  #variance in modelling
             
            if 1 : #i == 1 :   #Finding Boundary distance via interpolation (different conditions based on individual profiles)
                
                if i == 27  or i == 23 :
                    count = int(i/3)
                
                else :
                    count = 0
                    
                x_left, x_right= finding_boundary(position, value,count,  *fit)
                
                boundary_each.append (x_right - x_left)     # BOUNDARY Distance
                          
                center = (x_left + x_right)/2   #Centering the data between two boudnary points
                
         #       print (i, x_right-x_left, x_left)
                
                x_align = zeros(len(position))
    
                for k in range (0, len(position)) :
                    
                     x_align[k] = position[k] - center  # aligns the data, where the averaged center distance is set to 0, and the two minima are at equal distance from the averaged center value
              #  
                for l in range (0, 4) :   #aligning the modelled data
                    
                     fit[3*l + 1] = fit[3*l + 1] - center
            
            if i == 22 :  #DATA PROCESSING - PAPER
                
                ax10[1][1].plot(x_align, value  - fit[-1], '-o', label = "Data")
                ax10[1][1].plot(x_align, guass(x_align, *fit) - fit[-1], label = "Fit")
                ax10[1][1].axhline(y = 0, color = 'k')
                
                ax10[1][1].axvline (x = x_left - center , color = 'k' ,  linestyle = '--')#(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                ax10[1][1].axvline (x = x_right -center , color = 'k',  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                      
                
            if 1: #plotting each profile
                if i < 7 :
                   # print ("first", i,  files[i])
                    ax1[i].plot(x_align, value  - fit[-1]  , marker='o' )
                    ax1[i].plot (x_align , guass(x_align, *fit) - fit[-1] )
                    ax1[i].axhline (y = 0 , color = 'k')
                   
                    ax1[i].axvline (x = x_left - center , color = 'k' ,  linestyle = '--')#(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                    ax1[i].axvline (x = x_right -center , color = 'k',  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                    
                
                if i > 6 and i < 14 : 
                        
                 #   print ("second" , i , files[i])
                    ax2[i - 7].plot(x_align, value  - fit[-1]  , marker='o' )
                    ax2[i - 7].plot (x_align , guass(x_align, *fit) - fit[-1] )
                    ax2[i - 7].axhline (y = 0 , color = 'k')
                    
                    ax2[i - 7].axvline (x = x_left - center  , color = 'k' ,  linestyle = '--') #(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                    ax2[i - 7].axvline (x =x_right - center , color = 'k',  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                


                if i > 13 and i < 22 : 
                    
                    ax11[i - 14].plot(x_align, value  - fit[-1]  , marker='o' )
                    ax11[i - 14].plot (x_align , guass(x_align, *fit) - fit[-1] )
                    ax11[i - 14].axhline (y = 0 , color = 'k')
                    
                    ax11[i - 14].axvline (x = x_left - center , color = 'k' ,  linestyle = '--') #(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                    ax11[i - 14].axvline (x = x_right - center , color = 'k',  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                

                if i > 21 :
                    
                    ax12[i - 22].plot(x_align, value  - fit[-1]  , marker='o' )
                    ax12[i - 22].plot (x_align , guass(x_align, *fit) - fit[-1] )
                    ax12[i - 22].axhline (y = 0 , color = 'k')
                    
                    ax12[i - 22].axvline (x = x_left - center , color = 'k' ,  linestyle = '--') #(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                    ax12[i - 22].axvline (x = x_right - center , color = 'k',  linestyle = '--')   #x = -(x_align[ind_zero + ind_max_left] + x_align [ind_one + ind_max_left])*0.5 , color = 'k' , linestyle = '--')
                
                #    print (i, files_all[i])
                
            if (len(position %2) == 0 ) :
                   middle = int(len(position)/2)               
        
            else :
                   middle = int((len(position) + 1)/2)
            
            value = value - fit[-1]  #subtracting base line
            
           

    #%% Plotting individual profiles
    
            if 1 :
                if i < 7 :
                   
                    ax1[i].set_xlabel ("Distance (nm)")
                    ax1[0].set_ylabel ("Gray value")
                    ax1[i].set_title ("TProfile " + str(i+1))
                    
                    
                if i > 6 and i < 14  :
                    
                  
                    ax2[i-7].set_xlabel ("Distance (nm)")
                    ax2[0].set_ylabel ("Gray value")
                    ax2[i - 7].set_title ("TProfile " + str(i+1))
                #    print (str(i))
                
                if i > 13 and i < 22 : 
                    
                    
                     ax11[i-14].set_xlabel ("Distance (nm)")
                     ax11[0].set_ylabel ("Gray value")
                     ax11[i - 14].set_title ("TProfile " + str(i+1))
                    # print (i)
                     
                if i > 21 :
                    
                  
                     ax12[i-22].set_xlabel ("Distance (nm)")
                     ax12[0].set_ylabel ("Gray value")
                     ax12[i - 22].set_title ("TProfile " + str(i+1))
                     #print (i)
                    

            dip_dip_each.append(fit[7] - fit[4])  #inner 
            
            dip_dip_each_outer.append(fit[10] - fit[1])  #outer dip
            
            
            error_inner.append( np.sqrt(co[7]**2 + co[4]**2))  #error in inner dip to dip
            
            std_mean.append(sem(x_align))  #storing SEM of each profile
            
            SD_inner_left.append(fit[5])  #SD left inner dip  FWHM
            SD_inner_right.append(fit[8]) #SD right inner dip FWHM
            
            SD_outer_left.append(fit[2])  #SD left outer dip FWHM
            SD_outer_right.append(fit[11]) #SD right outer dip FWHM
            
            print (i, fit[10] , fit[1], fit[10] - fit[1])

    

#%% Averaging data

           # print (files[i] , len(x_align))
            if 1 :#i!= 3 :
             
                index_higher = where (x_align > 12)
                index_higher = index_higher[0][0]
                                               
                index_lower = where (x_align > -12.1  )
                index_lower = index_lower[0][0]
                
                x_align = x_align[index_lower : index_higher]
                value = value [index_lower : index_higher]
                
                x_new = linspace(-11.5,11.5,150)
               
                value_new  = interp1d(x_align, value)(x_new)  #splined gray value
                          
                if i == 0 :
                    
                    integrate = zeros(len(x_new))
                    integrate = value_new
                
                else :
                    
                    integrate = integrate + value_new
                
                average = integrate / (len(files_all)-1)


ax7[0].errorbar (x_new,  -1*average/min(average), yerr = sem(average)/min(average) , marker = 'o',  color ='red', label = "Defocus 2")
ax7[0].axhline ( y = 0, color ='k')

ax3.errorbar (x_new,  -1*average/min(average), yerr = sem(average)/min(average) , marker = 'o',  color ='blue', label = "DWNTs")
ax3.axhline ( y = 0, color ='k')

ax3.set_xlabel ("Distance (nm)")
ax3.set_ylabel ("Line profile contrast (arb. units)")


for y in range(0, 2 ):  #Labeling plot (data processing -PAPER)
    
    for p in range (0, 2) :
        ax10[y][p].set_xlabel("Distance (nm)")
        ax10[y][p].set_ylabel("Line profile contrast (arb. units)")
        ax10[y][p].legend()
                



#%%  Reading SW average 

SW_tem = loadtxt("SW_new_average.txt")
SW_dist = SW_tem[:,0]
SW_avg = SW_tem[:,1]

ax7[1].errorbar (SW_dist, -1*SW_avg/min(SW_avg), yerr = sem(SW_avg)/min(SW_avg) , marker = 'o', color = 'red' , label = "Defocus 2")
ax7[1].axhline ( y = 0, color ='k')
ax7[1].legend()

ax7[0].set_xlabel ("Distance (nm)")
ax7[0].set_ylabel ("Line profile contrast (arb. units)")

ax3.errorbar (SW_dist, -1*SW_avg/min(SW_avg), yerr = sem(SW_avg)/min(SW_avg) , marker = 'o', color = 'red' , label = "Flash-diluted NTs")
ax3.axhline ( y = 0, color ='k')
ax3.legend()
    


#%% Histograms



dip_dip_each_outer[23] = 11  #Only for NWO conference, problem with Gaussian

#rounding data to one decimal place
boundary_each = np.sort(boundary_each)
boundary_each = np.round(boundary_each, 1)

dip_dip_each = np.sort(dip_dip_each)
dip_dip_each = np.round(dip_dip_each, 1)

dip_dip_each_outer = np.sort(dip_dip_each_outer)
dip_dip_each_outer = np.round(dip_dip_each_outer, 1)

elements_count_boundary = collections.Counter(boundary_each)  #counts frequency of each diameter value
elements_count_dip = collections.Counter(dip_dip_each)
elements_count_dip_outer = collections.Counter(dip_dip_each_outer)


ax8[0].bar(elements_count_boundary.keys(), elements_count_boundary.values(), align='center', width=0.1,facecolor="brown", edgecolor='k' , alpha = 0.5,  lw = 2,  label = "Boundary distance" )  
ax8[0].bar(elements_count_dip.keys(), elements_count_dip.values(), align='center', width=0.1,facecolor="green", edgecolor='k' , alpha = 0.5,  lw = 2,  label = "Inner dip-to-dip distance" )  
ax8[0].bar(elements_count_dip_outer.keys(), elements_count_dip_outer.values(), align='center', width=0.1,facecolor="green", edgecolor='k' , alpha = 1,  linestyle = '--',  lw = 2,  label = "Outer dip-to-dip distance" )  

#ax9.bar(elements_count_boundary.keys(), elements_count_boundary.values(), align='center', width=0.1,facecolor="blue", edgecolor='none' , alpha = 0.3,  lw = 2,  label = "DWNTs" )  

print ("DWNTs Boundary distance " , round(mean(boundary_each),1) , " ± ", round(std(boundary_each), 1))
print ("DWNTs Inner dip " , round(mean(dip_dip_each),1) ," ± " , round(std(dip_dip_each),1))

print ("DWNTs left SD Inner dip " , round(mean(SD_inner_left),1) ," ± " , round(std(SD_inner_left),1))
print ("DWNTs right SD Inner dip " , round(mean(SD_inner_right),1) ," ± " , round(std(SD_inner_right),1))

print ("DWNTs left SD outer dip " , round(mean(SD_outer_left),1) ," ± " , round(std(SD_outer_left),1))
print ("DWNTs right SD outer dip " , round(mean(SD_outer_right),1) ," ± " , round(std(SD_outer_right),1))
print ("DWNTs Outer dip " , round(mean(dip_dip_each_outer),1) , " ± " ,  round(std(dip_dip_each_outer),1))


ax8[0].set_xlim(5.5, 14.5)
#ax8[0].legend()
ax8[0].set_ylim(0,5)
#ax8[0].set_xlabel ("Distance (nm)")
ax8[0].set_ylabel ("Number of line profiles")

#---------comparing with image 26
 
ax9[1].bar(elements_count_boundary.keys(), elements_count_boundary.values(), align='center', width=0.1,facecolor="blue", edgecolor='k' , alpha = 0.5,  lw = 2,  label = "DWNTs (boundary distance)" )  


#%% Copying distribution values of SW NTs

bound_sw = np.loadtxt("SW_dip_boundary_all.txt")

ax8[1].bar(bound_sw[:,0], bound_sw[:,1] ,  align='center', width=0.1,facecolor="brown", edgecolor='k' , alpha = 0.5, lw = 2, label = "Boundary distance" )

ax9[1].bar(bound_sw[:,0], bound_sw[:,1] ,  align='center', width=0.1, facecolor="red", edgecolor='k' , alpha = 0.5,  lw = 2, label = "Flash-diluted NTs (boundary distance)" )  #comparing with image 26

#ax9.bar(bound_sw[:,0], bound_sw[:,1] ,  align='center', width=0.1,facecolor="green", edgecolor='none' , alpha = 0.3,  lw = 2, label = "Flash diluted NTs" )

dip_sw = np.loadtxt("SW_dip_dip_all.txt")
ax8[1].bar(dip_sw[:,0], dip_sw[:,1],  align='center', width=0.1,facecolor="green", edgecolor='k' , alpha = 0.5,  lw = 2,  label = "Inner dip-to-dip distance" )

ax9[1].bar(dip_sw[:,0], dip_sw[:,1],  align='center', width=0.1,facecolor="red", edgecolor='k' , alpha = 0.5,  lw = 2, linestyle = '--' ,  label = "Flash-diluted NTs (dip-dip)" )


ax8[1].set_xlim(5.5, 14.5)
#ax8[1].legend(loc = 2)
ax8[1].set_ylim(0,5)
ax8[1].set_xlabel ("Distance (nm)")
ax8[1].set_ylabel ("Number of line profiles")


print ("SW_ boundary distance" , str(round(mean(bound_sw[:,0]),1)), " ± " , str(round(std(bound_sw[:,0]), 1 )))
print ("SW_ dio_dip distance" , str(round(mean(dip_sw[:,0]),1)), " ± " , str(round(std(dip_sw[:,0]), 1 )))

for k in range (0, 2 ) :
    
    for j in range(0, 2) :
    
        ax10[k][j].xaxis.label.set_size(15) 
        ax10[k][j].yaxis.label.set_size(15) 
        
        
        ax10[k][j].tick_params(axis='x', labelsize= 15 )
        ax10[k][j].tick_params(axis='y', labelsize= 15 )
        
    

    