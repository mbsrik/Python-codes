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
    
    if i == 2:  #chosing sundar (the sample used for TEM)
        
        WL, data = convert(file,specified_lines)
       
        inner_peak_dw = inner_peak_WL(WL, data)
        
        wavenumber = 10**4/WL
        ax2.plot(10**4/WL,data/data[inner_peak_dw], color = 'blue',  label =  "Double wall nanotubes") #
    
    
        wn_dw = 10**7/WL[inner_peak_dw]  #wavenumber peak value (cm-1)

        print ("Double wall NTs, inner peak WL ",  WL[inner_peak_dw] , " nm, " , round(10**7/WL[inner_peak_dw],0) , "cm-1 and OD " , round(data[inner_peak_dw],2))
        print ("Double wall NTs, outer peak WL ",  WL[where(data == max(data))[0][0]] , " nm, " , round(10**7/ WL[where(data == max(data))[0][0]] , 0), " cm-1")
        
        xini = [0.05,596.5,0.1,  0.08,589,1,  0.005,580,1,  0.005,577,0.5,  0.008,562,1,  0.004,540,0.2,   0.002,485,5,  0.001,500,5]
        xfinal = [0.4,600,20,    0.6,594,20, 0.3,585,20,    0.3,581,20,  0.3,570,20,  0.1,557,25,    0.5,525,25,  0.4,535,20 ]
             
        param, co = curve_fit(main, WL, data,  bounds = (xini,xfinal))
        co = np.sqrt(np.diag(co))
        
        ax4.plot(WL,data, label = "data")
        
        
        
        print ("DWNTs ", round(param[2],2) , " ± ", round(co[2],2))
        print (param)
        fit = main(WL,*param)
      
        fit1 = lorr(WL,param[0],param[1],param[2])
        fit2 = lorr(WL,param[3],param[4],param[5])
        fit3 = lorr(WL,param[6],param[7],param[8])
        fit4 = lorr(WL,param[9],param[10],param[11])
        fit5 = lorr(WL,param[12],param[13],param[14])
        fit6 = lorr(WL,param[15],param[16],param[17])
        fit7 = lorr(WL,param[18],param[19],param[20])
        fit8 = lorr(WL,param[21],param[22],param[23])
        
        ax4.plot(WL,fit/1,   label = "fit" , color = 'k')
        ax4.plot(WL,fit1/1, linestyle = "--", label = "1")
        ax4.plot(WL,fit2/1, linestyle = "--", label = "2")
        ax4.plot(WL,fit3/1, linestyle = "--", label = "3")
        ax4.plot(WL,fit4/1, linestyle = "--", label = "4")
        ax4.plot(WL,fit5/1, linestyle = "--", label = "5")
       
        ax4.plot(WL,fit7/1, linestyle = "--", label = "7")
        ax4.plot(WL,fit8/1, linestyle = "--", label = "8")
        
        ax4.set_xlabel ("Wavelength (nm)")
        ax4.set_ylabel ("Absorption")
       
        #for l in range (0, 6) : 
            
          #  print (param[3*l+1] , co[3*l+1], 10**7/param[3*l+1],  10**7/(param[3*l+1] - param[3*l+2]) - 10**7/(param[3*l+1] + param[3*l+2]) )

       # print ((param))
       
    if i == 1:  #chosing sundar (the sample used for TEM)  #monomer file
        
        monomer = loadtxt("monomer.sp")
        
        monomer_WL = monomer[:,0]
        monomer_count = monomer[:,1]
        
        ind_520 = where (monomer_WL == 520)
        ax2.plot(10**4/monomer_WL, (monomer_count - mean(monomer_count[0 : 30]))/monomer_count[ind_520] , color = 'brown',  label =  "Monomer")
        
        print ("Monomer max OD WL",  monomer[:,0][where(monomer[:,1] == max(monomer[:,1]))[0][0]] , " nm ")

ax4.legend()


#%% FD : 1 : 1.6 ratio (NT : FD)

ax2.axhline ( y = 0 , color = 'k' , linewidth = 0.5)
files_fd = glob.glob("FDbeforeTEM\\1t1p6\\*.txt")

files_fd = files_fd[ 5 : 35 ]


for i in range (0,len(files_fd)) :

    file = open(files_fd[i])
    
    specified_lines = [17,3664 ]  #start and end line of data
    
    WL, data = convert(file,specified_lines)
    
    
    
   # print (i, sem(WL))
    
    if i == 0 :
        
        integrate = zeros(len(WL))
        integrate = data 
    
    else :
        
        integrate = integrate + data
    
    average = integrate/len(files_fd)


xini = [  0.008,594,1,  0.005,580,1,  0.005,570,0.5,    0.004,540,0.2,   0.002,485,5,  0.001,500,5]
xfinal = [ 0.6,601,20, 0.3,587,20,    0.3,581,20,    0.1,557,25,    0.5,525,25,  0.4,535,20 ]

param, co = curve_fit(main_fd, WL, data,  bounds = (xini,xfinal))
co = np.sqrt(np.diag(co))

print ("FD NTs ", round(param[2],2) , " ±", round(co[2],2))
print (param)
ax5.plot(WL,data, label = "data")
   
fit = main_fd(WL,*param)
 
fit1 = lorr(WL,param[0],param[1],param[2])
fit2 = lorr(WL,param[3],param[4],param[5])
fit3 = lorr(WL,param[6],param[7],param[8])
fit4 = lorr(WL,param[9],param[10],param[11])
fit5 = lorr(WL,param[12],param[13],param[14])
fit6 = lorr(WL,param[15],param[16],param[17])
#fit7 = lorr(WL,param[18],param[19],param[20])


ax5.plot(WL,fit/1,   label = "fit" , color = 'k')
ax5.plot(WL,fit1/1, linestyle = "--", label = "1")
ax5.plot(WL,fit2/1, linestyle = "--", label = "2")
ax5.plot(WL,fit3/1, linestyle = "--", label = "3")
ax5.plot(WL,fit4/1, linestyle = "--", label = "4")
ax5.plot(WL,fit5/1, linestyle = "--", label = "5")
ax5.plot(WL,fit6/1, linestyle = "--", label = "6")
ax5.set_xlabel ("Wavelength (nm)")
ax5.set_ylabel ("Absorption")
     
ax5.legend()

inner_peak_fd = inner_peak_WL(WL, average)


ax2.plot(10**4/WL, average/average[inner_peak_fd] , color = 'red',  label = "Flash-diluted NTs")
# /




wn_fd = 10**7/WL[inner_peak_fd]  #wavenumber peak value (cm-1)

print ("Flash diluted NTs, inner peak WL ",  WL[inner_peak_fd] , " nm, ", round(10**7/WL[inner_peak_fd],0) , " cm-1 and OD " , round(average[inner_peak_fd],2))

print ("Blue shift (cm-1) " , wn_fd - wn_dw)

#ax2.plot(10**4/WL - 0*( wn_fd - wn_dw)/1000, average/average[inner_peak_fd] , color = 'red',  label = "Flash-diluted nanotubes")

secax = ax2.secondary_xaxis('top',functions=(f,g))

#secax = ax2.secondary_xaxis('top', functions=(function,inverse))

secax.set_xlabel("Wavelength (nm)")


#%%

ax2.set_xlabel (r'Wavenumber (10 $^3$  cm$^{-1}$)', size = MEDIUM_SIZE)
#ax2.set_ylabel ("Absorption (arb. units)")

ax2.set_ylabel ("Absorbance (normalised)")

ax2.set_ylim(-0.05,1.6)
ax2.set_xlim(10**4/650, 10**4/480)

#ax2.set_title ("SS absorption - portable_setup (100ms & 10 scans)")
ax2.legend()

print ("Dilution factor of flash diluted NTs is 2.6")
#print ("Percent of DWNTs completely dissolved ", (data[inner_peak_dw] - average[inner_peak_fd])

ax2.xaxis.label.set_size(15) 
ax2.yaxis.label.set_size(15) 

secax.yaxis.label.set_size(15) 
