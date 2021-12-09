# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 02:01:02 2021

@author: loaner
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 01:43:38 2021

@author: loaner
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 00:35:02 2021

@author: loaner
"""



import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.interpolate import make_interp_spline


Re_tau1 = [180, 550, 1000, 2000]
sparese = [0.02, 0.05, 0.1]

dummy_idx1 = 70
dummy_idx2 = 50

path = "raw_results/"

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

err_U_sps = []
err_uv_sps = []

#for sprse in sparese:
for Re_tau in Re_tau1:
    
    if Re_tau == 550:
         #dummy_idx2 -= 2
         dummy_idx1 -= 2
    
    
    dummy_idx2 += 2
    dummy_idx1 += 2
    
    """
    
    Get DNS data and Spline fitting
    
    """

    
    if Re_tau == 180:
          
        U_tau =  0.57231059E-01 
        nu = 1/0.32500000E+04
    
    
        #rho = 0.834  #rho_150C
        
    
        data = np.loadtxt('DNS_data_channel/ReTau='+np.str(Re_tau)+'.txt')
        
        #half channel data
    
        y_plus, U_plus, uv_plus, uu_plus, vv_plus = data[:,1], data[:,2], data[:,10], data[:,3], data[:,4]
        
        
        
    elif Re_tau == 550:
        
        U_tau = 0.48904658E-01  
        nu = 1/0.11180000E+05
     
        data = np.loadtxt('DNS_data_channel/ReTau='+np.str(Re_tau)+'.txt')
        
        #half channel data
    
        y_plus, U_plus, uv_plus, uu_plus, vv_plus = data[:,1], data[:,2], data[:,10], data[:,3], data[:,4]
        
        
        
    elif Re_tau == 950:
        
        Re_tau = 950
        U_tau = 0.45390026E-01 
        nu = 1/0.20580000E+05 
    
       
        
        data = np.loadtxt('DNS_data_channel/ReTau='+np.str(Re_tau)+'.txt')
        
        #half channel data
    
        y_plus, U_plus, uv_plus, uu_plus, vv_plus = data[:,1], data[:,2], data[:,10], data[:,3], data[:,4]
    
    
        
        
        
    elif Re_tau == 1000:
        
        U_tau = 0.0499
        nu = 5E-5
        dPdx = 0.0025
    
    
        #import data
        data = np.loadtxt('DNS_data_channel/ReTau='+np.str(Re_tau)+'.txt')
        
        
        #half channel data
        
        y_plus, U_plus, uv_plus, uu_plus, vv_plus = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
        
    
    
        
        
    elif Re_tau == 2000:
        
        U_tau = 0.41302030E-01 
        nu = 1/0.48500000E+05
    
    
        #rho = (1.026 + 0.994) / 2    #rho_160F + rho_180F / 2
    
    
        data = np.loadtxt('DNS_data_channel/ReTau='+np.str(Re_tau)+'.txt')
            
        #half channel data
    
        y_plus, U_plus, uv_plus, uu_plus, vv_plus = data[:,1], data[:,2], data[:,10], data[:,3], data[:,4]
    
        
    
    
        
    elif Re_tau == 5200:
    
        U_tau =  4.14872e-02 
        nu = 8.00000e-06
    
    
        #import data
        data = np.loadtxt('DNS_data_channel/ReTau='+np.str(Re_tau)+'.txt')
    
        #half channel data
    
        y_plus, U_plus = data[:,1], data[:,2]
        
        
    else:
        
        raise "Valid Re_tau = 180, 550, 950, 1000, 2000, 5200"
    
    
    
    new_Re_tau = y_plus[-1]
    dPdx_plus = -1/ new_Re_tau
    
    spl_U = make_interp_spline(y_plus, U_plus)
    spl_uv = make_interp_spline(y_plus, uv_plus)
    
    
    plt.semilogx (y_plus, U_plus/np.max(U_plus) , 'k--', label = r"$U_{dns}$")
    plt.semilogx (y_plus.reshape((-1,1)), uv_plus , 'b--', label = r"$uv_{dns}$")

    #for Re_tau in Re_tau1:
    #for sprse in sparese:
#
#    dummy_idx2 += 2
#    dummy_idx1 += 2

    
    
    data_sparse = np.loadtxt(path+'Re_tau ='+np.str(Re_tau)+'_coeff-aux-pts_beta='+np.str(dummy_idx1)+'/'+np.str(Re_tau)+'_coeff-aux-pts='+np.str(dummy_idx1)+'_alpha_.txt')
    
    yp_sps_nf, U_sps_nf, uv_sps_nf = data_sparse[:,0], data_sparse[:, 1], data_sparse[:,2]
    
    
    data_sparse = np.loadtxt(path+'Re_tau ='+np.str(Re_tau)+'_coeff-aux-pts_beta='+np.str(dummy_idx2)+'/'+np.str(Re_tau)+'_coeff-aux-pts='+np.str(dummy_idx2)+'_alpha_.txt')
    
    yp_sps_nom, U_sps_nom, uv_sps_nom = data_sparse[:,0], data_sparse[:, 1], data_sparse[:,2]
    
    
    
    plt.semilogx (yp_sps_nf.reshape((-1,1)), U_sps_nf.reshape(-1)/np.max(U_plus), label = r"NF: $U_{nn}$")
    plt.semilogx (yp_sps_nf.reshape((-1,1)), uv_sps_nf.reshape(-1), label = r"NF: $uv_{nn}$")
                 
    
    plt.semilogx (yp_sps_nom.reshape((-1,1)), U_sps_nom.reshape(-1)/np.max(U_plus), label = r"NoModel: $U_{nn}$")
    plt.semilogx (yp_sps_nom.reshape((-1,1)), uv_sps_nom.reshape(-1), label = r"NoModel: $uv_{nn}$")
                 
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel(r"$y^+$")
    plt.ylabel("values")
    plt.title(r"Channel : Only Bondary Data, $Re_{\tau}$ = "+np.str(Re_tau))

    plt.tight_layout()
    plt.savefig('pics/bc_channe_Re_'+np.str(Re_tau)+'.png', dpi=300)
    
    plt.show()
    #plt.close(fig)


