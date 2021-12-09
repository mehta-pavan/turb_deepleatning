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


Re_tau1 = [125, 180, 250, 550]
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

    
    str1 = 'DNS_data/Couette_Retau'+np.str(Re_tau)+'.dat'
    data = np.loadtxt(str1)
    
    y_h, y_plus, U_plus, uv_plus = data[:,0], data[:,1], data[:,2], data[:,9]
    
    
    new_Re_tau = y_plus[-1]/2
    
    spl_U = make_interp_spline(y_plus, U_plus)
    spl_uv = make_interp_spline(y_plus, uv_plus)
    idx = np.where(y_plus <new_Re_tau+0.01 )
    
    plt.semilogx (y_plus[idx], U_plus[idx]*2/np.max(U_plus) , 'k--', label = r"$U_{dns}$")
    plt.semilogx (y_plus[idx].reshape((-1,1)), uv_plus[idx] , 'b--', label = r"$uv_{dns}$")

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
    plt.title(r"Couette : Only Bondary Data, $Re_{\tau}$ = "+np.str(Re_tau))

    plt.tight_layout()
    plt.savefig('pics/bc_cou_Re_'+np.str(Re_tau)+'.png', dpi=300)
    
    plt.show()
    #plt.close(fig)


