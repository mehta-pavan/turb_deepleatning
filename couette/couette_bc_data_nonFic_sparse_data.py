# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 03:58:22 2021

@author: loaner
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:59:02 2021

@author: loaner
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:00:44 2021

@author: loaner
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:36:44 2021

@author: Pavan Pranjivan Mehta

Distribution and use is not permitted with the knowledge of the author
"""


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.interpolate import make_interp_spline



"""
Control parameters and NN intitilaisaions

"""

Re_tau1 = [125, 180, 250, 550]
sparese = [0.02, 0.05, 0.1]

Re_tau1 = [180]

loss_tol = 1e-20

dummy_idx = 200


layers = [1]+[20]*3 +[2] #DNN layers

num_iter = 50000 #max DNN iteations 
print_skip = 1000 #printing NN outpluts after every "nth" iteration


AdaAF = True
AdaN = 5.
Ini_a = 1./AdaN

#-----------------------------------------------------------------------------------------------------------------


for sprse in sparese:
   
    
    dummy_idx += 2

    
    for Re_tau in Re_tau1:
        
        """
        
        Get DNS data and Spline fitting
        
        """
        
        
        str1 = 'DNS_data/Couette_Retau'+np.str(Re_tau)+'.dat'
        data = np.loadtxt(str1)
        
        y_h, y_plus, U_plus, uv_plus = data[:,0], data[:,1], data[:,2], data[:,9]
        
        
        new_Re_tau = y_plus[-1]/2
        
        #Curve fitting
        shape = U_plus.shape
        shape = shape[0]
        
        U = np.zeros((shape), dtype = "float64")
        uv = np.zeros((shape), dtype = "float64")
        #
        #for i in range(shape):
        #    U[i] = U_plus[shape-i-1]
        #    uv[i] = -uv_plus[shape-i-1]
        #
        #U_plus = np.hstack((U_plus[:-1], U)).reshape(-1)
        #uv_plus = np.hstack((uv_plus[:-1], uv)).reshape(-1)
        #
        #y_plus_max = np.max(y_plus)
        #y_plus = np.hstack((y_plus[:-1], 2*new_Re_tau-y_plus[::-1] )).reshape(-1)
        
        
        spl_U = make_interp_spline(y_plus, U_plus)
        spl_uv = make_interp_spline(y_plus, uv_plus)
        
        
        
        dUdy_plus = (U_plus[1:] - U_plus[:-1]) / (y_plus[1:] - y_plus[:-1])
        
        y_spasse_data = np.linspace(0, new_Re_tau, num=int(new_Re_tau*2*sprse), endpoint = True)
    
        yp_train = np.linspace(0.01, new_Re_tau-0.01, num=int(new_Re_tau*2), endpoint = True)
        yp_train = np.sort(np.unique(np.hstack((yp_train, y_spasse_data))))
    
        
        
        
        
        num_training_pts = yp_train.shape
        num_training_pts = num_training_pts[0]
        
        #-----------------------------------------------------------------------------------------------------------------
        
        
        """
        Neural Network
            1. Full connected architechure
            
        """
        
        
        def xavier_init(size): # weight intitailing 
            in_dim = size[0]
            out_dim = size[1]        
            xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
            #variable creatuion inn tensor flow - intilatisation
            return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,dtype=tf.float64,seed=1704), dtype=tf.float64)
        
            
        """A fully-connected NN"""
        def DNN(X, layers,weights,biases):
            L = len(layers)
            H = X 
            for l in range(0,L-2): # (X*w(X*w + b) + b)...b) Full conected neural network
                W = weights[l] 
                b = biases[l]
                #H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
                H = tf.tanh(AdaN*a*tf.add(tf.matmul(H, W), b) )# H - activation function? 
                #H = tf.tanh(a*tf.add(tf.matmul(H, W), b)) 
                #H = tf.nn.tan(tf.add(tf.matmul(H, W), b))
            #the loops are not in the same hirecachy as the loss functions
            W = weights[-1]
            b = biases[-1]
            Y = tf.add(tf.matmul(H, W), b) # Y - output - final yayer
            return Y
        
        
        
        if AdaAF:
            a = tf.Variable(Ini_a, dtype=tf.float64)
        else:
            a = tf.constant(Ini_a, dtype=tf.float64)
        
        
        L = len(layers)
        weights = [xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]   
        biases = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]
        
        
        
        #-----------------------------------------------------------------------------------------------------------------
        
        
        
        dnn_out = DNN((yp_train.reshape(-1,1)), layers, weights, biases) #fractional order - aplha
        
        U_train = dnn_out[:,0]
        nu_non_fic = dnn_out[:,1]
        
        rhs =   np.ones(num_training_pts)
        
        #yp_train = tf.stack(yp_train)
        
        #U_x_train = tf.gradients(U_train, yp_train)[0]
        U_x_train = (U_train[1:] - U_train[:-1])/ (yp_train[1:] - yp_train[:-1])
        
        loc = nu_non_fic[1:]*U_x_train
        uv_nf_strain = nu_non_fic[1:] + loc #((loc[1:] - loc[:-1]) / (yp_train[2:] - yp_train[:-2]))
        
        uv_Nf = -tf.abs(uv_nf_strain )
        
        eq1 = U_x_train - uv_Nf
        
        idx = []
        for yp in y_spasse_data[1:-1]:
            idx.append(np.where(yp_train == yp))
        idx = np.array(idx).reshape(1, -1)
        
        
        
        U_loss = tf.square(U_train[0] - U_plus[0])  + tf.square(U_train[-1] - spl_U(yp_train[-1]))
        uv_loss = tf.square(uv_Nf[0] - uv_plus[0])  + tf.square(uv_Nf[-1] - spl_uv(yp_train[-1]))
        
        U_loss =  tf.reduce_mean(tf.square( tf.gather(U_train, idx) - spl_U(y_spasse_data[1:-1]))) + U_loss
        uv_loss = tf.reduce_mean(tf.square(tf.gather(uv_Nf, idx) - spl_uv(y_spasse_data[1:-1]))) + uv_loss
        #uv_Nf_loss = tf.abs(uv_Nf[0])
        
        loss = 100*tf.reduce_mean(tf.abs(eq1 - rhs[1:])) + U_loss + uv_loss
        
        
        optimizer = tf.train.AdamOptimizer(learning_rate=1.0E-4).minimize(loss)
        
        loss_max = 1.0e16 
            
        lss = []
        
        
        os.mkdir('raw_results/Re_tau ='+np.str(Re_tau)+'_coeff-aux-pts_beta='+np.str(dummy_idx))
        
        
        with tf.Session() as sess:
                        
            sess.run(tf.global_variables_initializer())
                        
            for i in range(num_iter+1):
                sess.run(optimizer)
                loss_val = sess.run(loss)
                lss.append([i, loss_val])
                #if i % print_skip == 0:
                
               
                if loss_val > loss_tol:
                   
                    if i % print_skip == 0:
                        
                        U_val = np.array(sess.run(U_train))
                        uv_val = np.array(sess.run(uv_Nf))
                        loss_val = sess.run(loss)
                        print("loss = "+np.str(loss_val)+"; iter ="+np.str(i))
                                                        
                        fig= plt.figure()
                        plt.semilogx (yp_train.reshape((-1,1)), U_val.reshape(-1)/np.max(U_plus), 'r', label = "U_val")
                        plt.semilogx (yp_train.reshape((-1,1)), spl_U(yp_train)/np.max(U_plus) , 'k--', label = "U_dns")
                        
                        plt.semilogx (yp_train[1:].reshape((-1,1)), uv_val.reshape(-1), 'g', label = "uv_val")
                        plt.semilogx (yp_train.reshape((-1,1)), spl_uv(yp_train) , 'b--', label = "uv_dns")
                                       
                        plt.legend()
                        plt.xlabel("y+")
                        plt.ylabel("U(y+)/uv(y+)")
                        #plt.title('Couette Flow Re_tau ='+np.str(Re_tau)+'_coeff-aux-pts_beta='+np.str(dummy_idx))
                        plt.savefig('raw_results/Re_tau ='+np.str(Re_tau)+'_coeff-aux-pts_beta='+np.str(dummy_idx)+'/'+np.str(Re_tau)+'_coeff-aux-pts='+np.str(dummy_idx)+'.png', dpi=100)
                        #plt.show()
                        #plt.close(fig)
        
                    else:
                        pass
                else:
                    
                    pass
                
                continue
            
            
                
                
    
        data = np.stack((yp_train[1:].reshape(-1), U_val[1:].reshape(-1), uv_val.reshape(-1)), axis=1)
        np.savetxt('raw_results/Re_tau ='+np.str(Re_tau)+'_coeff-aux-pts_beta='+np.str(dummy_idx)+'/Channel_Re_tau ='+np.str(Re_tau)+'_coeff-aux-pts='+np.str(dummy_idx)+'_alpha_.txt', data)
                          
            
