
Solving ill-posed problem for trubulent flows

#-----------------------------------------------------------------
Author : Pavan Pranjivan Mehta
Email : pavan_pranjivan_mehta@brown.edu / mehtapavanp@gmail.com
Licenced by Brown University

December 2021
#-----------------------------------------------------------------


#-----------------------------------------------------------------
Case: Channel Flow
#-----------------------------------------------------------------

Simplified RANS Equation : grad_y^+(U^+) - uv^+ = 1 + grad_x^+(P^+)*y^+


#-----------------------------------------------------------------
Files
#-----------------------------------------------------------------
1. channel_bench_mark.py -- well posed problem computed U^+, uv^+ suppied
2. channel_bc_data_sparse.py  -- computes for sparse data
3. channel_bc_data_nonFic_sparse_data.py -- computes for sparse data
4. channel_bc_data_nonFic.py  - only boundary conditions applied
5. channel_bc_data.py - only boundary conditions applied



#-----------------------------------------------------------------
Non-Fickian Diffusion
#-----------------------------------------------------------------
1. channel_bc_data_nonFic_sparse_data.py -- computes for sparse data
2. channel_bc_data_nonFic.py -- only boundary conditions applied

Non-Ficakian Law : uv^+ = -|nu_nf(y^+) +  nu_nf(y^+)*grad_y^+(U^+)|

where, nu_nf is the non-Fickian diffusivity 
as a function of wall normal distance (y^+)


#-----------------------------------------------------------------
No Model 
#-----------------------------------------------------------------
1. channel_bc_data_sparse.py  -- computes for sparse data
2. channel_bc_data.py - only boundary conditions applied

Imposes a non-positive constarint on uv^+


#-----------------------------------------------------------------
References
#-----------------------------------------------------------------

1. Eivazi, H., Tahani, M., Schlatter, P. and Vinuesa, R., 2021. Physics-informed neural networks for solving Reynolds-averaged Navier $\unicode {x2013} $ Stokes equations. arXiv preprint arXiv:2107.10711.
2. Mehta, P.P., Pang, G., Song, F. and Karniadakis, G.E., 2019. Discovering a universal variable-order fractional model for turbulent Couette flow using a physics-informed neural network. Fractional Calculus and Applied Analysis, 22(6), pp.1675-1688.
3. Mehta, P.P., 2021. Fractional models of Reynolds-averaged Navier-Stokes equations for Turbulent flows. arXiv preprint arXiv:2105.03646.



