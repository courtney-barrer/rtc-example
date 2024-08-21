#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 21:23:29 2024

@author: bencb
"""

import numpy as np
import glob 
import copy 
from astropy.io import fits
import scipy 
import os 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.append('simBaldr/' )
#sys.path.append('pyBaldr/' )
#from pyBaldr import utilities as util
import baldr_simulation_functions as baldrSim
import data_structure_functions as config


def project_matrix( CM , projection_vector_list ):
    """
    create two new matrices CM_TT, and CM_HO from CM, 
    where CM_TT projects any "Img" onto the column space of vectors in 
    projection_vector_list  vectors, 
    and CM_HO which projects any "Img" to the null space of CM_TT that is within CM.
    
    Typical use is to seperate control matrix to tip/tilt reconstructor (CM_TT) and
    higher order reconstructor (CM_HO)

    Note vectors in projection_vector_list  must be 
    
    Parameters
    ----------
    CM : TYPE matrix
        DESCRIPTION. 
    projection_vector_list : list of vectors that are in col space of CM
        DESCRIPTION.

    Returns
    -------
    CM_TT , CM_HO

    """

    # Step 1: Create the matrix T from projection_vector_list (e.g. tip and tilt vectors )
    projection_vector_list
    T = np.column_stack( projection_vector_list )  # T is Mx2
    
    # Step 2: Calculate the projection matrix P
    #P = T @ np.linalg.inv(T.T @ T) @ T.T  # P is MxM <- also works like this 
    # Step 2: Compute SVD of T
    U, S, Vt = np.linalg.svd(T, full_matrices=False)
    
    # Step 2.1: Compute the projection matrix P using SVD
    P = U @ U.T  # U @ U.T gives the projection matrix onto the column space of T
    
    # Step 3: Compute CM_TT (projection onto tip and tilt space)
    CM_TT = P @ CM  # CM_TT is MxN
    
    # Step 4: Compute the null space projection matrix and CM_HO
    I = np.eye(T.shape[0])  # Identity matrix of size MxM
    CM_HO = (I - P) @ CM  # CM_HO is MxN

    return( CM_TT , CM_HO )
    
#%%
#
# setting up the hardware and software modes of our ZWFS
tel_config =  config.init_telescope_config_dict(use_default_values = True)
phasemask_config = config.init_phasemask_config_dict(use_default_values = True) 

# -------- trialling this 
phasemask_config['on-axis phasemask depth'] = 4.210526315789474e-05
phasemask_config['off-axis phasemask depth'] = 4.122526315789484e-05

phasemask_config['phasemask_diameter'] = 1.5 * (phasemask_config['fratio'] * 1.65e-6)

#---------------------

DM_config_square = config.init_DM_config_dict(use_default_values = True) 
DM_config_bmc = config.init_DM_config_dict(use_default_values = True) 
DM_config_bmc['DM_model'] = 'BMC-multi3.5' #'square_12'
DM_config_square['DM_model'] = 'square_12'#'square_12'

# default is 'BMC-multi3.5'
detector_config = config.init_detector_config_dict(use_default_values = True)


# the only thing we need to be compatible is the pupil geometry and Npix, Dpix 
tel_config['pup_geometry'] = 'disk'

# define a hardware mode for the ZWFS 
mode_dict_bmc = config.create_mode_config_dict( tel_config, phasemask_config, DM_config_bmc, detector_config)
mode_dict_square = config.create_mode_config_dict( tel_config, phasemask_config, DM_config_square, detector_config)

#create our zwfs object with square DM
zwfs = baldrSim.ZWFS(mode_dict_square)
# now create another one with the BMC DM 
zwfs_bmc = baldrSim.ZWFS(mode_dict_bmc)


# Cold stops have to be updated for both FPM and FPM_off!!!!!!!
zwfs.FPM.update_cold_stop_parameters( None )
zwfs.FPM_off.update_cold_stop_parameters( None )

zwfs_bmc.FPM.update_cold_stop_parameters( None )
zwfs_bmc.FPM_off.update_cold_stop_parameters( None )

# define an internal calibration source 
calibration_source_config_dict = config.init_calibration_source_config_dict(use_default_values = True)
calibration_source_config_dict['temperature']=1900 #K (Thorlabs SLS202L/M - Stabilized Tungsten Fiber-Coupled IR Light Source )
calibration_source_config_dict['calsource_pup_geometry'] = 'Disk'


# Build control matricies  (zwfs or zwfs_bmc)
lab = 'test_zernike'
z = copy.deepcopy( zwfs )
z.dm.update_shape( np.zeros( z.dm.N_act ) )
z.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=20, \
                                  modal_basis='zernike', pokeAmp = 150e-9 , label=lab, replace_nan_with=0, without_piston=True)

#%% TESTING TIP/TILT PROJECTION 

pokeamp = z.control_variables[lab ]['pokeAmp']
M2C =  pokeamp* np.array( z.control_variables[lab ]['control_basis']).T #.reshape(control_basis.shape[0],control_basis.shape[1]*control_basis.shape[2]).T
I2M =  np.array( z.control_variables[lab ]['I2M'] ).T
IM = np.array(z.control_variables[lab ]['IM'] )
I0 = np.array(z.control_variables[lab ]['sig_on_ref'].signal )
N0 = np.array(z.control_variables[lab ]['sig_off_ref'].signal )
control_basis = np.array(z.control_variables[lab ]['control_basis'] )
#CM = M2C @ I2M

#1. create non-aberrated field field 
test_field = baldrSim.init_a_field( Hmag=-3, mode=0, wvls=zwfs.wvls, \
                                   pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'],\
                                       dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'])

    
"""# initial test  (make sure we are reconstructing correctly )
mode_idx = 1
cmd =  pokeamp * control_basis[mode_idx] #a_tt * M2C[:,i_tt] + a_ho1 * M2C[:,i_ho1] #note that M2C was scaled by poke amplitude - so coefficients are relative to this
#plt.figure(); plt.imshow( cmd.reshape(12,12) ); plt.colorbar(); plt.savefig( 'data/delme.png')
z.dm.update_shape( cmd )

i = z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
o = z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )

sig = i.signal / np.sum( o.signal) - I0 / np.sum( N0 )
#plt.figure(); plt.imshow( cmd.reshape(12,12)  ); plt.colorbar(); plt.savefig( 'data/delme.png')
#plt.figure(); plt.imshow( sig ); plt.colorbar();# plt.savefig( 'data/delme.png')
#plt.hist( sig.reshape(-1) , label='sig', bins=20,alpha=0.5); plt.hist( IM[mode_idx] , label='IM', bins=20, alpha=0.5); plt.legend() 
mode_reco = I2M @ sig.reshape(-1)
print(mode_reco)"""



# if we consider modal basis tip = [1,0,0,....0], tilt = [0,1,0,...,0]
# if we consider DM space then we have the actual commands
# tests modal space first 
tip = np.zeros(len(M2C.T) )  # Replace with your tip vector (size M)
tip[0] = 1
tilt = np.zeros(len(M2C.T) ) # Replace with your tilt vector (size M)
tilt[1] = 1
#other = np.zeros(len(M2C.T) ) # Replace with your tilt vector (size M)
#other[8] = 1
CM = I2M #np.array([...])  # Replace with your CM matrix (size MxN)

projection_vector_list = [tip, tilt] #, other] 
CM_TT, CM_HO = project_matrix( CM , projection_vector_list )
"""# Step 1: Create the matrix T from tip and tilt vectors
T = np.column_stack((tip, tilt))  # T is Mx2

# Step 2: Calculate the projection matrix P
#P = T @ np.linalg.inv(T.T @ T) @ T.T  # P is MxM

U, S, Vt = np.linalg.svd(T, full_matrices=False)

# Step 3: Compute the projection matrix P using SVD
P = U @ U.T  # U @ U.T gives the projection matrix onto the column space of T


# Step 3: Compute CM_TT (projection onto tip and tilt space)
CM_TT = P @ CM  # CM_TT is MxN

# Step 4: Compute the null space projection matrix and CM_HO
I = np.eye(T.shape[0])  # Identity matrix of size MxM
CM_HO = (I - P) @ CM  # CM_HO is MxN

# note this ends up giving same result as CM_HO = CM - (( CM.T @ T ) @ np.linalg.pinv(T )).T 
# which is the expression used in 


# CM_TT and CM_HO are the desired matrices
print("CM_TT:\n", CM_TT)
print("CM_HO:\n", CM_HO)
  
# Now to test create put a tip mode + some HO modes on DM, get image and see if we can reconstruct them with 
# CM_TT and CM_HO
"""

a_tt = 1
i_tt = 1

a_ho1 = [0.5, 0.1, 1.2]
i_ho1 = [5,8,2]

app_mode = np.zeros( len(M2C.T) ) 
app_mode[i_tt] = a_tt
for a,i in zip( a_ho1, i_ho1):
    app_mode[i] = a

cmd =  a_tt * M2C[:,i_tt]
for a,i in zip( a_ho1, i_ho1):
    cmd += a * M2C[:,i] #note that M2C was scaled by poke amplitude - so coefficients are relative to this


#plt.figure(); plt.imshow( cmd.reshape(12,12) ); plt.colorbar(); # plt.savefig( 'data/delme.png')

z.dm.update_shape( cmd )

i = z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
o = z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )

sig = i.signal / np.sum( o.signal) - I0 / np.sum( N0 )
#plt.figure(); plt.imshow( cmd.reshape(12,12)  ); plt.colorbar(); plt.savefig( 'data/delme.png')
plt.figure(); plt.imshow( sig ); plt.colorbar(); #plt.savefig( 'data/delme.png')

# reconstruct mode amplitudes 
mode_reco_HO = CM_HO @ sig.reshape(-1) 
mode_reco_TT = CM_TT @ sig.reshape(-1)  
mode_reco_full = CM @ sig.reshape(-1) 



plt.figure(figsize=(8,5))


plt.plot( mode_reco_HO, 'x', label='HO reconstruction',markersize=12)
plt.plot( mode_reco_TT, 'x',label='TT reconstruction',markersize=12)
#plt.plot( mode_reco_full, 'x', label='full reconstruction')
plt.plot( app_mode, label='input mode amplitudes',color='k',ls=':' )
plt.xlabel('mode index',fontsize=15)
plt.ylabel('mode amplitude',fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.legend(fontsize=15)
#plt.savefig( 'data/delme.png',dpi=300, bbox_inches='tight')







#%% Ok now test in DM command space 

#Try with Fourier basis 
#fbasis =  baldrSim.create_control_basis(z.dm, N_controlled_modes=20, basis_modes='fourier')
#tip = fbasis[0]  # Replace with your tip vector (size M)
#tilt =  fbasis[3] # Replace with your tilt vector (size M)

# if we consider modal basis tip = [1,0,0,....0], tilt = [0,1,0,...,0]
# if we consider DM space then we have the actual commands
# tests modal space first 
tip = control_basis[0]  # Replace with your tip vector (size M)
tilt =  control_basis[1] # Replace with your tilt vector (size M)


CM = M2C @ I2M #np.array([...])  # Replace with your CM matrix (size MxN)

# Step 1: Create the matrix T from tip and tilt vectors
T = np.column_stack((tip, tilt))  # T is Mx2

# Step 2: Calculate the projection matrix P
#P = T @ np.linalg.inv(T.T @ T) @ T.T  # P is MxM <- also works like this 
# Step 2: Compute SVD of T
U, S, Vt = np.linalg.svd(T, full_matrices=False)

# Step 2.1: Compute the projection matrix P using SVD
P = U @ U.T  # U @ U.T gives the projection matrix onto the column space of T


# Step 3: Compute CM_TT (projection onto tip and tilt space)
CM_TT = P @ CM  # CM_TT is MxN

# Step 4: Compute the null space projection matrix and CM_HO
I = np.eye(T.shape[0])  # Identity matrix of size MxM
CM_HO = (I - P) @ CM  # CM_HO is MxN

# note this ends up giving same result as CM_HO = CM - (( CM.T @ T ) @ np.linalg.pinv(T )).T 
# which is the expression used in 


# CM_TT and CM_HO are the desired matrices
print("CM_TT:\n", CM_TT)
print("CM_HO:\n", CM_HO)
  
# Now to test create put a tip mode + some HO modes on DM, get image and see if we can reconstruct them with 
# CM_TT and CM_HO



a_tt = 1
i_tt = 1

a_ho1 = [0.5, 0.1, 1.2]
i_ho1 = [5,8,2]

cmd =  a_tt * M2C[:,i_tt]
for a,i in zip( a_ho1, i_ho1):
    cmd += a * M2C[:,i] #note that M2C was scaled by poke amplitude - so coefficients are relative to this

# look at higher order and tip/tilt components seperately 
cmd_HO =  np.zeros( len(cmd) )
for a,i in zip( a_ho1, i_ho1):
    cmd_HO += a * M2C[:,i] #note that M2C was scaled by poke amplitude - so coefficients are relative to this

cmd_TT = a_tt * M2C[:,i_tt]

#plt.figure(); plt.imshow( cmd.reshape(12,12) ); plt.colorbar(); # plt.savefig( 'data/delme.png')

z.dm.update_shape( cmd )

i = z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
o = z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )

sig = i.signal / np.sum( o.signal) - I0 / np.sum( N0 )
#plt.figure(); plt.imshow( cmd.reshape(12,12)  ); plt.colorbar(); plt.savefig( 'data/delme.png')
plt.figure(); plt.imshow( sig ); plt.colorbar(); #plt.savefig( 'data/delme.png')

# reconstruct mode amplitudes 
cmd_reco_HO = CM_HO @ sig.reshape(-1) 
cmd_reco_TT = CM_TT @ sig.reshape(-1)  
cmd_reco_full = CM @ sig.reshape(-1) 


fig,ax = plt.subplots( 1,3 )
im0= ax[0].imshow( cmd_reco_TT.reshape(12,12) )
im1=ax[1].imshow( cmd_TT.reshape(12,12) )
im2=ax[2].imshow( cmd_TT.reshape(12,12) - cmd_reco_TT.reshape(12,12) )
plt.colorbar(im0, ax=ax[0])
plt.colorbar(im1, ax=ax[1])
plt.colorbar(im2, ax=ax[2])
ax[0].set_title('TT reconstructed')
ax[1].set_title('TT applied')
ax[2].set_title('residual')


fig,ax = plt.subplots( 1,3 )
im0=ax[0].imshow( cmd_reco_HO.reshape(12,12) )
im1=ax[1].imshow( cmd_HO.reshape(12,12) )
im2=ax[2].imshow( cmd_HO.reshape(12,12) - cmd_reco_HO.reshape(12,12) )
plt.colorbar(im0, ax=ax[0])
plt.colorbar(im1, ax=ax[1])
plt.colorbar(im2, ax=ax[2])
ax[0].set_title('HO reconstructed')
ax[1].set_title('HO applied')
ax[2].set_title('residual')
                










