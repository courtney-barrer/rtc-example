#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:26:23 2024

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
sys.path.append('pyBaldr/' )
from pyBaldr import utilities as util
import baldr_simulation_functions as baldrSim
import data_structure_functions as config

# sys.path.append('/Users/bencb/Documents/rtc-example/simBaldr/' )
# dont import pyBaldr since dont have locally FLI / BMC 





# ========== PLOTTING STANDARDS 
def nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None):

    n = len(im_list)
    fs = fontsize
    fig = plt.figure(figsize=(5*n, 5))

    for a in range(n) :
        ax1 = fig.add_subplot(int(f'1{n}{a+1}'))
        ax1.set_title(title_list[a] ,fontsize=fs)

        if vlims!=None:
            im1 = ax1.imshow(  im_list[a] , vmin = vlims[a][0], vmax = vlims[a][1])
        else:
            im1 = ax1.imshow(  im_list[a] )
        ax1.set_title( title_list[a] ,fontsize=fs)
        ax1.set_xlabel( xlabel_list[a] ,fontsize=fs) 
        ax1.set_ylabel( ylabel_list[a] ,fontsize=fs) 
        ax1.tick_params( labelsize=fs ) 

        if axis_off:
            ax1.axis('off')
        divider = make_axes_locatable(ax1)
        if cbar_orientation == 'bottom':
            cax = divider.append_axes('bottom', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
                
        elif cbar_orientation == 'top':
            cax = divider.append_axes('top', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
                
        else: # we put it on the right 
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='vertical')  
        
   
        cbar.set_label( cbar_label_list[a], rotation=0,fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)
    if savefig!=None:
        plt.savefig( savefig , bbox_inches='tight', dpi=300) 

    plt.show() 
    
    
def AO_iteration( z, test_field , controller_label, plot=True): 

  
    
    control_basis =  np.array(z.control_variables[controller_label]['control_basis'])
    M2C = z.control_variables[controller_label ]['pokeAmp'] *  control_basis.T #.reshape(control_basis.shape[0],control_basis.shape[1]*control_basis.shape[2]).T
    I2M = np.array( z.control_variables[controller_label ]['I2M'] ).T  
    IM = np.array(z.control_variables[controller_label]['IM'] )
    I0 = np.array(z.control_variables[controller_label]['sig_on_ref'].signal )
    N0 = np.array(z.control_variables[controller_label]['sig_off_ref'].signal )
    
    
    i = z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    o = z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    
    sig = i.signal / np.sum( o.signal ) - I0 / np.sum( N0 )
    
    cmd = -1 * M2C @ (I2M @ sig.reshape(-1) ) 
    
    #plt.figure() 
    #plt.imshow ( cmd.reshape(12,12)); plt.colorbar()
    
    z.dm.update_shape( cmd - np.mean( cmd ) )
    
    post_dm_field = test_field.applyDM( z.dm )
    
    if plot:
        wvl_i = 0 
        if 'square' in z.dm.DM_model:
            im_list = [ 1e9 * z.wvls[0]/(2*np.pi) * test_field.phase[z.wvls[0]], sig, 1e9 * cmd.reshape(12,12), 1e9 * z.wvls[0]/(2*np.pi) * post_dm_field.phase[z.wvls[0]] ]
        elif z.dm.DM_model == 'BMC-multi3.5':
            im_list = [ 1e9 * z.wvls[0]/(2*np.pi) * test_field.phase[z.wvls[0]], sig, 1e9 * baldrSim.get_BMCmulti35_DM_command_in_2D( cmd ), 1e9 * z.wvls[0]/(2*np.pi) * post_dm_field.phase[z.wvls[0]] ]
        xlabel_list = ['' for _ in range(len(im_list))]
        ylabel_list = ['' for _ in range(len(im_list))]
        title_list = ['phase pre DM','detector signal', 'DM surface','phase post DM']
        cbar_label_list = ['OPD [nm]', 'intensity [adu]', 'OPD [nm]', 'phase [nm]']
        nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None)
    

    return( sig, cmd, post_dm_field )


#%%
#===================== SIMULATION 

# =========== Setup simulation using both a square DM and a BMC multi 3.5 DM using various control basis

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



# -------- trialling this 

# Cold stops have to be updated for both FPM and FPM_off!!!!!!!
zwfs.FPM.update_cold_stop_parameters( None )
zwfs.FPM_off.update_cold_stop_parameters( None )

zwfs_bmc.FPM.update_cold_stop_parameters( None )
zwfs_bmc.FPM_off.update_cold_stop_parameters( None )


#---------------------
# define an internal calibration source 
calibration_source_config_dict = config.init_calibration_source_config_dict(use_default_values = True)
calibration_source_config_dict['temperature']=1900 #K (Thorlabs SLS202L/M - Stabilized Tungsten Fiber-Coupled IR Light Source )
calibration_source_config_dict['calsource_pup_geometry'] = 'Disk'

nbasismodes = 20
basis_labels = [ 'zernike','fourier', 'KL']
control_labels = [f'control_{nbasismodes}_{b}_modes' for b in basis_labels]

for b,lab in zip( basis_labels, control_labels):
    # square DM  
    zwfs.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=nbasismodes, modal_basis=b, pokeAmp = 150e-9 , label=lab, replace_nan_with=0)
    # BMC multi3.5 DM 
    zwfs_bmc.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=nbasismodes, modal_basis=b, pokeAmp = 150e-9 , label=lab,replace_nan_with=0)




test_field = baldrSim.init_a_field( Hmag=0, mode='Kolmogorov', wvls=zwfs.wvls, \
                                   pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'],\
                                       dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'], \
                                           r0=0.1, L0 = 25, phase_scale_factor=1.3)

"""test_field = baldrSim.init_a_field( Hmag=-10, mode=10, wvls=zwfs.wvls, \
                                   pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'],\
                                       dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'] ,\
                                           phase_scale_factor=1)
"""


#########
### CHECKING THAT WE GET AN OPEN LOOP CORRECTION FOR EACH DM AND EACH BASIS CONSTRUCTION  
plt.imshow(baldrSim.get_BMCmulti35_DM_command_in_2D( zwfs_bmc.control_variables['control_20_fourier_modes']['M2C'][5] ))
# first check the basis 
for zz in [zwfs]: #, zwfs_bmc ]: 
    print(zz.dm.DM_model,'----\n')
    for lab in control_labels:
        print( '   ', lab )
        sig, cmd, post_dm_field = AO_iteration( z = zz , test_field = test_field, controller_label= lab, plot=True  )
        print( 'strehl before = ',np.exp( -np.nanvar( test_field.phase[zwfs.wvls[0]][zwfs.pup>0] ) ) )
        print( 'strehl after = ', np.exp( -np.nanvar( post_dm_field.phase[zwfs.wvls[0]][zwfs.pup>0] ) ) ) 


#

# NOW LETS SEE CLOSED LOOP ON A STATIC ABERRATION

z = copy.deepcopy( zwfs) 
lab = 'control_20_zernike_modes' # 'control_20_fourier_modes'
control_basis =  np.array(z.control_variables[lab ]['control_basis'])
M2C = z.control_variables[lab ]['pokeAmp'] *  control_basis.T #.reshape(control_basis.shape[0],control_basis.shape[1]*control_basis.shape[2]).T
I2M = np.array( z.control_variables[lab ]['I2M'] ).T  
IM = np.array(z.control_variables[lab ]['IM'] )
I0 = np.array(z.control_variables[lab ]['sig_on_ref'].signal )
N0 = np.array(z.control_variables[lab ]['sig_off_ref'].signal )


kp = 1 * np.ones( len(M2C.T) )
kp[0] = 0 # filter out 
ki = 0. * np.ones( len(M2C.T) )
kd = np.zeros( len(M2C.T) )
upper_limit = 100 *  np.ones( len(M2C.T) )
lower_limit = -100 * np.ones( len(M2C.T) )
setpoint = np.zeros( len(M2C.T) )

pid = baldrSim.PIDController(kp, ki, kd, upper_limit, lower_limit, setpoint)


test_field = baldrSim.init_a_field( Hmag=0, mode='Kolmogorov', wvls=zwfs.wvls, \
                                   pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'],\
                                       dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'], \
                                           r0=0.1, L0 = 25, phase_scale_factor=1.)

s_list = []
e_list = []
u_list = []
c_list = []
strehl_list = []
residual_list = []

wvl_ref = zwfs.wvls[0] # where to calculate Strehl ratio etc
pid.reset() 
for _ in range(40):
    
    strehl = np.exp( -np.nanvar( test_field.phase[wvl_ref][zwfs.pup>0] ) )
    strehl_list.append( strehl )
    
    i = z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    o = z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    
    sig = i.signal / np.sum( o.signal ) - I0 / np.sum( N0 )

    e = I2M @ sig.reshape(-1)
    
    #print( e )
    u = pid.process( I2M @ sig.reshape(-1) )
    
    c = 1 * M2C @ u 
    
    z.dm.update_shape( c - np.mean( c ) )
    
    test_field = test_field.applyDM( z.dm )
    
    
    
    s_list.append( sig )
    e_list.append( e )
    u_list.append( u )
    c_list.append( c )
    
    residual_list.append( test_field.phase[z.wvls[0]] )
    
    print( strehl )


plt.figure()
plt.plot( strehl_list )
plt.ylabel(f'Strehl Ratio at {round(1e6*wvl_ref,2)}um')
plt.xlabel('iterations')

 





#%% BUG FIXING 


## ============================
# fixing bug for DM normalization

test_field = baldrSim.init_a_field( Hmag=0, mode=0, wvls=zwfs.wvls, pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'], dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'])

# put a mode on the DM 
square_basis = baldrSim.create_control_basis(zwfs.dm, N_controlled_modes=20, basis_modes='zernike')
bmc_basis = baldrSim.create_control_basis(zwfs_bmc.dm, N_controlled_modes=20, basis_modes='zernike')

sig_on_list = []
sig_off_list = []
for zz, bb in zip([zwfs, zwfs_bmc], [square_basis, bmc_basis]):
    b = bb[5] #zwfs.control_variables[lab ]['control_basis'][5]
    b.reshape(1,-1)
    zz.dm.update_shape(  b * 450e-9   )# zwfs.control_variables[lab ]['pokeAmp'] * M2C.T[5] )

    plt.figure()
    if zz.dm.DM_model=='square_12' :
        plt.imshow( zz.dm.surface.reshape(12,12) )
    elif zz.dm.DM_model=='BMC-multi3.5':
        plt.imshow( baldrSim.get_BMCmulti35_DM_command_in_2D(zz.dm.surface ) )
    plt.title('DM surface')
    # now apply DM to field 

    post_dm_field = test_field.applyDM( zz.dm )
    fig,ax = plt.subplots(1,3)
    im0 = ax[0].imshow(zz.pup * test_field.phase[zz.wvls[0]])
    plt.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(zz.pup * post_dm_field.phase[zz.wvls[0]])
    plt.colorbar(im1, ax=ax[1])
    ax[2].imshow(zz.pup * post_dm_field.flux[zz.wvls[0]])
    ax[0].set_title('field phase before DM')
    ax[1].set_title('field phase after DM')
    ax[2].set_title('field flux')
    
    #output =  zz.detection_chain( test_field )
    sig_on = zz.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 ) #zz.detection_chain( test_field, zz.dm, zz.FPM, zz.det, replace_nan_with=0)
    sig_off =  zz.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 ) #zz.detection_chain( test_field, zz.dm, zz.FPM_off, zz.det, replace_nan_with=0) #replace_nan_with=None
    
    # to test it also works using base function
    #sig_on = baldrSim._detection_chain( test_field, zz.dm, zz.FPM, zz.det, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0)
    #sig_off = baldrSim._detection_chain( test_field, zz.dm, zz.FPM_off, zz.det,include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0) #replace_nan_with=None
    
    sig_on_list.append( sig_on )
    sig_off_list.append( sig_off )

    fig,ax = plt.subplots(1,2)
    if zz.dm.DM_model=='square_12' :
        ax[0].imshow(zz.dm.surface.reshape(12,12))
    elif zz.dm.DM_model=='BMC-multi3.5':
        ax[0].imshow( baldrSim.get_BMCmulti35_DM_command_in_2D(zz.dm.surface ) )
    ax[1].imshow(sig_on.signal )
    ax[0].set_title('DM surface')
    ax[1].set_title('ZWFS intensity')
plt.show() 


## ============================
# Now fix bug of why zwfs.FPM_off and zwfs.FPM are same 

fig,ax = plt.subplots( 2, 1 )
ax[0].set_title('post FPM field phase')
ax[0].imshow( sig_on.signal  )
ax[1].imshow( sig_off.signal  )
ax[0].set_ylabel('FPM on')
ax[1].set_ylabel('FPM off')


#line 1151 and 1470 have two definitions of detection_chain
zwfs = baldrSim.ZWFS(mode_dict_square)
zwfs.FPM.update_cold_stop_parameters(None)
zwfs.FPM_off.update_cold_stop_parameters(None)

test_field = baldrSim.init_a_field( Hmag=0, mode=0, wvls=zwfs.wvls, pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'], dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'])

zwfs.dm.update_shape(  square_basis[5] * 450e-9   )

post_dm_field = test_field.applyDM( zwfs.dm )

print('FPM on : d_on = d_off? ', zwfs.FPM.d_on == zwfs.FPM.d_off )
print('FPM off : d_on = d_off? ', zwfs.FPM_off.d_on == zwfs.FPM_off.d_off )

square_basis = baldrSim.create_control_basis(zwfs.dm, N_controlled_modes=20, basis_modes='zernike')

# fields after phase mask 




test_out_on = zwfs.FPM.get_output_field( post_dm_field , keep_intermediate_products=False, replace_nan_with= 0 )
test_out_off = zwfs.FPM_off.get_output_field( post_dm_field, keep_intermediate_products=False, replace_nan_with= 0  )

fig,ax = plt.subplots( 2, 1 )
ax[0].set_title('post FPM field phase')
ax[0].imshow( test_out_on.phase[zwfs.wvls[0]] )
ax[1].imshow( test_out_off.phase[zwfs.wvls[0]] )
ax[0].set_ylabel('FPM on')
ax[1].set_ylabel('FPM off')

fig,ax = plt.subplots( 2, 1 )
ax[0].set_title('post FPM field flux')
ax[0].imshow( test_out_on.flux[zwfs.wvls[0]] )
ax[1].imshow( test_out_off.flux[zwfs.wvls[0]] )
ax[0].set_ylabel('FPM on')
ax[1].set_ylabel('FPM off')

# detect fields 
test_out_on.define_pupil_grid(dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'], D_pix=zwfs.mode['telescope']['telescope_diameter_pixels']) 
test_out_off.define_pupil_grid(dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['telescope_diameter_pixels'], D_pix=zwfs.mode['telescope']['telescope_diameter_pixels']) 

test_inten_on = zwfs.det.detect_field(test_out_on, include_shotnoise=True, ph_per_s_per_m2_per_nm=True,grids_aligned=True)
test_inten_off = zwfs.det.detect_field(test_out_off, include_shotnoise=True, ph_per_s_per_m2_per_nm=True,grids_aligned=True)

fig,ax = plt.subplots( 3, 1 )
ax[0].set_title('post FPM field flux')
ax[0].imshow( test_inten_on.signal )
ax[1].imshow( test_inten_off.signal )
ax[2].imshow(test_inten_on.signal - test_inten_off.signal )
ax[0].set_ylabel('FPM on')
ax[1].set_ylabel('FPM off')
ax[2].set_ylabel('difference')



## ============================
#testing multi3.5 DM interpolation 
def _get_corner_indices(N):
    # util for BMC multi 3.5 DM which has missing corners 
    return [
        (0, 0),        # Top-left
        (0, N-1),      # Top-right
        (N-1, 0),      # Bottom-left
        (N-1, N-1)     # Bottom-right
    ]


x = np.linspace(-1,1,12)
y =  np.linspace(-1,1,12)
X, Y = np.meshgrid(x,y)  
coor = np.vstack([X.ravel(), Y.ravel()]).T
z = np.sin( X+Y )
nearest_interp_fn = scipy.interpolate.LinearNDInterpolator( coor, z.reshape(-1) , fill_value = np.nan)
# works fine.. Now remove corners
x_flat = X.flatten()
y_flat = Y.flatten()
corner_indices = _get_corner_indices( len(x) )
corner_indices_flat = [i * 12 + j for i, j in corner_indices]

X2 = np.delete( x_flat, corner_indices_flat)
Y2 = np.delete( y_flat, corner_indices_flat)
coor2 = np.vstack([X2.ravel(), Y2.ravel()]).T
z2 =  np.sin( X2+Y2 )
nearest_interp_fn2 = scipy.interpolate.LinearNDInterpolator( coor, z.reshape(-1) , fill_value = np.nan)

nearest_interp_fn2(X,Y)


fig,ax = plt.subplots(2,1,sharex=True)
for zz,axx  in zip( [zwfs, zwfs_bmc], ax.reshape(-1) ):
    x = np.linspace(-test_field.dx * (test_field.nx_size //2) , test_field.dx * (test_field.nx_size //2) , zz.dm.Nx_act)
    y =  np.linspace(-test_field.dx * (test_field.nx_size //2) , test_field.dx * (test_field.nx_size //2) , zz.dm.Nx_act)
    
    if 'square' in zz.dm.DM_model:
      # simple square DM, DM values defined at each point on square grid
      X, Y = np.meshgrid(x, y)  
    
    elif 'BMC-multi3.5' == zz.dm.DM_model:
      #this DM is square with missing corners so need to handle corners 
      # (since no DM value at corners we delete the associated coordinates here 
      # before interpolation )
      X,Y = np.meshgrid( x, y) 
      x_flat = X.flatten()
      y_flat = Y.flatten()
      corner_indices = _get_corner_indices(zz.dm.Nx_act)
      corner_indices_flat = [i * 12 + j for i, j in corner_indices]
      
      X = np.delete(x_flat, corner_indices_flat)
      Y = np.delete(y_flat, corner_indices_flat)
      
    
    else:
      raise TypeError('DM model unknown (check DM.DM_model) in applyDM method')
    
    coordinates = np.vstack([X.ravel(), Y.ravel()]).T
    
    
    # This runs everytime... We should only build these interpolators once..
    if zz.dm.surface_type == 'continuous' :
        # DM.surface is now 1D so reshape(1,-1)[0] not necessary! delkete and test
        nearest_interp_fn = scipy.interpolate.LinearNDInterpolator( coordinates, zz.dm.surface.reshape(-1) , fill_value = np.mean(zz.dm.surface)  )
    elif zz.dm.surface_type == 'segmented':
        nearest_interp_fn = scipy.interpolate.NearestNDInterpolator( coordinates, zz.dm.surface.reshape(-1) , fill_value = np.mean(zz.dm.surface) )
    else:
        raise TypeError('\nDM object does not have valid surface_type\nmake sure DM.surface_type = "continuous" or "segmented" ')
    
    
    
    dm_at_field_pt = nearest_interp_fn( test_field.coordinates ) # these x, y, points may need to be meshed...and flattened
    
    dm_at_field_pt = dm_at_field_pt.reshape( test_field.nx_size, test_field.nx_size )
      
      
    phase_shifts = {w:2*np.pi/w * (2*np.cos(zz.dm.angle)) * dm_at_field_pt for w in test_field.wvl} # (2*np.cos(DM.angle)) because DM is double passed
    
    field_despues = copy.copy(test_field)
    
    field_despues.phase = {w: field_despues.phase[w] + phase_shifts[w] for w in field_despues.wvl}
    
    
    #im = axx.imshow( field_despues.phase[zwfs.wvls[0]] )
    #plt.colorbar(im, ax= axx)
    axx.hist( zz.dm.surface , alpha =0.4 )
    
    #Xn, Yn = np.meshgrid(x, y) 
    #coordinates_new = np.vstack([Xn.ravel(), Yn.ravel()]).T
    # changed I2M and M2C in simulation - check multiplication.
