#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:24:26 2024

@author: bencb

simulating P2C sinusoidal fits 

"""


import numpy as np
import glob 
import datetime
import copy 
from astropy.io import fits
import scipy
from scipy.optimize import curve_fit
import corner
import os 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.append('simBaldr/' )
sys.path.append('pyBaldr/' )
#from pyBaldr import utilities as util
import baldr_simulation_functions as baldrSim
import data_structure_functions as config

#sys.path.append('/Users/bencb/Documents/rtc-example/simBaldr/' )
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


def estimate_peak_frequency_and_phase(x, y):
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Compute the sampling frequency
    sampling_interval = np.mean(np.diff(x))  # Time between samples
    sampling_frequency = 1.0 / sampling_interval  # Samples per second (Hz)
    
    # Perform the Fast Fourier Transform (FFT)
    Y = np.fft.fft(y)
    
    # Compute the corresponding frequency bins
    freqs = np.fft.fftfreq(len(y), d=sampling_interval)
    
    # Compute the magnitude of the FFT (ignoring complex phase information)
    magnitude = np.abs(Y)
    
    # Ignore the zero frequency (DC component)
    zero_freq_index = np.argmin(np.abs(freqs))  # Find the index for zero frequency
    magnitude[zero_freq_index] = 0  # Set the magnitude at zero frequency to zero
    
    # Find the peak frequency (excluding zero frequency)
    peak_index = np.argmax(magnitude)
    peak_frequency = np.abs(freqs[peak_index])
    
    # Get the phase at the peak frequency
    peak_phase = np.angle(Y[peak_index])  # Phase at the peak frequency
    
    return peak_frequency, peak_phase


#%%
#===================== SIMULATION SETUP

# =========== Setup simulation using both a square DM and a BMC multi 3.5 DM using various control basis

# setting up the hardware and software modes of our ZWFS
tel_config =  config.init_telescope_config_dict(use_default_values = True)
phasemask_config = config.init_phasemask_config_dict(use_default_values = True) 

# -------- trialling this 
phasemask_config['on-axis phasemask depth'] = 4.210526315789474e-05
phasemask_config['off-axis phasemask depth'] = 4.122526315789484e-05

phasemask_config['phasemask_diameter'] = 1.3 * (phasemask_config['fratio'] * 1.65e-6)

# trying to understand I0 measured in sydney 
phasemask_config['on-axis_transparency'] = 1

#---------------------

#DM_config_square = config.init_DM_config_dict(use_default_values = True) 
DM_config_bmc = config.init_DM_config_dict(use_default_values = True) 
DM_config_bmc['DM_model'] = 'BMC-multi3.5' #'square_12'
#DM_config_square['DM_model'] = 'square_12'#'square_12'

# default is 'BMC-multi3.5'
detector_config = config.init_detector_config_dict(use_default_values = True)


# the only thing we need to be compatible is the pupil geometry and Npix, Dpix 
tel_config['pup_geometry'] = 'disk'

# define a hardware mode for the ZWFS 
mode_dict_bmc = config.create_mode_config_dict( tel_config, phasemask_config, DM_config_bmc, detector_config)
#mode_dict_square = config.create_mode_config_dict( tel_config, phasemask_config, DM_config_square, detector_config)

#create our zwfs object with square DM
# zwfs = baldrSim.ZWFS(mode_dict_square)
# now create another one with the BMC DM 
zwfs_bmc = baldrSim.ZWFS(mode_dict_bmc)



# -------- trialling this 

"""# Cold stops have to be updated for both FPM and FPM_off!!!!!!!
zwfs.FPM.update_cold_stop_parameters( None )
zwfs.FPM_off.update_cold_stop_parameters( None )

zwfs_bmc.FPM.update_cold_stop_parameters( None )
zwfs_bmc.FPM_off.update_cold_stop_parameters( None )
"""

#---------------------
# define an internal calibration source 
calibration_source_config_dict = config.init_calibration_source_config_dict(use_default_values = True)
calibration_source_config_dict['temperature']=1900 #K (Thorlabs SLS202L/M - Stabilized Tungsten Fiber-Coupled IR Light Source )
calibration_source_config_dict['calsource_pup_geometry'] = 'Disk'

nbasismodes = 140
basis_labels = [ 'zonal'] #, 'KL']
control_labels = [f'control_zonal' for b in basis_labels]

for b,lab in zip( basis_labels, control_labels):
    
    # BMC multi3.5 DM 
    zwfs_bmc.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=nbasismodes, \
                                      modal_basis=b, pokeAmp = 150e-9 , label=lab,replace_nan_with=0, without_piston=True)




#%% set up reference filed and write fits exactly how I did it in pyBaldr

zwfs = copy.deepcopy( zwfs_bmc )


number_amp_samples = 20 
amp_max = 0.7e-6 #0.2
number_images_recorded_per_cmd = 3
source_selector = None
take_mean_of_images = True
save_fits =  '/Users/bencb/Documents/rtc-example/data/simulation_test.fits' #None
#def GET_BDR_RECON_DATA_INTERNAL(zwfs,  number_amp_samples = 18, amp_max = 0.2, number_images_recorded_per_cmd = 10, source_selector = None,save_fits = None) :
    
#zwfs.dm_shapes['flat_dm'] = np.zeros( len( zwfs.dm.surface ))

#zwfs.dm_shapes['waffle'] =  pd.read_csv( '/Users/bencb/Documents/rtc-example/DMShapes/waffle.csv', header=None)[0].values
 


test_field = baldrSim.init_a_field( Hmag=-3, mode=0, wvls=zwfs.wvls, \
                                   pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'],\
                                       dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'])


off_field = baldrSim.init_a_field( Hmag=20, mode=0, wvls=zwfs.wvls, \
                                   pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'],\
                                       dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'])

    
if 1:
    """
    source_selector is motor to move light source for bias frame, if None we ask to manually move it
    """
    flat_dm_cmd = np.zeros( 140 ) #zwfs.dm_shapes['flat_dm']

    modal_basis = np.eye( len( flat_dm_cmd ) )
    
    ramp_values = np.linspace(-amp_max, amp_max, number_amp_samples)

    # ======== reference image with FPM OUT

    zwfs.dm.update_shape( flat_dm_cmd )
    #_ = input('MANUALLY MOVE PHASE MASK OUT OF BEAM, PRESS ENTER TO BEGIN' )
    #watch_camera(zwfs, frames_to_watch = 70, time_between_frames=0.05)
    
    N0_list = [] 
    for _ in range( number_images_recorded_per_cmd ):
        img_tmp = zwfs.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
        N0_list.append( img_tmp.signal ) 
    N0 = np.mean(  N0_list, axis = 0 )

    I0_list = []
    for _ in range( number_images_recorded_per_cmd ):
        img_tmp = zwfs.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
        I0_list.append( img_tmp.signal )
    I0 =  np.mean(  I0_list, axis = 0 )
    
    BIAS_list = []
    for _ in range( number_images_recorded_per_cmd ):
        img_tmp = zwfs.detection_chain( off_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    BIAS_list.append( img_tmp.signal )
    
    BIAS = np.mean(  BIAS_list, axis = 0 )


    #====== make references fits files
    I0_fits = fits.PrimaryHDU( I0 )
    N0_fits = fits.PrimaryHDU( N0 )
    BIAS_fits = fits.PrimaryHDU( BIAS_list )
    I0_fits.header.set('EXTNAME','FPM_IN')
    N0_fits.header.set('EXTNAME','FPM_OUT')
    BIAS_fits.header.set('EXTNAME','BIAS')

    flat_DM_fits = fits.PrimaryHDU( flat_dm_cmd )
    flat_DM_fits.header.set('EXTNAME','FLAT_DM_CMD')
    
    

    
    zwfs.dm.update_shape( flat_dm_cmd )

    
    _ = input('PRESS ENTER WHEN READY TO BUILD IM' )
    
    # --- creating sequence of dm commands
    _DM_command_sequence = [list(flat_dm_cmd + amp * modal_basis) for amp in ramp_values ]  
    # add in flat dm command at beginning of sequence and reshape so that cmd sequence is
    # [0, a0*b0,.. aN*b0, a0*b1,...,aN*b1, ..., a0*bM,...,aN*bM]
    DM_command_sequence = [flat_dm_cmd] + list( np.array(_DM_command_sequence).reshape(number_amp_samples*modal_basis.shape[0],modal_basis.shape[1] ) )

    # --- additional labels to append to fits file to keep information about the sequence applied 
    additional_labels = [('cp_x1','simulation'),('cp_x2','simulation'),('cp_y1','simulation'),('cp_y2','simulation'),('in-poke max amp', np.max(ramp_values)),('out-poke max amp', np.min(ramp_values)),('#ramp steps',number_amp_samples), ('seq0','flatdm'), ('reshape',f'{number_amp_samples}-{modal_basis.shape[0]}-{modal_basis.shape[1]}'),('Nmodes_poked',len(modal_basis)),('Nact',140)]


    
    image_list = [] #init list to hold images

    # NOTE THE CAMERA SHOULD ALREADY BE STARTED BEFORE BEGINNING - No checking here yet
    for cmd_indx, cmd in enumerate(DM_command_sequence):
        print(f'executing cmd_indx {cmd_indx} / {len(DM_command_sequence)}')
        # wait a sec        
        #time.sleep(sleeptime_between_commands)
        # ok, now apply command
        zwfs.dm.update_shape( flat_dm_cmd + cmd )
        # wait a sec        
        #time.sleep(sleeptime_between_commands)
        
        tmp_list = []
        for _ in range( number_images_recorded_per_cmd ):
            img_tmp = zwfs.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
            tmp_list.append( img_tmp.signal )

        image_list.append( [np.mean( tmp_list, axis=0 )] )


    # SAVING IN FITS FORMAT 
    data = fits.HDUList([]) #init main fits file to append things to
    
    # Camera data
    cam_fits = fits.PrimaryHDU( image_list )
    
    cam_fits.header.set('EXTNAME', 'SEQUENCE_IMGS' )
    #camera headers
    cam_fits.header.set('#images per DM command', number_images_recorded_per_cmd )
    cam_fits.header.set('take_mean_of_images', take_mean_of_images )
    
    cam_fits.header.set('cropping_corners_r1', 'simulation' )
    cam_fits.header.set('cropping_corners_r2', 'simulation' )
    cam_fits.header.set('cropping_corners_c1', 'simulation' )
    cam_fits.header.set('cropping_corners_c2', 'simulation' )
    
    
    for hw in zwfs.mode:
        for k,v in zwfs.mode[hw].items():
            cam_fits.header.set(k,v)

    for k,v in additional_labels :
        cam_fits.header.set(k,v)
    #if user specifies additional headers using additional_header_labels
    """if (additional_labels!=None): 
        if type(additional_header_labels)==list:
            for i,h in enumerate(additional_header_labels):
                cam_fits.header.set(h[0],h[1])
        else:
            cam_fits.header.set(additional_header_labels[0],additional_header_labels[1])
    """
    # add camera data to main fits
    data.append(cam_fits)
    

    # put commands in fits format
    dm_fits = fits.PrimaryHDU( DM_command_sequence )
    #DM headers 
    dm_fits.header.set('timestamp', str(datetime.datetime.now()) )
    dm_fits.header.set('EXTNAME', 'DM_CMD_SEQUENCE' )
    #dm_fits.header.set('DM', DM.... )
    #dm_fits.header.set('#actuators', DM.... )

    # append to the data
    data.append(dm_fits)
    
    # append FPM IN and OUT references (note FPM in reference is also first entry in recon_data so we can compare if we want!) 
    data.append( I0_fits ) 
    data.append( N0_fits ) 
    data.append( BIAS_fits )
    data.append( flat_DM_fits )
    
    
    if save_fits!=None:
        if type(save_fits)==str:
            data.writeto(save_fits) #, overwrite=True)
        else:
            raise TypeError('save_images needs to be either None or a string indicating where to save file')
        
    
#%%
# FITTING -------------------


recon_data = fits.open( '/Users/bencb/Documents/rtc-example/data/simulation_test_2.fits' ) # data
bad_pixels = ([],[])
active_dm_actuator_filter=None 
debug=True

fig_path = '/Users/bencb/Documents/rtc-example/data/' 
save_fits =  '/Users/bencb/Documents/rtc-example/data/simulation_test.fits' #None
    

        
# -- prelims of reading in and labelling data 
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

# poke values used in linear ramp
No_ramps = int(recon_data['SEQUENCE_IMGS'].header['#ramp steps'])
max_ramp = float( recon_data['SEQUENCE_IMGS'].header['in-poke max amp'] )
min_ramp = float( recon_data['SEQUENCE_IMGS'].header['out-poke max amp'] ) 
ramp_values = np.linspace( min_ramp, max_ramp, No_ramps)

flat_dm_cmd = recon_data['FLAT_DM_CMD'].data

Nmodes_poked = int(recon_data[0].header['HIERARCH Nmodes_poked']) # can also see recon_data[0].header['RESHAPE']

Nact =  int(recon_data[0].header['HIERARCH Nact'])  

N0 = recon_data['FPM_OUT'].data
#P = np.sqrt( pupil ) # 
I0 = recon_data['FPM_IN'].data

# the first image is another reference I0 with FPM IN and flat DM
poke_imgs = recon_data['SEQUENCE_IMGS'].data[1:].reshape(No_ramps, 140, I0.shape[0], I0.shape[1])
#poke_imgs = poke_imgs[1:].reshape(No_ramps, 140, I0.shape[0], I0.shape[1])

    
"""# some checks 

act = 65

plt.figure(); plt.imshow( poke_imgs[-1][act] - I0 );plt.colorbar()


i0 = np.argmax( abs( a[5][act] - I0 ) )
[ a[act].reshape(-1)[i0] for a in poke_imgs]   
    
plt.figure(); 
grads = []
for act in range(140):
    i0 = np.argmax( abs( poke_imgs[a0][act] - I0 ) )
    plt.plot( ramp_values, [ a[act].reshape(-1)[i0] for a in poke_imgs] ,alpha = 0.9)
    grads.append( np.diff( [ a[act].reshape(-1)[i0] for a in poke_imgs[4:-4]] )[0] )

plt.plot( ramp_values, 1e6 + 0.5e6 * np.sin(np.pi * 2 * 1/6e-7 * ramp_values  ) ,ls=':')



def estimate_peak_frequency_and_phase(x, y):
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Compute the sampling frequency
    sampling_interval = np.mean(np.diff(x))  # Time between samples
    sampling_frequency = 1.0 / sampling_interval  # Samples per second (Hz)
    
    # Perform the Fast Fourier Transform (FFT)
    Y = np.fft.fft(y)
    
    # Compute the corresponding frequency bins
    freqs = np.fft.fftfreq(len(y), d=sampling_interval)
    
    # Compute the magnitude of the FFT (ignoring complex phase information)
    magnitude = np.abs(Y)
    
    # Ignore the zero frequency (DC component)
    zero_freq_index = np.argmin(np.abs(freqs))  # Find the index for zero frequency
    magnitude[zero_freq_index] = 0  # Set the magnitude at zero frequency to zero
    
    # Find the peak frequency (excluding zero frequency)
    peak_index = np.argmax(magnitude)
    peak_frequency = np.abs(freqs[peak_index])
    
    # Get the phase at the peak frequency
    peak_phase = np.angle(Y[peak_index])  # Phase at the peak frequency
    
    return peak_frequency, peak_phase



TESTING FOR ONE ACTUATOR 
act = 65

theta = zwfs.FPM.phase_mask_phase_shift( np.mean( zwfs.wvls ) )
mu = np.arctan2( np.sin( np.deg2rad( theta ) ) , (np.cos( np.deg2rad( theta ) ) - 1) )
i0 = np.argmax( abs( poke_imgs[a0][act] - I0 ) )

s = np.array( [ a[act].reshape(-1)[i0]  for a in poke_imgs]  )
f_est, mu_est = estimate_peak_frequency_and_phase(ramp_values, s)
plt.plot( ramp_values, s ,alpha = 0.9)
plt.plot( ramp_values, np.mean( s )  + np.ptp( s )/2 * np.cos(1 * 2 * np.pi * f_est * ramp_values  - mu_est )) #- mu_est - 0.2) ,color='k',ls=':')



fit_dict = {'A':[],'B':[],'F':[],'mu':[],'psi_c^2':[], 'residuals':[]}
mu_theory = np.arctan2( np.sin( np.deg2rad( theta ) ) , (np.cos( np.deg2rad( theta ) ) - 1) )
for act in range(140):
    
    i0 = np.argmax( abs( poke_imgs[a0][act] - I0 ) )

    psi_A = N0.reshape(-1)[i0]
    s = np.array( [ a[act].reshape(-1)[i0]  for a in poke_imgs]  )
    F_est, mu_est = estimate_peak_frequency_and_phase(ramp_values, s)
    
    mu_est = abs( mu_est )
    
    A_est, B_est = np.mean( s ), np.ptp( s )/2 
    
    fit_dict['A'].append( A_est )
    fit_dict['B'].append( B_est )
    fit_dict['F'].append( F_est )
    fit_dict['mu'].append( mu_est )
    fit_dict['psi_c^2'].append(  np.mean( s ) - N0.reshape(-1)[i0] )
    fit_dict['residuals'].append( ( s - (A_est +  B_est * np.cos( 2*np.pi*F_est * ramp_values - mu_est)) ) / s )
    plt.figure()
    plt.title( f'{act}, \nA={ A_est },\n B = {B_est}, \nF={F_est} , \nmu={mu_est}' )
    plt.plot( ramp_values, s ,alpha = 0.9, label='meas')
    plt.plot( ramp_values, A_est +  B_est * np.cos( 2*np.pi*F_est * ramp_values - mu_est))
    

plt.figure()   
plt.hist( fit_dict['B'] )
#plt.axvline( np.mean( fit_dict['B'] ) - 2* np.std( fit_dict['B'] ))
plt.axvline( np.mean( fit_dict['B'] )  - 1* np.std( fit_dict['B'] ) , color='k')

filt_ptp = fit_dict['B'] > np.mean( fit_dict['B'] )  - 0.5* np.std( fit_dict['B'] )
filt_res = np.sum( abs(np.array( fit_dict['residuals'] ) ), axis=1 ) < 5
    

#  # I DONT KNOW WHY 2 * mu_theory - pi works!!!
plt.hist( fit_dict['mu'] , bins = 50); plt.axvline( 2 *mu_theory - np.pi, color='k' )


# look at good and bad fits 

    
plt.imshow(baldrSim.get_BMCmulti35_DM_command_in_2D(  np.sum( abs(np.array( fit_dict['residuals'] ) ) ) )
    
plt.imshow(baldrSim.get_BMCmulti35_DM_command_in_2D(  filt_ptp ) )

plt.imshow( baldrSim.get_BMCmulti35_DM_command_in_2D( fit_dict['F'] ) ); plt.colorbar()
plt.imshow( baldrSim.get_BMCmulti35_DM_command_in_2D( fit_dict['mu'] ) ); plt.colorbar()

plt.figure()
plt.hist( fit_dict['mu'] )
plt.axvline( mu_theory )    

"""


if len(bad_pixels[0]) > 0:
        
    bad_pixel_mask = np.ones(I0.shape)
    for ibad,jbad in list(zip(bad_pixels[0], bad_pixels[1])):
        bad_pixel_mask[ibad,jbad] = 0
        
    I0 *= bad_pixel_mask
    N0 *= bad_pixel_mask
    poke_imgs  = poke_imgs * bad_pixel_mask

a0 = len(ramp_values)//2 - 2 # which poke value (index) do we want to consider for finding region of influence. Pick a value near the center of the ramp (ramp values are from negative to positive) where we are in a linear regime.



if active_dm_actuator_filter==None:
    # ========================== !! 1 !! =====================
    #  == Then we let the user define the region of influence on DM where we will fit our models (by defining a threshold for I(epsilon)_max -I_0). This is important because the quality of the fits can go to shit around pupil/DM edges, we only need some good samples around the center to reconstruct what we need, setting this threshold is semi automated here  

    fig,ax= plt.subplots( 4, 4, figsize=(10,10))
    num_pixels = []
    candidate_thresholds = np.linspace(4 * np.std(abs(poke_imgs[a0,:,:,:] - I0)),np.max(abs(poke_imgs[a0,:,:,:] - I0)),16)
    for axx, thresh in zip(ax.reshape(-1),candidate_thresholds):

        dm_pupil_filt = thresh < np.array( [np.max( abs( poke_imgs[a0][act] - I0) ) for act in range(140)] ) 
        axx.imshow( get_DM_command_in_2D( dm_pupil_filt ) ) 
        axx.set_title('threshold = {}'.format(round( thresh )),fontsize=12) 
        axx.axis('off')
        num_pixels.append(sum(dm_pupil_filt)) 
        # we could use this to automate threshold decision.. look for where 
        # d num_pixels/ d threshold ~ 0.. np.argmin( abs( np.diff( num_pixels ) )[:10])
    plt.show()

    recommended_threshold = candidate_thresholds[np.argmin( abs( np.diff( num_pixels ) )[2:11]) + 1 ]
    print( f'\n\nrecommended threshold ~ {round(recommended_threshold)} \n(check this makes sense with the graph by checking the colored area is stable around changes in threshold about this value)\n\n')

    pupil_filt_threshold = float(input('input threshold of peak differences'))

    ### <---- THIS FILTER DETERMINES WHERE WE FIT THE MODELS (ONLY FIT WHERE DM HAS GOOD INFLUENCE!)
    dm_pupil_filt = pupil_filt_threshold < np.array( [np.max( abs( poke_imgs[a0][act] - I0) ) for act in range(Nact)] ) 

    if debug:
       plt.figure()
       plt.imshow( get_DM_command_in_2D( dm_pupil_filt ) )
       plt.title('influence region on DM where we will fit intensity models per actuator')
       plt.show()
       plt.savefig(  fig_path + f'process_fits_0_{tstamp}.png', bbox_inches='tight', dpi=300)






     
# ========================== !! 2 !! =====================
# ======== P2C 

Sw_x, Sw_y = 3,3 #+- pixels taken around region of peak influence. PICK ODD NUMBERS SO WELL CENTERED!   
act_img_mask_1x1 = {} #pixel with peak sensitivity to the actuator
act_img_mask_3x3 = {} # 3x3 region around pixel with peak sensitivity to the actuator
poor_registration_list = np.zeros(Nact).astype(bool) # list of actuators in control region that have poor registration 


# I should probably include a threshold filter here - that no registration is made if delta < threshold
# threshold set to 5 sigma above background (seems to work - but can be tweaked) 
registration_threshold = 5 * np.mean(np.std(abs(poke_imgs- I0),axis=(0,1)) )
# how to best deal with actuators that have poor registration ?
for act_idx in range(Nact):
    delta =  poke_imgs[a0][act_idx] - I0

    mask_3x3 = np.zeros( I0.shape )
    mask_1x1 = np.zeros( I0.shape )
    if dm_pupil_filt[act_idx]: #  if we decided actuator has strong influence on ZWFS image, we 
        peak_delta = np.max( abs(delta) ) 
        if peak_delta > registration_threshold:

            i,j = np.unravel_index( np.argmax( abs(delta) ), I0.shape )
      
            mask_3x3[i-Sw_x-1: i+Sw_x, j-Sw_y-1:j+Sw_y] = 1 # keep centered, 
            mask_1x1[i,j] = 1 
            #mask *= 1/np.sum(mask[i-Sw_x-1: i+Sw_x, j-Sw_y-1:j+Sw_y]) #normalize by #pixels in window 
            act_img_mask_3x3[act_idx] = mask_3x3
            act_img_mask_1x1[act_idx] = mask_1x1
        else:
            poor_registration_list[act_idx] = True
            act_img_mask_3x3[act_idx] = mask_3x3 
            act_img_mask_1x1[act_idx] = mask_1x1 
    else :
        act_img_mask_3x3[act_idx] = mask_3x3 
        act_img_mask_1x1[act_idx] = mask_1x1 
        #act_flag[act_idx] = 0 
if debug:
    plt.figure()
    plt.title('masked regions of influence per actuator')
    plt.imshow( np.sum( list(act_img_mask_3x3.values()), axis = 0 ) )
    #plt.show()
    plt.savefig(  fig_path + f'process_fits_1_{tstamp}.png', bbox_inches='tight', dpi=300)



# turn our dictionary to a big pixel to command matrix 
P2C_1x1 = np.array([list(act_img_mask_1x1[act_idx].reshape(-1)) for act_idx in range(Nact)])
P2C_3x3 = np.array([list(act_img_mask_3x3[act_idx].reshape(-1)) for act_idx in range(Nact)])

# we can look at filtering a particular actuator in image P2C_3x3[i].reshape(zwfs.get_image().shape)
# we can also interpolate signals in 3x3 grid to if DM actuator not perfectly registered to center of pixel. 




if debug: 
    im_list = [(I0 - N0) / np.mean(N0) ]
    xlabel_list = ['x [pixels]']
    ylabel_list = ['y [pixels]']
    title_list = ['']
    cbar_label_list = [r'$\frac{|\psi_C|^2 - |\psi_A|^2}{<|\psi_A|^2>}$']
    savefig = None# fig_path + f'pupil_FPM_IN-OUT_readout_mode-FULL_t{tstamp}.png'

    nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

    # check the active region 
    fig,ax = plt.subplots(1,1)
    ax.imshow( get_DM_command_in_2D( dm_pupil_filt  ) )
    ax.set_title('active DM region')
    ax.grid(True, which='minor',axis='both', linestyle='-', color='k' ,lw=3)
    ax.set_xticks( np.arange(12) - 0.5 , minor=True)
    ax.set_yticks( np.arange(12) - 0.5 , minor=True)

    plt.savefig(  fig_path + f'process_fits_2_{tstamp}.png', bbox_inches='tight', dpi=300)
    #plt.savefig( fig_path + f'active_DM_region_{tstamp}.png' , bbox_inches='tight', dpi=300) 
    # check the well registered DM region : 
    fig,ax = plt.subplots(1,1)
    ax.imshow( get_DM_command_in_2D( np.sum( P2C_1x1, axis=1 )))
    ax.set_title('well registered actuators')
    ax.grid(True, which='minor',axis='both', linestyle='-', color='k',lw=2 )
    ax.set_xticks( np.arange(12) - 0.5 , minor=True)
    ax.set_yticks( np.arange(12) - 0.5 , minor=True)

    plt.savefig(  fig_path + f'process_fits_3_{tstamp}.png', bbox_inches='tight', dpi=300)
  
    #plt.savefig( fig_path + f'poorly_registered_actuators_{tstamp}.png' , bbox_inches='tight', dpi=300) 

    # check poorly registered actuators: 
    fig,ax = plt.subplots(1,1)
    ax.imshow( get_DM_command_in_2D(poor_registration_list) )
    ax.set_title('poorly registered actuators')
    ax.grid(True, which='minor',axis='both', linestyle='-', color='k', lw=2 )
    ax.set_xticks( np.arange(12) - 0.5 , minor=True)
    ax.set_yticks( np.arange(12) - 0.5 , minor=True)

    plt.savefig(  fig_path + f'process_fits_4_{tstamp}.png', bbox_inches='tight', dpi=300)

    plt.show() 
    
    

    
#%%
# ======== FITTING 
# what do we fit ( I - N0 ) / N0 
savefits = '/Users/bencb/Documents/rtc-example/data/simulation_test.fits'

def Ic_model_constrained(x, A, B, F, mu):
    penalty = 0
    #if (F < 0) or (mu < 0): # F and mu can be correlated so constrain the quadrants 
    #    penalty = 1e3
    I = A + B * np.cos(F * x + mu) + penalty
    return I 



def model_func(x, A, B, F, mu):
    return A + B + A * B * np.sin(F * x + mu)

def fit_func(x, B, F, mu):
    penalty = 0
    #if (F < 0) or (B < 0): #(mu < 0): # F and mu can be correlated so constrain the quadrants 
    #    penalty = 1e50
    I = model_func(x, N_0, B, F, mu)
    return( I + penalty )





param_dict = {}
cov_dict = {}
fit_residuals = []
nofit_list = []

##
fit_3_parameter = False # e
##
if debug:
    Nrows = np.ceil( sum( dm_pupil_filt )**0.5).astype(int)
    fig,ax = plt.subplots(Nrows,Nrows,figsize=(20,20))
    axx = ax.reshape(-1)
    for aaa in axx:
        aaa.axis('off')
    j=0 #axx index



#mean_filtered_pupil = 1/(3*3)  * np.mean( P2C_3x3 @ N0.reshape(-1) )

for act_idx in range(len(flat_dm_cmd)): 

    #Note that if we use the P2C_3x3 we need to normalize it 1/(3*3) * P2C_3x3
    if dm_pupil_filt[act_idx] * ( ~poor_registration_list)[act_idx]:

        # -- we do this with matrix multiplication using  mask_matrix
        #P_i = np.sum( act_img_mask[act_idx] * pupil ) #Flat DM with FPM OUT 
        #P_i = mean_filtered_pupil.copy() # just consider mean pupil! 
    
        I_i = np.array( [P2C_1x1[act_idx] @ poke_imgs[i][act_idx].reshape(-1) for i in  range(len(ramp_values))] ) #np.array( [np.sum( act_img_mask[act_idx] * poke_imgs[i][act_idx] ) for i in range(len(ramp_values))] ) #spatially filtered sum of intensities per actuator cmds 
        I_0 = P2C_1x1[act_idx] @ I0.reshape(-1) #np.sum( act_img_mask[act_idx] * I0 ) # Flat DM with FPM IN  
        N_0 = P2C_1x1[act_idx] @ N0.reshape(-1) #np.sum( act_img_mask[act_idx] * N0 )
        # ================================
        #   THIS IS OUR MODEL!S=A+B*cos(F*x + mu)  
        #S = (I_i - I_0) / P_i # signal to fit!
        S = I_i  # A 
        #S = (I_i - N_0) / N_0 # signal to fit! <- should I take mean of total pupil?? 
        # THEORETICALLY THIS SIGNAL IS: S = |M0|^2/|A|^2 + M0/|A| * cos(F.c + mu)  
        # ================================

        #re-label and filter to capture best linear range 
        x_data = ramp_values[1:-1].copy()
        y_data = S[1:-1].copy()

        # HERE! 
        F_est, mu_est = estimate_peak_frequency_and_phase(x_data, y_data)
        #mu_est = abs( mu_est )
        
        A_est, B_est = np.mean( y_data ), np.ptp( y_data )/2 
        if not fit_3_parameter :
            initial_guess = [A_est, B_est , F_est, abs(mu_est)]  #[np.mean(S), (np.max(S)-np.min(S))/2,  15, 2.4]
        else:
            initial_guess = [B_est, F_est, abs(mu_est)] #[(np.max(S)-np.min(S))/2,  2 * np.pi  * 0.5/6e-7 , 2.3]
        
        #initial_guess = [7, 2, 15, 2.4] #[0.5, 0.5, 15, 2.4]  #A_opt, B_opt, F_opt, mu_opt  ( S = A+B*cos(F*x + mu) )

        try:
            # FIT 
            # HERE! 
            if not fit_3_parameter :
                popt, pcov = curve_fit(Ic_model_constrained, x_data, y_data, p0=initial_guess)
            else:
                popt, pcov = curve_fit( fit_func,  x_data, y_data, p0=initial_guess)
            
            # Extract the optimized parameters explictly to measure residuals
            # HERE!
            if not fit_3_parameter :
                A_opt, B_opt, F_opt, mu_opt = popt
            else:
                B_opt, F_opt, mu_opt = popt
            
            # STORE FITS 
            param_dict[act_idx] = popt
            cov_dict[act_idx] = pcov 
            
            # HERE! 
            # also record fit residuals 
            if not fit_3_parameter :
                fit_residuals.append( S - Ic_model_constrained(ramp_values, A_opt, B_opt, F_opt, mu_opt) )
            else:
                fit_residuals.append( S -  fit_func(ramp_values, B_opt, F_opt, mu_opt) )


            if debug: 

                #HERE 
                if not fit_3_parameter :
                    axx[j].plot( ramp_values, Ic_model_constrained(ramp_values, A_opt, B_opt, F_opt, mu_opt) ,label=f'fit (act{act_idx})') 
                else:
                    axx[j].plot( ramp_values, fit_func(ramp_values, B_opt, F_opt, mu_opt)  ,label=f'fit (act{act_idx})') 
                axx[j].plot( ramp_values, S ,label=f'measured (act{act_idx})' )
                #axx[j].set_xlabel( 'normalized DM command')
                #axx[j].set_ylabel( 'normalized Intensity')
                axx[j].legend(fontsize=6)
                #axx[j].set_title(act_idx,fontsize=5)
                #ins = axx[j].inset_axes([0.15,0.15,0.25,0.25])
                #ins.imshow(poke_imgs[3][act_idx] )
                #axx[j].axis('off')
                j+=1
        except:
            print(f'\n!!!!!!!!!!!!\nfit failed for actuator {act_idx}\n!!!!!!!!!!!!\nanalyse plot to try understand why')
            """nofit_list.append( act_idx ) 
            fig1, ax1 = plt.subplots(1,1)
            ax1.plot( ramp_values, S )
            ax1.set_title('could not fit this!') """
             
   

if debug:
    plt.savefig( fig_path + f'process_fits_5_{tstamp}.png' , bbox_inches='tight', dpi=300) 
    #plt.show() 

if debug:
    """ used to buff things out (adding new 0 normal noise variance to samples) 
    Qlst,Wlst,Flst,mulst = [],[],[],[]
    Q_est =  np.array(list( param_dict.values() ))[:, 0]
    W_est = np.array(list( param_dict.values() ))[:, 1] 
    F_est = np.array(list( param_dict.values() ))[:, 2]
    mu_est = np.array(list( param_dict.values() ))[:, 3] 
    for q,w,f,u in param_dict.values():
        Qlst.append( list( q + 0.01*np.mean(Q_est)*np.random.randn(100 ) ) )
        Wlst.append( list( w + 0.01*np.mean(W_est)*np.random.randn(100 ) ) )
        Flst.append( list( f + 0.01*np.mean(F_est)*np.random.randn(100 ) ) )
        mulst.append( list( u + 0.01*np.mean(mu_est)*np.random.randn(100 ) ) )

    #buffcorners = np.array( [list( param_dict.values() ) for _ in range(10)]).reshape(-1,4)
    buffcorners = np.array([np.array(Qlst).ravel(),np.array(Wlst).ravel(), np.array(Flst).ravel(),np.array(mulst).ravel()]).T
    corner.corner( buffcorners , quantiles=[0.16,0.5,0.84], show_titles=True, labels = ['Q [adu]', 'W [adu/cos(rad)]', 'F [rad/cmd]', r'$\mu$ [rad]'] ) 
    """
    # HERE! 
    #labels = ['Q', 'W', 'F', r'$\mu$']
    if not fit_3_parameter :
        corner.corner( np.array(list( param_dict.values() )), quantiles=[0.16,0.5,0.84], show_titles=True, \
                  labels = ['A', 'B', 'F', r'$\mu$']  ) #, range = [(0,2*np.mean(y_data)),(0, 10*(np.max(y_data)-np.min(y_data)) ) , (5,20), (0,6) ] ##range = [(2*np.min(S), 102*np.max(S)), (0, 2*(np.max(S) - np.min(S)) ), (5, 20), (-3,3)] ) #['Q [adu]', 'W [adu/cos(rad)]', 'F [rad/cmd]', r'$\mu$ [rad]']
    else:
        corner.corner( np.array(list( param_dict.values() )), quantiles=[0.16,0.5,0.84], show_titles=True, \
                      labels = ['B', 'F', r'$\mu$']  ) 
    plt.savefig( fig_path + f'process_fits_6_{tstamp}.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    


B_tmp = np.nan * np.zeros( 140 )
F_tmp = np.nan * np.zeros( 140 )
mu_tmp = np.nan * np.zeros( 140 )
for k in param_dict:
    param_dict[k][0]
    B_tmp[k] = param_dict[k][0]
    F_tmp[k] = param_dict[k][1]
    mu_tmp[k] = param_dict[k][2]

fig,ax = plt.subplots( 3,1 ,figsize=(15,5))
im0 = ax[0].imshow( baldrSim.get_BMCmulti35_DM_command_in_2D( B_tmp ) )
im1= ax[1].imshow( baldrSim.get_BMCmulti35_DM_command_in_2D( F_tmp ) )
im2 = ax[2].imshow( baldrSim.get_BMCmulti35_DM_command_in_2D( mu_tmp ) )
plt.colorbar(im2 , ax=ax[2])


    
output_fits = fits.HDUList( [] )

# reference images 
N0_fits = fits.PrimaryHDU( N0 )
N0_fits.header.set('EXTNAME','FPM OUT REF')
N0_fits.header.set('WHAT IS','ref int. with FPM out')

I0_fits = fits.PrimaryHDU( I0 )
I0_fits.header.set('EXTNAME','FPM IN REF')
I0_fits.header.set('WHAT IS','ref int. with FPM in')

# output fits files 
P2C_fits = fits.PrimaryHDU( np.array([P2C_1x1, P2C_3x3]) )
P2C_fits.header.set('EXTNAME','P2C')
P2C_fits.header.set('WHAT IS','pixel to DM actuator register')
P2C_fits.header.set('index 0','P2C_1x1') 
P2C_fits.header.set('index 1','P2C_3x3')    

#fitted parameters
param_fits = fits.PrimaryHDU( np.array(list( param_dict.values() )) )
param_fits.header.set('EXTNAME','FITTED_PARAMS')
"""param_fits.header.set('COL0','Q [adu]')
param_fits.header.set('COL1','W [adu/cos(rad)]')
param_fits.header.set('COL2','F [rad/cmd]')
param_fits.header.set('COL4','mu [rad]')"""
param_fits.header.set('COL0','A [adu]')
param_fits.header.set('COL1','B [adu]')
param_fits.header.set('COL2','F [rad/cmd]')
param_fits.header.set('COL4','mu [rad]')
if len(nofit_list)!=0:
    for i, act_idx in enumerate(nofit_list):
        param_fits.header.set(f'{i}_fit_fail_act', act_idx)
    
#covariances
cov_fits = fits.PrimaryHDU( np.array(list(cov_dict.values())) )
cov_fits.header.set('EXTNAME','FIT_COV')
# residuals 
res_fits = fits.PrimaryHDU( np.array(fit_residuals) )
res_fits.header.set('EXTNAME','FIT_RESIDUALS')

#DM regions 
dm_fit_regions = fits.PrimaryHDU( np.array( [dm_pupil_filt, dm_pupil_filt*(~poor_registration_list), poor_registration_list] ).astype(int) )
dm_fit_regions.header.set('EXTNAME','DM_REGISTRATION_REGIONS')
dm_fit_regions.header.set('registration_threshold',registration_threshold)
dm_fit_regions.header.set('index 0 ','active_DM_region')   
dm_fit_regions.header.set('index 1 ','well registered actuators') 
dm_fit_regions.header.set('index 2 ','poor registered actuators') 
 
for f in [N0_fits, I0_fits, P2C_fits, param_fits, cov_fits,res_fits, dm_fit_regions ]:
    output_fits.append( f ) 

if savefits!=None:
       
    output_fits.writeto( savefits, overwrite=True )  #data_path + 'ZWFS_internal_calibration.fits'












#%%



    
    
  









# 
dm_waffle_cmd = pd.read_csv( '/Users/bencb/Documents/rtc-example/DMShapes/waffle.csv', header=None)[0].values

shape_files = glob.glob( 'DMShapes/*.csv')


shape_name = file.split('/')[-1].split('.csv')[0]

shape = pd.read_csv(shape_files[-1], header=None)[0].values
