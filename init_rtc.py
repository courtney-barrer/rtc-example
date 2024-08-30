
from time import sleep
import rtc
import numpy as np
from datetime import timedelta
from astropy.io import fits 
import pickle
import matplotlib.pyplot as plt
import time
import os
import pandas as pd 
import glob
import datetime

from pyBaldr import ZWFS
from pyBaldr import phase_control
from pyBaldr import pupil_control
from pyBaldr import utilities as util

#pip install --no-build-isolation -e .

def convert_local_to_global_coordinates(relative_pixels_flat, row1, col1, n, m, global_shape, flatten = True):
    """
    Convert relative pixel coordinates from the cropped region to global coordinates in the original image.
    
    Parameters:
    - relative_pixels_flat: 1D list or array of indices in the flattened cropped array.
    - row1: The starting row index of the cropped region in the global image.
    - col1: The starting column index of the cropped region in the global image.
    - n: The number of rows in the cropped region.
    - m: The number of columns in the cropped region.
    - global_shape: shape of global image to reference back to
    Returns:
    - global_pixels: List of tuples [(row_global1, col_global1), (row_global2, col_global2), ...]
                     containing the global coordinates in the original image.
    """
    
    # Convert flattened relative pixel indices to 2D coordinates within the cropped region
    relative_coords = [(index // m, index % m) for index in relative_pixels_flat]
    
    # Convert relative 2D coordinates to global 2D coordinates
    global_pixels = [(row + row1, col + col1) for (row, col) in relative_coords]

    if flatten:
        #global_pixels = [row * M + col for row, col in global_pixels]
        rows, cols = zip(*global_pixels)
        global_pixels = np.ravel_multi_index((rows, cols), global_shape)

    return global_pixels





# TO DO 
# 

data_path = '/home/heimdallr/Documents/asgard-alignment/tmp/30-08-2024/iter_4_J3/fourier_20modes_map_reconstructor/' #'/home/heimdallr/Documents/asgard-alignment/tmp/30-08-2024/iter_3_J3/fourier_20modes_map_reconstructor/' #'/home/heimdallr/Documents/asgard-alignment/tmp/30-08-2024/iter_1_J3/fourier_20modes_map_reconstructor/'#'/home/heimdallr/Documents/asgard-alignment/tmp/29-08-2024/iter_14_J3/fourier_20modes_map_reconstructor/' #'~/Documents/asgard-alignment/tmp/29-08-2024/iter_13_J3/'

reconstructor_file = data_path + 'RECONSTRUCTORS_fourier_0.2pokeamp_in-out_pokes_map_DIT-0.002_gain_high_30-08-2024T13.32.11.fits'#'RECONSTRUCTORS_fourier_0.2pokeamp_in-out_pokes_map_DIT-0.002_gain_high_30-08-2024T09.43.22.fits' #'RECONSTRUCTORS_fourier_0.2pokeamp_in-out_pokes_map_DIT-0.001_gain_high_30-08-2024T07.51.19.fits' #'RECONSTRUCTORS_fourier_0.2pokeamp_in-out_pokes_map_DIT-0.001_gain_high_29-08-2024T23.48.48.fits' #'RECONSTRUCTORS_fourier_0.2pokeamp_in-out_pokes_map_DIT-0.001_gain_high_29-08-2024T22.59.26.fits'


#def init_rtc( reco_fits_file ):
if 1: 
    # ============== READ IN PUPIL PHASE RECONSTRUCTOR DATA
    config_fits = fits.open( reconstructor_file  ) 

    print( [reco.header['EXTNAME'] for reco in config_fits])

    # our nanobinded rtc object
    r = rtc.RTC()

    # init struct's
    cam_settings_tmp = rtc.camera_settings_struct()
    reconstructors_tmp = rtc.phase_reconstuctor_struct()
    pupil_regions_tmp = rtc.pupil_regions_struct()


    # camera settings used to build reconstructor 

    det_fps = float( config_fits['info'].header['camera_fps'] ) # frames per sec (Hz) 

    det_dit = float( config_fits['info'].header['camera_tint'] )  # integration time (seconds)

    det_gain = str( config_fits['info'].header['camera_gain'] ) # camera gain 

    # Image should neve be cropped unless do latency measurements 
    det_cropping_rows = str( config_fits['info'].header['cropping_rows'] ).split('rows: ')[1]
    det_cropping_cols = str( config_fits['info'].header['cropping_columns'] ).split('columns: ')[1]
    # to build reconstructors we do look at subregions (and reference pixels from within sub-region)
    # in-order to reduce file sizes of the reconstructors .. Here we read out these subregion corners
    # in the global frame 
    r1_subregion = config_fits['info'].header['cropping_corners_r1']
    r2_subregion = config_fits['info'].header['cropping_corners_r2']
    c1_subregion = config_fits['info'].header['cropping_corners_c1']
    c2_subregion = config_fits['info'].header['cropping_corners_c2']

    # full image dimensions
    img_height = 512
    img_width = 640  
    global_shape = (img_height, img_width)

    # pupil region classification data in local (sub-region)
    pupil_pixels_local = np.array( config_fits['pupil_pixels'].data, dtype=np.int32)
    secondary_pixels_local = np.array( config_fits['secondary_pixels'].data, dtype=np.int32)
    outside_pixels_local = np.array( config_fits['outside_pixels'].data, dtype=np.int32)
    #local_region_pixels_local = np.arange(0, len(config_fits['DARK'].data[0].reshape(-1)) )  #np.array( config_fits['outside_pixels'].data, dtype=np.int32)

    # converting back to the global image (in the reconstructor we likely (to reduce data size) looked at only a sub-region and referenced pixels from there)
    pupil_pixels = convert_local_to_global_coordinates(pupil_pixels_local, r1_subregion, \
        c1_subregion, r2_subregion-r1_subregion, c2_subregion-c1_subregion, global_shape, flatten = True)
    secondary_pixels = convert_local_to_global_coordinates(secondary_pixels_local, r1_subregion, \
        c1_subregion, r2_subregion-r1_subregion, c2_subregion-c1_subregion, global_shape, flatten = True)
    #outside_pixels_0 = convert_local_to_global_coordinates(outside_pixels_local, r1_subregion, \
    #    c1_subregion, r2_subregion-r1_subregion, c2_subregion-c1_subregion, global_shape, flatten = True)
    #local_region_pixels = convert_local_to_global_coordinates(local_region_pixels_local, r1_subregion, \
    #    c1_subregion, r2_subregion-r1_subregion, c2_subregion-c1_subregion, global_shape, flatten = True)

    # outside_pixels_0 is only outside in local region, defauls to 0 outside of this region so to correct this
    all_pixels = np.arange(0, img_height * img_width )
    outside_pixels = np.array( list( set( all_pixels ) - set( pupil_pixels ) ) ) 

    # to define the total local region used when developing reconstructor
    # this is important for consistent flux normalization!
    local_region_pixels = np.array( list(outside_pixels) + list(pupil_pixels)  )# could also do what we commented out above
    
    # reduction data used in reconstructor
    darkss = config_fits['DARK'].data 
    dark = darkss[0] # chose which one
    bad_pixels = config_fits['BAD_PIXELS'].data
    if 0 in config_fits['BAD_PIXELS'].data:
        bad_pixels = bad_pixels[1:] # first frame is tag for frame count - make sure it is not masked 

    # camera settings 
    cam_settings_tmp.det_fps = det_fps
    cam_settings_tmp.det_dit = det_dit
    cam_settings_tmp.det_gain = det_gain
    cam_settings_tmp.det_cropping_rows = det_cropping_rows
    cam_settings_tmp.det_cropping_cols = det_cropping_cols 
    cam_settings_tmp.dark = dark.reshape(-1) 
    cam_settings_tmp.bad_pixels = bad_pixels
    cam_settings_tmp.det_tag_enabled = True
    cam_settings_tmp.det_crop_enabled = False # True <- always false unless latency tests etc 


    # reconstructor data 

    R_TT = config_fits['R_TT'].data.astype(np.float32) #tip-tilt reconstructor

    R_HO = config_fits['R_HO'].data.astype(np.float32) #higher-oder reconstructor

    IM = config_fits['IM'].data.astype(np.float32) # interaction matrix (unfiltered)

    M2C = config_fits['M2C_4RECO'].data.astype(np.float32) # mode to command matrix normalized to poke amplitude used in IM construction

    I2M = np.transpose( config_fits['I2M'].data).astype(np.float32) # intensity (signal) to mode matrix  
    # (# transposed so we can multiply directly I2M @ signal)
        
    CM = config_fits['CM'].data.astype(np.float32)  # full control matrix 

    I0 = config_fits['I0'].data.astype(np.float32) # calibration source reference intensity (FPM IN)

    N0 = config_fits['N0'].data.astype(np.float32) # calibration source reference intensity (FPM OUT)

    #############
    dm_flat =  config_fits['FLAT_DM'].data.astype(np.float32)
    ##########

    reconstructors_tmp.IM.update(IM.reshape(-1))
    reconstructors_tmp.CM.update(CM.reshape(-1))
    reconstructors_tmp.R_TT.update(R_TT.reshape(-1))
    reconstructors_tmp.R_HO.update(R_HO.reshape(-1))
    reconstructors_tmp.M2C.update(M2C.reshape(-1))
    reconstructors_tmp.I2M.update(I2M.reshape(-1))
    reconstructors_tmp.I0.update(I0.reshape(-1)/np.mean( I0.reshape(-1) )) #normalized
    reconstructors_tmp.N0.update(N0.reshape(-1)/np.mean( N0.reshape(-1) )) #normalized # N0.reshape(-1)/np.mean( I0.reshape(-1)[pupil_pixels] )
    reconstructors_tmp.flux_norm.update(np.mean( I0.reshape(-1) ))   #normalized over mean of whole sub-region

    # COMMIT IT ALL 
    reconstructors_tmp.commit_all()
    # -----------------------------

    pupil_regions_tmp.pupil_pixels.update( pupil_pixels )
    pupil_regions_tmp.secondary_pixels.update( secondary_pixels ) 
    pupil_regions_tmp.outside_pixels.update( outside_pixels )
    pupil_regions_tmp.local_region_pixels.update( local_region_pixels )

    #filter(lambda a: not a.startswith('__'), dir(pupil_regions_tmp))

    # COMMIT IT ALL 
    pupil_regions_tmp.commit_all()
    # -----------------------------


    # pid and leaky integator

    Nmodes = I2M.shape[0] #M2C.shape[1]
    kp = np.zeros(Nmodes) # init all to zero to start 
    ki = np.zeros(Nmodes)
    kd = np.zeros(Nmodes)
    lower_limit = -100 * np.ones(Nmodes)
    upper_limit = 100 * np.ones(Nmodes)
    pid_setpoint =  np.zeros(Nmodes)

    pid_tmp = rtc.PIDController( kp, ki, kd, lower_limit, upper_limit , pid_setpoint)
    leaky_tmp = rtc.LeakyIntegrator( ki, lower_limit, upper_limit ) 

    # Append all classes and structures to our rtc object to put them in C
    r.pid = pid_tmp
    r.LeakyInt = leaky_tmp
    r.regions = pupil_regions_tmp
    r.reco = reconstructors_tmp
    r.camera_settings = cam_settings_tmp

    r.dm_disturb = np.zeros( 140 ) # no disturbance 
    r.dm_flat = dm_flat 
    # now set up the camera as required 
    r.apply_camera_settings()

    #return( r )



# update for testing 

# 30
# for iterations below using: 
#data_path = '/home/heimdallr/Documents/asgard-alignment/tmp/30-08-2024/iter_3_J3/fourier_20modes_map_reconstructor/' #'/home/heimdallr/Documents/asgard-alignment/tmp/30-08-2024/iter_1_J3/fourier_20modes_map_reconstructor/'#'/home/heimdallr/Documents/asgard-alignment/tmp/29-08-2024/iter_14_J3/fourier_20modes_map_reconstructor/' #'~/Documents/asgard-alignment/tmp/29-08-2024/iter_13_J3/'
#reconstructor_file = data_path + 'RECONSTRUCTORS_fourier_0.2pokeamp_in-out_pokes_map_DIT-0.002_gain_high_30-08-2024T09.43.22.fits' #'RECONSTRUCTORS_fourier_0.2pokeamp_in-out_pokes_map_DIT-0.001_gain_high_30-08-2024T07.51.19.fits' #'RECONSTRUCTORS_fourier_0.2pokeamp_in-out_pokes_map_DIT-0.001_gain_high_29-08-2024T23.48.48.fits' #'RECONSTRUCTORS_fourier_0.2pokeamp_in-out_pokes_map_DIT-0.001_gain_high_29-08-2024T22.59.26.fits'

# it 5 : with ki[0] =0.1, kp[0]=1
# it 6 : same but longer , telemetry doesnlt seem to clear or change 
# it 7 : adding kp[1] = 1 , worked out need to exit / enter every session 
# it 8 : ki[1] = 0.1
# it 9 : increasing ki = 0.5 for both 
# it 10 : starting 1 of the higher order modes with leaky int, ki_leak[2] = 0.1
# it 11 : realised HO terms commented out in RTC. Includde them and reset ki_leak = 0 to check 
# it 12 : ok now try ki_leaky[2] = 0.1 again 
# it 13: semi worked but unstable . added send_dm_cmd and close all. So build basis and add static offset on DM (need to add distubance in rtc to keep it there)
#    also for reference merged utilities from asgard alignment project 
# it 14: added dm_distrub vector (nanobinded) default to zero. So we can add DM disturbance. Also prior send_dm_cmd and close_all seem to work fine  
# it 15 . adding TT disturbance 
# it 16 . Add change disturbance after 100 telemetry entries
# it 17 : make disturb bigger! 
# it 18 ; fresh reco calibrator
it = 18

# basis to add aberrations 
basis =  util.construct_command_basis( basis='fourier_pinned_edges', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
# flatten DM first 
r.send_dm_cmd( dm_flat ) 

r.dm_disturb = 0 * 0.3 * basis.T[0]  # add a tip disturb 

max_mode = 3

Nmodes = I2M.shape[0] #M2C.shape[1]
kp = np.zeros(Nmodes)
ki = np.zeros(Nmodes)
ki_leak = np.zeros(Nmodes)
kd = np.zeros(Nmodes)




################
# tip 
kp[0] = 1
ki[0] = 0.5
# tilt 
kp[1] = 1
ki[1] = 0.5

# lets try leaky on a higher order mode 
for i in range(2,max_mode):
    ki_leak[i] = 0.1
################

lower_limit = -100 * np.ones(Nmodes)
upper_limit = 100 * np.ones(Nmodes)
pid_setpoint =  np.zeros(Nmodes)

pid_tmp = rtc.PIDController( kp, ki, kd, lower_limit, upper_limit , pid_setpoint)
leaky_tmp = rtc.LeakyIntegrator( ki_leak, lower_limit, upper_limit ) 

# Append all classes and structures to our rtc object to put them in C
r.pid = pid_tmp
r.LeakyInt = leaky_tmp



# start it 
no_tele = 1000
disturb_after = 100
r.enable_telemetry(1000)
# start a runner that calls latency function 
runner = rtc.AsyncRunner(r, period = timedelta(microseconds=1000))
runner.start()

while r.telemetry_cnt > 900: #no_tele-distub_after:
    print(r.telemetry_cnt)
runner.pause()

r.dm_disturb = 0.5 * basis.T[0]  # add a tip disturb 

time.sleep(0.2)
runner.resume()
while r.telemetry_cnt > 0:
    print(r.telemetry_cnt)

runner.pause()
# now add a disturbance
runner.stop()


# read out the telemetry 
t = rtc.get_telemetry()
tel_rawimg = np.array([tt.image_in_pupil for tt in t] ) # reduced image filtered in pupil (not signal)
#tel_imgErr = np.array([tt.image_err_signal for tt in t])
tel_modeErr = np.array([tt.mode_err for tt in t])
tel_reco = np.array([tt.dm_cmd_err for tt in t])


#runner.flush # not writing anything here yet, but could 
#rtc.clear_telemetry() 

# reconstruct 2D image from pupil filtered img

pupil_img_2D = []
for img_tmp in tel_rawimg:
    tmp = np.zeros( I0.shape )
    tmp.reshape(-1)[pupil_pixels_local] =  img_tmp 
    pupil_img_2D.append( tmp )


im_list = [] #[ pupil_img_2D]
title_list = [] #['initial','final']
xlabel_list = [] #[None, None]
ylabel_list = []# [None, None]
cbar_label_list = []#['DM units', 'DM units' ] 

look_at_it =[0] + [disturb_after-i for i in np.arange(-4,8,2)[::-1] ] + [-10,-1]
for i in look_at_it:
    im_list.append( pupil_img_2D[i] )
    title_list.append( f'iteration {i}' ) 
    xlabel_list.append( None ) 
    ylabel_list.append( None )
    cbar_label_list.append( 'DM units' ) 
savefig = data_path + f'rtc_RED_IMGS_it{it}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'
util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)


# save telemetry 
telemetry_fits = fits.HDUList( [] )
for tel, lab in zip ( [tel_rawimg, pupil_img_2D, tel_modeErr, tel_reco, r.pid.kp, r.pid.ki, r.LeakyInt.rho, r.dm_disturb], ['reduced_img_pupil', 'reduced_img_pupil_2D','mode_err', 'dm_cmd',  'r.pid.kp', 'r.pid.ki', 'r.LeakyInt.rho','r.dm_disturb']):

    frame_fits = fits.PrimaryHDU( tel ) 
    frame_fits.header.set('EXTNAME',f'{lab}')
    
    telemetry_fits.append( frame_fits )
telemetry_fits.writeto( data_path + f'telemetry_it{it}.fits' ) #, overwrite = True) # don't by default, overwrite = True)


plt.figure()
for m in range(2):
    plt.plot( tel_modeErr.T[m], label=f'mode {m}' )
plt.xlabel('iterations')
plt.ylabel('mode error amplitude')
plt.legend()
savefig = data_path + f'rtc_TT_ONLY_test_mode_err_it{it}.png'
plt.savefig( savefig , bbox_inches ='tight')


plt.figure()
for m in range(max_mode):
    plt.plot( tel_modeErr.T[m], alpha=0.3, label=f'mode {m}' )
plt.xlabel('iterations')
plt.ylabel('mode error amplitude')
plt.legend()
savefig = data_path + f'rtc_HO_test_mode_err_it{it}.png'
plt.savefig( savefig , bbox_inches ='tight')


im_list = [ util.get_DM_command_in_2D(tel_reco[0]-dm_flat), util.get_DM_command_in_2D(tel_reco[-1]-dm_flat)]
title_list = ['initial','final']
xlabel_list = [None, None]
ylabel_list = [None, None]
cbar_label_list = ['DM units', 'DM units' ] 
savefig = data_path + f'rtc_TT_test_cmds_inital-final_it{it}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'
util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)





# r.close_all() # to close/disconnect DM and camera safely  





#---- TESTS 

# coerrect mapping to global pixels works correctly (saves output in data input folder)
# get a full frame and reshape it 


for lab, reg in zip(['pupil_pixels', 'outside_pixels'],[pupil_pixels, outside_pixels]):
    pupil_reg_test = np.zeros(np.array( r.im2vec_test()).shape )
    #outside_reg_test = np.zeros(np.array( r.im2vec_test()).shape )
    pupil_reg_test[reg] = 1

    plt.figure() ; plt.imshow( pupil_reg_test.reshape(r.camera_settings.image_height, r.camera_settings.image_width)) ; plt.savefig(data_path+f'full_frame_{lab}_registration.png')


# checking reduced image 
img = np.array(r.reduceImg_test()).reshape( 512, 640 )
pupimg = pupil_reg_test.reshape(r.camera_settings.image_height, r.camera_settings.image_width)
fig,ax = plt.subplots(1,2) 
ax[0].imshow( img[ 100:300, 100:300] ) ; plt.savefig(data_path + 'delme.png')
ax[1].imshow( pupimg[ 100:300, 100:300] ) 
plt.savefig(data_path+f'full_frame_{lab}_registration_with_image_in_rtc.png')


# polling image and convert to vector 
test1 = r.im2vec_test()
if(len(test1) == r.camera_settings.full_image_length):
    print( ' passed im2vec_test')
else:
    print( ' FAILED --- im2vec_test')

# polling and reducing using the camera settings dark and bad pixels 

test1 = r.reduceImg_test()
if(len(test1) == r.camera_settings.full_image_length):
    print( ' passed reduceImg test')
else:
    print( ' FAILED --- reduceImg test')


# polling image and convert to vector and filter for pupil pixels 
test2 = r.im2filtered_im_test()
if(len(test2 ) == len(r.regions.pupil_pixels.current)):
    print( ' passed im2filtered_im_test')
else:
    print( ' FAILED --- im2filtered_im_test')

# filter reference intensity setpoint for pupil pixels 
test3 = r.im2filteredref_im_test()
if(len(test3 ) == len(r.regions.pupil_pixels.current)):
    print( ' passed test3 .im2filteredref_im_test()')
else:
    print( ' FAILED --- test3 .im2filteredref_im_test()')

# process image 
test4 = r.process_im_test()
if(len(test4 ) == len(r.regions.pupil_pixels.current)):
    print( ' passed test4 .process_im_test()')
else:
    print( ' FAILED --- test4 .process_im_test()')




