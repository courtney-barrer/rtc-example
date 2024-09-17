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

data_path = '/home/heimdallr/Documents/asgard-alignment/tmp/16-09-2024/iter_2_J3/'#'/home/heimdallr/Documents/asgard-alignment/tmp/15-09-2024/iter_5_J3/zonal_reconstructor/' #'/home/heimdallr/Documents/asgard-alignment/tmp/15-09-2024/iter_5_J3/zonal_reconstructor/'
#'/home/heimdallr/Documents/asgard-alignment/tmp/15-09-2024/iter5_J1/' # # 30-08-2024/iter_4_J3/fourier_20modes_map_reconstructor/' #'/home/heimdallr/Documents/asgard-alignment/tmp/30-08-2024/iter_3_J3/fourier_20modes_map_reconstructor/' #'/home/heimdallr/Documents/asgard-alignment/tmp/30-08-2024/iter_1_J3/fourier_20modes_map_reconstructor/'#'/home/heimdallr/Documents/asgard-alignment/tmp/29-08-2024/iter_14_J3/fourier_20modes_map_reconstructor/' #'~/Documents/asgard-alignment/tmp/29-08-2024/iter_13_J3/'

reconstructor_file = '/home/heimdallr/Documents/asgard-alignment/tmp/17-09-2024/iter_11_J3/fourier_90modes_map_reconstructor/RECONSTRUCTORS_fourier90_0.2pokeamp_in-out_pokes_map_DIT-0.0049_gain_high_17-09-2024T23.20.25.fits'
#'/home/heimdallr/Documents/asgard-alignment/tmp/17-09-2024/iter_11_J3/fourier_90modes_map_reconstructor/RECONSTRUCTORS_fourier90_0.2pokeamp_in-out_pokes_MAP_DIT-0.0049_gain_high_17-09-2024T22.41.03.fits'
#'/home/heimdallr/Documents/asgard-alignment/tmp/17-09-2024/iter_11_J3/fourier_90modes_pinv_reconstructor/RECONSTRUCTORS_fourier90_0.2pokeamp_in-out_pokes_pinv_DIT-0.0049_gain_high_17-09-2024T20.08.46.fits'
# FOURIER 

# ZONAL 
#'/home/heimdallr/Documents/asgard-alignment/tmp/17-09-2024/iter_11_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.04pokeamp_in-out_pokes_map_DIT-0.0049_gain_high_17-09-2024T20.06.24.fits'
#'/home/heimdallr/Documents/asgard-alignment/tmp/17-09-2024/iter_11_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.04pokeamp_in-out_pokes_map_DIT-0.0049_gain_high_17-09-2024T18.24.28.fits'
#'/home/heimdallr/Documents/asgard-alignment/tmp/17-09-2024/iter_11_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.04pokeamp_in-out_pokes_map_DIT-0.0049_gain_high_17-09-2024T17.42.10.fits'
#'/home/heimdallr/Documents/asgard-alignment/tmp/17-09-2024/iter_11_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.04pokeamp_in-out_pokes_map_DIT-0.0049_gain_high_17-09-2024T15.54.12.fits'
#'/home/heimdallr/Documents/asgard-alignment/tmp/17-09-2024/iter_11_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.04pokeamp_in-out_pokes_map_DIT-0.0049_gain_high_17-09-2024T15.13.27.fits'
#'/home/heimdallr/Documents/asgard-alignment/tmp/17-09-2024/iter_10_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.04pokeamp_in-out_pokes_map_DIT-0.0049_gain_high_17-09-2024T14.08.44.fits'
#'/home/heimdallr/Documents/asgard-alignment/tmp/16-09-2024/iter_10_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.04pokeamp_in-out_pokes_map_DIT-0.005_gain_high_16-09-2024T20.45.13.fits'
#'/home/heimdallr/Documents/asgard-alignment/tmp/16-09-2024/iter_10_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.04pokeamp_in-out_pokes_map_DIT-0.001_gain_high_16-09-2024T19.42.55.fits'
#'/home/heimdallr/Documents/asgard-alignment/tmp/16-09-2024/iter_2_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.07pokeamp_in-out_pokes_map_DIT-0.004_gain_high_16-09-2024T16.28.21.fits'
#data_path + 'RECONSTRUCTORS_zonal_0.07pokeamp_in-out_pokes_map_DIT-0.001_gain_high_15-09-2024T20.27.47.fits'


#def init_rtc( reco_fits_file ):
if 1: 
    # ============== READ IN PUPIL PHASE RECONSTRUCTOR DATA
    config_fits = fits.open( reconstructor_file ) 

    print( [reco.header['EXTNAME'] for reco in config_fits])

    # our nanobinded rtc object
    r = rtc.RTC()

    # init struct's
    cam_settings_tmp = rtc.camera_settings_struct()
    reconstructors_tmp = rtc.phase_reconstuctor_struct()
    pupil_regions_tmp = rtc.pupil_regions_struct()


    # camera settings used to build reconstructor 

    det_fps = float( config_fits['info'].header['camera_fps'] ) # frames per sec (Hz) 

    det_dit = float( config_fits['info'].header['camera_tint'] ) # integration time (seconds)

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
    #local_region_pixels_local = np.arange(0, len(config_fits['DARK'].data[0].reshape(-1)) ) #np.array( config_fits['outside_pixels'].data, dtype=np.int32)

    # converting back to the global image (in the reconstructor we likely (to reduce data size) looked at only a sub-region and referenced pixels from there)
    pupil_pixels = convert_local_to_global_coordinates(pupil_pixels_local, r1_subregion, \
    c1_subregion, r2_subregion-r1_subregion, c2_subregion-c1_subregion, global_shape, flatten = True)
    secondary_pixels = convert_local_to_global_coordinates(secondary_pixels_local, r1_subregion, \
    c1_subregion, r2_subregion-r1_subregion, c2_subregion-c1_subregion, global_shape, flatten = True)
    #outside_pixels_0 = convert_local_to_global_coordinates(outside_pixels_local, r1_subregion, \
    # c1_subregion, r2_subregion-r1_subregion, c2_subregion-c1_subregion, global_shape, flatten = True)
    #local_region_pixels = convert_local_to_global_coordinates(local_region_pixels_local, r1_subregion, \
    # c1_subregion, r2_subregion-r1_subregion, c2_subregion-c1_subregion, global_shape, flatten = True)

    # outside_pixels_0 is only outside in local region, defauls to 0 outside of this region so to correct this
    all_pixels = np.arange(0, img_height * img_width )
    outside_pixels = np.array( list( set( all_pixels ) - set( pupil_pixels ) ) ) 

    # to define the total local region used when developing reconstructor
    # this is important for consistent flux normalization!
    local_region_pixels = np.array( list(all_pixels) + list(pupil_pixels) ) #- set( outside_pixels_0)) ) #np.array( list(outside_pixels) + list(pupil_pixels) )# could also do what we commented out above

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
    
    # use zonal to build TT 
    tmp_config = fits.open( '/home/heimdallr/Documents/asgard-alignment/tmp/17-09-2024/iter_11_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.04pokeamp_in-out_pokes_map_DIT-0.0049_gain_high_17-09-2024T23.18.03.fits' )
    #'/home/heimdallr/Documents/asgard-alignment/tmp/17-09-2024/iter_11_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.04pokeamp_in-out_pokes_map_DIT-0.0049_gain_high_17-09-2024T22.31.57.fits' )
    #'/home/heimdallr/Documents/asgard-alignment/tmp/17-09-2024/iter_11_J3/zonal_reconstructor/RECONSTRUCTORS_zonal_0.04pokeamp_in-out_pokes_map_DIT-0.0049_gain_high_17-09-2024T20.06.24.fits')
    M2C = tmp_config['M2C_4RECO'].data.astype(np.float32) # mode to command matrix normalized to poke amplitude used in IM construction
    I2M = np.transpose( tmp_config['I2M'].data ).astype(np.float32)
    IM = tmp_config['IM'].data.astype(np.float32)
    M2C_0 = M2C.T
    U, S, Vt = np.linalg.svd( IM.T, full_matrices=False)

    Smax = 40
    R = (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)]) @ U.T

    TT_vectors = util.get_tip_tilt_vectors()

    TT_space = M2C_0 @ TT_vectors

    U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

    I2M_TT = U_TT.T @ R 

    M2C_TT = M2C_0.T @ U_TT # since pinned need M2C to go back to 140 dimension vector 
    

    # ====================== THEN FOURIER FOR HO

    R_TT = config_fits['R_TT'].data.astype(np.float32) #tip-tilt reconstructor

    R_HO = config_fits['R_HO'].data.astype(np.float32) #higher-oder reconstructor

    IM = config_fits['IM'].data.astype(np.float32) # interaction matrix (unfiltered)

    M2C = config_fits['M2C_4RECO'].data.astype(np.float32) # mode to command matrix normalized to poke amplitude used in IM construction

    I2M = np.transpose( config_fits['I2M'].data ).astype(np.float32) # intensity (signal) to mode matrix 
    # (# transposed so we can multiply directly I2M @ signal)

    CM = config_fits['CM'].data.astype(np.float32) # full control matrix 

    I0 = config_fits['I0'].data.astype(np.float32) # calibration source reference intensity (FPM IN)

    N0 = config_fits['N0'].data.astype(np.float32) # calibration source reference intensity (FPM OUT)


    """


    # fourier projections - doesn't project great onto tip tilt 
    M2C_0 = M2C.T
    TT_vectors = util.get_tip_tilt_vectors()
    TT_space = M2C_0 @ TT_vectors # TT space in naitive basis (Fourier here)
    
    U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

    I2M_TT = U_TT.T @ I2M

    M2C_TT = M2C_0.T @ U_TT

    I2M_HO = (np.eye(U_TT.shape[0]) - U_TT @ U_TT.T) @ I2M

    M2C_HO = M2C_0.T #@ I2M_HO
    """

    
    # fourier simply assuming index 0, 1 are essentially TT 
    M2C_0 = M2C.T
    #I2M_TT = I2M[:2,:]
    I2M_HO = I2M[2:,:]
    #M2C_TT = M2C_0[:2,:].T
    M2C_HO = M2C_0[2:,:].T
    # issue here is that tip+tilt isnt right 
    



    """
    #using actuator push/pull 

    M2C_0 = M2C.T
    U, S, Vt = np.linalg.svd( IM.T, full_matrices=False)

    Smax = 40
    R = (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)]) @ U.T

    TT_vectors = util.get_tip_tilt_vectors()

    TT_space = M2C_0 @ TT_vectors

    U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

    I2M_TT = U_TT.T @ R 

    M2C_TT = M2C_0.T @ U_TT # since pinned need M2C to go back to 140 dimension vector 

    R_HO = (np.eye(U_TT.shape[0]) - U_TT @ U_TT.T) @ R

    # go to Eigenmodes for modal control in higher order reconstructor
    U_HO, S_HO, Vt_HO = np.linalg.svd( R_HO, full_matrices=False)
    I2M_HO = Vt_HO 
    M2C_HO = M2C_0.T @ (U_HO * S_HO) # since pinned need M2C to go back to 140 dimension vector
    
    """

    #############
    dm_flat = config_fits['FLAT_DM'].data.astype(np.float32)
    ##########

    reconstructors_tmp.IM.update(IM.reshape(-1))
    reconstructors_tmp.CM.update(CM.reshape(-1))
    reconstructors_tmp.R_TT.update(R_TT.reshape(-1))
    reconstructors_tmp.R_HO.update(R_HO.reshape(-1))
    reconstructors_tmp.M2C.update(M2C.reshape(-1))
    reconstructors_tmp.I2M.update(I2M.reshape(-1))

    reconstructors_tmp.I2M_TT.update(I2M_TT.reshape(-1))
    reconstructors_tmp.I2M_HO.update(I2M_HO.reshape(-1))
    reconstructors_tmp.M2C_TT.update(M2C_TT.reshape(-1))
    reconstructors_tmp.M2C_HO.update(M2C_HO.reshape(-1))


    reconstructors_tmp.I0.update(I0.reshape(-1)/np.mean( I0.reshape(-1) )) #normalized
    reconstructors_tmp.N0.update(N0.reshape(-1)/np.mean( N0.reshape(-1) )) #normalized # N0.reshape(-1)/np.mean( I0.reshape(-1)[pupil_pixels] )
    reconstructors_tmp.flux_norm.update(np.mean( I0.reshape(-1) )) #normalized over mean of whole sub-region

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

    #Nmodes = I2M.shape[0] #M2C.shape[1]
    kp = np.zeros(I2M_TT.shape[0]) # init all to zero to start 
    ki = np.zeros(I2M_TT.shape[0])
    kd = np.zeros(I2M_TT.shape[0])
    lower_limit = -100 * np.ones(I2M_TT.shape[0])
    upper_limit = 100 * np.ones(I2M_TT.shape[0])
    pid_setpoint = np.zeros(I2M_TT.shape[0])

    pid_tmp = rtc.PIDController( kp, ki, kd, lower_limit, upper_limit , pid_setpoint)

    rho_leak = np.zeros( I2M_HO.shape[0] )
    kp_leak = np.zeros( I2M_HO.shape[0] )
    lower_limit_leak = -100 * np.ones( I2M_HO.shape[0] )
    upper_limit_leak = 100 * np.ones( I2M_HO.shape[0] )
    leaky_tmp = rtc.LeakyIntegrator( rho_leak, kp_leak, lower_limit_leak, upper_limit_leak ) 

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




# quick check 
plt.figure(); plt.plot( (M2C_HO @ I2M_HO @ IM[65]) ); plt.savefig( 'data/17-09-2024FINAL/delme.png')
## Projecting TT vectors onto pixel space with our calibrated TT and HO matricies
# (to check if they work...)
plt.figure(); plt.imshow( util.get_DM_command_in_2D( M2C_HO @ U[0] ) ); plt.savefig( 'data/17-09-2024FINAL/delme.png')

# check tip/tilt modes on DM 
plt.figure(); plt.imshow( util.get_DM_command_in_2D( M2C_TT[:,0] ) ); plt.savefig( 'data/17-09-2024FINAL/delme.png')

i = 2

basis = util.construct_command_basis( basis='fourier_pinned_edges', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
#basis = np.eye(140)
TTonHO_img_2D = np.zeros( I0.shape )
TTonHO_img_2D.reshape(-1)[pupil_pixels_local] =  (I2M_HO.T @ (M2C_HO.T @ basis))[:,i]

TTonTT_img_2D  = np.zeros( I0.shape )
TTonTT_img_2D.reshape(-1)[pupil_pixels_local] =  (I2M_TT.T @ (M2C_TT.T @ basis))[:,i]

fig,ax = plt.subplots( 3,1,figsize=(10,5))

im_list = [ util.get_DM_command_in_2D( basis[:,i]),  TTonTT_img_2D,  TTonHO_img_2D  ]
xlabel_list = ['' for _ in im_list] 
ylabel_list = ['' for _ in im_list]
title_list = ['DM COMMAND', 'RECO TT IMG', 'RECO HO IMG']
cbar_label_list = ['DM UNITS', 'ADU', 'ADU']
savefig = 'data/16-09-2024FINAL/delme.png'
util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, \
                           cbar_label_list, fontsize=15, axis_off=True, \
                            cbar_orientation = 'bottom', savefig=savefig)





#####################
## OPENLOOP 
lab = 'OPENLOOP'
rtc.clear_telemetry()

# DISTURBANCE 
basis = util.construct_command_basis( basis='fourier_pinned_edges', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
r.dm_disturb = 0 * basis.T[1] # add a tip disturb 

kpTT = 0
kiTT = 0
rhoHO = 0 
kpHO = 0
no_HO_modes = 0

no_tele = 2000
r.enable_telemetry(2000)


explabel = f'OPENLOOP_HOmodes-{no_HO_modes}_kpTT-{kpTT}_kiTT-{kiTT}_rhoHO-{rhoHO}_kpHO-{kpHO}_impulse'
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

current_path = f'data/{tstamp.split("T")[0]}FINAL/{lab}/{explabel}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 

if not os.path.exists(current_path):
    os.makedirs(current_path)

# CONTROLLERS 
r.pid.kp = [0, 0]
r.pid.ki = [0, 0]


r.send_dm_cmd( dm_flat + r.dm_disturb )
time.sleep( 1 )

# start a runner that calls latency function 
runner = rtc.AsyncRunner(r, period = timedelta(microseconds=1000))
runner.start()
while r.telemetry_cnt > 0:
    continue #print( r.telemetry_cnt )
runner.pause()
runner.stop()

# read out the telemetry 
t = rtc.get_telemetry()
telem_dict = {
 "im_err" : np.array([tt.image_in_pupil for tt in t] ) ,
 "e_TT" : np.array([tt.e_TT for tt in t]),
 "e_HO" : np.array([tt.e_HO for tt in t]),
 "u_TT" : np.array([tt.u_TT for tt in t]),
 "u_HO" : np.array([tt.u_HO for tt in t]),
 "cmd_TT" : np.array([tt.cmd_TT for tt in t]),
 "cmd_HO" : np.array([tt.cmd_HO for tt in t]),
 "dm_disturb" : np.array([tt.dm_disturb for tt in t]),
 "t0" : np.array([tt.t0 for tt in t]),
 "t1" : np.array([tt.t1 for tt in t]),
 "I0" : I0,
 "pupil_pixels" : r.regions.pupil_pixels.current, #in global frame
 "local_pupil_pixels" : pupil_pixels_local, #in local frame 
 "pid.kp": r.pid.kp,
 "pid.ki": r.pid.ki,
 "pid.kd": r.pid.kd,
 "leak.kp": r.LeakyInt.kp,
 "leak.rho": r.LeakyInt.rho,
 "IM":IM
}

# reconstruct the error signal in the cropped pupil region 
pupil_img_2D = []
for img_tmp in telem_dict['im_err']:
    tmp = np.zeros( I0.shape )
    tmp.reshape(-1)[pupil_pixels_local] = img_tmp 
    pupil_img_2D.append( tmp )

telem_dict['signal_2D'] = pupil_img_2D



# Create a list of HDUs (Header Data Units)
hdul = fits.HDUList()

# Add each list to the HDU list as a new extension
for list_name, data_list in telem_dict.items():
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float) # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    hdul.append(hdu)

# Write the HDU list to a FITS file
hdul.writeto(current_path + f'{explabel}_{tstamp}.fits', overwrite=True)



# plot telemetry 
# plot telemetry 
fig, ax = plt.subplots(5,1,figsize=(10,20))

cmd_err = telem_dict['cmd_TT'] + telem_dict['cmd_HO'] + telem_dict['dm_disturb']
cmd_rmse = np.sqrt( np.mean( cmd_err**2, axis=1 ) )

ax[0].plot( telem_dict['im_err'] )
ax[0].set_ylabel(r'$\Delta I$')
ax[1].plot( cmd_err )
ax[1].set_ylabel(r'$\Delta C$')
ax[2].plot( telem_dict['e_TT'] )
ax[2].set_ylabel(r'$e_{TT}$')
ax[3].plot( telem_dict['e_HO'] )
ax[3].set_ylabel(r'$e_{HO}$')
ax[4].plot( np.sqrt( np.mean( cmd_err**2, axis=1 ) ) )
ax[4].set_ylabel(r'RMSE')

plt.savefig(current_path + f'telemetry_summary_{tstamp}.png')


plt.figure()
fig,ax = plt.subplots( 1,2)
ax[0].set_title('intial image')
ax[1].set_title('final image')
ax[0].imshow( telem_dict['signal_2D'][0] )
ax[1].imshow( telem_dict['signal_2D'][-1] )
#plt.savefig(fig_path + 'delme.png')
plt.savefig(current_path + f'initial_v_final_image_{explabel}_{tstamp}.png')
# write telemetry to file 


im_list = [ util.get_DM_command_in_2D(telem_dict['cmd_TT'][-1] + telem_dict['cmd_HO'][-1]), util.get_DM_command_in_2D(telem_dict['dm_disturb'][-1])]
title_list = ['final\nreconstruction','final\ndisturbance']
xlabel_list = [None, None]
ylabel_list = [None, None]
cbar_label_list = ['DM units', 'DM units' ] 
savefig = current_path + f'dm_final_disturb_v_reco_{explabel}_{tstamp}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'
util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)






#####################
## CLOSING TT ON NOTHING
lab = 'CLOSING_TT_ON_NOTHING'
for kpTT in np.linspace(0.2,1.5,6):
    for kiTT in np.linspace(0,0.99,5): 
        print( f'\n\nki = {kiTT}\n\nkp = {kpTT}\n\n')

        # start it 
        rtc.clear_telemetry()

        no_tele = 200
        close_after = 20
        r.enable_telemetry(200)

        #kpTT = 10.8
        #kiTT = 0.9
        rhoHO = 0 
        kpHO = 0
        no_HO_modes = 0

        explabel = f'close_after{close_after}_HOmodes-{no_HO_modes}_kpTT-{kpTT}_kiTT-{kiTT}_rhoHO-{rhoHO}_kpHO-{kpHO}_impulse'
        tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

        current_path = f'data/{tstamp.split("T")[0]}FINAL/{lab}/{explabel}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 

        if not os.path.exists(current_path):
            os.makedirs(current_path)

        # CONTROLLERS 
        r.pid.kp = [0, 0]
        r.pid.ki = [0, 0]

        # DISTURBANCE 
        basis = util.construct_command_basis( basis='fourier_pinned_edges', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)

        r.dm_disturb = 0 * basis.T[1] # add a tip disturb 
        r.send_dm_cmd( dm_flat + r.dm_disturb )
        time.sleep( 1 )

        # start a runner that calls latency function 
        runner = rtc.AsyncRunner(r, period = timedelta(microseconds=1000))
        runner.start()
        while r.telemetry_cnt > no_tele - close_after:
            print( r.telemetry_cnt )
        runner.pause()

        # CONTROLLERS 
        r.pid.kp = kpTT * np.ones( 2 ) 
        r.pid.ki = kiTT * np.ones( 2 )
        time.sleep(0.1)
        runner.start()

        while r.telemetry_cnt > 0:
            continue
        runner.pause()
        runner.stop()

        # read out the telemetry 
        t = rtc.get_telemetry()
        telem_dict = {
        "im_err" : np.array([tt.image_in_pupil for tt in t] ) ,
        "e_TT" : np.array([tt.e_TT for tt in t]),
        "e_HO" : np.array([tt.e_HO for tt in t]),
        "u_TT" : np.array([tt.u_TT for tt in t]),
        "u_HO" : np.array([tt.u_HO for tt in t]),
        "cmd_TT" : np.array([tt.cmd_TT for tt in t]),
        "cmd_HO" : np.array([tt.cmd_HO for tt in t]),
        "dm_disturb" : np.array([tt.dm_disturb for tt in t]),
        "t0" : np.array([tt.t0 for tt in t]),
        "t1" : np.array([tt.t1 for tt in t]),
        "I0" : I0,
        "pupil_pixels" : r.regions.pupil_pixels.current, #in global frame
        "local_pupil_pixels" : pupil_pixels_local, #in local frame 
        "pid.kp": r.pid.kp,
        "pid.ki": r.pid.ki,
        "pid.kd": r.pid.kd,
        "leak.kp": r.LeakyInt.kp,
        "leak.rho": r.LeakyInt.rho,
        "IM":IM
        }

        # reconstruct the error signal in the cropped pupil region 
        pupil_img_2D = []
        for img_tmp in telem_dict['im_err']:
        tmp = np.zeros( I0.shape )
        tmp.reshape(-1)[pupil_pixels_local] = img_tmp 
        pupil_img_2D.append( tmp )

        telem_dict['signal_2D'] = pupil_img_2D



        # Create a list of HDUs (Header Data Units)
        hdul = fits.HDUList()

        # Add each list to the HDU list as a new extension
        for list_name, data_list in telem_dict.items():
        # Convert list to numpy array for FITS compatibility
        data_array = np.array(data_list, dtype=float) # Ensure it is a float array or any appropriate type

        # Create a new ImageHDU with the data
        hdu = fits.ImageHDU(data_array)

        # Set the EXTNAME header to the variable name
        hdu.header['EXTNAME'] = list_name

        # Append the HDU to the HDU list
        hdul.append(hdu)

        # Write the HDU list to a FITS file
        hdul.writeto(current_path + f'{explabel}_{tstamp}.fits', overwrite=True)



        # plot telemetry 
        fig, ax = plt.subplots(4,1,figsize=(10,20))

        cmd_err = telem_dict['cmd_TT'] + telem_dict['cmd_HO'] - telem_dict['dm_disturb']
        ax[0].plot( cmd_err )
        ax[0].set_ylabel(r'$\Delta C$')
        ax[1].plot( telem_dict['e_TT'] )
        ax[1].set_ylabel(r'$e_{TT}$')
        ax[2].plot( telem_dict['e_HO'] )
        ax[2].set_ylabel(r'$e_{HO}$')
        ax[3].plot( np.sqrt( np.mean( cmd_err**2, axis=1 ) ) )
        ax[3].set_ylabel(r'RMSE')

        plt.savefig(current_path + f'telemetry_summary_{tstamp}.png')



        plt.figure()
        fig,ax = plt.subplots( 1,2)
        ax[0].set_title('intial image')
        ax[1].set_title('final image')
        ax[0].imshow( telem_dict['signal_2D'][0] )
        ax[1].imshow( telem_dict['signal_2D'][-1] )
        #plt.savefig(fig_path + 'delme.png')
        plt.savefig(current_path + f'initial_v_final_image_{explabel}_{tstamp}.png')
        # write telemetry to file 


        im_list = [ util.get_DM_command_in_2D(telem_dict['cmd_TT'][-1] + telem_dict['cmd_HO'][-1]), util.get_DM_command_in_2D(telem_dict['dm_disturb'][-1])]
        title_list = ['final\nreconstruction','final\ndisturbance']
        xlabel_list = [None, None]
        ylabel_list = [None, None]
        cbar_label_list = ['DM units', 'DM units' ] 
        savefig = current_path + f'dm_final_disturb_v_reco_{explabel}_{tstamp}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'
        util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)




#####################
## CLOSING TT ON STATIC
lab = 'CLOSING_TT_ON_STATIC_TT_fourierbasis'

# optimal seems to be kpTT = 1, kiTT = 0.125

# DISTURBANCE 
basis = util.construct_command_basis( basis='fourier_pinned_edges', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
TT_vectors = util.get_tip_tilt_vectors()
r.dm_disturb = 0.4 * TT_vectors[:,1] #+ 0.6 * TT_vectors[:,1]# add a tip disturb 



"""

r1,r2 = 180,225
c1,c2 = 115,175
r.send_dm_cmd( dm_flat  )#+ r.dm_disturb )
time.sleep(1)
atest_img = np.array ( r.reduceImg_test() ).reshape( 512,640 )[r1:r2, c1:c2]
plt.figure(); plt.imshow(atest_img); plt.colorbar(); plt.savefig( '/home/heimdallr/Documents/rtc-example/data/17-09-2024FINAL/delme.png' )

"""

for kpTT in [1.0]:
    for kiTT in [0.1, 0.2,0.5,0.7,0.9]: 
        # start it 
        rtc.clear_telemetry()

        no_tele = 200
        close_after = 20
        r.enable_telemetry(200)

        #kpTT = 10.8
        #kiTT = 0.9
        rhoHO = 0 
        kpHO = 0
        no_HO_modes = 0

        explabel = f'f_Smax{Smax}_close_after{close_after}_HOmodes-{no_HO_modes}_kpTT-{kpTT}_kiTT-{kiTT}_rhoHO-{rhoHO}_kpHO-{kpHO}_impulse'
        tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

        current_path = f'data/{tstamp.split("T")[0]}FINAL/{lab}/{explabel}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 

        if not os.path.exists(current_path):
            os.makedirs(current_path)

        # CONTROLLERS 
        r.pid.reset()
        r.pid.kp = [0, 0]
        r.pid.ki = [0, 0]
        
        r.LeakyInt.reset()
        r.LeakyInt.kp = np.zeros( len( r.LeakyInt.kp ))
        r.LeakyInt.rho = np.zeros( len( r.LeakyInt.rho ))

        r.send_dm_cmd( dm_flat + r.dm_disturb )
        time.sleep( 1 )

        # start a runner that calls latency function 
        runner = rtc.AsyncRunner(r, period = timedelta(microseconds=1000))
        runner.start()
        while r.telemetry_cnt > no_tele - close_after:
            continue #print( r.telemetry_cnt )
        runner.pause()

        # CONTROLLERS 
        #r.pid.reset()
        r.pid.kp = kpTT * np.ones( 2 ) 
        r.pid.ki = kiTT * np.ones( 2 )
        time.sleep(0.1)
        runner.start()

        while r.telemetry_cnt > 0:
            print( r.telemetry_cnt )
        runner.pause()
        runner.stop()

        # read out the telemetry 
        t = rtc.get_telemetry()
        telem_dict = {
        "im_err" : np.array([tt.image_in_pupil for tt in t] ) ,
        "e_TT" : np.array([tt.e_TT for tt in t]),
        "e_HO" : np.array([tt.e_HO for tt in t]),
        "u_TT" : np.array([tt.u_TT for tt in t]),
        "u_HO" : np.array([tt.u_HO for tt in t]),
        "cmd_TT" : np.array([tt.cmd_TT for tt in t]),
        "cmd_HO" : np.array([tt.cmd_HO for tt in t]),
        "dm_disturb" : np.array([tt.dm_disturb for tt in t]),
        "t0" : np.array([tt.t0 for tt in t]),
        "t1" : np.array([tt.t1 for tt in t]),
        "I0" : I0,
        "pupil_pixels" : r.regions.pupil_pixels.current, #in global frame
        "local_pupil_pixels" : pupil_pixels_local, #in local frame 
        "pid.kp": r.pid.kp,
        "pid.ki": r.pid.ki,
        "pid.kd": r.pid.kd,
        "leak.kp": r.LeakyInt.kp,
        "leak.rho": r.LeakyInt.rho,
        "IM":IM
        }

        # reconstruct the error signal in the cropped pupil region 
        pupil_img_2D = []
        for img_tmp in telem_dict['im_err']:
            tmp = np.zeros( I0.shape )
            tmp.reshape(-1)[pupil_pixels_local] = img_tmp 
            pupil_img_2D.append( tmp )

        telem_dict['signal_2D'] = pupil_img_2D



        # Create a list of HDUs (Header Data Units)
        hdul = fits.HDUList()

        # Add each list to the HDU list as a new extension
        for list_name, data_list in telem_dict.items():
            # Convert list to numpy array for FITS compatibility
            data_array = np.array(data_list, dtype=float) # Ensure it is a float array or any appropriate type

            # Create a new ImageHDU with the data
            hdu = fits.ImageHDU(data_array)

            # Set the EXTNAME header to the variable name
            hdu.header['EXTNAME'] = list_name

            # Append the HDU to the HDU list
            hdul.append(hdu)

            # Write the HDU list to a FITS file
        hdul.writeto(current_path + f'{explabel}_{tstamp}.fits', overwrite=True)


        # plot telemetry 
        fig, ax = plt.subplots(5,1,figsize=(10,20))

        cmd_err = telem_dict['cmd_TT'] + telem_dict['cmd_HO'] - telem_dict['dm_disturb']
        ax[0].plot( telem_dict['im_err'] )
        ax[0].set_ylabel(r'$\Delta I$')
        ax[1].plot( cmd_err )
        ax[1].set_ylabel(r'$\Delta C$')
        ax[2].plot( telem_dict['e_TT'] )
        ax[2].set_ylabel(r'$e_{TT}$')
        ax[3].plot( telem_dict['e_HO'] )
        ax[3].set_ylabel(r'$e_{HO}$')
        ax[4].plot( np.sqrt( np.mean( cmd_err**2, axis=1 ) ) )
        ax[4].set_ylabel(r'RMSE')

        plt.savefig(current_path + f'telemetry_summary_{tstamp}.png')


        plt.figure()
        fig,ax = plt.subplots( 1,2)
        ax[0].set_title('intial image')
        ax[1].set_title('final image')
        ax[0].imshow( telem_dict['signal_2D'][0] )
        ax[1].imshow( telem_dict['signal_2D'][-1] )
        #plt.savefig(fig_path + 'delme.png')
        plt.savefig(current_path + f'initial_v_final_image_{explabel}_{tstamp}.png')
        # write telemetry to file 


        im_list = [ util.get_DM_command_in_2D(telem_dict['cmd_TT'][-1] + telem_dict['cmd_HO'][-1]), util.get_DM_command_in_2D(telem_dict['dm_disturb'][-1])]
        title_list = ['final\nreconstruction','final\ndisturbance']
        xlabel_list = [None, None]
        ylabel_list = [None, None]
        cbar_label_list = ['DM units', 'DM units' ] 
        savefig = current_path + f'dm_final_disturb_v_reco_{explabel}_{tstamp}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'
        util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)



best_TTparam={}
best_TTparam['kpTT'] = 1
best_TTparam['kiTT'] = 0.125



#####################
## CLOSING TT ON STATIC LONG SERIES AT OPTIMAL GAINS
lab = 'CLOSING_TT_ON_STATIC_TT_LONG'

# optimal seems to be kpTT = 1, kiTT = 0.125

# DISTURBANCE 
basis = util.construct_command_basis( basis='fourier_pinned_edges', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
TT_vectors = util.get_tip_tilt_vectors()
r.dm_disturb = 0.5 * TT_vectors[:,1]# add a tip disturb 


"""

r1,r2 = 180,225
c1,c2 = 115,175
r.send_dm_cmd( dm_flat  + r.dm_disturb )
time.sleep(1)
atest_img = np.array ( r.reduceImg_test() ).reshape( 512,640 )[r1:r2, c1:c2]
plt.figure(); plt.imshow(atest_img); plt.colorbar(); plt.savefig( '/home/heimdallr/Documents/rtc-example/data/17-09-2024FINAL/delme.png' )

"""



if 1:

    kpTT = best_TTparam['kpTT']
    kiTT = best_TTparam['kiTT']

    # start it 
    rtc.clear_telemetry()

    no_tele = 5000
    close_after = 1000
    r.enable_telemetry(5000)

    #kpTT = 10.8
    #kiTT = 0.9
    rhoHO = 0 
    kpHO = 0
    no_HO_modes = 0

    explabel = f'close_after{close_after}_HOmodes-{no_HO_modes}_kpTT-{kpTT}_kiTT-{kiTT}_rhoHO-{rhoHO}_kpHO-{kpHO}_impulse'
    tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

    current_path = f'data/{tstamp.split("T")[0]}FINAL/{lab}/{explabel}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 

    if not os.path.exists(current_path):
        os.makedirs(current_path)

    # CONTROLLERS 
    r.LeakyInt.reset()
    r.pid.reset()
    r.pid.kp = [0, 0]
    r.pid.ki = [0, 0]

    r.send_dm_cmd( dm_flat + r.dm_disturb )
    time.sleep( 1 )

    # start a runner that calls latency function 
    runner = rtc.AsyncRunner(r, period = timedelta(microseconds=1000))
    runner.start()
    while r.telemetry_cnt > no_tele - close_after:
        print( r.telemetry_cnt )
    runner.pause()

    # CONTROLLERS 
    r.pid.kp = kpTT * np.ones( 2 ) 
    r.pid.ki = kiTT * np.ones( 2 )
    time.sleep(0.1)
    runner.start()

    while r.telemetry_cnt > 0:
        print( r.telemetry_cnt )

    runner.pause()
    runner.stop()
    r1,r2 = 180,225
    c1,c2 = 115,175
    #r.send_dm_cmd( dm_flat  + r.dm_disturb )
    time.sleep(1)
    atest_img = np.array ( r.reduceImg_test() ).reshape( 512,640 )[r1:r2, c1:c2]
    plt.figure(); plt.imshow(atest_img); plt.colorbar(); plt.savefig( current_path + 'not_telem_final_img.png' )


    # read out the telemetry 
    t = rtc.get_telemetry()
    telem_dict = {
    "im_err" : np.array([tt.image_in_pupil for tt in t] ) ,
    "e_TT" : np.array([tt.e_TT for tt in t]),
    "e_HO" : np.array([tt.e_HO for tt in t]),
    "u_TT" : np.array([tt.u_TT for tt in t]),
    "u_HO" : np.array([tt.u_HO for tt in t]),
    "cmd_TT" : np.array([tt.cmd_TT for tt in t]),
    "cmd_HO" : np.array([tt.cmd_HO for tt in t]),
    "dm_disturb" : np.array([tt.dm_disturb for tt in t]),
    "t0" : np.array([tt.t0 for tt in t]),
    "t1" : np.array([tt.t1 for tt in t]),
    "I0" : I0,
    "pupil_pixels" : r.regions.pupil_pixels.current, #in global frame
    "local_pupil_pixels" : pupil_pixels_local, #in local frame 
    "pid.kp": r.pid.kp,
    "pid.ki": r.pid.ki,
    "pid.kd": r.pid.kd,
    "leak.kp": r.LeakyInt.kp,
    "leak.rho": r.LeakyInt.rho,
    "IM":IM
    }

    # reconstruct the error signal in the cropped pupil region 
    pupil_img_2D = []
    for img_tmp in telem_dict['im_err']:
        tmp = np.zeros( I0.shape )
        tmp.reshape(-1)[pupil_pixels_local] = img_tmp 
        pupil_img_2D.append( tmp )

    telem_dict['signal_2D'] = pupil_img_2D

    # Create a list of HDUs (Header Data Units)
    hdul = fits.HDUList()

    # Add each list to the HDU list as a new extension
    for list_name, data_list in telem_dict.items():
        # Convert list to numpy array for FITS compatibility
        data_array = np.array(data_list, dtype=float) # Ensure it is a float array or any appropriate type

        # Create a new ImageHDU with the data
        hdu = fits.ImageHDU(data_array)

        # Set the EXTNAME header to the variable name
        hdu.header['EXTNAME'] = list_name

        # Append the HDU to the HDU list
        hdul.append(hdu)

    # Write the HDU list to a FITS file
    hdul.writeto(current_path + f'{explabel}_{tstamp}.fits', overwrite=True)

    # plot telemetry 
    fig, ax = plt.subplots(5,1,figsize=(10,20))

    cmd_err = telem_dict['cmd_TT'] + telem_dict['cmd_HO'] + telem_dict['dm_disturb']
    ax[0].plot( telem_dict['im_err'] )
    ax[0].set_ylabel(r'$\Delta I$')
    ax[1].plot( cmd_err )
    ax[1].set_ylabel(r'$\Delta C$')
    ax[2].plot( telem_dict['e_TT'] )
    ax[2].set_ylabel(r'$e_{TT}$')
    ax[3].plot( telem_dict['e_HO'] )
    ax[3].set_ylabel(r'$e_{HO}$')
    ax[4].plot( np.sqrt( np.mean( cmd_err**2, axis=1 ) ) )
    ax[4].set_ylabel(r'RMSE')

    plt.savefig(current_path + f'telemetry_summary_{tstamp}.png')


    plt.figure()
    fig,ax = plt.subplots( 1,2)
    ax[0].set_title('intial image')
    ax[1].set_title('final image')
    ax[0].imshow( telem_dict['signal_2D'][0] )
    ax[1].imshow( telem_dict['signal_2D'][-1] )
    #plt.savefig(fig_path + 'delme.png')
    plt.savefig(current_path + f'initial_v_final_image_{explabel}_{tstamp}.png')
    # write telemetry to file 


    im_list = [ util.get_DM_command_in_2D(telem_dict['cmd_TT'][-1] + telem_dict['cmd_HO'][-1]), util.get_DM_command_in_2D(telem_dict['dm_disturb'][-1])]
    title_list = ['final\nreconstruction','final\ndisturbance']
    xlabel_list = [None, None]
    ylabel_list = [None, None]
    cbar_label_list = ['DM units', 'DM units' ] 
    savefig = current_path + f'dm_final_disturb_v_reco_{explabel}_{tstamp}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'
    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)



#####################
## CLOSING HO MODES
lab = 'CLOSING_HO_FOURIER_MODES_ON_NOTHING'

# DISTURBANCE 
basis = util.construct_command_basis( basis='fourier_pinned_edges', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
TT_vectors = util.get_tip_tilt_vectors()
r.dm_disturb = 0. * TT_vectors.T[1] 



"""r1,r2 = 180,225
c1,c2 = 115,175
r.send_dm_cmd( dm_flat  + r.dm_disturb )
time.sleep(1)
atest_img = np.array ( r.reduceImg_test() ).reshape( 512,640 )[r1:r2, c1:c2]
plt.figure(); plt.imshow(atest_img); plt.colorbar(); plt.savefig( '/home/heimdallr/Documents/rtc-example/data/17-09-2024FINAL/delme.png' )

"""
#plt.figure(); plt.imshow( util.get_DM_command_in_2D(r.dm_disturb)); plt.savefig(current_path+'delme.png')

mode_grid = [0,3,5,10,15,20,30]
rho_grid = [0.1, 0.2, 0.5, 0.7]
kpHO_grid = [0.1, 0.5]
HO_gopt_dict = {} # dictionary to hold mean rmse for each mode and respective trialed gains
for m in mode_grid:
    HO_gopt_dict[m] = {}
    for rhoHO in rho_grid:
        for kpHO in kpHO_grid: 
            
            # use TT optimal gains 
            kpTT = best_TTparam['kpTT']
            kiTT = best_TTparam['kiTT']

            rtc.clear_telemetry()
            no_tele = 200
            close_after = 20
            r.enable_telemetry(200)
            no_HO_modes = m

            # NAME SAVE DIRECTORIES 
            explabel = f'fourier_HOgain_opt_close_after{close_after}_HOmodes-{no_HO_modes}_kpTT-{kpTT}_kiTT-{kiTT}_rhoHO-{rhoHO}_kpHO-{kpHO}_impulse'
            tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

            current_path = f'data/{tstamp.split("T")[0]}FINAL/{lab}/{explabel}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 

            if not os.path.exists(current_path):
                os.makedirs(current_path)

            # INIT CONTROLLERS 
            r.pid.reset()
            r.LeakyInt.reset()
            r.pid.kp = np.array([0, 0])
            r.pid.ki = np.array([0, 0])
            r.LeakyInt.rho = np.zeros( len(r.LeakyInt.rho ) )
            r.LeakyInt.kp = np.zeros( len(r.LeakyInt.kp ) )

            # ADD DISTURBANCE 
            r.send_dm_cmd( dm_flat + r.dm_disturb )
            time.sleep( 1 )

            # INIT RTC RUNNER 
            runner = rtc.AsyncRunner(r, period = timedelta(microseconds=1000))

            # RUN RTC FOR A BIT...
            runner.start()
            while r.telemetry_cnt > no_tele - close_after:
                print( r.telemetry_cnt )
            runner.pause()

            # TURN ON CONTROLLERS AND RESTART IT 
            r.pid.kp = kpTT * np.ones( 2 ) # = 1.0
            r.pid.ki = kiTT* np.ones( 2 ) # = 0.125

            r.LeakyInt.rho = [rhoHO if mtmp < m else 0 for mtmp in range(len(r.LeakyInt.rho))]
            r.LeakyInt.kp = [kpHO if mtmp < m else 0 for mtmp in range(len(r.LeakyInt.kp))]

            time.sleep(0.001)
            runner.start()

            while r.telemetry_cnt > 0:
                print( r.telemetry_cnt )
            runner.pause()
            runner.stop()

            # SAVE AND PLOT TELEMETRY 
            t = rtc.get_telemetry()
            telem_dict = {
            "im_err" : np.array([tt.image_in_pupil for tt in t] ) ,
            "e_TT" : np.array([tt.e_TT for tt in t]),
            "e_HO" : np.array([tt.e_HO for tt in t]),
            "u_TT" : np.array([tt.u_TT for tt in t]),
            "u_HO" : np.array([tt.u_HO for tt in t]),
            "cmd_TT" : np.array([tt.cmd_TT for tt in t]),
            "cmd_HO" : np.array([tt.cmd_HO for tt in t]),
            "dm_disturb" : np.array([tt.dm_disturb for tt in t]),
            "t0" : np.array([tt.t0 for tt in t]),
            "t1" : np.array([tt.t1 for tt in t]),
            "I0" : I0,
            "pupil_pixels" : r.regions.pupil_pixels.current, #in global frame
            "local_pupil_pixels" : pupil_pixels_local, #in local frame 
            "pid.kp": r.pid.kp,
            "pid.ki": r.pid.ki,
            "pid.kd": r.pid.kd,
            "leak.kp": r.LeakyInt.kp,
            "leak.rho": r.LeakyInt.rho,
            "IM":IM
            }

            # reconstruct the error signal in the cropped pupil region 
            pupil_img_2D = []
            for img_tmp in telem_dict['im_err']:
                tmp = np.zeros( I0.shape )
                tmp.reshape(-1)[pupil_pixels_local] = img_tmp 
                pupil_img_2D.append( tmp )

            telem_dict['signal_2D'] = pupil_img_2D

            # Create a list of HDUs (Header Data Units)
            hdul = fits.HDUList()

            # Add each list to the HDU list as a new extension
            for list_name, data_list in telem_dict.items():
                # Convert list to numpy array for FITS compatibility
                data_array = np.array(data_list, dtype=float) # Ensure it is a float array or any appropriate type

                # Create a new ImageHDU with the data
                hdu = fits.ImageHDU(data_array)

                # Set the EXTNAME header to the variable name
                hdu.header['EXTNAME'] = list_name

                # Append the HDU to the HDU list
                hdul.append(hdu)

            # Write the HDU list to a FITS file
            hdul.writeto(current_path + f'{explabel}_{tstamp}.fits', overwrite=True)

            # plot telemetry 
            fig, ax = plt.subplots(5,1,figsize=(10,20))
          
            cmd_err = telem_dict['cmd_TT'] + telem_dict['cmd_HO'] + telem_dict['dm_disturb']
            cmd_rmse = np.sqrt( np.mean( cmd_err**2, axis=1 ) )
           
            ax[0].plot( telem_dict['im_err'] )
            ax[0].set_ylabel(r'$\Delta I$')
            ax[1].plot( cmd_err )
            ax[1].set_ylabel(r'$\Delta C$')
            ax[2].plot( telem_dict['e_TT'] )
            ax[2].set_ylabel(r'$e_{TT}$')
            ax[3].plot( telem_dict['e_HO'] )
            ax[3].set_ylabel(r'$e_{HO}$')
            ax[4].plot( np.sqrt( np.mean( cmd_err**2, axis=1 ) ) )
            ax[4].set_ylabel(r'RMSE')

            plt.savefig(current_path + f'telemetry_summary_{tstamp}.png')



            plt.figure()
            fig,ax = plt.subplots( 1,2)
            ax[0].set_title('intial image')
            ax[1].set_title('final image')
            ax[0].imshow( telem_dict['signal_2D'][0] )
            ax[1].imshow( telem_dict['signal_2D'][-1] )
            #plt.savefig(fig_path + 'delme.png')
            plt.savefig(current_path + f'initial_v_final_image_{explabel}_{tstamp}.png')
            # write telemetry to file 


            im_list = [ util.get_DM_command_in_2D(telem_dict['cmd_TT'][-1] + telem_dict['cmd_HO'][-1]), util.get_DM_command_in_2D(telem_dict['dm_disturb'][-1])]
            title_list = ['final\nreconstruction','final\ndisturbance']
            xlabel_list = [None, None]
            ylabel_list = [None, None]
            cbar_label_list = ['DM units', 'DM units' ] 
            savefig = current_path + f'dm_final_disturb_v_reco_{explabel}_{tstamp}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'
            util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

            
            HO_gopt_dict[m][(rhoHO,kpHO)] = np.mean( cmd_rmse ) 






#####################
## CLOSING HO MODES
lab = 'CLOSING_HO_FOURIER_MODES_GAIN_OPT'

# DISTURBANCE 
basis = util.construct_command_basis( basis='fourier_pinned_edges', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
TT_vectors = util.get_tip_tilt_vectors()
r.dm_disturb = 0.4 * TT_vectors.T[1] + 0.2 * basis.T[4] # add a tip disturb 



"""r1,r2 = 180,225
c1,c2 = 115,175
r.send_dm_cmd( dm_flat  + r.dm_disturb )
time.sleep(1)
atest_img = np.array ( r.reduceImg_test() ).reshape( 512,640 )[r1:r2, c1:c2]
plt.figure(); plt.imshow(atest_img); plt.colorbar(); plt.savefig( '/home/heimdallr/Documents/rtc-example/data/17-09-2024FINAL/delme.png' )

"""
#plt.figure(); plt.imshow( util.get_DM_command_in_2D(r.dm_disturb)); plt.savefig(current_path+'delme.png')

mode_grid = [0,3,5,10,15,20, 30]
rho_grid = [0.1, 0.2, 0.5]
kpHO_grid = [0.1]
HO_gopt_dict = {} # dictionary to hold mean rmse for each mode and respective trialed gains
for m in mode_grid:
    HO_gopt_dict[m] = {}
    for rhoHO in rho_grid:
        for kpHO in kpHO_grid: 
            
            # use TT optimal gains 
            kpTT = best_TTparam['kpTT']
            kiTT = best_TTparam['kiTT']

            rtc.clear_telemetry()
            no_tele = 200
            close_after = 20
            r.enable_telemetry(200)
            no_HO_modes = m

            # NAME SAVE DIRECTORIES 
            explabel = f'a_HOgain_opt_close_after{close_after}_HOmodes-{no_HO_modes}_kpTT-{kpTT}_kiTT-{kiTT}_rhoHO-{rhoHO}_kpHO-{kpHO}_impulse'
            tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

            current_path = f'data/{tstamp.split("T")[0]}FINAL/{lab}/{explabel}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 

            if not os.path.exists(current_path):
                os.makedirs(current_path)

            # INIT CONTROLLERS 
            r.pid.reset()
            r.LeakyInt.reset()
            r.pid.kp = np.array([0, 0])
            r.pid.ki = np.array([0, 0])
            r.LeakyInt.rho = np.zeros( len(r.LeakyInt.rho ) )
            r.LeakyInt.kp = np.zeros( len(r.LeakyInt.kp ) )

            # ADD DISTURBANCE 
            r.send_dm_cmd( dm_flat + r.dm_disturb )
            time.sleep( 1 )

            # INIT RTC RUNNER 
            runner = rtc.AsyncRunner(r, period = timedelta(microseconds=1000))

            # RUN RTC FOR A BIT...
            runner.start()
            while r.telemetry_cnt > no_tele - close_after:
                print( r.telemetry_cnt )
            runner.pause()

            # TURN ON CONTROLLERS AND RESTART IT 
            r.pid.kp = kpTT * np.ones( 2 ) # = 1.0
            r.pid.ki = kiTT* np.ones( 2 ) # = 0.125

            r.LeakyInt.rho = [rhoHO if mtmp < m else 0 for mtmp in range(len(r.LeakyInt.rho))]
            r.LeakyInt.kp = [kpHO if mtmp < m else 0 for mtmp in range(len(r.LeakyInt.kp))]

            time.sleep(0.001)
            runner.start()

            while r.telemetry_cnt > 0:
                print( r.telemetry_cnt )
            runner.pause()
            runner.stop()

            # SAVE AND PLOT TELEMETRY 
            t = rtc.get_telemetry()
            telem_dict = {
            "im_err" : np.array([tt.image_in_pupil for tt in t] ) ,
            "e_TT" : np.array([tt.e_TT for tt in t]),
            "e_HO" : np.array([tt.e_HO for tt in t]),
            "u_TT" : np.array([tt.u_TT for tt in t]),
            "u_HO" : np.array([tt.u_HO for tt in t]),
            "cmd_TT" : np.array([tt.cmd_TT for tt in t]),
            "cmd_HO" : np.array([tt.cmd_HO for tt in t]),
            "dm_disturb" : np.array([tt.dm_disturb for tt in t]),
            "t0" : np.array([tt.t0 for tt in t]),
            "t1" : np.array([tt.t1 for tt in t]),
            "I0" : I0,
            "pupil_pixels" : r.regions.pupil_pixels.current, #in global frame
            "local_pupil_pixels" : pupil_pixels_local, #in local frame 
            "pid.kp": r.pid.kp,
            "pid.ki": r.pid.ki,
            "pid.kd": r.pid.kd,
            "leak.kp": r.LeakyInt.kp,
            "leak.rho": r.LeakyInt.rho,
            "IM":IM
            }

            # reconstruct the error signal in the cropped pupil region 
            pupil_img_2D = []
            for img_tmp in telem_dict['im_err']:
                tmp = np.zeros( I0.shape )
                tmp.reshape(-1)[pupil_pixels_local] = img_tmp 
                pupil_img_2D.append( tmp )

            telem_dict['signal_2D'] = pupil_img_2D

            # Create a list of HDUs (Header Data Units)
            hdul = fits.HDUList()

            # Add each list to the HDU list as a new extension
            for list_name, data_list in telem_dict.items():
                # Convert list to numpy array for FITS compatibility
                data_array = np.array(data_list, dtype=float) # Ensure it is a float array or any appropriate type

                # Create a new ImageHDU with the data
                hdu = fits.ImageHDU(data_array)

                # Set the EXTNAME header to the variable name
                hdu.header['EXTNAME'] = list_name

                # Append the HDU to the HDU list
                hdul.append(hdu)

            # Write the HDU list to a FITS file
            hdul.writeto(current_path + f'{explabel}_{tstamp}.fits', overwrite=True)

            # plot telemetry 
            fig, ax = plt.subplots(5,1,figsize=(10,20))

            cmd_err = telem_dict['cmd_TT'] + telem_dict['cmd_HO'] + telem_dict['dm_disturb']
            cmd_rmse = np.sqrt( np.mean( cmd_err**2, axis=1 ) )
            ax[0].plot( telem_dict['im_err'] )
            ax[0].set_ylabel(r'$\Delta I$')
            ax[1].plot( cmd_err )
            ax[1].set_ylabel(r'$\Delta C$')
            ax[2].plot( telem_dict['e_TT'] )
            ax[2].set_ylabel(r'$e_{TT}$')
            ax[3].plot( telem_dict['e_HO'] )
            ax[3].set_ylabel(r'$e_{HO}$')
            ax[4].plot( cmd_rmse )
            ax[4].set_ylabel(r'RMSE')

            plt.savefig(current_path + f'telemetry_summary_{tstamp}.png')



            plt.figure()
            fig,ax = plt.subplots( 1,2)
            ax[0].set_title('intial image')
            ax[1].set_title('final image')
            ax[0].imshow( telem_dict['signal_2D'][0] )
            ax[1].imshow( telem_dict['signal_2D'][-1] )
            #plt.savefig(fig_path + 'delme.png')
            plt.savefig(current_path + f'initial_v_final_image_{explabel}_{tstamp}.png')
            # write telemetry to file 


            im_list = [ util.get_DM_command_in_2D(telem_dict['cmd_TT'][-1] + telem_dict['cmd_HO'][-1]), util.get_DM_command_in_2D(telem_dict['dm_disturb'][-1])]
            title_list = ['final\nreconstruction','final\ndisturbance']
            xlabel_list = [None, None]
            ylabel_list = [None, None]
            cbar_label_list = ['DM units', 'DM units' ] 
            savefig = current_path + f'dm_final_disturb_v_reco_{explabel}_{tstamp}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'
            util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

            
            HO_gopt_dict[m][(rhoHO,kpHO)] = np.mean( cmd_rmse ) 


plt.figure() 
best_HOparam_dict = {}
for m in mode_grid:
    rmse_s = np.array( [v for _,v in HO_gopt_dict[m].items()] ) 
    rho_s = np.array( [k[0] for k,_ in HO_gopt_dict[m].items()] ) 
    kpHO_s = np.array( [k[1] for k,_ in HO_gopt_dict[m].items()] ) 

    plt.plot( rho_s, rmse_s , label=f'{m} HO modes')
    best_rho_s = np.array([r for r,_ in sorted(zip(rho_s, rmse_s))])  #np.argmin( rmse_s )
    best_kpHO_s = np.array([k for k,_ in sorted(zip(kpHO_s, rmse_s))]) 
    print( f'best for mode {m}\n========')
    for i,rmse_tmp in enumerate(sorted( rmse_s )):
        print( f'  -rmse={rmse_tmp}: kpHO={best_kpHO_s[i]}, rho={best_rho_s[i]}\n')

    # store for our final run
    best_HOparam_dict[m] = (best_rho_s[0] ,best_kpHO_s[0])

plt.legend(fontsize=12)
plt.xlabel(r'$\rho$',fontsize=15)
plt.ylabel('RMSE [DM units]',fontsize=15)
saven_tmp = f'data/{tstamp.split("T")[0]}FINAL/{lab}/RMSE_vs_RHO_vs_HOmodes_kpHO-{kpHO_s[0]}_{tstamp}.png'
plt.savefig(saven_tmp ,dpi=200, bbox_inches ='tight')







#####################
## CLOSING HO MODES ON LONG SERIES 
lab = 'LONGG_CLOSING_HO_MODES'


# DISTURBANCE 
basis = util.construct_command_basis( basis='fourier_pinned_edges', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
TT_vectors = util.get_tip_tilt_vectors()
r.dm_disturb = 0.6 * TT_vectors.T[1] + 0.5 * basis.T[6] # add a tip disturb 

plt.figure(); plt.imshow( util.get_DM_command_in_2D(r.dm_disturb)); plt.savefig(current_path+'delme.png')


for m in mode_grid:

    # use the found optimal gains
    kpTT = best_TTparam['kpTT']
    kiTT = best_TTparam['kiTT']
    rhoHO, kpHO = best_HOparam_dict[m]  

    rtc.clear_telemetry()
    no_tele = 2000
    close_after = 500
    r.enable_telemetry(2000)
    no_HO_modes = m

    # NAME SAVE DIRECTORIES 
    explabel = f'HOgain_opt_close_after{close_after}_HOmodes-{no_HO_modes}_kpTT-{kpTT}_kiTT-{kiTT}_rhoHO-{rhoHO}_kpHO-{kpHO}_impulse'
    tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

    current_path = f'data/{tstamp.split("T")[0]}FINAL/{lab}/{explabel}/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 

    if not os.path.exists(current_path):
        os.makedirs(current_path)

    # INIT CONTROLLERS 
    r.pid.reset()
    r.LeakyInt.reset()
    r.pid.kp = np.array([0, 0])
    r.pid.ki = np.arary([0, 0])
    r.LeakyInt.rho = np.zeros( len(r.LeakyInt.rho ) )
    r.LeakyInt.kp = np.zeros( len(r.LeakyInt.kp ) )

    # ADD DISTURBANCE 
    r.send_dm_cmd( dm_flat + r.dm_disturb )
    time.sleep( 1 )

    # INIT RTC RUNNER 
    runner = rtc.AsyncRunner(r, period = timedelta(microseconds=1000))

    # RUN RTC FOR A BIT...
    runner.start()
    while r.telemetry_cnt > no_tele - close_after:
        print( r.telemetry_cnt )
    runner.pause()

    # TURN ON CONTROLLERS AND RESTART IT 
    r.pid.kp = kpTT * np.ones( 2 ) # = 1.0
    r.pid.ki = kiTT * np.ones( 2 ) # = 0.125
    
    for mi in range(m):
        r.LeakyInt.rho[mi] = rhoHO
        r.LeakyInt.kp[mi] = kpHO 


    time.sleep(0.001)
    runner.start()

    while r.telemetry_cnt > 0:
        print( r.telemetry_cnt )
    runner.pause()
    runner.stop()

    # SAVE AND PLOT TELEMETRY 
    t = rtc.get_telemetry()
    telem_dict = {
    "im_err" : np.array([tt.image_in_pupil for tt in t] ) ,
    "e_TT" : np.array([tt.e_TT for tt in t]),
    "e_HO" : np.array([tt.e_HO for tt in t]),
    "u_TT" : np.array([tt.u_TT for tt in t]),
    "u_HO" : np.array([tt.u_HO for tt in t]),
    "cmd_TT" : np.array([tt.cmd_TT for tt in t]),
    "cmd_HO" : np.array([tt.cmd_HO for tt in t]),
    "dm_disturb" : np.array([tt.dm_disturb for tt in t]),
    "t0" : np.array([tt.t0 for tt in t]),
    "t1" : np.array([tt.t1 for tt in t]),
    "I0" : I0,
    "pupil_pixels" : r.regions.pupil_pixels.current, #in global frame
    "local_pupil_pixels" : pupil_pixels_local, #in local frame 
    "pid.kp": r.pid.kp,
    "pid.ki": r.pid.ki,
    "pid.kd": r.pid.kd,
    "leak.kp": r.LeakyInt.kp,
    "leak.rho": r.LeakyInt.rho,
    "IM":IM
    }

    # reconstruct the error signal in the cropped pupil region 
    pupil_img_2D = []
    for img_tmp in telem_dict['im_err']:
        tmp = np.zeros( I0.shape )
        tmp.reshape(-1)[pupil_pixels_local] = img_tmp 
        pupil_img_2D.append( tmp )

    telem_dict['signal_2D'] = pupil_img_2D

    # Create a list of HDUs (Header Data Units)
    hdul = fits.HDUList()

    # Add each list to the HDU list as a new extension
    for list_name, data_list in telem_dict.items():
    # Convert list to numpy array for FITS compatibility
    data_array = np.array(data_list, dtype=float) # Ensure it is a float array or any appropriate type

    # Create a new ImageHDU with the data
    hdu = fits.ImageHDU(data_array)

    # Set the EXTNAME header to the variable name
    hdu.header['EXTNAME'] = list_name

    # Append the HDU to the HDU list
    hdul.append(hdu)

    # Write the HDU list to a FITS file
    hdul.writeto(current_path + f'{explabel}_{tstamp}.fits', overwrite=True)

    # plot telemetry 
    fig, ax = plt.subplots(4,1,figsize=(10,20))

    cmd_err = telem_dict['cmd_TT'] + telem_dict['cmd_HO'] - telem_dict['dm_disturb']
    cmd_rmse = np.sqrt( np.mean( cmd_err**2, axis=1 ) )
    
    ax[0].plot( cmd_err )
    ax[0].set_ylabel(r'$\Delta C$')
    ax[1].plot( telem_dict['e_TT'] )
    ax[1].set_ylabel(r'$e_{TT}$')
    ax[2].plot( telem_dict['e_HO'] )
    ax[2].set_ylabel(r'$e_{HO}$')
    ax[3].plot( cmd_rmse )
    ax[3].set_ylabel(r'RMSE')

    plt.savefig(current_path + f'telemetry_summary_{tstamp}.png')


    plt.figure()
    fig,ax = plt.subplots( 1,2)
    ax[0].set_title('intial image')
    ax[1].set_title('final image')
    ax[0].imshow( telem_dict['signal_2D'][0] )
    ax[1].imshow( telem_dict['signal_2D'][-1] )
    #plt.savefig(fig_path + 'delme.png')
    plt.savefig(current_path + f'initial_v_final_image_{explabel}_{tstamp}.png')
    # write telemetry to file 


    im_list = [ util.get_DM_command_in_2D(telem_dict['cmd_TT'][-1] + telem_dict['cmd_HO'][-1]), util.get_DM_command_in_2D(telem_dict['dm_disturb'][-1])]
    title_list = ['final\nreconstruction','final\ndisturbance']
    xlabel_list = [None, None]
    ylabel_list = [None, None]
    cbar_label_list = ['DM units', 'DM units' ] 
    savefig = current_path + f'dm_final_disturb_v_reco_{explabel}_{tstamp}.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'
    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

    







