
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
# - add pid and leakys on init , see test 

data_path = '/home/heimdallr/Documents/asgard-alignment/tmp/29-08-2024/iter_14_J3/fourier_20modes_map_reconstructor/' #'~/Documents/asgard-alignment/tmp/29-08-2024/iter_13_J3/'

reconstructor_file = data_path + 'RECONSTRUCTORS_fourier_0.2pokeamp_in-out_pokes_map_DIT-0.001_gain_high_29-08-2024T23.48.48.fits' #'RECONSTRUCTORS_fourier_0.2pokeamp_in-out_pokes_map_DIT-0.001_gain_high_29-08-2024T22.59.26.fits'

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

    # converting back to the global image (in the reconstructor we likely (to reduce data size) looked at only a sub-region and referenced pixels from there)
    pupil_pixels = convert_local_to_global_coordinates(pupil_pixels_local, r1_subregion, \
        c1_subregion, r2_subregion-r1_subregion, c2_subregion-c1_subregion, global_shape, flatten = True)
    secondary_pixels = convert_local_to_global_coordinates(secondary_pixels_local, r1_subregion, \
        c1_subregion, r2_subregion-r1_subregion, c2_subregion-c1_subregion, global_shape, flatten = True)
    outside_pixels = convert_local_to_global_coordinates(outside_pixels_local, r1_subregion, \
        c1_subregion, r2_subregion-r1_subregion, c2_subregion-c1_subregion, global_shape, flatten = True)
    # to define the total local region used when developing reconstructor
    # this is important for consistent flux normalization!
    local_region_pixels = list(outside_pixels) + list(pupil_pixels) # could also 
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
    dm_flat =  0.5*np.ones(140 ) ################ config_fits['DM_FLAT'].data.astype(np.float)
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


    #filter(lambda a: not a.startswith('__'), dir(pupil_regions_tmp))

    # COMMIT IT ALL 
    pupil_regions_tmp.commit_all()
    # -----------------------------


    # pid and leaky integator

    Nmodes = I2M.shape[0] #M2C.shape[1]
    kp = np.zeros(Nmodes)
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
    r.dm_flat = dm_flat 
    # now set up the camera as required 
    r.apply_camera_settings()

    #return( r )



# start it 
r.enable_telemetry(1000)
# start a runner that calls latency function 
runner = rtc.AsyncRunner(r, period = timedelta(microseconds=1000))
runner.start()
time.sleep(1)
runner.pause()
runner.stop()


# read out the telemetry 

t = rtc.get_telemetry()
tel_rawimg = np.array([tt.image_in_pupil for tt in t] )
#tel_imgErr = np.array([tt.image_err_signal for tt in t])
tel_modeErr = np.array([tt.mode_err for tt in t])
tel_reco = np.array([tt.dm_cmd_err for tt in t])











#---- TESTS 

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






# Example usage:

# Assume the cropped region is of size n x m
n, m = 4, 4  # Example values for the cropped region size

# Assume the cropped region starts at (row1, col1) in the original image
row1, col1 = 2, 3

# Example flattened relative pixel indices within the cropped region
relative_pixels_flat = [0, 5, 10]  # These correspond to [(0,0), (1,1), (2,2)] in 2D within the cropped region

# Convert to global coordinates
global_pixels = convert_to_global_coordinates_flattened(relative_pixels_flat, row1, col1, n, m)

# Print the results
print("Global Pixel Coordinates:", global_pixels)



# with array 
def convert_to_global_coordinates_flattened(relative_pixels_flat, boolean_array_flat, row1, col1, n, m, N, M):
    """
    Convert relative pixel coordinates and a flattened boolean array from the cropped region
    to global coordinates in the original image.
    
    Parameters:
    - relative_pixels_flat: 1D list or array of indices in the flattened cropped array.
    - boolean_array_flat: 1D flattened boolean array indicating true values in the cropped region.
    - row1: The starting row index of the cropped region in the global image.
    - col1: The starting column index of the cropped region in the global image.
    - n: The number of rows in the cropped region.
    - m: The number of columns in the cropped region.
    - N: The number of rows in the original global image.
    - M: The number of columns in the original global image.
    
    Returns:
    - global_pixels: List of tuples [(row_global1, col_global1), (row_global2, col_global2), ...]
                     containing the global coordinates in the original image.
    - global_boolean_array: 2D numpy array of the same size as the original image with true values where the original
                            boolean array indicated in the cropped region.
    """
    
    # Convert flattened relative pixel indices to 2D coordinates within the cropped region
    relative_coords = [(index // m, index % m) for index in relative_pixels_flat]
    
    # Convert relative 2D coordinates to global 2D coordinates
    global_pixels = [(row + row1, col + col1) for (row, col) in relative_coords]
    
    # Initialize the global boolean array with False
    global_boolean_array = np.zeros((N, M), dtype=bool)
    
    # Ensure boolean_array_flat is of size n * m
    if len(boolean_array_flat) != n * m:
        raise ValueError(f"Size of boolean_array_flat ({len(boolean_array_flat)}) does not match n * m ({n * m}).")
    
    # Convert flattened boolean array to 2D boolean array of size n x m
    boolean_array_2d = np.reshape(boolean_array_flat, (n, m))
    
    # Place the cropped boolean array into the corresponding location in the global boolean array
    global_boolean_array[row1:row1 + n, col1:col1 + m] = boolean_array_2d
    
    return global_pixels, global_boolean_array

# Example usage:

# Assume we have an original image of size N x M
N, M = 10, 12

# Cropped region is from row 2 to row 5 and column 3 to column 6
row1, row2, col1, col2 = 2, 5, 3, 6

# Assume the cropped region is of size n x m
n, m = row2 - row1 + 1, col2 - col1 + 1  # n = 4, m = 4

# Example flattened relative pixel indices within the cropped region
relative_pixels_flat = [0, 5, 10]  # These correspond to [(0,0), (1,1), (2,2)] in 2D within the cropped region

# Example flattened boolean array within the cropped region
boolean_array_flat = np.array([True, False, False, False, False, True, False, False, False, False, True, False, False, False, False, True])

# Convert to global coordinates
global_pixels, global_boolean_array = convert_to_global_coordinates_flattened(relative_pixels_flat, boolean_array_flat, row1, col1, n, m, N, M)

# Print the results
print("Global Pixel Coordinates:", global_pixels)
print("Global Boolean Array:")
print(global_boolean_array)
