
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

# TO DO 
# - add pid and leakys on init , see test 


def init_rtc( reco_fits_file ):
    # ============== READ IN PUPIL PHASE RECONSTRUCTOR DATA
    config_fits = fits.open( reco_fits_file  ) 

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

    det_cropping_rows = str( config_fits['info'].header['cropping_rows'] ).split('rows: ')[1]

    det_cropping_cols = str( config_fits['info'].header['cropping_columns'] ).split('columns: ')[1]

    darkss = config_fits['DARK'].data
    dark = darkss[0] # chose which one
    bad_pixels = config_fits['BAD_PIXELS'].data
    if 0 in config_fits['BAD_PIXELS'].data:
        bad_pixels = bad_pixels[2:] # first frame is tag for frame count - make sure it is not masked 


    cam_settings_tmp.det_fps = det_fps
    cam_settings_tmp.det_dit = det_dit
    cam_settings_tmp.det_gain = det_gain
    cam_settings_tmp.det_cropping_rows = det_cropping_rows
    cam_settings_tmp.det_cropping_cols = det_cropping_cols 
    cam_settings_tmp.dark = dark.reshape(-1) 
    cam_settings_tmp.bad_pixels = bad_pixels
    cam_settings_tmp.det_tag_enabled = True
    cam_settings_tmp.det_crop_enabled = True


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


    # pupil region classification data 

    pupil_pixels = np.array( config_fits['pupil_pixels'].data, dtype=np.int32)

    secondary_pixels = np.array( config_fits['secondary_pixels'].data, dtype=np.int32)

    outside_pixels = np.array( config_fits['outside_pixels'].data, dtype=np.int32)


    reconstructors_tmp.IM.update(IM.reshape(-1))
    reconstructors_tmp.CM.update(CM.reshape(-1))
    reconstructors_tmp.R_TT.update(R_TT.reshape(-1))
    reconstructors_tmp.R_HO.update(R_HO.reshape(-1))
    reconstructors_tmp.M2C.update(M2C.reshape(-1))
    reconstructors_tmp.I2M.update(I2M.reshape(-1))
    reconstructors_tmp.I0.update(I0.reshape(-1)/np.mean( I0.reshape(-1)[pupil_pixels] )) #normalized
    reconstructors_tmp.N0.update(N0.reshape(-1)/np.mean( I0.reshape(-1)[pupil_pixels] )) #normalized
    reconstructors_tmp.flux_norm.update(np.mean( I0.reshape(-1)[pupil_pixels] ))   #normalized

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
    # now set up the camera as required 
    r.apply_camera_settings()

    return( r )

data_path = '~/Documents/asgard-alignment/tmp/29-08-2024/iter_13_J3/'

reconstructor_file = data_path + ''


r = init_rtc( reconstructor_file )







