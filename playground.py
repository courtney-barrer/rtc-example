
import numpy as np
import glob 
from astropy.io import fits
import os 
import matplotlib.pyplot as plt 

import rtc
from pyBaldr import utilities as util
from pyBaldr import ZWFS
from pyBaldr import phase_control
from pyBaldr import pupil_control

"""
1)
init rtc
read in reconstructor configuration (fits) file and configure rtc struc with it

2) 
set up simulation signals if rtc_state.camera_simulation_mode = true

3)
test single compute with the simulated signals to ensure we get correct behaviour

4) 
t



"""

#%% 1) ------------------------------------------------------------------
r = rtc.RTC() 
conig_file_name = None #if None we just get most recent reconstructor file in /data/ path

# Note we do not do this as a function because the \
# we get memory errors in RTC struc when manipulating in 
# local scope of python function

#states_tmp = rtc.rtc_state_struct() 
#sim_signals = rtc.simulated_signals_struct()
cam_settings_tmp = rtc.camera_settings_struct()
reconstructors_tmp = rtc.phase_reconstuctor_struct()
pupil_regions_tmp = rtc.pupil_regions_struct()

if conig_file_name==None:
    # get most recent 
    list_of_recon_files = glob.glob('data/' + 'RECONS*')
    conig_file_name = max(list_of_recon_files, key=os.path.getctime) #'RECONSTRUCTORS_debugging_DIT-0.002005_gain_medium_10-07-2024T22.21.55.fits'#"RECONSTRUCTORS_TEST_RTC_DIT-0.002005_gain_medium_10-07-2024T19.51.53.fits" #"RECONSTRUCTORS_test_DIT-0.002004_gain_high_05-07-2024T10.09.47.fits"#"RECONSTRUCTORS_try2_DIT-0.002003_gain_medium_04-06-2024T12.40.05.fits"
    #/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data
    #reco_filename = reco_filename


config_fits = fits.open( conig_file_name  ) 

# camera settings used to build reconstructor 
det_fps = float(config_fits['info'].header['camera_fps']) # frames per sec (Hz) 

det_dit = float( config_fits['info'].header['camera_tint'] )  # integration time (seconds)

det_gain = str(config_fits['info'].header['camera_gain']) # camera gain 

det_cropping_rows = str( config_fits['info'].header['cropping_rows'] ).split('rows: ')[1]

det_cropping_cols = str( config_fits['info'].header['cropping_columns'] ).split('columns: ')[1]

cam_settings_tmp.det_fps = det_fps
cam_settings_tmp.det_dit = det_dit
cam_settings_tmp.det_gain = det_gain
cam_settings_tmp.det_cropping_rows = det_cropping_rows
cam_settings_tmp.det_cropping_cols = det_cropping_cols 


# reconstructor data 
R_TT = config_fits['R_TT'].data.astype(np.float32) #tip-tilt reconstructor

R_HO = config_fits['R_HO'].data.astype(np.float32) #higher-oder reconstructor

IM = config_fits['IM'].data.astype(np.float32) # interaction matrix (unfiltered)

M2C = config_fits['M2C'].data.astype(np.float32) # mode to command matrix 

I2M = np.transpose( config_fits['I2M'].data).astype(np.float32) # intensity (signal) to mode matrix  
# (# transposed so we can multiply directly I2M @ signal)
    
CM = config_fits['CM'].data.astype(np.float32)  # full control matrix 

I0 = config_fits['I0'].data.astype(np.float32) # calibration source reference intensity (FPM IN)

N0 = config_fits['N0'].data.astype(np.float32) # calibration source reference intensity (FPM OUT)

reconstructors_tmp.IM.update(IM.reshape(-1))
reconstructors_tmp.CM.update(CM.reshape(-1))
reconstructors_tmp.R_TT.update(R_TT.reshape(-1))
reconstructors_tmp.R_HO.update(R_HO.reshape(-1))
reconstructors_tmp.M2C.update(M2C.reshape(-1))
reconstructors_tmp.I2M.update(I2M.reshape(-1))
reconstructors_tmp.I0.update(I0.reshape(-1))
reconstructors_tmp.N0.update(N0.reshape(-1))

# COMMIT IT ALL 
reconstructors_tmp.commit_all()
# -----------------------------

# pupil region classification data 
pupil_pixels = np.array( config_fits['pupil_pixels'].data, dtype=np.int32)

secondary_pixels = np.array( config_fits['secondary_pixels'].data, dtype=np.int32)

outside_pixels = np.array( config_fits['outside_pixels'].data, dtype=np.int32)

pupil_regions_tmp.pupil_pixels.update( pupil_pixels )
pupil_regions_tmp.secondary_pixels.update( secondary_pixels ) 
pupil_regions_tmp.outside_pixels.update( outside_pixels )
#filter(lambda a: not a.startswith('__'), dir(pupil_regions_tmp))

# COMMIT IT ALL 
pupil_regions_tmp.commit_all()
# -----------------------------

# Simple check 
if len( pupil_pixels ) != CM.shape[1]:
    raise TypeError("number of pupil pixels (for control) does not match\
    control matrix size!!! CHeck why, did you input correct files?")

# update our RTC object 
# _RTC.regions = pupil_regions_tmp
# _RTC.reco = reconstructors_tmp
# _RTC.camera_settings = cam_settings_tmp

r.regions = pupil_regions_tmp
r.reco = reconstructors_tmp
r.camera_settings = cam_settings_tmp
# do we return it or is it static?
#return(_RTC)


# check it updated correcetly 
#print('current CM for r:', r.reco.CM.current )
#print('current pupil_pixels for r:', r.regions.pupil_pixels.current )

r.apply_camera_settings()


#%% 2) ------------------------------------------------------------------
if r.rtc_state.camera_simulation_mode :
    r.rtc_simulation_signals.simulated_image = I0.reshape(-1) # simulated image
    r.rtc_simulation_signals.simulated_signal = IM[0] # simulated signal (processed image) - 
    # if r.rtc_state.signal_simulation_mode=true than this simulated signal over-rides any 
    # images (simulated or not) as input to the controllers.
    r.rtc_simulation_signals.simulated_dm_cmd = M2C.T[0] # simulated DM command 

# to check it
#plt.figure();plt.imshow( util.get_DM_command_in_2D( r.rtc_simulation_signals.simulated_dm_cmd) ); plt.show()


#%% 3) ------------------------------------------------------------------
# test some functionality 

# polling image and convert to vector 
test1 = r.im2vec_test()
if(len(test1) == r.camera_settings.full_image_length):
    print( ' passed im2vec_test')
else:
    print( ' FAILED --- im2vec_test')

# polling image and convert to vector and filter for pupil pixels 
test2 = r.im2filtered_im_test()
if(len(test2 ) == len(r.regions.pupil_pixels.current)):
    print( ' passed im2vec_test')
else:
    print( ' FAILED --- im2vec_test')


# filter reference intensity setpoint for pupil pixels 

# process image 

# matrix multiplication ( to go to command space)

# matrix multiplication (to go to modal space )






















# we can print, edit nested struc members in RTC struct directly. e.g. 
print( r.rtc_state.close_loop_mode)
# r.rtc_state.close_loop_mode = True # example of setting this field directly 

# or best we init nested data structures as indpendent structs to edit and directly update  
states = rtc.rtc_state_struct()
sim_signals = rtc.simulated_signals_struct()
cam_settings = rtc.camera_settings_struct()


fits.open( )

cam_settings.det_dit = 0.0018

# set a rtc structure 
r.camera_settings = cam_settings

print( r.camera_settings.det_dit )

# these use updatable fields (see cpp code definition of updatable)


reconstructors = rtc.phase_reconstuctor_struct()
pupil_regions = rtc.pupil_regions_struct()


#r.regions.outside_pixels.update([1,2,3])

# this puts the values as next to commit to current 
pupil_regions.outside_pixels.commit()

pupil_regions.outside_pixels.current
pupil_regions.outside_pixels.next

# DONT KNOW WHY DOESN"T allow me to do it for the reconstructor struc 
reconstructors.I2M.update([1,2,3]) # doesn't work for reconstructors?? 


# doesnt work 









# set up camera/DM with same settings as reconstructor file 
r = rtc.RTC()

# ============ 
# i have to wait for this set up otherwise below code screws up

# -- update camera settings 
r.set_det_dit( det_dit )
r.set_det_fps( det_fps )
r.set_det_gain( det_gain )
r.set_det_tag_enabled( True ) # we want this on to count frames 
r.set_det_crop_enabled( True )
r.set_det_cropping_rows( det_cropping_rows ) 
r.set_det_cropping_cols( det_cropping_cols ) 

r.commit_camera_settings()

r.update_camera_settings()

# ============ 
# i have to wait for this set up otherwise below code screws up
time.sleep(1)

r.set_ctrl_matrix(  CM.reshape(-1) )  # check r.get_ctrl_matrix()
r.set_TT_reconstructor(  R_TT.reshape(-1) )  # check r.get_ctrl_matrix
r.set_HO_reconstructor(  R_HO.reshape(-1) )  
r.set_I2M(  I2M.reshape(-1) )  # 
r.set_M2C(  M2C.reshape(-1) )  # 

# set I0, bias, etc <- long term I should create a function to do this 
# for now use reconstructor file 
frame = r.get_last_frame() 

r.set_bias( 0*r.get_last_frame()  )

r.set_I0( (I0.reshape(-1) / np.mean(I0) ).astype(np.float32)  )
# try with this one since updatable is going crazy
r.set_I0_vec( ((I0.reshape(-1) / np.mean(I0))[pupil_pixels]).astype(np.float32)  ) 
r.set_fluxNorm(  np.mean(np.array(r.get_last_frame(), dtype=np.float32) ) )

# init the rtc. Could have been done using constructor but requires to code it.
#r.set_slope_offsets(slope_offsets[0])
#r.set_gain(1.1)
#r.set_offset(2)
r.set_pupil_pixels(pupil_pixels)
# none of the above commands are executed yet until we commit.
# It's safe to do it because the rtc is not running yet.
time.sleep(0.05)
r.commit()