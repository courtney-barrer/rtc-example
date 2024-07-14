
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

from pyBaldr import ZWFS
from pyBaldr import phase_control
from pyBaldr import pupil_control
from pyBaldr import utilities as util

def print_n_last_lines(s: str, n: int = 10):
    lines = s.split('\n')
    for l in lines[-n:]:
        print(l)



fig_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 

pupil_classification_filename = 'pupil_classification_10-07-2024T22.21.55.pickle' #"pupil_classification_31-05-2024T15.26.52.pickle"
# get most recent 
list_of_recon_files = glob.glob(data_path + 'RECONS*')
reco_filename = max(list_of_recon_files, key=os.path.getctime) #'RECONSTRUCTORS_debugging_DIT-0.002005_gain_medium_10-07-2024T22.21.55.fits'#"RECONSTRUCTORS_TEST_RTC_DIT-0.002005_gain_medium_10-07-2024T19.51.53.fits" #"RECONSTRUCTORS_test_DIT-0.002004_gain_high_05-07-2024T10.09.47.fits"#"RECONSTRUCTORS_try2_DIT-0.002003_gain_medium_04-06-2024T12.40.05.fits"

"""
set up camera and DM settings based on reconstruction fits file
"""


# ============== READ IN PUPIL CLASSIFICATION DATA
#with open(data_path + pupil_classification_filename, 'rb') as handle:
#    pup_classification = pickle.load(handle)

# ============== READ IN PUPIL PHASE RECONSTRUCTOR DATA
reco_fits = fits.open( reco_filename  ) 

# reconstructor data 
R_TT = reco_fits['R_TT'].data #tip-tilt reconstructor
R_HO = reco_fits['R_HO'].data #higher-oder reconstructor

IM = reco_fits['IM'].data.astype(np.float32)
M2C = reco_fits['M2C'].data.astype(np.float32) # mode to command matrix 
I2M = reco_fits['I2M'].data.astype(np.float32) # intensity (signal) to mode matrix  
CM = reco_fits['CM'].data.astype(np.float32)  # full control matrix 
I0 = reco_fits['I0'].data # calibration source reference intensity (FPM IN)
N0 = reco_fits['N0'].data # calibration source reference intensity (FPM OUT)

# camera settings used to build reconstructor 
det_fps = float(reco_fits['info'].header['camera_fps']) # frames per sec (Hz) 
det_dit = float( reco_fits['info'].header['camera_tint'] )  # integration time (seconds)
det_gain = str(reco_fits['info'].header['camera_gain']) # camera gain 

# cropping - NEED TO TEST WITHOUT CROPPING - WHAT GOES INTO HEADERS? WILL IT CRASH?
det_cropping_rows = str( reco_fits['info'].header['cropping_rows'] ).split('rows: ')[1]
det_cropping_cols = str( reco_fits['info'].header['cropping_columns'] ).split('columns: ')[1]

# need to also deal with tagging - mask if tagging is on - at what level does this get done?
# should probably remove these pixels from any of the pixel filters!!! 
# cannot assume this was done at development - because we may want to turn this feature
# on and off at any moment. 



# JUST USE A DUMMY FOR NOW OF THE FULL FRAME SIZE 
#nmodes = 140
#CM = np.zeros( [140 , 640*512], dtype=np.float32).reshape(-1) # just use zeros

#pupil_pixels = np.ones(CM.shape[0], dtype=np.int32)# np.array( pup_classification['pupil_pixels'][:-2], dtype=np.int32)
#pupil_pixels = np.array( pup_classification['pupil_pixels'], dtype=np.int32)
pupil_pixels = np.array( reco_fits['pupil_pixels'].data, dtype=np.int32)

if len( pupil_pixels ) != CM.shape[1]:
    raise TypeError("number of pupil pixels (for control) does not match\
    control matrix size!!! CHeck why, did you input correct files?")



# create 2 slope offsets buffer.
#slope_offsets = np.ones((2, 15), dtype=np.float32)

# set the first slope offset to 1 and the second to 2
#slope_offsets[1] = 2
#slope_offsets[0] = 1

# =================================
# =========== INIT RTC
# set up camera/DM with same settings as reconstructor file 
r = rtc.RTC()

# ============ 
# i have to wait for this set up otherwise below code screws up

# -- update camera settings 
r.set_det_dit( det_dit )
r.set_det_fps( det_fps )
r.set_det_gain( det_gain )
r.set_det_tag_enabled( False )
r.set_det_crop_enabled( True )
r.set_det_cropping_rows( det_cropping_rows ) 
r.set_det_cropping_cols( det_cropping_cols ) 

r.commit_camera_settings()

r.update_camera_settings()

# ============ 
# i have to wait for this set up otherwise below code screws up
time.sleep(1)


r.set_ctrl_matrix(  CM.reshape(-1) )  # check r.get_reconstructor()

# set I0, bias, etc <- long term I should create a function to do this 
# for now use reconstructor file 
frame = r.get_last_frame() 

r.set_bias( 0*r.get_last_frame()  )

r.set_I0( (I0.reshape(-1) / np.mean(I0) ).astype(np.float32)  )
r.set_fluxNorm(  np.mean(np.array(r.get_last_frame(),dtype=np.float32) ) )

# init the rtc. Could have been done using constructor but requires to code it.
#r.set_slope_offsets(slope_offsets[0])
#r.set_gain(1.1)
#r.set_offset(2)
r.set_pupil_pixels(pupil_pixels)
# none of the above commands are executed yet until we commit.
# It's safe to do it because the rtc is not running yet.
time.sleep(0.05)
r.commit()

# ----------- HELP
# I dont know why this has to keep getting set / committed!!!!! 
#r.set_I0( (I0.reshape(-1) / np.mean(I0) ).astype(np.float32)  )
#r.commit() 

"""
IT seems to happen after I call, or do an operation on I0. e.g.
r.set_I0( (I0.reshape(-1) / np.mean(I0) ).astype(np.float32)  )
r.commit() 

"""

#==================== LATENCY TEST 

# -- update camera settings 
#r.set_det_dit( det_dit )
#r.set_det_fps( det_fps )
#r.set_det_gain( det_gain )
r.set_det_tag_enabled( True )
#r.set_det_crop_enabled( True )
#r.set_det_cropping_rows( det_cropping_rows ) 
#r.set_det_cropping_cols( det_cropping_cols ) 

r.commit_camera_settings()

r.update_camera_settings()

iteration_nb = 100
r.enable_telemetry(iteration_nb)

r.apply_dm_shape( np.zeros(140) )
w,h = r.get_img_width(),r.get_img_height() #image width height 
# to check
#f = r.get_last_frame()
#plt.imshow( f.reshape(h,w) ); plt.show()

# start a runner that calls latency function 
#runner = rtc.AsyncRunner(r, period = timedelta(microseconds=1000))
#runner.start()

#runner.pause()

t = rtc.get_telemetry()
tel_rawimg = np.array([tt.image_raw for tt in t] )

tel_rawimg
#tel_signal = np.array([tt.image_proc for tt in t])
#tel_reco =   np.array([tt.reco_dm_err for tt in t])

# basic dimensionality check of control matrix and pupil filtered image 
try:
    i0 = I0.reshape(-1)[pupil_pixels]/np.mean(I0 )
    f = r.get_last_frame()
    i = f[pupil_pixels]/np.mean(f)
    cmd_test = r.get_reconstructor().reshape(CM.shape) @ (i-i0)


    print(f"DIMENSIONS CHECK OUT (ie, can multiply!!\nresulting cmd length = {len(cmd_test)}. should be 140 for BMC multi3.5 DM ")
except:
    print("SOMETHING WRONG WITH INPUT DIMENSIONS!! ")


#========== COMPARE RECONSTRUCTORS WITH/WITHOUT DISTURBANCE
# flat DM 
#r.apply_dm_shape( np.ones( 140 ) * 0.5 ) 
r.flatten_dm()
time.sleep(0.5)
i0 = I0.reshape(-1)[pupil_pixels]/np.mean( I0 )
f = r.get_last_frame()
i = f[pupil_pixels]/np.mean(f)
cmd_test_flat = r.get_reconstructor().reshape(CM.shape) @ (i-i0)

# now put on a mode 
mode_idx = 5
amp = 0.2
mode_cmd = 0.5*np.ones( 140 ) + amp*M2C[:,mode_idx]
r.apply_dm_shape( mode_cmd ) 
time.sleep(0.5)
f = r.get_last_frame()
i = f[pupil_pixels]/np.mean(f)
cmd_test_mode = r.get_reconstructor().reshape(CM.shape) @ (i-i0)

im_list = [util.get_DM_command_in_2D(mode_cmd ), util.get_DM_command_in_2D( cmd_test_flat ) ,util.get_DM_command_in_2D( cmd_test_mode  ) ]
xlabel_list = [None, None, None]
ylabel_list = [None, None, None]
title_list = [f'dist with {amp} x mode {mode_idx}','reco with flat DM input', f'reco with {amp} x mode {mode_idx}']
cbar_label_list = ['DM command', 'DM command' , 'DM command' ] 
savefig = None # fig_path + f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

# does it even register the mode in I2M 
reco_mode_amps = I2M.T @  (i-i0)
plt.figure() 
plt.plot( reco_mode_amps ) 
plt.axvline( mode_idx , ls=':');plt.show()

###################
# ok its shit.. lets try re-filter to see if we can improve 

U, S, Vt = np.linalg.svd( IM , full_matrices=True)
S[0] = np.min(S)
S[50:] = np.min(S)
Sigma = np.zeros( IM.shape)
np.fill_diagonal(Sigma, S, wrap=False)
I2M_new =  np.linalg.pinv( U @ Sigma @ Vt ) # C = A @ M #1/abs(poke_amp)
print(  np.linalg.cond( I2M_new) )
#control matrix (note in zonal method M2C is just identity matrix)
CM_new = M2C @ I2M_new.T


# flat DM 
r.apply_dm_shape( np.ones( 140 ) * 0.5 ) 
time.sleep(0.5)
i0 = I0.reshape(-1)[pupil_pixels]/np.mean(I0 )
f = r.get_last_frame()
i = f[pupil_pixels]/np.mean(f)
cmd_test_flat = CM_new @ (i-i0)

# now put on a mode 
mode_idx = 10
amp = 0.08
mode_cmd = np.ones( 140 ) * 0.5 + amp*M2C[:,mode_idx]
r.apply_dm_shape( mode_cmd ) 
time.sleep(0.5)
f = r.get_last_frame()
i = f[pupil_pixels]/np.mean(f)
cmd_test_mode = CM_new @ (i-i0)

im_list = [util.get_DM_command_in_2D(mode_cmd ), util.get_DM_command_in_2D( cmd_test_flat ) ,util.get_DM_command_in_2D( cmd_test_mode  ) ]
xlabel_list = [None, None, None]
ylabel_list = [None, None, None]
title_list = [f'dist with {amp} x mode {mode_idx}','reco with flat DM input', f'reco with {amp} x mode {mode_idx}']
cbar_label_list = ['DM command', 'DM command' , 'DM command' ] 
savefig = None # fig_path + f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)

# does it even register the mode in I2M 
reco_mode_amps = I2M_new.T @  (i-i0)
plt.figure() 
plt.plot( reco_mode_amps ) 
plt.axvline( mode_idx , ls=':');plt.show()







# to get some telemetry when we run compute
no_tele = 10
r.enable_telemetry(no_tele)

# test 12 iteration of closed loop 
for i in range(15):
    r.test() 

t = rtc.get_telemetry()

#t[no_tele+1]
#t[no_tele+2] # fails cause empty .. so iterates one more than the specified telemetry entry

# get second iteration  
tel_rawimg = np.array([tt.image_raw for tt in t] )
tel_signal = np.array([tt.image_proc for tt in t])
tel_reco =   np.array([tt.reco_dm_err for tt in t])
#tel_dmcmd =  [tt.dm_command for tt in t]

# reconstruct raw image, (cannot do processed since its filtered, but could do this using I0 etc)
#tel_rawimg[1].reshape(116,128)




#%% - some verification 

# lets have a look at a frame. 
plt.imshow( r.get_last_frame().reshape(116,160)); plt.show() # reshape (height, width)


# ========verify image poke and setting DM shape 
# poke sequential actuators and look at differential images 

w,h = r.get_img_width(),r.get_img_height()
fig,ax = plt.subplots(3,3)
r.apply_dm_shape(0.5*np.ones(140)) # need to put check in apply shape, especially for length! 
time.sleep(0.5)
im_a = r.get_last_frame().astype(float)
for i,axx in enumerate(ax.reshape(-1)):

    #plt.imshow( (im_a).reshape( 116,160 ) ); plt.show()
    r.poke_dm_actuator(60+i,0.45)
    time.sleep(0.5)
    im_b = r.get_last_frame().astype(float) 
    #im_list.append( r.get_last_frame().astype(float) ) 
    axx.imshow( (im_a - im_b).reshape( h,w ) )
    r.apply_dm_shape(0.5*np.ones(140))
    time.sleep(0.5)
plt.show() 

# ========verify applyin DM shape 

fourtorres = pd.read_csv("/home/baldr/Documents/baldr/DMShapes/four_torres_4.csv") 
waffle = pd.read_csv("/home/baldr/Documents/baldr/DMShapes/waffle.csv") 

# reference image 
r.apply_dm_shape(0.5*np.ones(140)) # need to put check in apply shape, especially for length! 
time.sleep(0.5)
im_a = r.get_last_frame().astype(float)

#r.apply_dm_shape( waffle.values.ravel()*0.02+0.5) #*0.2 + 0.5 )
r.apply_dm_shape( fourtorres.values.ravel()*0.1+0.5) #*0.2 + 0.5 )
time.sleep(0.5)
im_c = r.get_last_frame().astype(float)
plt.imshow( (im_a - im_c).reshape( h, w ) ); 
plt.show()

#another test 
play_shape1 = np.ones(140) * 0.5
play_shape1[65] = 0.4
r.apply_dm_shape(0.5*np.ones(140)) 
time.sleep(0.5)
im_a = r.get_last_frame().astype(float)
r.apply_dm_shape( play_shape1 ) #*0.2 + 0.5 )
time.sleep(0.5)
im_c = r.get_last_frame().astype(float)
plt.imshow( (im_a - im_c).reshape( h, w ) ); plt.show()

# ========verify getting image, signal processing and reconstructor of DM CMD
# using r.test() 
# to be safe we enforce CM is just zero! so apply zero vector to DM 
#r.set_ctrl_matrix( 0 * CM.reshape(-1) )
#r.commit() 
time.sleep(0.5)
#test time , gets and image, filters based on classified pupil region,
# processes a signal, matrix multiplies to get a DM command and applied it to DM
# returned value is the cmd applied
cmd_test = r.test() 


# ok lets put test in compute ! then run



# for playing with reconstructor / testing  
# test filtering and matrix mult new image 
r.test()#CM.reshape(-1))


# Create an async runner. This component will run the rtc in a separate thread.
runner = rtc.AsyncRunner(r, period = timedelta(microseconds=1000))


runner.start()

sleep(1)
print_n_last_lines(runner.flush(), 6)


r.set_slope_offsets(slope_offsets[1])
r.set_gain(0)
r.set_offset(-1)

# request a commit. The runner will commit the new values at the next iteration.
r.request_commit()

sleep(.2)

# pause keep the thread alive but stop the execution of the rtc.
# this can be resume later using runner.resume()
runner.pause()

# get the output of the runner but just keep the last 6 lines.
print_n_last_lines(runner.flush(), 6)

# kill the thread. A new thread can still be recreated using `start` later.
runner.stop()

# `del runner`` will also stop the thread.

