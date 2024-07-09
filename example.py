
from time import sleep
import rtc
import numpy as np
from datetime import timedelta
from astropy.io import fits 
import pickle
import matplotlib.pyplot as plt
def print_n_last_lines(s: str, n: int = 10):
    lines = s.split('\n')
    for l in lines[-n:]:
        print(l)


"""
Questions for Julien - how to pass DM / camera object between python a c++? 
given input reconstructor files we need to update camera settings - in python or C++,
makes sense in C++.

function - update_camera() based on reconstructotr input which has camera settings 
git commits?

how to save camera config file? 

with updateables - why next and not current after commit. 


"""
fig_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = '/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 

pupil_classification_filename = "pupil_classification_31-05-2024T15.26.52.pickle"
reco_filename = "RECONSTRUCTORS_test_DIT-0.002004_gain_high_05-07-2024T10.09.47.fits"#"RECONSTRUCTORS_try2_DIT-0.002003_gain_medium_04-06-2024T12.40.05.fits"

"""
set up camera and DM settings based on reconstruction fits file
"""


# ============== READ IN PUPIL CLASSIFICATION DATA
with open(data_path + pupil_classification_filename, 'rb') as handle:
    pup_classification = pickle.load(handle)

# ============== READ IN PUPIL PHASE RECONSTRUCTOR DATA
reco_fits = fits.open(data_path + reco_filename  ) 

# reconstructor data 
R_TT = reco_fits['R_TT'].data #tip-tilt reconstructor
R_HO = reco_fits['R_HO'].data #higher-oder reconstructor
CM = reco_fits['CM'].data.astype(np.float32)  # full control matrix 
I0 = reco_fits['FPM_IN'].data # calibration source reference intensity (FPM IN)
N0 = reco_fits['FPM_OUT'].data # calibration source reference intensity (FPM OUT)

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
nmodes = 140
#CM = np.zeros( [140 , 640*512], dtype=np.float32).reshape(-1) # just use zeros

pupil_pixels = np.ones(CM.shape[0], dtype=np.int32)# np.array( pup_classification['pupil_pixels'][:-2], dtype=np.int32)

if len( pupil_pixels ) != CM.shape[0]:
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


# -- update camera settings 
r.set_det_dit( det_dit )
r.set_det_fps( det_fps )
r.set_det_gain( det_gain )
r.set_det_tag_enabled( True )
r.set_det_crop_enabled( True )
r.set_det_cropping_rows( det_cropping_rows ) 
r.set_det_cropping_cols( det_cropping_cols ) 

# -- update control matrix 
r.set_ctrl_matrix( CM.reshape(-1) )

# set I0, bias, etc <- long term I should create a function to do this 
# for now use reconstructor file 
frame = r.get_last_frame() 
r.set_bias( r.get_last_frame()  )
r.set_I0(  np.array(r.get_last_frame(),dtype=np.float32) )
r.set_fluxNorm(  np.sum(np.array(r.get_last_frame(),dtype=np.float32) ) )

# init the rtc. Could have been done using constructor but requires to code it.
#r.set_slope_offsets(slope_offsets[0])
#r.set_gain(1.1)
#r.set_offset(2)
#r.set_pupil_pixels(pupil_pixels)
# none of the above commands are executed yet until we commit.
# It's safe to do it because the rtc is not running yet.
r.commit()

r.update_camera_settings()

#%%
plt.imshow( r.get_last_frame().reshape(512,640)); plt.show()


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

