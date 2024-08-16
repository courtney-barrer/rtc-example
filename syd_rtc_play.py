
import numpy as np
import glob 
from astropy.io import fits
import os 
import matplotlib.pyplot as plt 
import rtc
import sys
sys.path.append('simBaldr/' )
sys.path.append('pyBaldr/' )
from pyBaldr import utilities as util
from pyBaldr import ZWFS
from pyBaldr import phase_control
from pyBaldr import pupil_control



"""
- reads in reconstructors / pupil region classifications to init rtc object in C++
- run some basic tests to check operational 
- set up simulation and interface with Baldr RTC in simulation mode  


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


# first lets just start with basic SDK


# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

debug = True # plot some intermediate results 


#sw = 8 # 8 for 12x12, 16 for 6x6 
#pupil_crop_region = [157-sw, 269+sw, 98-sw, 210+sw ] #[165-sw, 261+sw, 106-sw, 202+sw ] #one pixel each side of pupil.  #tight->[165, 261, 106, 202 ]  #crop region around ZWFS pupil [row min, row max, col min, col max] 
#readout_mode = '12x12' # '6x6'
#pupil_crop_region = pd.read_csv('/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/' + f'T1_pupil_region_{readout_mode}.csv',index_col=[0])['0'].values

pupil_crop_region = [None, None, None, None]

#init our ZWFS (object that interacts with camera and DM) (old path = home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/)
zwfs = ZWFS.ZWFS(DM_serial_number=DM_serial_number, cameraIndex=0, DMshapes_path = 'DMShapes/', pupil_crop_region=pupil_crop_region ) 




#%% 1) ------------------------------------------------------------------
r = rtc.RTC() 

# Note we do not do this as a function because the \
# we get memory errors in RTC struc when manipulating in 
# local scope of python function

#states_tmp = rtc.rtc_state_struct() 
#sim_signals = rtc.simulated_signals_struct()
cam_settings_tmp = rtc.camera_settings_struct()
reconstructors_tmp = rtc.phase_reconstuctor_struct()
pupil_regions_tmp = rtc.pupil_regions_struct()

r.regions = pupil_regions_tmp
r.reco = reconstructors_tmp
r.camera_settings = cam_settings_tmp
# do we return it or is it static?

r.apply_camera_settings()

#get an image
a = np.array( r.im2vec_test() )


plt.imshow( a.reshape(r.camera_settings.image_height, r.camera_settings.image_width) )
plt.show()