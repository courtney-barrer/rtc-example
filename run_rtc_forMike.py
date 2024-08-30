

import rtc
import numpy as np
from datetime import timedelta 
import matplotlib.pyplot as plt
import time

r = rtc.RTC() # <- use default constructor to set up camera etc 
r.dm_flat = 0.5 * np.ones( 140 ) # or what ever you want. See DMShapes folder for calibrated flats 

# set up camera  
cam_settings_tmp = rtc.camera_settings_struct()
cam_settings_tmp.det_fps = 200
cam_settings_tmp.det_dit = 0.002
cam_settings_tmp.det_gain = 'high'
cam_settings_tmp.det_cropping_rows = '0-639'
cam_settings_tmp.det_cropping_cols = '0-511' 
#cam_settings_tmp.dark = dark.reshape(-1) 
#cam_settings_tmp.bad_pixels = bad_pixels
cam_settings_tmp.det_tag_enabled = True
cam_settings_tmp.det_crop_enabled = False # True <- always false unless latency tests etc 

r.camera_settings = cam_settings_tmp # append to rtc object

r.apply_camera_settings() # apply them 

### START RTC 
# start a runner (does calls compute on repeat from src/rtc.cpp). Does nothing to DM while gains are zero (which they are by default)
runner = rtc.AsyncRunner(r, period = timedelta(microseconds=1000))

runner.start()

time.sleep(1) # wait a sec

runner.pause()
runner.stop()

# get telemetry 
r.enable_telemetry(10) # get 10 frames worth

runner.start()

time.sleep(2)


runner.pause()
runner.stop()


# read out the telemetry 
t = rtc.get_telemetry()
imgs = np.array([tt.image_in_pupil for tt in t] ) 

print( imgs )

# close things nicely 
r.close_all()
