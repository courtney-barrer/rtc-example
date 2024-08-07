
import sys

#sys.path.insert(1, '/home/baldr/Documents/baldr/rtc_example/pyBaldr/')

from pyBaldr import ZWFS
from pyBaldr import phase_control
from pyBaldr import pupil_control
from pyBaldr import utilities as util

import os 
import pickle
import numpy as np
import glob
import matplotlib.pyplot as plt 
import time 
import datetime
from astropy.io import fits
import pandas as pd 


fig_path = 'data/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = 'data/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 
DM_serial_number = '17DW019#122'# Syd = '17DW019#122', ANU = '17DW019#053'
###
#    TAKE INSPIRATION FROM /BALDR/A_RECONSTRUCTOR_PIPELINE

#important cmds 
#zwfs.restore_default_settings()
#zwfs.get_image() 
#zwfs.shutdown()
###

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

#zwfs.restore_default_settings()

# ,------------------ AVERAGE OVER 8X8 SUBWIDOWS SO 12X12 PIXELS IN PUPIL
#zwfs.pixelation_factor = sw #8 # sum over 8x8 pixel subwindows in image
# HAVE TO PROPAGATE THIS TO PUPIL COORDINATES 
#zwfs._update_image_coordinates( )



zwfs.start_camera() 

gain = 'high'
zwfs.set_sensitivity(gain)

zwfs.enable_frame_tag(tag = False)

fps_grid = [50, 200,  600]
dit_grid = 1e-3 * np.logspace(-1,1.5 , 10)

save_path = '/home/heimdallr/Desktop/'
save_fits = save_path + f'scan_dit_fps_gain-{gain}_widegrid.fits'



data = fits.HDUList([]) 
number_images_recorded_per_cmd = 10
for i,fps in enumerate(fps_grid):
    print( i/len(fps_grid), '%' )
    zwfs.set_camera_fps(fps) # set the FPS 
    time.sleep(0.1)
    for dit in dit_grid:
        zwfs.set_camera_dit(dit)
        time.sleep(0.5)

        img_list = []
        for _ in range(number_images_recorded_per_cmd):
            img_list.append( zwfs.get_image() )
            time.sleep(0.01)

        mean_im = np.mean( img_list, axis = 0)

        tmp_fits = fits.PrimaryHDU(  img_list )

        camera_info_dict = util.get_camera_info(zwfs.camera)
        for k,v in camera_info_dict.items():
            tmp_fits.header.set(k,v)  

        data.append( tmp_fits )

data.writeto(save_fits,overwrite=True)


tint = np.array( [float(d.header['camera_tint']) for d in data] )
fps = np.array( [float(d.header['camera_fps']) for d in data] )
gains = np.array( [d.header['camera_gain'] for d in data] )
signal_mean = np.array( [np.mean(d.data) for d in data]   )

filt_50fps = fps < 60
filt_200fps = fps == 200
filt_600fps = fps > 300

plt.plot( 1e3 * tint[filt_50fps], signal_mean[filt_50fps] ,'-o',label = f'FPS={np.unique(fps[filt_50fps])}'); 
plt.plot( 1e3 * tint[filt_200fps], signal_mean[filt_200fps] ,'-o',label = f'FPS={np.unique(fps[filt_200fps])}'); 
plt.plot( 1e3 * tint[filt_600fps], signal_mean[filt_600fps] ,'-o',label = f'FPS={np.unique(fps[filt_600fps])}'); 
plt.legend()
plt.xlabel( 'integration time [ms]' ); plt.ylabel(f'mean ADU'); 

plt.savefig(save_path + f'adu_mean_vs_dit_gain-{np.unique(gains)[0]}.png' ) 

plt.show()

#print( np.mean( np.diff(signal_mean[filt_50fps]) ) / np.mean( np.diff( tint[filt_50fps] )) )

print( (signal_mean[filt_50fps][-1] - signal_mean[filt_50fps][0] ) /  ( tint[filt_50fps][-1] - tint[filt_50fps][0] ))

minD, maxD = zwfs.get_dit_limits()

zwfs.set_camera_dit(float(maxD))  # set the max DIT 

zwfs.set_sensitivity('high')
#zwfs.set_camera_cropping(r1=224, r2=287, c1=224, c2=287) #zwfs.set_camera_cropping(r1=152, r2=267, c1=96, c2=223) #
zwfs.enable_frame_tag(tag = False) # first 1-3 pixels count frame number etc


# measure dark current

zwfs.start_camera() # start camera

data = fits.HDUList([]) 


zwfs.stop_camera()  # stop camera

if save_fits!=None:
    if type(save_fits)==str:
        data.writeto(save_fits)
    else:
        raise TypeError('save_images needs to be either None or a string indicating where to save file')
    
return(data)

[np.mean(d.data) for d in data]



zwfs.build_bias(256)
zwfs.bias_on()


# TO EXIT CAMERA AND DM SO THEY CAN BE RE-INITIALIZED 
#zwfs.exit_camera()
#zwfs.exit_dm()

##
##    START CAMERA 
zwfs.start_camera()

zwfs.get_image()

5782 / 30e-3

#FliSdk_V2.GetProcessedImage(context, -1) 


# can I build my own bias 
im_list = []
for i in range(100):
    im_list.append( zwfs.get_image() )
    time.sleep(0.1)
dark_med = np.median( im_list , axis = 0 )
dark_mean = np.mean( im_list, axis = 0)


im_list = []
for i in range(100):
    im_list.append( zwfs.get_image() )
    time.sleep(0.1)
im_med = np.median( im_list , axis = 0 )
im_mean = np.mean( im_list, axis=0)

plt.figure(); plt.imshow( im_mean - dark_mean  ); plt.show()


# ----------------------
# look at the image for a second
util.watch_camera(zwfs, frames_to_watch=20)


# --- testing bias on / off 

im_bias_on = zwfs.get_image()

zwfs.stop_camera()

zwfs.bias_off()
time.sleep(2)
zwfs.start_camera()
time.sleep(1)

im_bias_off = zwfs.get_image()

fig, ax = plt.subplots( 1,2)
im1 = ax[0].imshow(im_bias_on ) 
im2 = ax[1].imshow(im_bias_off ) 
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.show()

#-----------------------------


im_list = []
for i in range(100):
    im_list.append( zwfs.get_image() )
    time.sleep(0.1)



im_med = np.median( im_list , axis = 0 )


im_high_gain  = zwfs.get_image()
#plt.imshow( im) ; plt.colorbar(); plt.show()

zwfs.stop_camera()

zwfs.set_sensitivity('low')
time.sleep(1)
zwfs.start_camera()

time.sleep(1)

im_low_gain = zwfs.get_image()

fig, ax = plt.subplots( 1,2)
im1 = ax[0].imshow(im_low_gain ) 
im2 = ax[1].imshow(im_high_gain ) 
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])

