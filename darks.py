
import numpy as np
import glob 
from astropy.io import fits
import time
import os 
import matplotlib.pyplot as plt 

import sys
import datetime
sys.path.append('pyBaldr/' )
from pyBaldr import utilities as util
from pyBaldr import ZWFS

save_path = 'data/'

it = 2 # iteration 
# SET UP SYSTEM AS DESIRED (dark / flat / bright etc)


# over whole detector
pupil_crop_region = [None, None, None, None] #[204,268,125, 187] #

DM_serial_number = '17DW019#122'# Syd = '17DW019#122', ANU = '17DW019#053'

#init our ZWFS (object that interacts with camera and DM) (old path = home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/)
zwfs = ZWFS.ZWFS(DM_serial_number=DM_serial_number, cameraIndex=0, DMshapes_path = 'DMShapes/', pupil_crop_region=pupil_crop_region ) 

# set high gain with no bias or flat (raw image), frame taggin on 
gain = 'high'
zwfs.set_sensitivity(gain)
zwfs.bias_off()
zwfs.flat_off()
zwfs.enable_frame_tag(tag = True)

zwfs.start_camera() 

# name to save output fits file
save_fits = save_path + f'scan_dit_gain-{gain}_widegrid_it{it}.fits'

data = fits.HDUList([]) 

fps_grid = [100, 200, 400]
for i,fps in enumerate(fps_grid):
    
    print( i/len(fps_grid), '%' )
    zwfs.set_camera_fps(fps) # set the FPS 
    time.sleep(0.1)

    minDIT, maxDIT = zwfs.get_dit_limits()
    # get range of possible DITs given fps (logspace)
    dit_grid = np.logspace( -4 , np.log10( 0.9 * float(maxDIT) ), 5 )
    for dit in dit_grid:

        zwfs.set_camera_dit(dit)
        time.sleep(0.5)

        # get some frames from each setup as list
        img_list = zwfs.get_some_frames(number_of_frames = 200, apply_manual_reduction = False )

        # put in fits
        tmp_fits = fits.PrimaryHDU(  img_list )
        # add headers based on camera settings 
        camera_info_dict = util.get_camera_info(zwfs.camera)
        for k,v in camera_info_dict.items():
            tmp_fits.header.set(k,v)  

        data.append( tmp_fits )

data.writeto(save_fits, overwrite=True)


#================
# quick analysis 
tint = np.array( [float(d.header['camera_tint']) for d in data] )
fps = np.array( [float(d.header['camera_fps']) for d in data] )
gains = np.array( [d.header['camera_gain'] for d in data] )

# build rough pixel mask
# define index (settings) to use to build our pixel mask
indx = np.where( (fps == np.median(fps_grid)) & (tint==np.median(tint)) )[0][0]
pixelwise_var = np.var( data[indx].data, axis =0 )
bad_pix_threshold = 10 * np.std( pixelwise_var ) 
bad_pix_mask =  ~( pixelwise_var > bad_pix_threshold) + ( pixelwise_var == 0 ) 
#pixelwise_var = np.var( data_cube  ,axis=0 )
print( f'{np.sum(~bad_pix_mask)}/{len(bad_pix_mask.reshape(-1))} pixels masked')

# get statistics  pixel w data[0].data.shape = (100, 512, 640)

# get expect mean and variance over each pixels (aggregated over many frames)
pixel_signal_mean = np.array( [np.mean(d.data,axis=0)  for d in data]   )
pixel_signal_var = np.array( [np.var(d.data,axis=0) for d in data]   )

#plt.figure(); plt.imshow( pixel_signal_var[0] ); plt.colorbar(); plt.savefig( save_path+ 'delme.png')

# get expect pixel mean and variance agregated over all temporally averaged pixels 
signal_mean = np.array( [np.mean(d[bad_pix_mask])  for d in pixel_signal_mean]   )
signal_var = np.array( [np.mean(d[bad_pix_mask]) for d in pixel_signal_var]   )


# PLOTTING EXPECTED INTENSITIES 
plt.figure(figsize=(8,5))
for fps_pt in fps_grid:
    fps_filt = abs(fps - fps_pt) < 0.1 # we don't equate just in case of precision error in writing etc

    plt.plot( 1e3 * tint[fps_filt], signal_mean[fps_filt] ,'-o',label = f'FPS={np.unique(fps[fps_filt])[0]}Hz'); 
plt.gca().set_xscale('log')
plt.legend()
plt.xlabel( 'integration time [ms]' ,fontsize=15); plt.ylabel(r'expected pixel intensity',fontsize=15); 

plt.savefig(save_path + f'adu_mean_vs_dit_gain_{it}-{np.unique(gains)[0]}.png' ) 

# PLOTTING EXPECTED INTENSITY VARIANCES 
plt.figure(figsize=(8,5))
for fps_pt in fps_grid:
    fps_filt = abs(fps - fps_pt) < 0.1 # we don't equate just in case of precision error in writing etc

    plt.plot( 1e3 * tint[fps_filt], signal_var[fps_filt] ,'-o',label = f'FPS={np.unique(fps[fps_filt])[0]}Hz'); 
plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')
plt.legend()
plt.xlabel( 'integration time [ms]' ,fontsize=15); plt.ylabel(f'expected pixel variance',fontsize=15); 

plt.savefig(save_path + f'adu_var_vs_dit_gain_{it}-{np.unique(gains)[0]}.png' ) 


plt.figure(figsize=(8,5))
for fps_pt in [fps_grid[0]]:
    fps_filt = abs(fps - fps_pt) < 0.1 # we don't equate just in case of precision error in writing etc

    m, c = np.polyfit(1e3 * tint[fps_filt], signal_var[fps_filt], 1)
    print( m ,c )
    ymodel = m * 1e3 * tint[fps_filt] + c
    plt.plot( 1e3 * tint[fps_filt], signal_var[fps_filt] ,'-o',label = f'FPS={np.unique(fps[fps_filt])[0]}Hz'); 
    plt.plot( 1e3 * tint[fps_filt], ymodel ,'-o',label = f'model FPS={np.unique(fps[fps_filt])[0]}Hz'); 
plt.legend()
plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')
plt.legend()
plt.xlabel( 'integration time [ms]' ,fontsize=15); plt.ylabel(f'expected pixel variance',fontsize=15); 

plt.savefig(save_path + f'delme.png' ) 





