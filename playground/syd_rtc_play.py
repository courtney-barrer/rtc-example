
import numpy as np
import glob 
from astropy.io import fits
import time
import os 
import matplotlib.pyplot as plt 
import rtc
import sys
import datetime
import importlib #e.g. to reload after change importlib.reload(phase_control) 
sys.path.append('simBaldr/' )
sys.path.append('pyBaldr/' )  
from pyBaldr import utilities as util
from pyBaldr import ZWFS
from pyBaldr import phase_control
from pyBaldr import pupil_control

sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')

import bmc
import FliSdk_V2


"""
ethernet:
heimdallr@10.66.100.141
wifi: 
heimdallr@10.17.6.140


0. pull git 
1. Turn on camera, DM . Ask adam to set things up so can move between on/off sources
2. set up, get proper images with bias etc. Check SDK works ok
3. DO things purely in python first with ZWFS
    - set up with pupil_crop_regions (since final system will have to use this)
    - plot how stable mean is over region (can we normalize with this?)

    - run through aquisition script.. USE  Tidy up as we go. Set pupil pixels manually. 


"""

fig_path = 'data/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/figures/' 
data_path = 'data/' #'/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data/' 
DM_serial_number = '17DW019#122'# Syd = '17DW019#122', ANU = '17DW019#053'

# first lets just start with basic SDK

# timestamp
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

debug = True # plot some intermediate results 


#sw = 8 # 8 for 12x12, 16 for 6x6 
#pupil_crop_region = [157-sw, 269+sw, 98-sw, 210+sw ] #[165-sw, 261+sw, 106-sw, 202+sw ] #one pixel each side of pupil.  #tight->[165, 261, 106, 202 ]  #crop region around ZWFS pupil [row min, row max, col min, col max] 
#readout_mode = '12x12' # '6x6'
#pupil_crop_region = pd.read_csv('/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/' + f'T1_pupil_region_{readout_mode}.csv',index_col=[0])['0'].values


pupil_crop_region = [204,268,125, 187] #[None, None, None, None] #[0, 192, 0, 192] 

#init our ZWFS (object that interacts with camera and DM) (old path = home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/)
zwfs = ZWFS.ZWFS(DM_serial_number=DM_serial_number, cameraIndex=0, DMshapes_path = 'DMShapes/', pupil_crop_region=pupil_crop_region ) 

# the sydney BMC multi-3.5 calibrated flat seems shit! Try with just a 

zwfs.set_camera_dit( 0.001 );time.sleep(0.2)
zwfs.set_camera_fps( 200 );time.sleep(0.2)
zwfs.set_sensitivity('high');time.sleep(0.2)
zwfs.enable_frame_tag(tag = True);time.sleep(0.2)
zwfs.bias_off();time.sleep(0.2)
zwfs.flat_off();time.sleep(0.2)

zwfs.dm_shapes['flat_dm'] = 0.5 * np.ones(140)

zwfs.start_camera()

# !!!! TAKE OUT SOURCE !!!! 
# at sydney move 01 X-LSM150A-SE03 to 133.07mm


zwfs.build_manual_dark()

# get our bad pixels 
bad_pixels = zwfs.get_bad_pixel_indicies( no_frames = 1000, std_threshold = 50 , flatten=False)

# update zwfs bad pixel mask and flattened pixel values 
zwfs.build_bad_pixel_mask( bad_pixels , set_bad_pixels_to = 0)

# test getting full detector covariance over 1000 frame sample
#cov_test = zwfs.estimate_noise_covariance(  number_of_frames = 1000, where = 'whole_image' )


#check them 
plt.figure() ; plt.title('dark'); plt.imshow( zwfs.reduction_dict['dark'][0] ); plt.colorbar() ; plt.savefig(fig_path + 'delme.png' )

plt.figure() ;  plt.title('bad pixels'); plt.imshow( zwfs.bad_pixel_filter.reshape(zwfs.reduction_dict['dark'][0].shape) ); plt.savefig(fig_path + 'delme.png' )


# !!!! PUT IN SOURCE !!!! 
# quick check that dark subtraction works
I0 = zwfs.get_image( apply_manual_reduction  = True)
plt.figure(); plt.title('test image \nwith dark subtraction \nand bad pixel mask'); plt.imshow( I0 )
plt.savefig( fig_path + 'delme.png')
"""
# build a quick fourier basis and define tip/tilt - check symmetry one probing each quadrant in tip/tilt
fourier_basis = util.construct_command_basis( basis='fourier', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
tip = fourier_basis[:,0]
tilt = fourier_basis[:,5]

# try just the four quadrants 
flatdm = zwfs.dm_shapes['flat_dm']
cmd_dict = {'tip+' : flatdm + 0.4 * tip, 'tip-':flatdm - 0.4 * tip, 'tilt+':flatdm + 0.4 * tilt, 'tilt-':flatdm - 0.4 * tilt}
img_dict = {}
fig,ax = plt.subplots(2,2, figsize=(10,10))
for axx, (lab, cmd) in zip(ax.reshape(-1), cmd_dict.items()):
    zwfs.dm.send_data(cmd)
    time.sleep(0.5)
    img_dict[lab] = zwfs.get_image( apply_manual_reduction  = True)
    axx.imshow( img_dict[lab] - I0 )
    axx.set_title( lab )
plt.savefig( fig_path + 'delme.png')"""

zwfs.dm.send_data(zwfs.dm_shapes['flat_dm'])
# --- testing reconstruction 


#init our phase controller (object that processes ZWFS images and outputs DM commands)
phase_ctrl = phase_control.phase_controller_1(config_file = None) 
#phase_ctrl.change_control_basis_parameters( controller_label = ctrl_method_label, number_of_controlled_modes=phase_ctrl.config['number_of_controlled_modes'], basis_name='Zonal' , dm_control_diameter=None, dm_control_center=None)

#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)


# --- linear ramps 
# use baldr.
recon_data = util.GET_BDR_RECON_DATA_INTERNAL(zwfs, number_amp_samples = 20, amp_max = 0.2, number_images_recorded_per_cmd = 4, save_fits = data_path+f'pokeramp_data_MASK_3_sydney_{tstamp}.fits') 
#recon_data = fits.open( data_path+'recon_data_LARGE_SECONDARY_19-04-2024T12.19.22.fits' )

# process recon data to get a bunch of fits, DM actuator to pupil registration etc
internal_cal_fits =  util.PROCESS_BDR_RECON_DATA_INTERNAL(recon_data , bad_pixels = bad_pixels, active_dm_actuator_filter=phase_ctrl.config['active_actuator_filter'], debug=True, savefits=data_path + f'processed_recon_data_{tstamp}.fits'  )


# 1.2) analyse pupil and decide if it is ok
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)

if pupil_report['pupil_quality_flag'] == 1: 
    # I think this needs to become attribute of ZWFS as the ZWFS object is always passed to pupil and phase control as an argunment to take pixtures and ctrl DM. The object controlling the camera should provide the info on where a controller object should look to apply control algorithm. otherwise pupil and phase controller would always need to talk to eachother. Also we will have 4 controllers in total

    zwfs.update_reference_regions_in_img( pupil_report ) # 

# build and update detector noise model once we have defined the pupil  (we can do this with the light source on!)
phase_ctrl.update_noise_model( zwfs, number_of_frames = 1000 )


# 1.3) builds our control model with the zwfs
#control_model_report
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
ctrl_method_label = 'ctrl_1'

"""# MANUALLY ADJUST THIS JUST FOR TESTING ! 
zwfs.pupil_pixel_filter = ~zwfs.bad_pixel_filter
zwfs.pupil_pixels = np.where( ~zwfs.bad_pixel_filter )[0]
"""
# TRY model_2 WITH  method='single_side_poke', or 'double_sided_poke'
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
time.sleep( 0.1 )

#phase_ctrl.change_control_basis_parameters(  number_of_controlled_modes=140, basis_name ='Zonal', dm_control_diameter=None, dm_control_center=None,controller_label=None)
pokeamp = -0.2 
phase_ctrl.build_control_model_2(zwfs, poke_amp = pokeamp, label='ctrl_1', poke_method='single_sided_poke',\
 inverse_method='pinv',  debug = True) # inverse_method='pinv' or 'MAP'
#phase_ctrl.build_control_model( zwfs , poke_amp = -0.15, label='ctrl_1', debug = True)  

phase_ctrl.plot_SVD_modes( zwfs, 'ctrl_1', save_path=fig_path)

### Then try some reconstruction open loop!!


# Try reconstruct modes on a different basis to the one developed in the IM 

#ab_basis = util.construct_command_basis( basis='Zernike', number_of_modes = 50, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)

ctrl_method_label = 'ctrl_1'
mode_basis = phase_ctrl.config['M2C'] 
M2C = phase_ctrl.ctrl_parameters[ctrl_method_label]['M2C_4reco'] # readability  # phase_ctrl.ctrl_parameters[ctrl_method_label]['M2C_4reco']#

I2M = phase_ctrl.ctrl_parameters[ctrl_method_label]['I2M']

IM = phase_ctrl.ctrl_parameters[ctrl_method_label]['IM'] # readability 
# unfiltered CM
CM = phase_ctrl.ctrl_parameters[ctrl_method_label]['CM'] # readability 

I0 = phase_ctrl.ctrl_parameters[ctrl_method_label]['ref_pupil_FPM_in']

poke_amp = phase_ctrl.ctrl_parameters[ctrl_method_label]['poke_amp']
# plt.figure(); plt.plot(I2M.T @ IM[1]); plt.savefig('data/delme.png')
# This needs to be perfect if pinv.. not necessarily in MAP !! 

# get tip/tilt and higher order reconstructors (signal -> modal amplitude)
tip = np.zeros( I2M.shape[1] )
tip[0] = 1
tilt = np.zeros( I2M.shape[1] )
tilt[1] = 1    
R_TT, R_HO = util.project_matrix( I2M.T , projection_vector_list = [tip, tilt] )
#e.g.  R_TT @ signal should give us a vector dimension = number of modes with non-zero entries at tip/tilt mode indicies
# TT reconstruction in DM space : M2C @ R_TT @ signal 
# full reconstruction DM space : M2C @ I2M.T

""" #FOR ZONAL BASIS
# try filtering here
IM = phase_ctrl.ctrl_parameters[ctrl_method_label]['IM']
U, S, Vt = np.linalg.svd( IM ) 
S_filt = np.array([i < np.pi * (10/2)**2 for i,_ in enumerate(S) ]) # we consider the highest eigenvalues/vectors up to the number_of_controlled_modes
Sigma = np.zeros( np.array(IM).shape, float)
np.fill_diagonal(Sigma, S[S_filt], wrap=False) 
#filtered CM 
CM =  np.linalg.pinv( U @ Sigma @ Vt )
"""

# put a mode on DM and reconstruct it with our CM 
amp = -0.15
#mode_indx = 11

for mode_indx in range( len(M2C)-1 ) :  

    mode_aberration = mode_basis.T[mode_indx]#   M2C.T[mode_indx]
    #plt.imshow( util.get_DM_command_in_2D(amp*mode_aberration));plt.colorbar();plt.show()
    
    dm_cmd_aber = zwfs.dm_shapes['flat_dm'] + amp * mode_aberration 

    zwfs.dm.send_data( dm_cmd_aber )
    time.sleep(0.1)
    raw_img_list = []
    for i in range( 10 ) :
        raw_img_list.append( zwfs.get_image() ) # @D, remember for control_phase method this needs to be flattened and filtered for pupil region
    raw_img = np.median( raw_img_list, axis = 0) 
    # plt.figure() ; plt.imshow( raw_img ) ; plt.savefig( fig_path + f'delme.png') # <- signal?
    
    err_img = phase_ctrl.get_img_err( 1/np.mean(raw_img) * raw_img.reshape(-1)[zwfs.pupil_pixel_filter]  ) 
    # plt.figure() ; plt.hist( err_img, label='meas', alpha=0.3 ) ; plt.hist( IM[mode_indx] , label='from IM', alpha=0.3); plt.legend() ; plt.savefig( fig_path + f'delme.png') # <- should be around zeros

    #mode_res_test : inject err_img from interaction matrix to I2M .. should result in perfect reconstruction  
    #plt.figure(); plt.plot( I2M.T @ IM[2] ); plt.savefig( fig_path + f'delme.png')
    #plt.figure(); plt.plot( I2M.T @ IM[mode_indx]  ,label='reconstructed amplitude'); plt.axvline(mode_indx  , ls=':', color='k', label='mode applied') ; plt.xlabel('mode index'); plt.ylabel('mode amplitude'); plt.legend(); plt.savefig( fig_path + f'delme.png')
    mode_res =  I2M.T @ err_img 


    plt.figure(); plt.plot( mode_res ); plt.axvline(mode_indx  , ls=':', color='k') ; plt.savefig( fig_path + f'delme.png')
    plt.figure(figsize=(8,5));
    plt.plot( mode_res  ,label='reconstructed amplitude');
    app_amp = np.zeros( len( mode_res ) ) 

    app_amp[mode_indx] = amp / poke_amp

    plt.plot( app_amp ,'x', label='applied amplitude');
    plt.axvline(mode_indx  , ls=':', color='k', label='mode applied') ; plt.xlabel('mode index',fontsize=15); 
    plt.ylabel('mode amplitude',fontsize=15); plt.gca().tick_params(labelsize=15) ; plt.legend();
    plt.savefig( fig_path + f'delme.png')

    _ = input('press when ready to see mode reconstruction')
    
    cmd_res = 1/poke_amp * M2C @ mode_res
    
    # WITH RESIDUALS 
    
    im_list = [util.get_DM_command_in_2D( mode_aberration ),1/np.mean(raw_img) * raw_img - I0/np.mean(I0),  util.get_DM_command_in_2D( cmd_res ) ,util.get_DM_command_in_2D( mode_aberration - cmd_res ) ]
    xlabel_list = [None, None, None, None]
    ylabel_list = [None, None, None, None]
    title_list = ['Aberration on DM', 'ZWFS signal', 'reconstructed DM cmd', 'residual']
    cbar_label_list = ['DM command', 'ADU (Normalized)', 'DM command' , 'DM command' ] 
    savefig = fig_path + 'delme.png' #f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
    
    _ = input('press when ready to go to next moce ')

    # time.sleep(0.01) 
    # #reco.append( util.get_DM_command_in_2D( cmd_res ) )

    # reco_dict = {} # for saving data for processing later if needed. 

    # # =============== plotting 
    # if mode_indx < 10:
    #     """
    #     # with ZWFS IMAGE
    #     im_list = [util.get_DM_command_in_2D( mode_aberration ), raw_img.T/np.max(raw_img), util.get_DM_command_in_2D( cmd_res)  ]
    #     xlabel_list = [None, None, None]
    #     ylabel_list = [None, None, None]
    #     title_list = ['Aberration on DM', 'ZWFS Pupil', 'reconstructed DM cmd']
    #     cbar_label_list = ['DM command', 'Normalized intensity' , 'DM command' ] 
    #     #savefig = fig_path + f'mode_reconstruction_images/phase_reconstruction_example_mode-{mode_indx}_basis-{phase_ctrl.config["basis"]}_ctrl_modes-{phase_ctrl.config["number_of_controlled_modes"]}ctrl_act_diam-{phase_ctrl.config["dm_control_diameter"]}_readout_mode-12x12.png'

    #     util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=savefig)
    #     """
        

     
# 

# ESTIMATING NOISE IN IMAGE, ADD IT TO IM SIGNAL 
zwfs.enable_frame_tag( True )
time.sleep(0.5)
#zwfs.get_image_in_another_region([0,1,0,4])
i=0
no_frames = 1000
dark_list = []
ref_img_list = []
while len( ref_img_list  ) < no_frames: # poll 1000 individual images
    full_img = zwfs.get_image_in_another_region() # we can also specify region (#zwfs.get_image_in_another_region([0,1,0,4]))
    current_frame_number = full_img[0][0] #previous_frame_number
    if i==0:
        previous_frame_number = current_frame_number
    if current_frame_number > previous_frame_number:
        if current_frame_number == 65535:
            previous_frame_number = -1 #// catch overflow case for int16 where current=0, previous = 65535
        else:
            previous_frame_number = current_frame_number 
            ref_img_list.append( zwfs.get_image( apply_manual_reduction  = True) - ref_img  )
    i+=1

ref_img = np.mean( ref_img_list )

while len( dark_list ) < no_frames: # poll 1000 individual images
    full_img = zwfs.get_image_in_another_region() # we can also specify region (#zwfs.get_image_in_another_region([0,1,0,4]))
    current_frame_number = full_img[0][0] #previous_frame_number
    if i==0:
        previous_frame_number = current_frame_number
    if current_frame_number > previous_frame_number:
        if current_frame_number == 65535:
            previous_frame_number = -1 #// catch overflow case for int16 where current=0, previous = 65535
        else:
            previous_frame_number = current_frame_number 
            dark_list.append( zwfs.get_image( apply_manual_reduction  = True) - ref_img  )
    i+=1
dark_std = np.std( dark_list ,axis=0)
dark_mean = np.mean( dark_list ,axis=0)

im_list = [  dark_std ]
xlabel_list = [None]
ylabel_list = [None]
title_list = ['FPM IN, SOURCE ON']
cbar_label_list = [r'<$\sigma>/<\mu>_{pixels}$' ] 
util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15, axis_off=True, cbar_orientation = 'bottom', savefig=fig_path+'delme.png')

Std = np.mean(  dark_std.reshape(-1)[zwfs.pupil_pixel_filter]  )

plt.figure(figsize=(8,5)); plt.plot( I2M.T @ ( IM[mode_indx] + Std * np.random.randn( len( IM[mode_indx] ) ) ) ,label='reconstructed amplitude');
plt.axvline(mode_indx  , ls=':', color='k', label='mode applied') ; plt.xlabel('mode index',fontsize=15); 
plt.ylabel('mode amplitude',fontsize=15); plt.gca().tick_params(labelsize=15) ; plt.legend(); plt.savefig( fig_path + f'delme.png')


plt.figure(figsize=(8,5)); plt.plot( I2M.T @ ( IM[mode_indx] + Std * np.random.randn( len( IM[mode_indx] ) ) ) ,label='reconstructed amplitude');
plt.axvline(mode_indx  , ls=':', color='k', label='mode applied') ; plt.xlabel('mode index',fontsize=15); 
plt.ylabel('mode amplitude',fontsize=15); plt.gca().tick_params(labelsize=15) ; plt.legend(); plt.savefig( fig_path + f'delme.png')

sigma_over_mu = np.mean(  dark_std.reshape(-1)[zwfs.pupil_pixel_filter]  ) / np.mean(  dark_mean.reshape(-1)[zwfs.pupil_pixel_filter]  )
print( sigma_over_mu  )


# looking at covariance of pixel noise 
dark_list = np.array( dark_list )
dark_list_filtered = np.array( [d.reshape(-1)[zwfs.pupil_pixel_filter] for d in dark_list])

cov_pupil = np.cov( np.array(dark_list_filtered ) ,ddof=1, rowvar = False ) # rowvar = False => rows are samples, cols variables 
plt.figure() ; plt.imshow( cov_pupil ) 
plt.colorbar(); plt.savefig( fig_path + f'delme.png')

#plt.figure() ; plt.imshow( np.cov( np.array( dark_list ).reshape(  1000, -1 ),ddof=1) );  plt.savefig( fig_path + f'delme.png')

#plt.figure() ; plt.imshow( np.cov( np.array( dark_list_filtered.T ),ddof=1) ;  plt.savefig( fig_path + f'delme.png')


IM = np.array( IM )
I2M_new = IM @ np.linalg.inv(IM.T @ IM + cov_pupil)


plt.figure(figsize=(8,5)); plt.plot( I2M_new @ ( IM[mode_indx] + 0 * np.random.randn( len( IM[mode_indx] ) ) ) ,label='reconstructed amplitude');
plt.axvline(mode_indx  , ls=':', color='k', label='mode applied') ; plt.xlabel('mode index',fontsize=15); 
plt.ylabel('mode amplitude',fontsize=15); plt.gca().tick_params(labelsize=15) ; plt.legend(); plt.savefig( fig_path + f'delme.png')

mode_indx = 4
ideal = np.zeros( len( IM ) )
ideal[mode_indx] = 1
map_rmse = []
pinv_rmse = []
for i in range( 50 ):
    amps_MAP = I2M_new @ ( IM[mode_indx] + i * np.random.randn( len( IM[mode_indx] ) ) ) # minimum variance of maximum posterior estimator 
    amps_pinv = I2M.T @ ( IM[mode_indx] + i * np.random.randn( len( IM[mode_indx] ) ) ) # psuedo inverse 

    map_rmse.append( np.sqrt( np.mean( ( amps_MAP - ideal )**2 ) ) )
    pinv_rmse.append( np.sqrt( np.mean( ( amps_pinv - ideal )**2 ) ) )

plt.figure(figsize=(8,5)) 
plt.plot(map_rmse, label='MAP')
plt.plot(pinv_rmse, label='pinv')
plt.xlabel(r'$\sigma_{ADU}$',fontsize=15)
plt.ylabel('mode amplitude RMSE',fontsize=15)
plt.legend(fontsize=15) 
plt.gca().tick_params(labelsize=15)
plt.savefig( fig_path + f'MAP_vs_PINV_{mode_indx}.png')





save_fits = save_path + f'scan_dit_fps_gain-{gain}_widegrid.fits'

fps_grid = [100, 200, 300, 400]
dit_grid = 1e-3 * np.logspace(-1,1.5 , 10)


data = fits.HDUList([]) 
number_images_recorded_per_cmd = 10
for i,fps in enumerate(fps_grid):
    print( i/len(fps_grid), '%' )

    # set DITs based on fps constraints
    minDIT, maxDIT = zwfs.get_dit_limits()
    dit_grid = np.logspace( -4, np.logspace(maxDIT), 10)

    zwfs.set_camera_fps(fps) # set the FPS 
    time.sleep(0.1)
    for dit in dit_grid:
        zwfs.set_camera_dit(dit)
        time.sleep(0.5)

        img_list = []
        for _ in range(number_images_recorded_per_cmd):
            img_list.append( zwfs.get_image(apply_manual_reduction  = True) )
            time.sleep(0.01)

        mean_im = np.mean( img_list, axis = 0)

        tmp_fits = fits.PrimaryHDU(  img_list )

        camera_info_dict = util.get_camera_info(zwfs.camera)
        for k,v in camera_info_dict.items():
            tmp_fits.header.set(k,v)  

        data.append( tmp_fits )

#data.writeto(save_fits,overwrite=True)


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







recomended_bad_pixels = np.where( (np.std( poke_imgs ,axis = (0,1)) > 100) + (np.std( poke_imgs ,axis = (0,1)) == 0 ))


#fps = 100 
"""
fps = 100 
DIT = 2e-3 #s integration time 

zwfs.send_fli_cmd("set unsigned on")

FliSdk_V2.FliSerialCamera.SendCommand(zwfs.camera, "aduoffset")
zwfs.send_fli_cmd("set aduoffset 1000")

zwfs.send_fli_cmd( "set sensibility high")
zwfs.send_fli_cmd( f"set fps {fps}")
zwfs.send_fli_cmd( f"set tint {DIT}")
zwfs.send_fli_cmd( "set imagetags on")
zwfs.send_fli_cmd( "set bias off")
zwfs.send_fli_cmd( "exec buildbias")

FliSdk_V2.FliSerialCamera.SendCommand(self.camera, cmd)


"""
##    START CAMERA 
zwfs.start_camera()

#first thing to build our own dark  ( move dichroic first!)
# NOTE I MADE IT DEFAULT TO apply_manual_reduction IN ZWFS
zwfs.build_manual_dark()

a = zwfs.get_image( apply_manual_reduction  = True)
plt.savefig( fig_path + 'delme.png')

# put baldr dichroic motor to H (temporary block) pos = 133.07mm
bias_list = []
for _ in range(256):
    time.sleep(1/fps)
    bias_list.append( zwfs.get_image() )
bias = np.median( bias_list ,axis = 0)
plt.figure()
plt.imshow( bias )
plt.savefig(fig_path + 'bias.png')

# put baldr dichroic motor to J (mirror) pos =  63.07mm
img_list = []
for _ in range(256):
    time.sleep(1/fps)
    img_list.append( zwfs.get_image() )
img = np.median( img_list ,axis = 0)
plt.figure()
plt.imshow( img )
plt.savefig(fig_path + 'img.png')


fig, ax = plt.subplots( 1,2)
im1 = ax[0].imshow(img - bias ) 
im2 = ax[1].imshow(img ) 
#plt.colorbar(img - bias, ax=ax[0])
#plt.colorbar(img, ax=ax[1])
plt.savefig(fig_path + 'img-bias.png')

# --- testing bias on / off 

zwfs.bias_on()
time.sleep(0.5)
im_bias_on = zwfs.get_image()

#zwfs.stop_camera()

zwfs.bias_off()
#time.sleep(2)
#zwfs.start_camera()
time.sleep(1)

im_bias_off = zwfs.get_image()

fig, ax = plt.subplots( 1,2)
im1 = ax[0].imshow(im_bias_on ) 
im2 = ax[1].imshow(im_bias_off ) 
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
#plt.show()
plt.savefig(fig_path + 'delme.png')

# issue with overflow - CRED 2 with firmware 2.2.7 (very old) and cannot change aduoffset not asign pixels to signed integers.

# 2 solutions : 
# - take manual bias, dark etc 
# - fix manually (im_bias_on[im_bias_on > 2**15]


# --- darks

save_fits = save_path + f'scan_dit_fps_gain-{gain}_widegrid.fits'

fps_grid = [100, 200, 300, 400]
dit_grid = 1e-3 * np.logspace(-1,1.5 , 10)


data = fits.HDUList([]) 
number_images_recorded_per_cmd = 10
for i,fps in enumerate(fps_grid):
    print( i/len(fps_grid), '%' )

    # set DITs based on fps constraints
    minDIT, maxDIT = zwfs.get_dit_limits()
    dit_grid = np.logspace( -4, np.logspace(maxDIT), 10)

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

#data.writeto(save_fits,overwrite=True)


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
#-----------------------------
# Now set reasonable pupil_crop_region 


#init our phase controller (object that processes ZWFS images and outputs DM commands)
phase_ctrl = phase_control.phase_controller_1(config_file = None) 
#phase_ctrl.change_control_basis_parameters( controller_label = ctrl_method_label, number_of_controlled_modes=phase_ctrl.config['number_of_controlled_modes'], basis_name='Zonal' , dm_control_diameter=None, dm_control_center=None)

# --- linear ramps 
# use baldr.
recon_data = util.GET_BDR_RECON_DATA_INTERNAL(zwfs, number_amp_samples = 18, amp_max = 0.2, number_images_recorded_per_cmd = 2, save_fits = data_path+f'pokeramp_data_sydney_{tstamp}.fits') 
#recon_data = fits.open( data_path+'recon_data_LARGE_SECONDARY_19-04-2024T12.19.22.fits' )

# process recon data to get a bunch of fits, DM actuator to pupil registration etc
internal_cal_fits =  util.PROCESS_BDR_RECON_DATA_INTERNAL(recon_data ,active_dm_actuator_filter=phase_ctrl.config['active_actuator_filter'], debug=True, savefits=data_path + f'processed_recon_data_{tstamp}.fits'  )


# --- testing reconstruction 


#init our phase controller (object that processes ZWFS images and outputs DM commands)
phase_ctrl = phase_control.phase_controller_1(config_file = None) 
#phase_ctrl.change_control_basis_parameters( controller_label = ctrl_method_label, number_of_controlled_modes=phase_ctrl.config['number_of_controlled_modes'], basis_name='Zonal' , dm_control_diameter=None, dm_control_center=None)

#init our pupil controller (object that processes ZWFS images and outputs VCM commands)
pupil_ctrl = pupil_control.pupil_controller_1(config_file = None)


# 1.2) analyse pupil and decide if it is ok
pupil_report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True)


if pupil_report['pupil_quality_flag'] == 1: 
    # I think this needs to become attribute of ZWFS as the ZWFS object is always passed to pupil and phase control as an argunment to take pixtures and ctrl DM. The object controlling the camera should provide the info on where a controller object should look to apply control algorithm. otherwise pupil and phase controller would always need to talk to eachother. Also we will have 4 controllers in total

    zwfs.update_reference_regions_in_img( pupil_report ) # 


# 1.3) builds our control model with the zwfs
#control_model_report
zwfs.dm.send_data( zwfs.dm_shapes['flat_dm'] )
ctrl_method_label = 'ctrl_1'

# TRY model_2 WITH  method='single_side_poke', or 'double_sided_poke'
phase_ctrl.build_control_model_2(self, ZWFS, poke_amp = -0.15, label='ctrl_1', method='single_side_poke',  debug = True):
#phase_ctrl.build_control_model( zwfs , poke_amp = -0.15, label=ctrl_method_label, debug = True)  







### WITH RTC
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