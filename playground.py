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
import baldr_simulation_functions as baldrSim
import data_structure_functions as config


#%%
#===================== SIMULATION 

# =========== Setup simulation 

throughput = 0.01
Hmag = 0
#Hmag_at_vltiLab = Hmag  - 2.5*np.log10(throughput)
#flux_at_vltilab = baldr.star2photons('H',Hmag_at_vltiLab,airmass=1,k=0.18,ph_m2_s_nm=True) #ph/m2/s/nm

# setting up the hardware and software modes of our ZWFS
tel_config =  config.init_telescope_config_dict(use_default_values = True)
phasemask_config = config.init_phasemask_config_dict(use_default_values = True) 

# -------- trialling this 
#phasemask_config['on-axis phasemask depth'] = 4.210526315789474e-05
#phasemask_config['off-axis phasemask depth'] = 4.122526315789484e-05

#phasemask_config['phasemask_diameter'] = 1.5 * (phasemask_config['fratio'] * 1.65e-6)

#---------------------

DM_config = config.init_DM_config_dict(use_default_values = True) 
detector_config = config.init_detector_config_dict(use_default_values = True)


# the only thing we need to be compatible is the pupil geometry and Npix, Dpix 
tel_config['pup_geometry'] = 'disk'

# define a hardware mode for the ZWFS 
mode_dict = config.create_mode_config_dict( tel_config, phasemask_config, DM_config, detector_config)

#create our zwfs object
zwfs = baldrSim.ZWFS(mode_dict)

# define an internal calibration source 
calibration_source_config_dict = config.init_calibration_source_config_dict(use_default_values = True)
calibration_source_config_dict['temperature']=1900 #K (Thorlabs SLS202L/M - Stabilized Tungsten Fiber-Coupled IR Light Source )
calibration_source_config_dict['calsource_pup_geometry'] = 'Disk'

# -------- trialling this 
zwfs.FPM.update_cold_stop_parameters(None)


#---------------------

lab = 'control_20_zernike_modes'
zwfs.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=20, modal_basis='zernike', pokeAmp = 50e-9 , label=lab)

print( zwfs.control_variables[lab ].keys() )

control_basis =  np.array(zwfs.control_variables[lab ]['control_basis'])
M2C = control_basis.reshape(control_basis.shape[0],control_basis.shape[1]*control_basis.shape[2]).T
I2M = np.array( zwfs.control_variables[lab ]['CM'] ).T  
IM = np.array(zwfs.control_variables[lab ]['IM'] )
I0 = np.array(zwfs.control_variables[lab ]['sig_on_ref'].signal )
N0 = np.array(zwfs.control_variables[lab ]['sig_off_ref'].signal )


# ---------------
"""
testing updates: simulation is not compatiple with BMC muli3.5 DM (square without corners).
To update this we first
- make sure zwfs.dm.surface is always a 1D array (apply DM interpolates this and reshapes appropiately)
  DM coordinates are always set up relative to the input field in field.applyDM method such that spreads across entire input field
- we could define method for plotting in 2D (although not necessary)
- update zwfs.mode[dm][nact] to be total number of actuators - from this redefine how to build basis (maybe include
   a DM geometry field for this!! two options: square, BMC multi3.5 )
"""


#zwfs.control_variables[lab ]['control_basis'] = np.array( zwfs.control_variables[lab ]['control_basis'] ).reshape(-1,12,12)

## TEST 1. verification after making dm.surface is always 1D (previously it was 2D)
test_field = baldrSim.init_a_field( Hmag=0, mode=0, wvls=zwfs.wvls, pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'], dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'])

# put a mode on the DM 
b = zwfs.control_variables[lab ]['control_basis'][5]
b.reshape(1,-1)
zwfs.dm.update_shape( b * zwfs.control_variables[lab ]['pokeAmp']   )# zwfs.control_variables[lab ]['pokeAmp'] * M2C.T[5] )

plt.figure()
plt.imshow( zwfs.dm.surface.reshape(12,12) )
plt.title('DM surface')
# now apply DM to field 

post_dm_field = test_field.applyDM( zwfs.dm )
fig,ax = plt.subplots(1,2)
ax[0].imshow(zwfs.pup * test_field.phase[zwfs.wvls[0]])
ax[1].imshow(zwfs.pup * post_dm_field.phase[zwfs.wvls[0]])
ax[0].set_title('field before DM')
ax[1].set_title('field after DM')


#output =  zwfs.detection_chain( test_field )
sig_off = zwfs.detection_chain( test_field, zwfs.dm, zwfs.FPM_off, zwfs.det, replace_nan_with=None)
sig_on = zwfs.detection_chain( test_field, zwfs.dm, zwfs.FPM, zwfs.det, replace_nan_with=None)

fig,ax = plt.subplots(1,2)
ax[0].imshow(zwfs.dm.surface.reshape(12,12))
ax[1].imshow(sig_on.signal)
ax[0].set_title('DM surface')
ax[1].set_title('ZWFS intensity')
plt.show() 

## TEST 2. upodate zwfs.mode['DM'] and data_structure_functions such that N_act is not row actuators but total! 
# then redefine how we build our basis 

# test using both square and BMC multi-3.5 DM 

# test building each basis
 


# set up RTC controller in C++
r = rtc.RTC() 

cam_settings_tmp = rtc.camera_settings_struct()
reconstructors_tmp = rtc.phase_reconstuctor_struct()
pupil_regions_tmp = rtc.pupil_regions_struct()

# SET PUPIL REGIONS
pupil_regions_tmp.pupil_pixels.update( np.arange( len( I0.reshape(-1))).astype(int) )
pupil_regions_tmp.secondary_pixels.update( [len( I0.reshape(-1))//2] ) 
pupil_regions_tmp.outside_pixels.update( [] )

# COMMIT IT ALL 
pupil_regions_tmp.commit_all()
# append to our rtc 
r.regions = pupil_regions_tmp

# SET OUR RECO 
reconstructors_tmp = rtc.phase_reconstuctor_struct()

reconstructors_tmp.IM.update(IM.reshape(-1))
reconstructors_tmp.CM.update((M2C @ I2M).reshape(-1))

reconstructors_tmp.R_TT.update((M2C @ I2M).reshape(-1))
reconstructors_tmp.R_HO.update((M2C @ I2M).reshape(-1))

reconstructors_tmp.M2C.update(M2C.reshape(-1))
reconstructors_tmp.I2M.update(I2M.reshape(-1))
reconstructors_tmp.I0.update(I0.reshape(-1)/np.mean( I0.reshape(-1)[r.regions.pupil_pixels.current] )) #normalized
reconstructors_tmp.N0.update(N0.reshape(-1)/np.mean( I0.reshape(-1)[r.regions.pupil_pixels.current] )) #normalized
reconstructors_tmp.flux_norm.update(np.mean( I0.reshape(-1)[r.regions.pupil_pixels.current] ))   #normalized

# COMMIT IT ALL 
reconstructors_tmp.commit_all()
# append to our rtc 
r.reco = reconstructors_tmp


# Now generate input and run simulation 
# ====================

# TO DO 
# - won't be able to interact properly unless DM in simulation matches DM shape here (removing corners)

# generate static mode
#dx = zwfs.mode['telescope']['telescope_diameter'] / zwfs.mode['telescope']['telescope_diameter_pixels']
#input_field = baldrSim.init_a_field( Hmag=0, mode=5, wvls=zwfs.wvls, pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'], dx=dx)
#input_field = baldrSim.init_a_field( Hmag=0, mode='Kolmogorov', wvls=zwfs.wvls, pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'], dx=dx,r0=0.1, L0=25, phase_scale_factor = 1)
test_field = baldr.init_a_field( Hmag=0, mode=0, wvls=zwfs.wvls, pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'], dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'])

# put a mode on the DM 
zwfs.dm.update_shape( zwfs.control_variables[lab ]['pokeAmp'] * M2C[5] )

#output =  zwfs.detection_chain( test_field )
sig_off = zwfs.detection_chain( test_field, zwfs.dm, zwfs.FPM_off, zwfs.det, replace_nan_with=None)
sig_on = zwfs.detection_chain( test_field, zwfs.dm, zwfs.FPM, zwfs.det, replace_nan_with=None)

#sig = sig_on.signal / np.sum( sig_off.signal)  -  I0 / np.sum(N0)  

# do we reconstruct the mode? 
#I2M.T @ sig.reshape(-1)

Nph_obj = np.sum( sig_off.signal) 
Nph_cal = zwfs.control_variables[lab ]['Nph_cal']

#sig = sig_on.signal / np.sum( sig_off.signal)  -  I0 / np.sum(N0)  

mode_err = I2M.T @ (  1/Nph_obj * (sig_on.signal - Nph_obj/Nph_cal * I0) ).reshape(-1) 
#cmd_err = M2C.T @ mode_err 





plt.figure()
plt.imshow( input_field.phase[zwfs.wvls[0]] )
plt.show()


signal



Nmodes = M2C.shape[1]
kp = 1 * np.ones(Nmodes)
ki = 0.1 * np.ones(Nmodes)
kd = 0 * np.ones(Nmodes)
lower_limit = -100 * np.ones(Nmodes)
upper_limit = 100 * np.ones(Nmodes)

pid_tmp = rtc.PIDController( kp, ki, kd, lower_limit, upper_limit )
leaky_tmp = rtc.LeakyIntegrator( ki, lower_limit, upper_limit ) 


r = rtc.RTC()
r.reco = reconstructors_tmp
r.regions = pupil_regions_tmp 
r.pid = pid_tmp 
r.LeakyInt = leaky_tmp







#
# test 
#test_field = baldr.init_a_field( Hmag=0, mode='Kolmogorov', wvls=zwfs.wvls, pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'], dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'], r0=0.1, L0=25, phase_scale_factor = 9e-1)
test_field = baldr.init_a_field( Hmag=0, mode=0, wvls=zwfs.wvls, pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'], dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'])

zwfs.dm.update_shape( zwfs.control_variables[lab ]['pokeAmp'] * M2C[5] )

#output =  zwfs.detection_chain( test_field )

sig_off = zwfs.detection_chain( test_field, zwfs.dm, zwfs.FPM_off, zwfs.det, replace_nan_with=None)
sig_on = zwfs.detection_chain( test_field, zwfs.dm, zwfs.FPM, zwfs.det, replace_nan_with=None)

#sig = sig_on.signal / np.sum( sig_off.signal)  -  I0 / np.sum(N0)  

# do we reconstruct the mode? 
#I2M.T @ sig.reshape(-1)

Nph_obj = np.sum( sig_off.signal) 
Nph_cal = zwfs.control_variables[lab ]['Nph_cal']

#sig = sig_on.signal / np.sum( sig_off.signal)  -  I0 / np.sum(N0)  

mode_err = I2M.T @ (  1/Nph_obj * (sig_on.signal - Nph_obj/Nph_cal * I0) ).reshape(-1) 
#cmd_err = M2C.T @ mode_err 






PID 




























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