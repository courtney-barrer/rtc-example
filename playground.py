import numpy as np
import glob 
from astropy.io import fits
import os 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import copy 

import rtc
sys.path.append('simBaldr/' )
sys.path.append('pyBaldr/' )
from pyBaldr import utilities as util
import baldr_simulation_functions as baldrSim
import data_structure_functions as config

# sys.path.append('/Users/bencb/Documents/rtc-example/simBaldr/' )
# dont import pyBaldr since dont have locally FLI / BMC 

"""
Testing the RTC in simulation mode using realistic inputs/outputs from baldrSim  

"""



#%%
#===================== SIMULATION 

# =========== Setup simulation 

#throughput = 0.01
#Hmag = 0
#Hmag_at_vltiLab = Hmag  - 2.5*np.log10(throughput)
#flux_at_vltilab = baldr.star2photons('H',Hmag_at_vltiLab,airmass=1,k=0.18,ph_m2_s_nm=True) #ph/m2/s/nm

# setting up the hardware and software modes of our ZWFS
tel_config =  config.init_telescope_config_dict(use_default_values = True)
phasemask_config = config.init_phasemask_config_dict(use_default_values = True) 

# -------- trialling this 
phasemask_config['on-axis phasemask depth'] = 4.210526315789474e-05
phasemask_config['off-axis phasemask depth'] = 4.122526315789484e-05

phasemask_config['phasemask_diameter'] = 1.1 * (phasemask_config['fratio'] * 1.65e-6)

#---------------------

DM_config_square = config.init_DM_config_dict(use_default_values = True) 
DM_config_bmc = config.init_DM_config_dict(use_default_values = True) 
DM_config_bmc['DM_model'] = 'BMC-multi3.5' #'square_12'
DM_config_square['DM_model'] = 'square_12'#'square_12'

# default is 'BMC-multi3.5'
detector_config = config.init_detector_config_dict(use_default_values = True)


# the only thing we need to be compatible is the pupil geometry and Npix, Dpix 
tel_config['pup_geometry'] = 'disk'

# define a hardware mode for the ZWFS 
mode_dict_bmc = config.create_mode_config_dict( tel_config, phasemask_config, DM_config_bmc, detector_config)
mode_dict_square = config.create_mode_config_dict( tel_config, phasemask_config, DM_config_square, detector_config)

#create our zwfs object with square DM
zwfs = baldrSim.ZWFS(mode_dict_square)
# now create another one with the BMC DM 
zwfs_bmc = baldrSim.ZWFS(mode_dict_bmc)



# -------- trialling this 

# Cold stops have to be updated for both FPM and FPM_off!!!!!!!
zwfs.FPM.update_cold_stop_parameters( None )
zwfs.FPM_off.update_cold_stop_parameters( None )

zwfs_bmc.FPM.update_cold_stop_parameters( None )
zwfs_bmc.FPM_off.update_cold_stop_parameters( None )


#---------------------
# define an internal calibration source 
calibration_source_config_dict = config.init_calibration_source_config_dict(use_default_values = True)
calibration_source_config_dict['temperature']=1900 #K (Thorlabs SLS202L/M - Stabilized Tungsten Fiber-Coupled IR Light Source )
calibration_source_config_dict['calsource_pup_geometry'] = 'Disk'

nbasismodes = 20
basis_labels = [ 'zernike','fourier', 'KL']
control_labels = [f'control_{nbasismodes}_{b}_modes' for b in basis_labels]



for b,lab in zip( basis_labels, control_labels):
    # square DM  
    # zwfs.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=nbasismodes, modal_basis=b, pokeAmp = 150e-9 , label=lab, replace_nan_with=0)
    # BMC multi3.5 DM 
    zwfs_bmc.setup_control_parameters(  calibration_source_config_dict, N_controlled_modes=nbasismodes, modal_basis=b, \
                                      pokeAmp = 150e-9 , label=lab,replace_nan_with=0, without_piston=True)




# ==================== A few basic checks for our ZWFS 

test_field = baldrSim.init_a_field( Hmag=10, mode=0, wvls=zwfs.wvls, pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'], dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'])

print( 'phase shift (deg) at 1.6um = ', zwfs.FPM.phase_mask_phase_shift( 1.6e-6 ))


psi_b = zwfs.FPM.get_output_field( test_field, keep_intermediate_products=True)

im_list = [ zwfs.FPM.phase_shift_region, abs( zwfs.FPM.Psi_B[0] )**2 ]
xlabel_list = ['' for _ in range(len(im_list))]
ylabel_list = ['' for _ in range(len(im_list))]
title_list = ['phase shift region','PSF \n(diffraction limited)']
cbar_label_list = ['[1/0]', 'intensity [adu]']
util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None)



#===================== SET-UP RTC (IN SIMULATION MODE)  

# set up RTC controller in C++
r = rtc.RTC() 

z = copy.deepcopy( zwfs_bmc ) # general name so we could potentially do this in for loop

lab = f'control_{nbasismodes}_zernike_modes' #f'control_{nbasismodes}_fourier_modes' #'control_20_zernike_modes' # 'control_20_fourier_modes'

control_basis =  np.array(z.control_variables[lab ]['control_basis'])
M2C = z.control_variables[lab ]['pokeAmp'] *  control_basis.T #.reshape(control_basis.shape[0],control_basis.shape[1]*control_basis.shape[2]).T
I2M = np.array( z.control_variables[lab ]['I2M'] ).T  
IM = np.array(z.control_variables[lab ]['IM'] )
I0 = np.array(z.control_variables[lab ]['sig_on_ref'].signal )
N0 = np.array(z.control_variables[lab ]['sig_off_ref'].signal )

#lets have a look at basis functions 
fig,ax = plt.subplots(int(np.sqrt(len(control_basis))),int(np.sqrt(len(control_basis))),figsize=(20,20))
for m, axx in zip(control_basis, ax.reshape(-1)):
    axx.imshow( baldrSim.get_BMCmulti35_DM_command_in_2D( m ))
plt.show()

# ---------------
# populating rtc structures
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
# I0 should be normalized by the reference with phase mask out, 
# we can use I0 for normalization only when considering large  enough area
reconstructors_tmp.I0.update(I0.reshape(-1)/np.sum( N0.reshape(-1)[r.regions.pupil_pixels.current] )) #normalized
reconstructors_tmp.N0.update(N0.reshape(-1)) #normalized
# THIS HAS TO BE UPDATED ON ANY NEW SOURCE!!! 
# HERE IS ONLY PLACEHOLDER 
reconstructors_tmp.flux_norm.update(np.sum( N0.reshape(-1)[r.regions.pupil_pixels.current] ))   #normalized

# COMMIT IT ALL 
reconstructors_tmp.commit_all()
# append to our rtc 
r.reco = reconstructors_tmp



# SET CONTROLLERS (PID and leaky integrator)
Nmodes = M2C.shape[1]
# pid
kp = list( 1.0 * np.ones(Nmodes) )
ki = list( 0.0 * np.ones(Nmodes) )
kd = list( 0.0 * np.ones(Nmodes) )
pid_setpoint = list( np.zeros( Nmodes ))
# leaky
rho = list( 1.0 * np.ones(Nmodes))

lower_limit = list( -100.0 * np.ones(Nmodes) )
upper_limit = list( 100.0 * np.ones(Nmodes) )

pid_tmp = rtc.PIDController( kp, ki, kd, lower_limit, upper_limit, pid_setpoint  )
leaky_tmp = rtc.LeakyIntegrator( rho, lower_limit, upper_limit ) 

r.pid = pid_tmp 
r.LeakyInt = leaky_tmp

# ====================
# Now generate input and run simulation 
# ====================
# Notes: DM active actuators and field don't have 1:1 overlap - so putting a field mode in (on the same basis)
# as DM is not necesarily orthogonal and can cause cross coupling. Using Fourier improves this for some modes! 
# (at least for lower order modes). Really need to move to Eigenbasis.
# also try removing tip/tilt?

# 0) set up simulation state and create a test input field 
# 1) in Baldrsim generate a detector intensity from input field and u
#   update normalization flux 
# 2) set this to the rtc simulated intensity 
# 3) do a single_compute that from:
#   intensity -> signal -> modal space -> apply controller -> DM command. Returns DM command
# 4) get the DM command, apply it to the simulated field, 
# 5) repeat from step 1 with the updated field.

# -- (0)
# we only simulate the input images and dm (do not simulated the processed signal)
r.rtc_state.camera_simulation_mode = True
r.rtc_state.signal_simulation_mode = False
r.rtc_state.dm_simulation_mode = True

mode_idx = 5
test_field = baldrSim.init_a_field( Hmag=2, mode=mode_idx, wvls=zwfs.wvls, pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'], dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'])

# -- (1)
# flatten DM ! 
z.dm.update_shape(  np.zeros( M2C.shape[0] ) )
# phase mask in 
i = z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
# phase mask out
o = z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
# check saturation
if np.max( i.signal ) > 2**16:
    raise TypeError('Saturation! max adu for detector in rtc is uint16. peak signal here > 2**16. Consider making test field fainter.')
# update our flux normalization based on FPM out measurement (simulated)
r.reco.flux_norm.update( np.sum( o.signal.reshape(-1)[r.regions.pupil_pixels.current] )) 
r.reco.commit_all()

# -- (2)
if r.rtc_state.camera_simulation_mode :
    r.rtc_simulation_signals.simulated_image = list( i.signal.reshape(-1) )# simulated image
    print(' updated simulated image in simulation mode')

# -- (3)
# single compute argunment has case numbers for testing different functionalities / modes of RTC
# case 113 corresponds to :
#    intensity -> signal -> modal space -> apply PID -> DM command.
# case 123 corresponds to :
#    intensity -> signal -> modal space -> apply leaky integrator -> DM command.
# where the intensity here is the simulated input 

r.pid.reset()
r.LeakyInt.reset()
# The reset doesn't really reset output!!!! 

#r.pid.kp[0] = 0
#r.LeakyInt.rho[0] = 0
cmd = r.single_compute( 123 )


# -- (4)
z.dm.update_shape(  np.mean( cmd ) - cmd )
post_dm_field = test_field.applyDM( z.dm )

wvl_ref = z.wvls[0] # wvl wher eto plot and compute metrics
im_list = [ zwfs.pup * 1e9 * wvl_ref /(2*np.pi) * test_field.phase[wvl_ref], i.signal, \
           1e9 * baldrSim.get_BMCmulti35_DM_command_in_2D( cmd ), \
            zwfs.pup * 1e9 * wvl_ref/(2*np.pi) * post_dm_field.phase[wvl_ref] ]
xlabel_list = ['' for _ in range(len(im_list))]
ylabel_list = ['' for _ in range(len(im_list))]
title_list = ['phase pre DM','detector signal', 'DM surface reco.','phase post DM']
cbar_label_list = ['OPD [nm]', 'intensity [adu]', 'OPD [nm]', 'phase [nm]']
savename = f'tmp/reco_input-Z{mode_idx}_basis-{lab}.png' #None
util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, \
                           title_list,cbar_label_list, fontsize=15, \
                            cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=savename)

print( f'strehl_before ({round(1e6*wvl_ref,2)}um) = ',np.exp( -np.nanvar( test_field.phase[wvl_ref][zwfs.pup>0] ) ))
print( f'strehl_after ({round(1e6*wvl_ref,2)}um)= ',np.exp( -np.nanvar( post_dm_field.phase[wvl_ref][zwfs.pup>0] ) ))


# -- (5)
# repeat  


# =========================
# Trying closed loop 


# SET CONTROLLERS (PID and leaky integrator)
Nmodes = M2C.shape[1]

"""
bug one ki !+ 0 
"""

# pid
kp = list( 0.5 * np.ones(Nmodes) )
ki = list( 0.0 * np.ones(Nmodes) )
kd = list( 0.0 * np.ones(Nmodes) )
pid_setpoint = list( np.zeros( Nmodes ))
# leaky
rho = list( 0.8 * np.ones(Nmodes))

lower_limit = list( -100.0 * np.ones(Nmodes) )
upper_limit = list( 100.0 * np.ones(Nmodes) )

pid_tmp = rtc.PIDController( kp, ki, kd, lower_limit, upper_limit, pid_setpoint  )
leaky_tmp = rtc.LeakyIntegrator( rho, lower_limit, upper_limit ) 

r.pid = pid_tmp 
r.LeakyInt = leaky_tmp



test_field = baldrSim.init_a_field( Hmag=1, mode='Kolmogorov', wvls=zwfs.wvls, \
                                   pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'],\
                                       dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'], \
                                           r0=0.1, L0 = 25, phase_scale_factor=0.9)


c_list = []
strehl_list = []
residual_list = []
u_list = []
e_list_real = []

wvl_ref = zwfs.wvls[0] # where to calculate Strehl ratio etc
r.pid.reset()
r.LeakyInt.reset()

#initial cmd 
c = np.zeros( M2C.shape[0] )
z.dm.update_shape( c )

#r.pid.kp[0] = 0
#r.pid.kp[1] = 0

#r.LeakyInt.rho[0]  = 0
#r.LeakyInt.rho[1]  = 0

# aberrations vs estimated aberrations 
field_basis = baldrSim.create_control_basis(None, N_controlled_modes=20, basis_modes=lab.split('_')[-2],\
                                    without_piston=True, not_associated_with_DM=zwfs.pup.shape[0])

plot=False
for it in range(100):
    
    strehl = np.exp( -np.nanvar( test_field.phase[wvl_ref][zwfs.pup>0] ) )
    print( strehl )
    strehl_list.append( strehl )
    
    # the real field aberation error 
    e_real = [np.nansum( test_field.phase[wvl_ref].reshape(-1) * a ) for a in field_basis ]

    i = z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    o = z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
    
    # check saturation
    if np.max( i.signal ) > 2**16:
        raise TypeError('Saturation! max adu for detector in rtc is uint16. peak signal here > 2**16. Consider making test field fainter.')
    # update our flux normalization based on FPM out measurement (simulated)
    r.reco.flux_norm.update( np.sum( o.signal.reshape(-1)[r.regions.pupil_pixels.current] )) 
    r.reco.commit_all()

    if r.rtc_state.camera_simulation_mode :
        r.rtc_simulation_signals.simulated_image = list( i.signal.reshape(-1) )# simulated image
        print(' updated simulated image in simulation mode')

    if plot:
        im_list = [ zwfs.pup * 1e9 * wvl_ref /(2*np.pi) * test_field.phase[wvl_ref], i.signal, \
            1e9 * baldrSim.get_BMCmulti35_DM_command_in_2D(  np.mean( c ) - c  ) ]
        xlabel_list = ['' for _ in range(len(im_list))]
        ylabel_list = ['' for _ in range(len(im_list))]
        title_list = ['phase', 'signal', 'DM surface reco.']
        cbar_label_list = ['OPD [nm]', 'intensity [adu]', 'OPD [nm]']
        savename = f'tmp/telem_basis-{lab}_it{it}.png' #None
        util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, \
                                title_list,cbar_label_list, fontsize=15, \
                                    cbar_orientation = 'bottom', axis_off=True, vlims=[[-500,500],[1000,6000], [-200,200]], savefig=savename)


    # with PID: 113, with leaky integrator: 123
    # leaky integrator works great after removing piston from basis function
    c = r.single_compute( 113 )


    z.dm.update_shape( c - np.mean( c ) )#  - c ) #
    
    test_field = test_field.applyDM( z.dm )
    
    c_list.append( c )
    u_list.append( r.pid.output )
    e_list_real.append( e_real )
    residual_list.append( test_field.phase[z.wvls[0]] )
    

fig,ax = plt.subplots(3,1,sharex=True,figsize=(5,10))
#plt.figure(); 
ax[0].plot( np.array(u_list)[:,:] )
ax[0].set_ylabel(f'modal residual (post controller)')
ax[1].plot( np.array(e_list_real)[:,:] )
ax[1].set_ylabel(f'real modal residual')
ax[2].plot( strehl_list )
ax[2].set_ylabel(f'Strehl Ratio at {round(1e6*wvl_ref,2)}um')
ax[2].set_xlabel('iterations')
plt.savefig( 'tmp/temp2.png' )



"""
fig,ax = plt.subplots(2,1,sharex=True)
#plt.figure(); 
ax[0].plot( np.array(u_list)[:,:] )
ax[0].set_ylabel(f'modal residual')
ax[1].plot( strehl_list )
ax[1].set_ylabel(f'Strehl Ratio at {round(1e6*wvl_ref,2)}um')
ax[1].set_xlabel('iterations')
plt.savefig( 'tmp/temp2.png' )
"""





"""
# -- (2.1) double checking in local simulation (not using RTC processes)

kp = 1 * np.ones( len(M2C.T) )
kp[0] = 0 # filter out 
ki = 0. * np.ones( len(M2C.T) )
kd = np.zeros( len(M2C.T) )
upper_limit = 100 *  np.ones( len(M2C.T) )
lower_limit = -100 * np.ones( len(M2C.T) )
setpoint = np.zeros( len(M2C.T) )

pid = baldrSim.PIDController(kp, ki, kd, upper_limit, lower_limit, setpoint)

i = z.detection_chain( test_field, FPM_on=True, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )
o = z.detection_chain( test_field, FPM_on=False, include_shotnoise=True, ph_per_s_per_m2_per_nm=True, grids_aligned=True, replace_nan_with=0 )

# this works 
sig = i.signal / np.sum( o.signal ) - I0 / np.sum( N0 )
# this does not work 
#sig = i.signal / np.sum( N0 ) - I0 / np.sum( N0 )

e = I2M @ sig.reshape(-1)

#print( e )
u = pid.process( I2M @ sig.reshape(-1) )

c = 1 * M2C @ u 

im_list = [ 1e9 * z.wvls[0]/(2*np.pi) * test_field.phase[z.wvls[0]], zwfs_intensity.signal, 1e9 * baldrSim.get_BMCmulti35_DM_command_in_2D( c ), 1e9 * z.wvls[0]/(2*np.pi) * post_dm_field.phase[z.wvls[0]] ]
xlabel_list = ['' for _ in range(len(im_list))]
ylabel_list = ['' for _ in range(len(im_list))]
title_list = ['phase pre DM','detector signal', 'DM surface', 'phase post DM']
cbar_label_list = ['OPD [nm]', 'intensity [adu]', 'OPD [nm]','OPD [nm]']
util.nice_heatmap_subplots(im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None)

"""



mode_num = 5
r.rtc_state.signal_simulation_mode = True
r.rtc_simulation_signals.simulated_signal = IM[mode_num] 
mode_reco_test = np.array( r.single_compute(3) ) 








# TO DO 
# - won't be able to interact properly unless DM in simulation matches DM shape here (removing corners)

# generate static mode
#dx = zwfs.mode['telescope']['telescope_diameter'] / zwfs.mode['telescope']['telescope_diameter_pixels']
#input_field = baldrSim.init_a_field( Hmag=0, mode=5, wvls=zwfs.wvls, pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'], dx=dx)
#input_field = baldrSim.init_a_field( Hmag=0, mode='Kolmogorov', wvls=zwfs.wvls, pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'], dx=dx,r0=0.1, L0=25, phase_scale_factor = 1)
test_field = baldrSim.init_a_field( Hmag=0, mode=0, wvls=zwfs.wvls, pup_geometry='disk', D_pix=zwfs.mode['telescope']['pupil_nx_pixels'], dx=zwfs.mode['telescope']['telescope_diameter']/zwfs.mode['telescope']['pupil_nx_pixels'])

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

mode_err = I2M.T @ ( 1/Nph_obj * (sig_on.signal - Nph_obj/Nph_cal * I0) ).reshape(-1) 
#cmd_err = M2C.T @ mode_err 





plt.figure()
plt.imshow( input_field.phase[zwfs.wvls[0]] )
plt.show()




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



<<<<<<< HEAD
r.apply_camera_settings()  # <------ segmentation fault here 
=======
>>>>>>> ebcb198b61353925d5194db419b1082549946db8






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