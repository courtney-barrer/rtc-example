import rtc
import numpy as np
import glob 
from astropy.io import fits

# init rtc 
r = rtc.RTC() 

# we can print, edit nested struc members in RTC struct directly. e.g. 
print( r.rtc_state.close_loop_mode)
# r.rtc_state.close_loop_mode = True # example of setting this field directly 

# or best we init nested data structures as indpendent structs to edit and directly update  
states = rtc.rtc_state_struct()
sim_signals = rtc.simulated_signals_struct()
cam_settings = rtc.camera_settings_struct()

cam_settings.det_dit = 0.0018

# set a rtc structure 
r.camera_settings = cam_settings

print( r.camera_settings.det_dit )

# these use updatable fields (see cpp code definition of updatable)


reconstructors = rtc.phase_reconstuctor_struct()
pupil_regions = rtc.pupil_regions_struct()


r.regions.outside_pixels.update([1,2,3])
pupil_regions.outside_pixels.update([2,2,2])
# this puts the values as next to commit to current 
pupil_regions.outside_pixels.commit()

pupil_regions.outside_pixels.current
pupil_regions.outside_pixels.next

# DONT KNOW WHY DOESN"T allow me to do it for the reconstructor struc 
reconstructors.I2M.update([1,2,3]) # doesn't work for reconstructors?? 
reconstructors.CM.next




def configure_RTC( _RTC , conig_file_name=None ):

    states_tmp = rtc.rtc_state_struct()
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
    det_fps = float(reco_fits['info'].header['camera_fps']) # frames per sec (Hz) 

    det_dit = float( reco_fits['info'].header['camera_tint'] )  # integration time (seconds)

    det_gain = str(reco_fits['info'].header['camera_gain']) # camera gain 

    det_cropping_rows = str( reco_fits['info'].header['cropping_rows'] ).split('rows: ')[1]

    det_cropping_cols = str( reco_fits['info'].header['cropping_columns'] ).split('columns: ')[1]

    # reconstructor data 
    R_TT = reco_fits['R_TT'].data.astype(np.float32) #tip-tilt reconstructor
    
    R_HO = reco_fits['R_HO'].data.astype(np.float32) #higher-oder reconstructor

    IM = reco_fits['IM'].data.astype(np.float32) # interaction matrix (unfiltered)
    
    M2C = reco_fits['M2C'].data.astype(np.float32) # mode to command matrix 
    
    I2M = np.transpose( reco_fits['I2M'].data).astype(np.float32) # intensity (signal) to mode matrix  
    # (# transposed so we can multiply directly I2M @ signal)
      
    CM = reco_fits['CM'].data.astype(np.float32)  # full control matrix 

    I0 = reco_fits['I0'].data.astype(np.float32) # calibration source reference intensity (FPM IN)
    
    N0 = reco_fits['N0'].data.astype(np.float32) # calibration source reference intensity (FPM OUT)

    # pupil region classification data 
    pupil_pixels = np.array( reco_fits['pupil_pixels'].data, dtype=np.int32)

    secondary_pixels = np.array( reco_fits['secondary_pixels'].data, dtype=np.int32)

    outside_pixels = np.array( reco_fits['outside_pixels'].data, dtype=np.int32)

    pupil_regions_tmp.pupil_pixels.update( pupil_pixels )
    pupil_regions_tmp.pupil_pixels.commit()

    pupil_regions_tmp.secondary_pixels.update( secondary_pixels ) 
    pupil_regions_tmp.secondary_pixels.commit()
    
    pupil_regions_tmp.outside_pixels.update( outside_pixels )
    pupil_regions_tmp.outside_pixels.commit()
    #filter(lambda a: not a.startswith('__'), dir(pupil_regions_tmp))

    pupil_regions_tmp.commit() 
    # Simple check 
    if len( pupil_pixels ) != CM.shape[1]:
        raise TypeError("number of pupil pixels (for control) does not match\
        control matrix size!!! CHeck why, did you input correct files?")


    # update our RTC object 
    _RTC.regions =  pupil_regions_tmp







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