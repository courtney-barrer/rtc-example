
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

#%% 1) ------------------------------------------------------------------
r = rtc.RTC() 
conig_file_name = None #if None we just get most recent reconstructor file in /data/ path

# Note we do not do this as a function because the \
# we get memory errors in RTC struc when manipulating in 
# local scope of python function

#states_tmp = rtc.rtc_state_struct() 
#sim_signals = rtc.simulated_signals_struct()
cam_settings_tmp = rtc.camera_settings_struct()
reconstructors_tmp = rtc.phase_reconstuctor_struct()
pupil_regions_tmp = rtc.pupil_regions_struct()

conig_file_name = None
if conig_file_name==None:
    # get most recent 
    list_of_recon_files = glob.glob('data/' + 'RECONS*')
    conig_file_name = max(list_of_recon_files, key=os.path.getctime) #'RECONSTRUCTORS_debugging_DIT-0.002005_gain_medium_10-07-2024T22.21.55.fits'#"RECONSTRUCTORS_TEST_RTC_DIT-0.002005_gain_medium_10-07-2024T19.51.53.fits" #"RECONSTRUCTORS_test_DIT-0.002004_gain_high_05-07-2024T10.09.47.fits"#"RECONSTRUCTORS_try2_DIT-0.002003_gain_medium_04-06-2024T12.40.05.fits"
    #/home/baldr/Documents/baldr/ANU_demo_scripts/BALDR/data
    #reco_filename = reco_filename


config_fits = fits.open( conig_file_name  ) 

# camera settings used to build reconstructor 
det_fps = float( config_fits['info'].header['camera_fps'] ) # frames per sec (Hz) 

det_dit = float( config_fits['info'].header['camera_tint'] )  # integration time (seconds)

det_gain = str( config_fits['info'].header['camera_gain'] ) # camera gain 

det_cropping_rows = str( config_fits['info'].header['cropping_rows'] ).split('rows: ')[1]

det_cropping_cols = str( config_fits['info'].header['cropping_columns'] ).split('columns: ')[1]

cam_settings_tmp.det_fps = det_fps
cam_settings_tmp.det_dit = det_dit
cam_settings_tmp.det_gain = det_gain
cam_settings_tmp.det_cropping_rows = det_cropping_rows
cam_settings_tmp.det_cropping_cols = det_cropping_cols 


# reconstructor data 
R_TT = config_fits['R_TT'].data.astype(np.float32) #tip-tilt reconstructor

R_HO = config_fits['R_HO'].data.astype(np.float32) #higher-oder reconstructor

IM = config_fits['IM'].data.astype(np.float32) # interaction matrix (unfiltered)

M2C = config_fits['M2C'].data.astype(np.float32) # mode to command matrix 

I2M = np.transpose( config_fits['I2M'].data).astype(np.float32) # intensity (signal) to mode matrix  
# (# transposed so we can multiply directly I2M @ signal)
    
CM = config_fits['CM'].data.astype(np.float32)  # full control matrix 

I0 = config_fits['I0'].data.astype(np.float32) # calibration source reference intensity (FPM IN)

N0 = config_fits['N0'].data.astype(np.float32) # calibration source reference intensity (FPM OUT)


# pupil region classification data 
pupil_pixels = np.array( config_fits['pupil_pixels'].data, dtype=np.int32)

secondary_pixels = np.array( config_fits['secondary_pixels'].data, dtype=np.int32)

outside_pixels = np.array( config_fits['outside_pixels'].data, dtype=np.int32)


reconstructors_tmp.IM.update(IM.reshape(-1))
reconstructors_tmp.CM.update(CM.reshape(-1))
reconstructors_tmp.R_TT.update(R_TT.reshape(-1))
reconstructors_tmp.R_HO.update(R_HO.reshape(-1))
reconstructors_tmp.M2C.update(M2C.reshape(-1))
reconstructors_tmp.I2M.update(I2M.reshape(-1))
reconstructors_tmp.I0.update(I0.reshape(-1)/np.mean( I0.reshape(-1)[pupil_pixels] )) #normalized
reconstructors_tmp.N0.update(N0.reshape(-1)/np.mean( I0.reshape(-1)[pupil_pixels] )) #normalized
reconstructors_tmp.flux_norm.update(np.mean( I0.reshape(-1)[pupil_pixels] ))   #normalized

# COMMIT IT ALL 
reconstructors_tmp.commit_all()
# -----------------------------

pupil_regions_tmp.pupil_pixels.update( pupil_pixels )
pupil_regions_tmp.secondary_pixels.update( secondary_pixels ) 
pupil_regions_tmp.outside_pixels.update( outside_pixels )
#filter(lambda a: not a.startswith('__'), dir(pupil_regions_tmp))

# COMMIT IT ALL 
pupil_regions_tmp.commit_all()
# -----------------------------

# Simple check 
if len( pupil_pixels ) != CM.shape[1]:
    raise TypeError("number of pupil pixels (for control) does not match\
    control matrix size!!! CHeck why, did you input correct files?")

# update our RTC object 
# _RTC.regions = pupil_regions_tmp
# _RTC.reco = reconstructors_tmp
# _RTC.camera_settings = cam_settings_tmp

r.regions = pupil_regions_tmp
r.reco = reconstructors_tmp
r.camera_settings = cam_settings_tmp
# do we return it or is it static?
#return(_RTC)


# check it updated correcetly 
#print('current CM for r:', r.reco.CM.current )
#print('current pupil_pixels for r:', r.regions.pupil_pixels.current )

r.apply_camera_settings()

# Controllers 
# we could use for example PID for tip/tilt reconstructor and leaky integrator for HO 
Nmodes = M2C.shape[1]
kp = np.zeros(Nmodes)
ki = np.zeros(Nmodes)
kd = np.zeros(Nmodes)
lower_limit = -100 * np.ones(Nmodes)
upper_limit = 100 * np.ones(Nmodes)
pid_setpoint = np.zeros(Nmodes)
pid_tmp = rtc.PIDController( kp, ki, kd, lower_limit, upper_limit , pid_setpoint)
leaky_tmp = rtc.LeakyIntegrator( ki, lower_limit, upper_limit ) 

# set them up with our RTC object 
r.pid = pid_tmp
r.LeakyInt = leaky_tmp






#%% 2) ------------------------------------------------------------------
if r.rtc_state.camera_simulation_mode :
    r.rtc_simulation_signals.simulated_image = I0.reshape(-1) # simulated image
    r.rtc_simulation_signals.simulated_signal = IM[0] # simulated signal (processed image) - 
    # if r.rtc_state.signal_simulation_mode=true than this simulated signal over-rides any 
    # images (simulated or not) as input to the controllers.
    r.rtc_simulation_signals.simulated_dm_cmd = M2C.T[0] # simulated DM command 

# to check it
#plt.figure();plt.imshow( util.get_DM_command_in_2D( r.rtc_simulation_signals.simulated_dm_cmd) ); plt.show()


#%% 3) ------------------------------------------------------------------
# test some functionality 

# polling image and convert to vector 
test1 = r.im2vec_test()
if(len(test1) == r.camera_settings.full_image_length):
    print( ' passed im2vec_test')
else:
    print( ' FAILED --- im2vec_test')

# polling and reducing using the camera settings dark and bad pixels 

test1 = r.reduceImg_test()
if(len(test1) == r.camera_settings.full_image_length):
    print( ' passed reduceImg test')
else:
    print( ' FAILED --- reduceImg test')


# polling image and convert to vector and filter for pupil pixels 
test2 = r.im2filtered_im_test()
if(len(test2 ) == len(r.regions.pupil_pixels.current)):
    print( ' passed im2filtered_im_test')
else:
    print( ' FAILED --- im2filtered_im_test')

# filter reference intensity setpoint for pupil pixels 
test3 = r.im2filteredref_im_test()
if(len(test3 ) == len(r.regions.pupil_pixels.current)):
    print( ' passed test3 .im2filteredref_im_test()')
else:
    print( ' FAILED --- test3 .im2filteredref_im_test()')

# process image 
test4 = r.process_im_test()
if(len(test4 ) == len(r.regions.pupil_pixels.current)):
    print( ' passed test4 .process_im_test()')
else:
    print( ' FAILED --- test4 .process_im_test()')



# test 6 - controllers 
# we could use for example PID for tip/tilt reconstructor and leaky integrator for HO 
Nmodes = M2C.shape[1]
kp = np.zeros(Nmodes)
ki = np.zeros(Nmodes)
kd = np.zeros(Nmodes)
lower_limit = -100 * np.ones(Nmodes)
upper_limit = 100 * np.ones(Nmodes)
pid_setpoint =  np.zeros(Nmodes)

pid_tmp = rtc.PIDController( kp, ki, kd, lower_limit, upper_limit , pid_setpoint)
leaky_tmp = rtc.LeakyIntegrator( ki, lower_limit, upper_limit ) 

# set them up with our RTC object 
r.pid = pid_tmp
r.LeakyInt = leaky_tmp


# example of getting controller output (note pid first arg is input, second is setpoint)
out_pid = r.pid.process( np.ones(Nmodes) ) # 
out_leaky = r.LeakyInt.process( np.ones(Nmodes) ) #
print('passed test5.1.1 output memeber matches actual output for LeakyInt', out_leaky == r.LeakyInt.output)

# check output memeber matches actual output 
r.pid.kp = np.ones( Nmodes )
out_pid2 = r.pid.process( np.ones(Nmodes) ) # 
print('passed test5.1.2 output memeber matches actual output for PID', out_pid2 == r.pid.output)


if(len( out_pid ) == Nmodes):
    print( f' passed test6 r.pid.process( np.ones(Nmodes), np.zeros(Nmodes) )')
else:
    print( f' FAILED --- test6 r.pid.process( np.ones(Nmodes), np.zeros(Nmodes) )')

if(len( out_leaky ) == Nmodes):
    print( f' passed test6 r.LeakyInt.process( np.ones(Nmodes) )')
else:
    print( f' FAILED --- test6 r.LeakyInt.process( np.ones(Nmodes) )')



# Now test cases in single compute - we have have int return_case variables which is used in switch statement 
# to test different things. Generally we follow that if return_case is in the:
# - ones - matrix multiplication to cmd or modal space
# - hundreds - apply PID controller in respective spaces after matrix multiplcation 
# - thousands - reconstruct final DM command


# ONES: matrix multiplication test with all our phase reconstructors
for i, (lens, lab) in enumerate(zip([R_TT.shape[0],R_HO.shape[0],I2M.shape[0],CM.shape[0]],['TT','HO','I2M','CM'])):
    test5 = r.single_compute(i+1)
    print(len(test5 ))
    if(len(test5 ) == lens):
        print( f' passed test5 .single_compute({i}) ({lab} reconstructor multiplication)')
    else:
        print( f' FAILED --- test5 .single_compute({i}) ({lab} reconstructor multiplication)')


# Test if we input directly a signal from the interaction matrix that we reconstruct that mode perfectly with PID
r.rtc_state.camera_simulation_mode = True 
r.rtc_state.dm_simulation_mode = True
r.rtc_state.signal_simulation_mode = True 

mode_num = 5
r.rtc_simulation_signals.simulated_image = I0.reshape(-1) # simulated image
r.rtc_simulation_signals.simulated_signal = IM[mode_num] # simulated signal (processed image) - 
r.rtc_simulation_signals.simulated_dm_cmd = M2C.T[0] # simulated DM command 

mode_reco_test = np.array( r.single_compute(3) ) 
signal_on_mode = abs( mode_reco_test[5] - 1) < 0.1
nosignal_outside_mode = np.all( abs( mode_reco_test[np.arange(len(mode_reco_test))!=5]) < 0.01 )
if signal_on_mode & nosignal_outside_mode :
    print( f'--\npassed mode reconstruction test of interaction matrix signal injection for mode {mode_num}' )
elif not signal_on_mode :
    print( f'--\nfailed mode reconstruction test : signal on mode {mode_num} injection not close to 1' )
elif not nosignal_outside_mode:
    print( f'--\nfailed mode reconstruction test : signal on mode other modes !={mode_num} (mode injected) not close to 0' )
"""plt.plot( mode_reco_test )
plt.axvline( mode_num ); plt.show()"""





# TENS: testing PID controller

# always have to set up PID so things have right length 
def init_pid_to_right_size_given_casenumber( base_case_number ):
    """
    see rtc.cpp RTC.single_compute switch cases.
    casenumber here should be in the ones (i.e. 0< case_number < 10)
    """
    # tip/tilt 
    tmp_cmd_err = r.single_compute(base_case_number)
    # init PID correctly 
    r.pid.kp = 1 * np.ones(len(tmp_cmd_err))
    r.pid.ki = 0 * np.ones(len(tmp_cmd_err))
    r.pid.kd = 0 * np.ones(len(tmp_cmd_err))
    r.pid.lower_limit = -100 * np.ones(len(tmp_cmd_err))
    r.pid.upper_limit = 100 * np.ones(len(tmp_cmd_err))
    r.pid.setpoint = 0 * np.ones(len(tmp_cmd_err))

def init_leakyInt_to_right_size_given_casenumber( base_case_number ):
    """
    see rtc.cpp RTC.single_compute switch cases.
    casenumber here should be in the ones (i.e. 0< case_number < 10)
    """
    # tip/tilt 
    tmp_cmd_err = r.single_compute(base_case_number)
    # init PID correctly 
    r.LeakyInt.rho = 1 * np.ones(len(tmp_cmd_err))
    r.LeakyInt.lower_limit = -100 * np.ones(len(tmp_cmd_err))
    r.LeakyInt.upper_limit = 100 * np.ones(len(tmp_cmd_err))




# TESTING THE 10s (PID output lengths)
for i, (lens, lab) in enumerate(zip([R_TT.shape[0],R_HO.shape[0],I2M.shape[0],CM.shape[0]],['TT','HO','I2M','CM'])):
    init_pid_to_right_size_given_casenumber( base_case_number = i+1 )
    test5 = r.single_compute((i+1)+10)
    print(len(test5 ))
    if(len(test5 ) == lens):
        print( f' passed test5 .single_compute({i+1+10}) ({lab} reconstructor multiplication+PID output)')
    else:
        print( f' FAILED --- test5 .single_compute({i+1+10}) ({lab} reconstructor multiplication+PID output)')



"""

Testing PID controller in simulation mode
- double kp (with ki,kd =0 ) and see double in the output
- ki=1 (kp,kd=0) and no actuation leads to integration in output

"""

r.rtc_state.camera_simulation_mode = True 
r.rtc_state.dm_simulation_mode = True
r.rtc_state.signal_simulation_mode = True 

mode_num = 5
r.rtc_simulation_signals.simulated_image = I0.reshape(-1) # simulated image
r.rtc_simulation_signals.simulated_signal = IM[mode_num] # simulated signal (processed image) - 
r.rtc_simulation_signals.simulated_dm_cmd = M2C.T[0] # simulated DM command 

init_pid_to_right_size_given_casenumber( base_case_number = 3 )
r.pid.kp = np.ones( len( r.pid.ki ))
r.pid.ki = np.zeros( len( r.pid.kp ))
r.pid.kd = np.zeros( len( r.pid.kp ))
mode_reco_test_pid_1 = np.array( r.single_compute(13) )
# now double  kp and ensure output doubles 
r.pid.kp = 2* np.ones( len( r.pid.ki ))
mode_reco_test_pid_2 = np.array( r.single_compute(13) )

double_pid_test = abs(mode_reco_test_pid_2[mode_num] / mode_reco_test_pid_1[mode_num] - 2) < 0.01 
"""
plt.figure()
plt.plot( mode_reco_test_pid_1 ,'o')
plt.plot( mode_reco_test_pid_2 , 'o')
plt.axvline( mode_num ); plt.show()
"""
# Check errs integrate properly (ki=1, no change in measurement so model errs should grow with integral term)
r.pid.reset()
r.pid.kp = np.zeros( len( r.pid.kp ))
r.pid.ki = np.ones( len( r.pid.ki ))
r.pid.kd = np.zeros( len( r.pid.kp ))
mode_err_test_list = []
for _ in range(10):
    mode_err_test_list.append( r.single_compute(13) )

pid_ki_test = np.array([m[mode_num] for m in mode_err_test_list])

if np.mean( np.diff(pid_ki_test)) > 0.5:
    print( 'pass pid_ki_test. Check errs integrate properly (ki=1, no change in measurement so model errs should grow with integral term)')

"""

Testing Leaky integrator controller in simulation mode acting on modal space
- rho=0 leads output = input
- rho=1 with no actuation leads to integration in the signals 
here we using simulated input signal directly from IM so should perfectly reconstruct mode amplitude

"""
# Testing leaky integration behaviour
mode_num = 5
r.rtc_simulation_signals.simulated_image = I0.reshape(-1) # simulated image
r.rtc_simulation_signals.simulated_signal = IM[mode_num] # simulated signal (processed image) - 
r.rtc_simulation_signals.simulated_dm_cmd = M2C.T[0] # simulated DM command 

init_pid_to_right_size_given_casenumber( base_case_number = 3 )
r.LeakyInt.reset()
r.LeakyInt.rho = np.zeros( len( r.LeakyInt.rho ) )

rho_equal_zero_list = []
rho_equal_one_list = []
for _ in range(10):
    rho_equal_zero_list.append( np.array( r.single_compute(23) ) ) # case 23: using simulated signal, multiply by I2M, apply leaky int.

r.LeakyInt.reset()
r.LeakyInt.rho = np.ones( len( r.LeakyInt.rho ) )
for _ in range(10):
    rho_equal_one_list.append( np.array( r.single_compute(23) ) ) # case 23: using simulated signal, multiply by I2M, apply leaky int.

plt.figure()
plt.plot([r[mode_num] for r in rho_equal_zero_list], label='rho=0' )
plt.plot([r[mode_num] for r in rho_equal_one_list], label='rho=1' )
plt.title('should see integration for rho=1, constant for rho=0')
plt.legend()



# TESTING THE 20s (Leaky integrator output lengths)
for i, (lens, lab) in enumerate(zip([R_TT.shape[0],R_HO.shape[0],I2M.shape[0],CM.shape[0]],['TT','HO','I2M','CM'])):
    init_leakyInt_to_right_size_given_casenumber( base_case_number = i+1 )
    test5 = r.single_compute((i+1)+20)
    print(len(test5 ))
    if(len(test5 ) == lens):
        print( f' passed test5 .single_compute({i+1+20}) ({lab} reconstructor multiplication+leaky integrator output)')
    else:
        print( f' FAILED --- test5 .single_compute({i+1+20}) ({lab} reconstructor multiplication+leaky integrator output)')



# Testing Telemetry 

# Testing leaky integration behaviour
mode_num = 5
r.rtc_simulation_signals.simulated_image = I0.reshape(-1) # simulated image
r.rtc_simulation_signals.simulated_signal = IM[mode_num] # simulated signal (processed image) - 
r.rtc_simulation_signals.simulated_dm_cmd = M2C.T[0] # simulated DM command 

r.enable_telemetry(10)
print( 'initial telemetry_cnt = ',r.telemetry_cnt)
init_leakyInt_to_right_size_given_casenumber( base_case_number = 3 )
test_telemetry = r.single_compute( 223 )
print( 'telemetry_cnt after 1 call = ', r.telemetry_cnt)

# TO GET TELEMETRY 
t = rtc.get_telemetry()
tel_img = np.array([tt.image_in_pupil for tt in t] )
tel_imgErr = np.array([tt.image_err_signal for tt in t])
tel_modeErr = np.array([tt.mode_err for tt in t])
tel_dmErr = np.array([tt.dm_cmd_err for tt in t])



#plt.imshow( util.get_DM_command_in_2D( tel_dmErr[0] ) ); plt.show()
signal_on_mode = abs( tel_modeErr[0][mode_num ] - 1) < 0.1
nosignal_outside_mode = np.all( abs( tel_modeErr[0][np.arange(len(mode_reco_test))!=mode_num ]) < 0.01 )
if signal_on_mode & nosignal_outside_mode :
    print( f'--\npassed mode reconstruction test WITH TELEMETRY DATA using interaction matrix signal injection for mode {mode_num}' )
elif not signal_on_mode :
    print( f'--\nfailed mode reconstruction test  WITH TELEMETRY DATA : signal on mode {mode_num} injection not close to 1' )
elif not nosignal_outside_mode:
    print( f'--\nfailed mode reconstruction test  WITH TELEMETRY DATA : signal on mode other modes !={mode_num} (mode injected) not close to 0' )


r.enable_telemetry(30)
for _ in range(31):
    r.latency_test()
    print( 'telemetry_cnt = ', r.telemetry_cnt)

t = rtc.get_telemetry()
tel_img = np.array([tt.image_in_pupil for tt in t] )
tel_imgErr = np.array([tt.image_err_signal for tt in t])
tel_modeErr = np.array([tt.mode_err for tt in t])
tel_dmErr = np.array([tt.dm_cmd_err for tt in t])


print( 'tel_img.shape = ', tel_img.shape )