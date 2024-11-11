from baldr import _baldr as ba
from baldr import sardine as sa

import copy
import subprocess
import numpy as np
import time
import datetime
import sys
from pathlib import Path
import os 
from astropy.io import fits
import json
import numpy as np
import matplotlib.pyplot as plt


from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage
import pyqtgraph as pg

#from astropy.io import fits

cred1_command_dict = {
    "all raw": "Display, colon-separated, camera parameters",
    "powers": "Get all camera powers",
    "powers raw": "raw printing",
    "powers getter": "Get getter power",
    "powers getter raw": "raw printing",
    "powers pulsetube": "Get pulsetube power",
    "powers pulsetube raw": "raw printing",
    "temperatures": "Get all camera temperatures",
    "temperatures raw": "raw printing",
    "temperatures motherboard": "Get mother board temperature",
    "temperatures motherboard raw": "raw printing",
    "temperatures frontend": "Get front end temperature",
    "temperatures frontend raw": "raw printing",
    "temperatures powerboard": "Get power board temperature",
    "temperatures powerboard raw": "raw printing",
    "temperatures water": "Get water temperature",
    "temperatures water raw": "raw printing",
    "temperatures ptmcu": "Get pulsetube MCU temperature",
    "temperatures ptmcu raw": "raw printing",
    "temperatures cryostat diode": "Get cryostat temperature from diode",
    "temperatures cryostat diode raw": "raw printing",
    "temperatures cryostat ptcontroller": "Get cryostat temperature from pulsetube controller",
    "temperatures cryostat ptcontroller raw": "raw printing",
    "temperatures cryostat setpoint": "Get cryostat temperature setpoint",
    "temperatures cryostat setpoint raw": "raw printing",
    "fps": "Get frame per second",
    "fps raw": "raw printing",
    "maxfps": "Get the max frame per second regarding current camera configuration",
    "maxfps raw": "raw printing",
    "peltiermaxcurrent": "Get peltiermaxcurrent",
    "peltiermaxcurrent raw": "raw printing",
    "ptready": "Get pulsetube ready information",
    "ptready raw": "raw printing",
    "pressure": "Get cryostat pressure",
    "pressure raw": "raw printing",
    "gain": "Get gain",
    "gain raw": "raw printing",
    "bias": "Get bias correction status",
    "bias raw": "raw printing",
    "flat": "Get flat correction status",
    "flat raw": "raw printing",
    "imagetags": "Get tags in image status",
    "imagetags raw": "raw printing",
    "led": "Get LED status",
    "led raw": "raw printing",
    "sendfile bias <bias image file size> <file MD5>": "Interpreter waits for bias image binary bytes; timeout restarts interpreter.",
    "sendfile flat <flat image file size> <file MD5>": "Interpreter waits for flat image binary bytes.",
    "getflat <url>": "Retrieve flat image from URL.",
    "getbias <url>": "Retrieve bias image from URL.",
    "gettestpattern <url>": "Retrieve test pattern images tar.gz file from URL for testpattern mode.",
    "testpattern": "Get testpattern mode status.",
    "testpattern raw": "raw printing",
    "events": "Camera events sending status",
    "events raw": "raw printing",
    "extsynchro": "Get external synchro usage status",
    "extsynchro raw": "raw printing",
    "rawimages": "Get raw images (no embedded computation) status",
    "rawimages raw": "raw printing",
    "getter nbregeneration": "Get getter regeneration count",
    "getter nbregeneration raw": "raw printing",
    "getter regremainingtime": "Get time remaining for getter regeneration",
    "getter regremainingtime raw": "raw printing",
    "cooling": "Get cooling status",
    "cooling raw": "raw printing",
    "standby": "Get standby mode status",
    "standby raw": "raw printing",
    "mode": "Get readout mode",
    "mode raw": "raw printing",
    "resetwidth": "Get reset width",
    "resetwidth raw": "raw printing",
    "nbreadworeset": "Get read count without reset",
    "nbreadworeset raw": "raw printing",
    "cropping": "Get cropping status (active/inactive)",
    "cropping raw": "raw printing",
    "cropping columns": "Get cropping columns config",
    "cropping columns raw": "raw printing",
    "cropping rows": "Get cropping rows config",
    "cropping rows raw": "raw printing",
    "aduoffset": "Get ADU offset",
    "aduoffset raw": "raw printing",
    "version": "Get all product versions",
    "version raw": "raw printing",
    "version firmware": "Get firmware version",
    "version firmware raw": "raw printing",
    "version firmware detailed": "Get detailed firmware version",
    "version firmware detailed raw": "raw printing",
    "version firmware build": "Get firmware build date",
    "version firmware build raw": "raw printing",
    "version fpga": "Get FPGA version",
    "version fpga raw": "raw printing",
    "version hardware": "Get hardware version",
    "version hardware raw": "raw printing",
    "status": (
        "Get camera status. Possible statuses:\n"
        "- starting: Just after power on\n"
        "- configuring: Reading configuration\n"
        "- poorvacuum: Vacuum between 10-3 and 10-4 during startup\n"
        "- faultyvacuum: Vacuum above 10-3\n"
        "- vacuumrege: Getter regeneration\n"
        "- ready: Ready to be cooled\n"
        "- isbeingcooled: Being cooled\n"
        "- standby: Cooled, sensor off\n"
        "- operational: Cooled, taking valid images\n"
        "- presave: Previous usage error occurred"
    ),
    "status raw": "raw printing",
    "status detailed": "Get last status change reason",
    "status detailed raw": "raw printing",
    "continue": "Resume camera if previously in error/poor vacuum state.",
    "save": "Save current settings; cooling/gain not saved.",
    "save raw": "raw printing",
    "ipaddress": "Display camera IP settings",
    "cameratype": "Display camera information",
    "exec upgradefirmware <url>": "Upgrade firmware from URL",
    "exec buildbias": "Build the bias image",
    "exec buildbias raw": "raw printing",
    "exec buildflat": "Build the flat image",
    "exec buildflat raw": "raw printing",
    "exec redovacuum": "Start vacuum regeneration",
    "set testpattern on": "Enable testpattern mode (loop of 32 images).",
    "set testpattern on raw": "raw printing",
    "set testpattern off": "Disable testpattern mode",
    "set testpattern off raw": "raw printing",
    "set fps <fpsValue>": "Set the frame rate",
    "set fps <fpsValue> raw": "raw printing",
    "set gain <gainValue>": "Set the gain",
    "set gain <gainValue> raw": "raw printing",
    "set bias on": "Enable bias correction",
    "set bias on raw": "raw printing",
    "set bias off": "Disable bias correction",
    "set bias off raw": "raw printing",
    "set flat on": "Enable flat correction",
    "set flat on raw": "raw printing",
    "set flat off": "Disable flat correction",
    "set flat off raw": "raw printing",
    "set imagetags on": "Enable tags in image",
    "set imagetags on raw": "raw printing",
    "set imagetags off": "Disable tags in image",
    "set imagetags off raw": "raw printing",
    "set led on": "Turn on LED; blinks purple if operational.",
    "set led on raw": "raw printing",
    "set led off": "Turn off LED",
    "set led off raw": "raw printing",
    "set events on": "Enable camera event sending (error messages)",
    "set events on raw": "raw printing",
    "set events off": "Disable camera event sending",
    "set events off raw": "raw printing",
    "set extsynchro on": "Enable external synchronization",
    "set extsynchro on raw": "raw printing",
    "set extsynchro off": "Disable external synchronization",
    "set extsynchro off raw": "raw printing",
    "set rawimages on": "Enable embedded computation on images",
    "set rawimages on raw": "raw printing",
    "set rawimages off": "Disable embedded computation",
    "set rawimages off raw": "raw printing",
    "set cooling on": "Enable cooling",
    "set cooling on raw": "raw printing",
    "set cooling off": "Disable cooling",
    "set cooling off raw": "raw printing",
    "set standby on": "Enable standby mode (cools camera, sensor off)",
    "set standby on raw": "raw printing",
    "set standby off": "Disable standby mode",
    "set standby off raw": "raw printing",
    "set mode globalreset": "Set global reset mode (legacy compatibility)",
    "set mode globalresetsingle": "Set global reset mode (single frame)",
    "set mode globalresetcds": "Set global reset correlated double sampling",
    "set mode globalresetbursts": "Set global reset multiple non-destructive readout mode",
    "set mode rollingresetsingle": "Set rolling reset (single frame)",
    "set mode rollingresetcds": "Set rolling reset correlated double sampling (compatibility)",
    "set mode rollingresetnro": "Set rolling reset multiple non-destructive readout",
    "set resetwidth <resetwidthValue>": "Set reset width",
    "set resetwidth <resetwidthValue> raw": "raw printing",
    "set nbreadworeset <nbreadworesetValue>": "Set read count without reset",
    "set nbreadworeset <nbreadworesetValue> raw": "raw printing",
    "set cropping on": "Enable cropping",
    "set cropping on raw": "raw printing",
    "set cropping off": "Disable cropping",
    "set cropping off raw": "raw printing",
    "set cropping columns <columnsValue>": "Set cropping columns selection; format: e.g., '1,3-9'.",
    "set cropping columns <columnsValue> raw": "raw printing",
    "set cropping rows <rowsValue>": "Set cropping rows selection; format: e.g., '1,3,9'.",
    "set cropping rows <rowsValue> raw": "raw printing",
    "set aduoffset <aduoffsetValue>": "Set ADU offset",
    "set aduoffset <aduoffsetValue> raw": "raw printing",
}


class fli( ):

    def __init__(self, cameraIndex=0 , roi=[None, None, None, None], config_file = 'default'):
        
        
        #self.camera = fli() #FliSdk_V2.Init() # init camera object

        # Set up shared memory frame
        self.init_camera(config_file)
        # Start the external process
        self.start_external_process()

        #self.setup_shared_memory()
        self.pupil_crop_region = roi 
        
        self.reduction_dict = {'bias':[], 'dark':[],'flat':[],'bad_pixel_mask':[]}
        
        self.command_dict = cred1_command_dict


    def init_camera(self, config_file = 'default'):

        if config_file is not None:
            cam_type = 'fli'  # using real camera
            if config_file == 'default':
                config_file = "rtc_config_jsons/cred1_camera_default_config.json"
                print(f'using default configuration file {config_file}')
            with open( config_file ) as f:
                cam_config = json.load(f)
        else:
            cam_type = 'fake'  # using fake camera
            cam_config = {
                'size' : frame_size_h * frame_size_w,
                'number': 100, # number of random frame rotating to be copied in shm
                'latency': 1000, # latency in μsec
            } 

        frame_size_h = 320
        frame_size_w = 256
        self.cam_command = ba.Command.create(ba.Cmd.pause)
        self.frame = sa.region.host.open_or_create('frames', shape=[frame_size_h, frame_size_w], dtype=np.uint16)
        self.frame_url = sa.url_of(self.frame)

        self.frame_lock = ba.SpinLock.create()
        self.frame_lock_url = sa.url_of(self.frame_lock)


        cam_config = {
            'component': 'camera',
            'type': cam_type,
            'config':  cam_config,
            'io': {
                'frame': self.frame_url.geturl(),
            },
            'sync': {
                'notify': self.frame_lock_url.geturl(),
            },
            'command': sa.url_of(self.cam_command).geturl(),
        }


        with open("baldr_config.json", "+w") as baldr_config_file :

            json.dump([cam_config], baldr_config_file) 

        #baldr_config_file.close()
        self.config_file = [cam_config] #copy.copy( baldr_config_file ) #.copy()


    # def setup_shared_memory(self):
    #     """Load configuration file and set up shared memory frame."""
    #     with open("baldr_config.json", "r") as f:
    #         config = json.load(f)
    #     frame_url = config[0]["io"]["frame"]

    #     # Access shared memory frame using sardine
    #     self.frame = sa.from_url(np.ndarray, frame_url)
    #     print(f"Shared memory frame accessed at: {frame_url}")

    def start_external_process(self):
        """Start the baldr_main process using subprocess."""
        # command = ["build/Release/baldr_main", "--config", "baldr_config.json"]
        # self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # print("Process started with PID:", self.process.pid)

        # print('start running camera')
        # time.sleep(0.2)
        # #self.cam_command.run()

        """Start the baldr_main process using subprocess and verify it's running."""
        try:
            # Start the process and redirect stdout and stderr
            command = ["build/Release/baldr_main", "--config", "baldr_config.json"]
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"Process started with PID: {self.process.pid}")

            # Give the process some time to start up
            time.sleep(0.5)  # Adjust this if needed based on the process startup time
            
            # Check if the process is still running
            if self.process.poll() is not None:
                # If the process has already terminated, read error output
                stdout, stderr = self.process.communicate()
                print("Error: Process failed to start.")
                print(f"Stdout: {stdout.decode('utf-8')}")
                print(f"Stderr: {stderr.decode('utf-8')}")
                self.process = None  # Clear the process variable if it failed
            else:
                print("Process started successfully.")
                # Run initial command to confirm connection
                self.cam_command.run()
        
        except Exception as e:
            print(f"Failed to start process: {str(e)}")
            self.process = None

    # send FLI command (based on firmware version)
    def send_fli_cmd(self, cmd ):
        val = 1 # FliSdk_V2.FliSerialCamera.SendCommand(self.camera, cmd)
        #if not val:
        #    print(f"Error with command {cmd}")
        return val 
    

    def print_camera_commands(self):
        """Prints all available commands and their descriptions in a readable format."""
        print('Available Camera Commands with "send_fli_cmd()" method:')
        print("=" * 30)
        for command, description in self.command_dict.items():
            print(f"{command}: {description}")
        print("=" * 30)


    def configure_camera( self, config_file , sleep_time = 0.2):
        """
        config_file must be json and follow convention
        that the cameras firmware CLI accepts the command
        > "set {k} {v}"
        where k is the config file key and v is the value
        """
        # stop the process 
        #self.process.terminate()
        self.exit_camera()

        # reconfigure (make new baldr_config?)
        self.init_camera( config_file = config_file)

        # run again 
        self.start_external_process()
        
    # basic wrapper functions
    def start_camera(self):
        self.cam_command.run() #FliSdk_V2.Start(self.camera)
        print('starting')
        return 1
    
    def stop_camera(self):
        self.cam_command.pause() #FliSdk_V2.Stop(self.camera)
        print('stopping')
        return 1
    
    def exit_camera(self):
        self.cam_command.exit()
        self.frame_lock.unlock()
        #self.process.terminate()
        #FliSdk_V2.Exit(self.camera)

    def get_last_raw_image_in_buffer(self):
        return self.frame
    

    def get_camera_config(self):
        # config_dict = {
        #     'mode':self.send_fli_cmd('mode raw' )[1], 
        #     'fps': self.send_fli_cmd('fps raw' )[1],
        #     'gain': self.send_fli_cmd('gain raw' )[1],
        #     "cropping_state": self.send_fli_cmd('cropping raw' )[1],
        #     "reset_width":self.send_fli_cmd('resetwidth raw' )[1],
        #     "aduoffset":self.send_fli_cmd( 'aduoffset raw' )[1],
        #     "resetwidth":self.send_fli_cmd( "resetwidth raw")[1]
        # } 

        # read in default_cred1_config

         
        # open the default config file to get the keys 
        #with open(os.path.join( self.config_file_path , "default_cred1_config.json"), "r") as file:
        #    default_cred1_config = json.load(file)  # Parses the JSON content into a Python dictionary

        #config_dict = {}
        #for k, v in default_cred1_config.items():
        #    config_dict[k] = self.send_fli_cmd( f"{k} raw" )[1].strip() # reads the state
        return( self.config_file )
     

    # some custom functions

    def build_manual_dark( self , no_frames = 100 ):
        
        # full frame variables here were used in previous rtc. 
        # maybe redundant now. 
        #fps = float( self.send_fli_cmd( "fps")[1] )
        #dark_fullframe_list = []
        
        #dark_list = []
        #for _ in range(no_frames):
        #    time.sleep(1/fps)
        #    dark_list.append( self.get_image(apply_manual_reduction  = False) )
        #    #dark_fullframe_list.append( self.get_image_in_another_region() ) 
        print('...getting frames')
        dark_list = self.get_some_frames(number_of_frames = no_frames, apply_manual_reduction=False, timeout_limit = 20000 )
        print('...aggregating frames')
        dark = np.median(dark_list ,axis = 0).astype(int)
        # dark_fullframe = np.median( dark_fullframe_list , axis=0).astype(int)

        if len( self.reduction_dict['bias'] ) > 0:
            print('...applying bias')
            dark -= self.reduction_dict['bias'][0]

        #if len( self.reduction_dict['bias_fullframe']) > 0 :
        #    dark_fullframe -= self.reduction_dict['bias_fullframe'][0]
        print('...appending dark')
        self.reduction_dict['dark'].append( dark )
        #self.reduction_dict['dark_fullframe'].append( dark_fullframe )



    def get_bad_pixel_indicies( self, no_frames = 100, std_threshold = 100 , flatten=False):
        # To get bad pixels we just take a bunch of images and look at pixel variance 
        #self.enable_frame_tag( True )
        time.sleep(0.5)
        #zwfs.get_image_in_another_region([0,1,0,4])
        
        dark_list = self.get_some_frames( number_of_frames = no_frames , apply_manual_reduction  = False  ) #[]
        #i=0
        # while len( dark_list ) < no_frames: # poll 1000 individual images
        #     full_img = self.get_image_in_another_region() # we can also specify region (#zwfs.get_image_in_another_region([0,1,0,4]))
        #     current_frame_number = full_img[0][0] #previous_frame_number
        #     if i==0:
        #         previous_frame_number = current_frame_number
        #     if current_frame_number > previous_frame_number:
        #         if current_frame_number == 65535:
        #             previous_frame_number = -1 #// catch overflow case for int16 where current=0, previous = 65535
        #         else:
        #             previous_frame_number = current_frame_number 
        #             dark_list.append( self.get_image( apply_manual_reduction  = False) )
        #     i+=1
        dark_std = np.std( dark_list ,axis=0)
        # define our bad pixels where std > 100 or zero variance
        #if not flatten:
        bad_pixels = np.where( (dark_std > std_threshold) + (dark_std == 0 ))
        #else:  # flatten is useful for when we filter regions by flattened pixel indicies
        bad_pixels_flat = np.where( (dark_std.reshape(-1) > std_threshold) + (dark_std.reshape(-1) == 0 ))

        #self.bad_pixels = bad_pixels_flat

        if not flatten:
            return( bad_pixels )
        else:
            return( bad_pixels_flat )


    def build_bad_pixel_mask( self, bad_pixels , set_bad_pixels_to = 0):
        """
        bad_pixels = tuple of array of row and col indicies of bad pixels.
        Can create this simply by bad_pixels = np.where( <condition on image> )
        gets a current image to generate bad_pixel_mask shape
        - Note this also updates zwfs.bad_pixel_filter  and zwfs.bad_pixels
           which can be used to filterout bad pixels in the controlled pupil region 
        """
        i = self.get_image(apply_manual_reduction = False )
        bad_pixel_mask = np.ones( i.shape )
        for ibad,jbad in list(zip(bad_pixels[0], bad_pixels[1])):
            bad_pixel_mask[ibad,jbad] = set_bad_pixels_to

        self.reduction_dict['bad_pixel_mask'].append( bad_pixel_mask )

        badpixel_bool_array = np.zeros(i.shape , dtype=bool)
        for ibad,jbad in list(zip(bad_pixels[0], bad_pixels[1])):
            badpixel_bool_array[ibad,jbad] = True
        
        self.bad_pixel_filter = badpixel_bool_array.reshape(-1)
        self.bad_pixels = np.where( self.bad_pixel_filter )[0]


    def get_image(self, apply_manual_reduction  = True, which_index = -1 ):

        # I do not check if the camera is running. Users should check this 
        # gets the last image in the buffer
        if not apply_manual_reduction:
            img = self.frame
            cropped_img = img[self.pupil_crop_region[0]:self.pupil_crop_region[1],self.pupil_crop_region[2]: self.pupil_crop_region[3]].astype(int)  # make sure int and not uint16 which overflows easily     
        else :
            img = self.frame
            cropped_img = img[self.pupil_crop_region[0]:self.pupil_crop_region[1],self.pupil_crop_region[2]: self.pupil_crop_region[3]].astype(int)  # make sure 

            if len( self.reduction_dict['bias'] ) > 0:
                cropped_img -= self.reduction_dict['bias'][which_index] # take the most recent bias. bias must be set in same cropping state 

            if len( self.reduction_dict['dark'] ) > 0:
                cropped_img -= self.reduction_dict['dark'][which_index] # take the most recent dark. Dark must be set in same cropping state 

            if len( self.reduction_dict['flat'] ) > 0:
                cropped_img /= np.array( self.reduction_dict['flat'][which_index] , dtype = type( cropped_img[0][0]) ) # take the most recent flat. flat must be set in same cropping state 

            if len( self.reduction_dict['bad_pixel_mask'] ) > 0:
                # enforce the same type for mask
                cropped_img *= np.array( self.reduction_dict['bad_pixel_mask'][which_index] , dtype = type( cropped_img[0][0]) ) # bad pixel mask must be set in same cropping state 

        return(cropped_img)    

    def get_image_in_another_region(self, crop_region=[None,None,None,None]):
        # useful if we want to look outside of the region of interest 
        # defined by self.pupil_crop_region

        img = self.frame
        cropped_img = img[crop_region[0]:crop_region[1],crop_region[2]: crop_region[3]].astype(int)  # make sure int and not uint16 which overflows easily     
        
        #if type( self.pixelation_factor ) == int : 
        #    cropped_img = util.block_sum(ar=cropped_img, fact = self.pixelation_factor)
        #elif self.pixelation_factor != None:
        #    raise TypeError('ZWFS.pixelation_factor has to be of type None or int')
        return( cropped_img )    
    

    def get_some_frames(self, number_of_frames = 100, apply_manual_reduction=True, timeout_limit = 20000 ):
        """
        poll sequential frames (no repeats) and store in list  
        """
        ref_img_list = []
        i=0
        timeout_counter = 0 
        timeout_flag = 0
        while (len( ref_img_list  ) < number_of_frames) and not timeout_flag: # poll  individual images
            if timeout_counter > timeout_limit: # we have done timeout_limit iterations without a frame update
                timeout_flag = 1 
                raise TypeError('timeout! timeout_counter > 10000')

            full_img = self.frame # empty argunment for full frame
            current_frame_number = full_img[0][0] #previous_frame_number
            if i==0:
                previous_frame_number = current_frame_number
            if current_frame_number > previous_frame_number:
                timeout_counter = 0 # reset timeout counter
                if current_frame_number == 65535:
                    previous_frame_number = -1 #// catch overflow case for int16 where current=0, previous = 65535
                else:
                    previous_frame_number = current_frame_number 
                    ref_img_list.append( self.frame )
            i+=1
            timeout_counter += 1
            
        return( ref_img_list )  


    def save_fits( self , fname ,  number_of_frames=10, apply_manual_reduction=True ):

        hdulist = fits.HDUList([])

        frames = self.get_some_frames( number_of_frames=number_of_frames, apply_manual_reduction=apply_manual_reduction,timeout_limit=20000)
        
        # Convert list to numpy array for FITS compatibility
        data_array = np.array(frames, dtype=float)  # Ensure it is a float array or any appropriate type

        # Create a new ImageHDU with the data
        hdu = fits.ImageHDU( np.array(frames) )

        # Set the EXTNAME header to the variable name
        hdu.header['EXTNAME'] = 'FRAMES'
        #hdu.header['config'] = config_file_name

        config_tmp = self.get_camera_config()
        for k, v in config_tmp.items():
            hdu.header[k] = v
        # Append the HDU to the HDU list
        hdulist.append(hdu)

        # append reduction info
        for k, v in self.reduction_dict.items():
            if len(v) > 0 :
                hdu = fits.ImageHDU( v[-1] )
                hdu.header['EXTNAME'] = k
                hdulist.append(hdu)
            else: # we just append empty list to show that its empty!
                hdu = fits.ImageHDU( v )
                hdu.header['EXTNAME'] = k
                hdulist.append(hdu)

        hdulist.writeto(fname, overwrite=True)


    # ensures we exit safely and set gain to unity
    def __del__(self):
        print("killing all")
        self.cam_command.exit()
        self.frame_lock.unlock()





class AOControlApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        # Initialize FLI camera
        self.camera = fli(0, roi=[None, None, None, None], config_file = 'default')
        #config_file_name = os.path.join(self.camera.config_file_path, "default_cred1_config.json")
        #self.camera.configure_camera(config_file_name)
        #self.camera.start_camera()


        # Main layout as a 2x2 grid
        main_layout = QtWidgets.QGridLayout(self)

        # Text input fields for min and max cut values
        self.min_cut_input = QtWidgets.QLineEdit()
        self.min_cut_input.setPlaceholderText("Min Cut")
        self.min_cut_input.returnPressed.connect(self.update_image_cuts)

        self.max_cut_input = QtWidgets.QLineEdit()
        self.max_cut_input.setPlaceholderText("Max Cut")
        self.max_cut_input.returnPressed.connect(self.update_image_cuts)

        # Button for auto cuts
        self.auto_cut_button = QtWidgets.QPushButton("Auto Cuts")
        self.auto_cut_button.clicked.connect(self.auto_cut)

        # Checkbox for image reduction
        self.reduce_image_checkbox = QtWidgets.QCheckBox("Reduce Image")
        self.reduce_image_checkbox.setChecked(True)  # Default to checked

        # Control layout for cut settings
        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(QtWidgets.QLabel("Min Cut"))
        control_layout.addWidget(self.min_cut_input)
        control_layout.addWidget(QtWidgets.QLabel("Max Cut"))
        control_layout.addWidget(self.max_cut_input)
        control_layout.addWidget(self.auto_cut_button)
        control_layout.addWidget(self.reduce_image_checkbox)

        # Camera frame placeholder (G[0,0])
        self.camera_view = pg.PlotWidget()
        self.camera_image = pg.ImageItem()
        self.camera_view.addItem(self.camera_image)

        self.camera_image.setLevels((0, 255))  # Set initial display range to 0-255 or any typical default range
        self.min_cut_input.setText("0")
        self.max_cut_input.setText("255")

        self.camera_view.setAspectLocked(True)
        main_layout.addWidget(self.camera_view, 0, 0)

        # Pixel value display label
        self.pixel_value_label = QtWidgets.QLabel("Pixel Value: N/A")
        main_layout.addWidget(self.pixel_value_label, 1, 0)

        # Connect mouse movement over image to show pixel values
        #self.proxy = pg.SignalProxy(self.camera_view.scene().sigMouseMoved, rateLimit=60, slot=self.show_pixel_value)

        # DM images in a vertical column (G[0,1])
        dm_layout = QtWidgets.QVBoxLayout()
        self.dm_images = []
        for i in range(4):
            dm_label = QtWidgets.QLabel(f"DM {i + 1}")
            dm_view = pg.PlotWidget()
            dm_image = pg.ImageItem()
            dm_view.addItem(dm_image)
            dm_view.setAspectLocked(True)
            dm_layout.addWidget(dm_label)
            dm_layout.addWidget(dm_view)
            self.dm_images.append(dm_image)
        main_layout.addLayout(dm_layout, 0, 1)

        # Command line prompt area (G[1,0])
        command_layout = QtWidgets.QVBoxLayout()
        self.prompt_history = QtWidgets.QTextEdit()
        self.prompt_history.setReadOnly(True)
        self.prompt_input = QtWidgets.QLineEdit()
        self.prompt_input.returnPressed.connect(self.handle_command)
        command_layout.addWidget(self.prompt_history)
        command_layout.addWidget(self.prompt_input)
        main_layout.addLayout(command_layout, 1, 0)

        # 4x2 button grid (G[1,1])
        button_layout = QtWidgets.QGridLayout()
        button_functions = [
            ("Start Camera", self.camera.start_camera),
            ("Stop Camera", self.camera.stop_camera),
            ("Save Config", self.save_camera_config),
            ("Load Config", self.load_camera_config),
            ("Save Images", self.save_images),
            ("Build Dark", self.build_dark),
            ("Get Bad Pixels", self.get_bad_pixels)
        ]
        for i, (text, func) in enumerate(button_functions):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(func)
            button_layout.addWidget(button, i // 2, i % 2)
        main_layout.addLayout(button_layout, 1, 1)

        # Integrate control layout (with cut settings) into the main layout
        main_layout.addLayout(control_layout, 2, 0, 1, 2)  # Positioned below the main camera view and button grid

        # Adjust row and column stretch to set proportions
        main_layout.setRowStretch(0, 3)
        main_layout.setRowStretch(1, 1)
        main_layout.setColumnStretch(0, 3)
        main_layout.setColumnStretch(1, 1)

        # Status LED
        self.status_led = QtWidgets.QLabel()
        self.status_led.setFixedSize(20, 20)
        self.update_led(False)
        main_layout.addWidget(self.status_led, 1, 1, QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight)

        # Timer for camera updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_camera_image)
        self.timer.start(100)

        # Command history and completer
        self.command_history = []
        self.history_index = -1
        self.completer = QtWidgets.QCompleter(self)
        self.prompt_input.setCompleter(self.completer)
        self.update_completer()

    # #redundant?
    # def update_camera_frame(self):
    #     """Update the main camera image from shared memory."""
    #     try:
    #         img_data = np.array(self.camera.frame, copy=False)  # Access shared memory directly
    #         self.camera_view.setImage(img_data, autoLevels=True)  # Update camera view
    #     except Exception as e:
    #         print(f"Failed to update camera frame: {e}")

    def auto_cut(self):
        """Automatically adjust the cut range for optimal viewing based on percentiles."""
        apply_manual_reduction = self.reduce_image_checkbox.isChecked()
        img = self.camera.get_image(apply_manual_reduction=apply_manual_reduction, which_index=-1)


        if img is not None:
            min_cut = np.percentile(img, 1)
            max_cut = np.percentile(img, 99)

            # Update text inputs with calculated cuts
            self.min_cut_input.setText(str(int(min_cut)))
            self.max_cut_input.setText(str(int(max_cut)))

            # Apply the new cut range
            self.update_image_cuts()
            self.prompt_history.append("Auto cut applied based on the 1st and 99th percentiles of the image.")
        else:
            self.prompt_history.append("No image data available for auto cut.")

    def update_image_cuts(self):
        """Update the min and max cuts for the displayed image based on text input values."""
        try:
            min_cut = int(self.min_cut_input.text())
            max_cut = int(self.max_cut_input.text())
            self.camera_image.setLevels((min_cut, max_cut))
        except ValueError:
            self.prompt_history.append("Invalid input for min or max cut.")

    def update_camera_image(self):
        apply_manual_reduction = self.reduce_image_checkbox.isChecked()
        img = self.camera.get_image(apply_manual_reduction=apply_manual_reduction, which_index=-1)
        if img is not None:
            self.camera_image.setImage(img)
            self.update_image_cuts()  # Apply current cut values to image

    def update_led(self, running):
        self.status_led.setStyleSheet("background-color: green;" if running else "background-color: red;")

    # def show_pixel_value(self, event):
    #     """Display pixel value at mouse hover position."""
    #     pos = event[0]  # event[0] holds the QtGui.QGraphicsSceneMouseEvent
    #     mouse_point = self.camera_view.getViewBox().mapSceneToView(pos)
    #     x, y = int(mouse_point.x()), int(mouse_point.y())

    #     # Check if mouse is within the image bounds
    #     if 0 <= x < self.camera_image.width() and 0 <= y < self.camera_image.height():
    #         img_data = self.camera_image.image  # Assuming the image data is stored in this attribute
    #         pixel_value = img_data[y, x]  # Note: y comes first for row-major order
    #         self.pixel_value_label.setText(f"Pixel Value: {pixel_value}")
    #     else:
    #         self.pixel_value_label.setText("Pixel Value: N/A")

    def handle_command(self):
        command = self.prompt_input.text()
        self.command_history.append(command)
        self.history_index = len(self.command_history)
        self.prompt_history.append(f"> {command}")
        
        try:
            if command.lower() == "autocuts":
                self.auto_cut()
                self.prompt_history.append("Image cut ranges updated based on current image.")
            else:
                response = self.camera.send_fli_cmd(command)
                self.prompt_history.append(f"Command executed. Reply:\n{response}")
        except Exception as e:
            self.prompt_history.append(f"Error: {str(e)}")
    
    def update_completer(self):
        model = QtGui.QStandardItemModel(self.completer)
        for cmd in self.camera.command_dict.keys():
            item = QtGui.QStandardItem(cmd)
            model.appendRow(item)
        self.completer.setModel(model)

    def reinitialize_camera(self, config_file_path = 'default'):
        self.camera.stop_camera()
        self.camera.exit_camera()
        self.camera = fli(0, roi=[None, None, None, None], config_file_path=config_file_path )
        #self.camera.configure_camera(config_file_path)
        self.camera.start_camera()
        self.prompt_history.append(f"Camera reinitialized with config: {config_file_path}")

    def save_camera_config(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Config", "", "Config Files (*.json)")
        if file_name:
            config_file = self.camera.get_camera_config()
            with open(file_name, 'w') as f:
                json.dump(config_file, f)

    def load_camera_config(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Config", "", "Config Files (*.json)")
        if file_name:
            self.reinitialize_camera(file_name)

    def save_images(self):
        self.timer.stop()
        self.camera.start_camera() # gui freezes if camera is off 
        apply_manual_reduction = self.reduce_image_checkbox.isChecked()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Images", "", "Image Files (*.fits)")
        self.camera.save_fits(fname=file_name, number_of_frames=10, apply_manual_reduction=apply_manual_reduction)
        self.timer.start(100)

    def build_dark(self):
        self.timer.stop()
        self.camera.start_camera() # gui freezes if camera is off 
        self.camera.build_manual_dark(no_frames=100)
        self.timer.start(100)

    def get_bad_pixels(self):
        self.timer.stop()
        self.camera.start_camera() # gui freezes if camera is off 
        bad_pixels = self.camera.get_bad_pixel_indicies(no_frames=100, std_threshold=100, flatten=False)
        self.camera.build_bad_pixel_mask(bad_pixels=bad_pixels, set_bad_pixels_to=0)
        self.timer.start(100)

    def closeEvent(self, event):
        self.camera.send_fli_cmd("set gain 1")
        self.camera.stop_camera()
        self.camera.exit_camera()
        event.accept()

    def __del__(self):
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.send_fli_cmd("set gain 1")
            self.camera.stop_camera()
            self.camera.exit_camera()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = AOControlApp()
    window.setWindowTitle("AO Control GUI")
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
