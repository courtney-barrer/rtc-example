#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:53:02 2024

@author: bencb

ZWFS class can only really get an image and send a command to the DM and/or update the camera settings

it does have an state machine, any processes interacting with ZWFS object
must do the logic to check and update state

"""


import numpy as np
import os
import glob 
import sys  
import pandas as pd 
import datetime 

sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')

import bmc
import FliSdk_V2
from . import utilities as util

                


class ZWFS():
    # for now one camera, one DM per object - real system will have 1 camera, 4 DMs!
    def __init__(self, DM_serial_number, cameraIndex=0, DMshapes_path='/home/baldr/Documents/baldr/DMShapes/', pupil_crop_region=[None, None, None, None] ):
       
        # connecting to camera
        camera = FliSdk_V2.Init() # init camera object
        listOfGrabbers = FliSdk_V2.DetectGrabbers(camera)
        listOfCameras = FliSdk_V2.DetectCameras(camera)
        # print some info and exit if nothing detected
        if len(listOfGrabbers) == 0:
            print("No grabber detected, exit.")
            exit()
        if len(listOfCameras) == 0:
            print("No camera detected, exit.")
            exit()
        for i,s in enumerate(listOfCameras):
            print("- index:" + str(i) + " -> " + s)
       
        print(f'--->using cameraIndex={cameraIndex}')
        # set the camera
        camera_err_flag = FliSdk_V2.SetCamera(camera, listOfCameras[cameraIndex])
        if not camera_err_flag:
            print("Error while setting camera.")
            exit()
        print("Setting mode full.")
        FliSdk_V2.SetMode(camera, FliSdk_V2.Mode.Full)
        print("Updating...")
        camera_err_flag = FliSdk_V2.Update(camera)
        if not camera_err_flag:
            print("Error while updating SDK.")
            exit()
           
        # connecting to DM
       
        dm = bmc.BmcDm() # init DM object
        dm_err_flag  = dm.open_dm(DM_serial_number) # open DM
        if dm_err_flag :
            print('Error initializing DM')
            raise Exception(dm.error_string(dm_err_flag))
       
        # ========== CAMERA & DM
        self.camera = camera
        self.dm = dm
        self.dm_number_of_actuators = 140

        # ========== DM shapes
        shapes_dict = {}
        if os.path.isdir(DMshapes_path):
            filenames = glob.glob(DMshapes_path+'*.csv')
            if filenames == []:
                print('no valid files in path. Empty dictionary appended to ZWFS.dm_shapes')
            else:
                for file in filenames:
                
                    try:
                        shape_name = file.split('/')[-1].split('.csv')[0]
                        shape = pd.read_csv(file, header=None)[0].values
                        if len(shape)==self.dm_number_of_actuators:
                            shapes_dict[shape_name] = np.array( (shape) )
                        else: 
                            print(f'file: {file}\n has {len(shape)} values which does not match the number of actuators on the DM ({self.dm_number_of_actuators})')

                    except:
                        print(f'falided to read and/or append shape corresponding to file: {file}')
        else:
            print('DMshapes path does not exist. Empty dictionary appended to ZWFS.dm_shapes')

        self.dm_shapes = shapes_dict
        
        #  we can aggregate the output image over a sub window WxW where W = pixelation_factor.
        # we use this sometimes when oversample pupil, so can use this to get to expected number of pixels with some careful choices
        self.pixelation_factor = None 
        

        # this is placed as an attribute here since from 1 image we will have 4 pupils 
        # which will control 4 different DMs.
        # So we need to define where to find each pupil for any given image taken. 
        if not hasattr(pupil_crop_region, '__len__'):
            #pupil_crop_region == None:
            pupil_crop_region = [None, None, None, None] # no cropping of the image
        elif hasattr(pupil_crop_region, '__len__'):
            if len( pupil_crop_region ) == 4:
                #if all(isinstance(x, int) for x in pupil_crop_region):
                self.pupil_crop_region = pupil_crop_region  # [row1 , row2, col1 , col2]
                #else:
                #    self.pupil_crop_region = [None, None, None, None]
                #    print('pupil_crop_region has INVALID entries, it needs to be integers')
            else:
                self.pupil_crop_region = [None, None, None, None]
                print('pupil_crop_region has INVALID length, it needs to have length = 4')
        else:
            self.pupil_crop_region = [None, None, None, None]
            print('pupil_crop_region has INVALID type. Needs to be list of integers of length = 4')

        # define our image coordinates based on cropping and pixelisation
        self._update_image_coordinates( )

        # define our region where we can robustly expect to uniquely find the ZWFS pupil. 

        # ========== initiate info on reference regions that controllers will need to work.
        """
        # this can be updated later after DM, pupil centering, for example using: 
        report = pupil_control.analyse_pupil_openloop( zwfs, debug = True, return_report = True) 
        as input to zwfs.update_reference_regions_in_img( report ).
        This has to be updated before any controller can be applied!
        """

        #boolean filters for reference regions 
        self.pupil_pixel_filter = [np.nan]
        self.secondary_pixel_filter = [np.nan]
        self.outside_pixel_filter = [np.nan]
        self.refpeak_pixel_filter = [np.nan]
        # pixels in reference regions  
        self.pupil_pixels = [np.nan]  
        self.secondary_pixels = [np.nan]
        self.outside_pixels = [np.nan]
        self.refpeak_pixels = [np.nan]
        # reference centers 
        self.pupil_center_ref_pixels = (np.nan, np.nan)
        self.dm_center_ref_pixels = (np.nan, np.nan)
        self.secondary_center_ref_pixels =  (np.nan, np.nan)

        # DM coordinates in pixel space relative to self.row_coords, self.col_coords
        self.dm_col_coordinates_in_pixels = [np.nan]
        self.dm_row_coordinates_in_pixels = [np.nan]

        # Note this is duplicated in phase_control object... need to decide where best to keep it! 
        # both pupil and phase controllers need this reference - so maybe best here?
        self.N0   =  None #2D array of intensity when FPM OUT 
        self.I0   =  None #2D array of intensity when FPM IN 


        # ========== CONTROLLERS
        self.phase_controllers = [] # initiate with no controllers
        self.pupil_controllers = [] # initiate with no controllers

        # ========== STATES
        """
        Notes:
        - Any process that takes ZWFS object as input is required to update ZWFS states
        - before using any controller, the ZWFS states should be compared with controller
        configuration file to check for consistency (i.e. can controller be applied in the current state?)
        """
        self.states = {}
       
        self.states['simulation_mode'] = 0 # 1 if we have in simulation mode
       
        self.states['telescopes'] = ['AT1'] #
        self.states['phase_ctrl_state'] = 0 # 0 open loop, 1 closed loop
        self.states['pupil_ctrl_state'] = 0 # 0 open loop, 1 closed loop
        self.states['source'] = 0 # 0 for no source, 1 for internal calibration source, 2 for stellar source
        self.states['sky'] = 0 # 1 if we are on a sky (background), 0 otherwise
        self.states['fpm'] = 0 # 0 for focal plance mask out, positive number for in depending on which one)
        self.states['busy'] = 0 # 1 if the ZWFS is doing something
        # etc 
       



    def _update_image_coordinates(self): 
        # define coordinates in our image based on cropping and pixelisation factors 
        
        #define pixel coordinates in this cropped region  
        r1, r2, c1, c2 = self.pupil_crop_region
        if (np.all(self.pupil_crop_region != [None, None, None, None])) & (self.pixelation_factor == None) :
            self.row_coords = np.linspace( (r1 - r2)/2 , (r2 - r1)/2, r2-r1)  #rows
            self.col_coords = np.linspace( (c1 - c2)/2 , (c2 - c1)/2, c2-c1)  #columns
        elif (np.all(self.pupil_crop_region != [None, None, None, None])) & (self.pixelation_factor != None) :
            pf = self.pixelation_factor
            self.row_coords = np.linspace( 1/pf * (r1 - r2)/2 , 1/pf * (r2 - r1)/2, int(1/pf * (r2-r1) ) )  #rows
            self.col_coords = np.linspace( 1/pf * (c1 - c2)/2 , 1/pf * (c2 - c1)/2, int(1/pf * (c2-c1) ) )  #columns

        elif np.all((self.pupil_crop_region == [None, None, None, None])) & (self.pixelation_factor == None) :# full image 
            r1 = 0; c1 = 0
            c2, r2 = FliSdk_V2.GetCurrentImageDimension(self.camera)
            self.row_coords = np.linspace( (r1 - r2)/2 , (r2 - r1)/2, r2-r1)  #rows
            self.col_coords = np.linspace( (c1 - c2)/2 , (c2 - c1)/2, c2-c1)  #columns

        elif np.all((self.pupil_crop_region == [None, None, None, None])) & (self.pixelation_factor != None) :
            r1 = 0; c1 = 0
            c2, r2 = FliSdk_V2.GetCurrentImageDimension(self.camera)
            pf = self.pixelation_factor
            self.row_coords = np.linspace( 1/pf * (r1 - r2)/2 , 1/pf * (r2 - r1)/2, int(1/pf * (r2-r1) ) )  #rows
            self.col_coords = np.linspace( 1/pf * (c1 - c2)/2 , 1/pf * (c2 - c1)/2, int(1/pf *(c2-c1) ) )  #columns




    def send_cmd(self, cmd):
       
        self.dm_err_flag = self.dm.send_data(cmd)
       
       
   
    def propagate_states(self, simulation_mode = False):
       
        if not self.states['simulation_mode']:
           
            for state, value in self.states.items():
                try:
                    print('check the systems state relative to current ZWFS state and \
                      rectify (e.g. send command to some motor) if any discrepency')
                except:
                    print('raise an error or implement some workaround if the requested state cannot be realised')

    def get_image(self):

        # I do not check if the camera is running. Users should check this 
        # gets the last image in the buffer
        img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)
        cropped_img = img[self.pupil_crop_region[0]:self.pupil_crop_region[1],self.pupil_crop_region[2]: self.pupil_crop_region[3]].astype(int)  # make sure int and not uint16 which overflows easily     
        
        if type( self.pixelation_factor ) == int : 
            cropped_img = util.block_sum(ar=cropped_img, fact = self.pixelation_factor)
        elif self.pixelation_factor != None:
            raise TypeError('ZWFS.pixelation_factor has to be of type None or int')
        return(cropped_img)    

    def get_image_in_another_region(self, crop_region=[0,-1,0,-1]):
        
        # I do not check if the camera is running. Users should check this 
        # gets the last image in the buffer
        img = FliSdk_V2.GetRawImageAsNumpyArray( self.camera , -1)
        cropped_img = img[crop_region[0]:crop_region[1],crop_region[2]: crop_region[3]].astype(int)  # make sure int and not uint16 which overflows easily     
        
        #if type( self.pixelation_factor ) == int : 
        #    cropped_img = util.block_sum(ar=cropped_img, fact = self.pixelation_factor)
        #elif self.pixelation_factor != None:
        #    raise TypeError('ZWFS.pixelation_factor has to be of type None or int')
        return( cropped_img )    


    def start_camera(self):
        FliSdk_V2.Start(self.camera)

    def stop_camera(self):
        FliSdk_V2.Stop(self.camera)

    def get_camera_dit(self):
        camera_err_flag, DIT = FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "tint raw")
        return( DIT ) 

    def get_camera_fps(self):
        camera_err_flag, fps = FliSdk_V2.FliSerialCamera.GetFps(self.camera)
        return( fps ) 

    def get_dit_limits(self):
               
        camera_err_flag, minDIT = FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "mintint raw")
        self.camera_err_flag = camera_err_flag
       
        camera_err_flag, maxDIT = FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "maxtint raw")
        self.camera_err_flag = camera_err_flag
       
        return(minDIT, maxDIT)   

    def set_camera_dit(self, DIT):
        # set detector integration time (DIT). input in seconds
        minDit, maxDit = self.get_dit_limits()
        if (DIT >= float(minDit)) & (DIT <= float(maxDit)):
            FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set tint " + str(float(DIT)))
        else:
            print(f"requested DIT {1e3*DIT}ms outside DIT limits {(1e3*minDit,1e3*maxDit)}ms.\n Cannot change DIT to this value")
    
    def set_camera_fps(self, fps):
        FliSdk_V2.FliSerialCamera.SetFps(self.camera, fps)

    def set_camera_cropping(self, r1, r2, c1, c2 ): 
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set cropping off")
        # cropped columns must be multiple of 32 - multiple of 32 minus 1
        #e.g. fli.FliSerialCamera.SendCommand(camera, "set cropping columns 64-287")
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, f"set cropping columns {c1}-{c2}")
        # cropped rows must be multiple of 4 - multiple of 4 minus 1
        #e.g. fli.FliSerialCamera.SendCommand(camera, "set cropping rows 120-299")
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, f"set cropping rows {r1}-{r2}")
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set cropping on")
    
    def deactive_cropping(self):
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set cropping off")

    def shutdown(self):
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "shutdown")

    def enable_frame_tag(self, tag = True):
        if tag:
            FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set imagetags on")
        else:
            FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "set imagetags off")

    def restore_default_settings(self): 
        FliSdk_V2.FliSerialCamera.SendCommand(self.camera, "restorefactory")

    def update_pupil_pixel_filter(self): 
        # updates self.pupil_pixel_filter, self.pupil_pixels, 
        # Need to decide if we also have to update central reference points here? - I dont think so
        return(None)
 
    def update_reference_regions_in_img(self, pupil_report ):
       
        self.pupil_pixel_filter = pupil_report['pupil_pixel_filter']
        self.secondary_pixel_filter = pupil_report['secondary_pupil_pixel_filter']
        self.outside_pixel_filter = pupil_report['outside_pupil_pixel_filter']
        self.refpeak_pixel_filter = pupil_report['reference_field_peak_filter']

        self.pupil_pixels = pupil_report['pupil_pixels']  
        self.secondary_pixels = pupil_report['secondary_pupil_pixels']
        self.outside_pixels = pupil_report['outside_pupil_pixels']
        self.refpeak_pixels = pupil_report['reference_field_peak_pixels']

        self.pupil_center_ref_pixels = pupil_report['pupil_center_ref_pixels']
        self.dm_center_ref_pixels = pupil_report['dm_center_ref_pixels']
        self.secondary_center_ref_pixels =  pupil_report['secondary_center_ref_pixels']

        # DM coordinates in pixel space relative to self.row_coords, self.col_coords
        self.dm_col_coordinates_in_pixels = pupil_report['dm_x_coords_in_pixels']
        self.dm_row_coordinates_in_pixels = pupil_report['dm_y_coords_in_pixels']

        # Note this is duplicated in phase_control object... need to decide where best to keep it! 
        self.N0   =  pupil_report['N0'] #2D array of intensity when FPM OUT 
        self.I0   =  pupil_report['I0'] #2D array of intensity when FPM IN 


    
    def plot_classified_pupil_region(self)
        # to check pupil regions are defined ok. 
        # If you want to check
        fig,ax = plt.subplots(1,5,figsize=(20,4))

        ax[0].imshow( self.pupil_pixel_filter.reshape(self.I0_2D.shape)) # cp_x2-cp_x1, cp_y2-cp_y1) )
        ax[1].imshow( self.outside_pupil_pixel_filter.reshape(self.I0_2D.shape))#( cp_x2-cp_x1, cp_y2-cp_y1) )
        ax[2].imshow( self.secondary_pupil_pixel_filter.reshape(self.I0_2D.shape))#( cp_x2-cp_x1, cp_y2-cp_y1) )
        ax[3].imshow( I0_2D.shape )
        ax[4].imshow( N0_2D.shape )
        
        for axx,l in zip(ax, ['inside pupil','outside pupil','secondary','I0','N0']):
            axx.set_title(l)
        plt.show()
    
    def write_reco_fits( self, phase_controller, ctrl_label , save_path):
        """
        phase_controller object from phase_control module
        ctrl_label is a string indicating the label used 
        when calibrating the phase_controller. see phase_control 
        module for details. 

        """
        # timestamp
        tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

        # == info about set up 
        info_fits = fits.PrimaryHDU( [] ) #[pupil_classification_file,WFS_response_file] )

        info_fits.header.set('EXTNAME','INFO')
        #info_fits.header.set('IM_construction_method',f'{reconstruction_method}')
        #info_fits.header.set('DM poke amplitude[normalized]',f'{poke_amps[amp_idx]}')
        #info_fits.header.set('what is?','names of input files for building IM')
        
        # INFO - camera info
        camera_info_dict = util.get_camera_info(self.camera)
        for k,v in camera_info_dict.items():
            info_fits.header.set(k,v)
        info_fits.header.set('#images per DM command', number_images_recorded_per_cmd )
        info_fits.header.set('take_median_of_images', take_median_of_images )
        
        # NOTE!! cropping_corners define cropping post camera read-out
        # they are different to the camera cropping 
        # IMPORTANT: if not NONE, pixel filters (e.g. pupil_pixel) are defined 
        # relative to this crop region! 
        info_fits.header.set('cropping_corners_r1', zwfs.pupil_crop_region[0] )
        info_fits.header.set('cropping_corners_r2', zwfs.pupil_crop_region[1] )
        info_fits.header.set('cropping_corners_c1', zwfs.pupil_crop_region[2] )
        info_fits.header.set('cropping_corners_c2', zwfs.pupil_crop_region[3] )

        # INFO - control basis info
        info_fits.header.set('control_basis', phase_controller.config['basis'] )
        info_fits.header.set('number_of_controlled_modes', phase_controller.config['number_of_controlled_modes'] )
        info_fits.header.set('dm_control_diameter', phase_controller.config['dm_control_diameter'] )
        info_fits.header.set('dm_control_center', phase_controller.config['dm_control_center'] )

        info_fits.header.set('CM_build_method', 'FILL ME' ) # how did we build the CM 
        # push, pull, push-pull ? 

        # INTERACTION MATRIX 
        IM_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]["IM"]  )  # unfiltered interaction matrix  
        IM_fits.header.set('what is?','unfiltered interaction matrix (IM)')
        IM_fits.header.set('EXTNAME','IM')      

        # CONTROL MATRIX
        CM_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]["CM"]   )  # filtered control matrix 
        CM_fits.header.set('what is?','filtered control matrix (CM)')
        CM_fits.header.set('EXTNAME','CM')
        
        # REFERENCE INTENSITY WITH FPM OUT (N0)
        # normalized and filtered with defined pupil (1D)
        #N0_filt_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]["ref_pupil_FPM_out"]  )
        #N0_filt_fits.header.set('what is?','pupil filtered FPM OUT')
        #N0_filt_fits.header.set('EXTNAME','N0_filt') 

        # normalized, unfiltered (2D)
        N0_fits = fits.PrimaryHDU( self.N0  )
        N0_fits.header.set('what is?','FPM OUT')
        N0_fits.header.set('EXTNAME','N0')        
        
        # REFERENCE INTENSITY WITH FPM IN (I0)
        # normalized and filtered with defined pupil (1D)
        #I0_filt_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]["ref_pupil_FPM_in"]  )
        #I0_filt_fits.header.set('what is?','pupil filtered FPM IN')
        #I0_filt_fits.header.set('EXTNAME','I0_filt')       

        # normalized, unfiltered (2D)
        I0_fits = fits.PrimaryHDU( self.I0  )
        I0_fits.header.set('what is?','FPM IN')
        I0_fits.header.set('EXTNAME','I0')        

        # MODE TO DM COMMAND MATRIX (normalized <C|C> = 1)
        M2C_fits = fits.PrimaryHDU( phase_controller.config['M2C']  ) # mode to commands
        M2C_fits.header.set('what is?','mode to dm cmd matrix')
        M2C_fits.header.set('EXTNAME','M2C')


        # NOTE: if not cropping_corners!=NONE, pixel filters (e.g. pupil_pixel) are defined 
        # relative to this crop region! which is cropped post frame read-out.

        # pixel inside pupil
        pupil_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]['pupil_pixels']  )
        pupil_fits.header.set('what is?','pixels_inside_pupil')
        pupil_fits.header.set('EXTNAME','pupil_pixels')

        # secondary 
        secondary_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]['secondary_pupil_pixels']  )
        secondary_fits.header.set('what is?','pixels_inside_secondary obstruction')
        secondary_fits.header.set('EXTNAME','secondary_pixels')

        #pixel outside pupil and not in secondary obstruction
        outside_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]['outside_pupil_pixels'] )
        outside_fits.header.set('what is?','pixels_outside_pupil')
        outside_fits.header.set('EXTNAME','outside_pixels')

        #DM center pixel during CM calibration
        dm_pixel_center_fits = fits.PrimaryHDU( phase_controller.ctrl_parameters[ctrl_label]['dm_center_ref_pixels']  )
        dm_pixel_center_fits .header.set('what is?','dm_center_reference_pixels')
        dm_pixel_center_fits .header.set('EXTNAME','dm_center_ref')

        # TO DO... Depends on modal basis used 
        RTT_fits = fits.PrimaryHDU( np.zeros(CM.shape) )
        RTT_fits.header.set('what is?','tip-tilt reconstructor')
        RTT_fits.header.set('EXTNAME','R_TT')

        RHO_fits = fits.PrimaryHDU(  np.zeros(CM.shape) )
        RHO_fits.header.set('what is?','higher-oder reconstructor')
        RHO_fits.header.set('EXTNAME','R_HO')

        fits_list = [info_fits, IM_fits, CM_fits, N0_fits, I0_fits, M2C_fits,\
        pupil_fits, secondary_fits, outside_fits, dm_pixel_center_fits,\
        RTT_fits,RH0_fits]


        reconstructor_fits = fits.HDUList( [] )
        for f in fits_list:
            reconstructor_fits.append( f )

        reconstructor_fits.writeto( save_path + f'RECONSTRUCTORS_DIT-{round(float(info_fits.header["camera_tint"]),6)}_gain_{info_fits.header["camera_gain"]}_{tstamp}.fits',overwrite=True )  