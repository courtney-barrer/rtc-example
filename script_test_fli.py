from baldr import _baldr as ba
from baldr import sardine as sa
import numpy as np
import json

global DM_COMMAND_SIZE = 140

with open("cred1_camera_default_config.json") as f:
    cred1_cam_config = json.load(f)

with open("bmc_DM_default_config.json") as f:
    bmc_dm_config = json.load(f)
    
with open("rtc_default_config.json") as f:
    rtc_config = json.load(f)

# for now we configure 1 RTC per DM .. ensure this is the case
assert len( bmc_dm_config ) == len( rtc_config )


###############
# DM
###############
commands_dict = {}
commands_url_dict = {}
dm_command_dict = {}
commands_lock_dict = {}
commands_lock_url_dict = {}
dm_config_list = []
for beam in range( len( bmc_dm_config )):
    commands_dict[beam] = sa.region.host.open_or_create(f'beam{beam}_commands', shape=[DM_COMMAND_SIZE], dtype=np.double)
    commands_url_dict[beam]  = sa.url_of(commands_dict[beam])
    dm_command_dict[beam] = ba.Command.create(ba.Cmd.pause)
    commands_lock_dict[beam] = ba.SpinLock.create()
    commands_lock_url_dict[beam] = sa.url_of(commands_lock_dict[beam])
    
    
dm_config_list = []
for beam, key in enumerate( bmc_dm_config ):
    tmp_config = {
        'beam': f'{beam}',
        'component': 'dm',
        'type': 'bmc',
        'config': bmc_dm_config[key], 
        'io': {
            'commands': commands_url_dict[beam].geturl(),
        },
        'sync': {
            'wait': commands_lock_url_dict[beam].geturl(),
        },
        'command': sa.url_of(dm_command_dict[beam]).geturl(),
    }
    dm_config_list.append(tmp_config)

###############
# CAMERA 
###############
frame_size_h = int( cred1_cam_config['image_height'] )
frame_size_w = int( cred1_cam_config['image_height'] )

commands_size = frame_size_h * frame_size_w

frame = sa.region.host.open_or_create('frames', shape=[frame_size_h, frame_size_w], dtype=np.uint16)    
frame_url = sa.url_of(frame)

cam_command = ba.Command.create(ba.Cmd.pause)

frame_lock = ba.SpinLock.create()

frame_lock_url = sa.url_of(frame_lock)



cam_config = {
    'component': 'camera',
    'type': 'fli',
    'config': cred1_cam_config,
    'io': {
        'frame': frame_url.geturl(), # where to write data 
    },
    'sync': {
        'notify': frame_lock_url.geturl(), # where to notify, rtc will wait on this lock to read frame 
    },
    'command': sa.url_of(cam_command).geturl(), # choice of commands pause, run, exit 
}



###############
# RTC
###############
rtc_command_dict = {}
rtc_config_list = []
for beam in range( len( bmc_dm_config ) ):
    
    rtc_command_dict[beam] = ba.Command.create(ba.Cmd.pause)
    
    tmp_rtc_config = {
        'component': 'rtc',
        'type': 'ben',
        'config': rtc_config[beam] ,
        'io': {
            'frame': frame_url.geturl(),
            'commands': commands_url_dict[beam].geturl(),
        },
        'sync': {
            'wait': frame_lock_url.geturl(),
            'notify': commands_lock_url_dict[beam].geturl(),
        },
        'command': sa.url_of(rtc_command_dict[beam]).geturl(),
    }
    rtc_config_list.append(rtc_config)
    
    
    
baldr_config_file = open("baldr_config.json", "+w")

json.dump([cam_config, rtc_config] +\
    [d for d in dm_config_list] +\
    [r for r in rtc_config_list],\
            baldr_config_file)

baldr_config_file.close()

class clean_exit:
    def __del__(self):
        print("killing all")
        cam_command.exit()
        frame_lock.unlock()
        for beam in rtc_command_dict:
            rtc_command_dict[beam].exit() 
        for beam in dm_command_dict:
            dm_command_dict[beam].exit()
            commands_lock_dict[beam].unlock()

        


# Will request all component to exit
_ = clean_exit()
