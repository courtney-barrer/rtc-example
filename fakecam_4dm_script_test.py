from baldr import _baldr as ba
from baldr import sardine as sa
import numpy as np

import json


###############
# CAMERA
###############
frame_size_h = 320
frame_size_w = 256

frame = sa.region.host.open_or_create('frames', shape=[frame_size_h, frame_size_w], dtype=np.uint16)
frame_url = sa.url_of(frame)

cam_command = ba.Command.create(ba.Cmd.pause)

frame_lock = ba.SpinLock.create()

frame_lock_url = sa.url_of(frame_lock)


fake_cam_config = {
    'size' : frame_size_h * frame_size_w,
    'number': 100, # number of random frame rotating to be copied in shm
    'latency': 1000, # latency in μsec
}


cam_config = {
    'component': 'camera',
    'type': 'fake',
    'config': fake_cam_config,
    'io': {
        'frame': frame_url.geturl(),
    },
    'sync': {
        'notify': frame_lock_url.geturl(),
    },
    'command': sa.url_of(cam_command).geturl(),
}



###############
# DM
###############

dm_command_size = 140 

# where it reads commands from
commands = sa.region.host.open_or_create('commands', shape=[dm_command_size], dtype=np.double)
commands_url = sa.url_of(commands)

# where we control the DMs
#dm_command = ba.Command.create(ba.Cmd.pause)
commands_lock = ba.SpinLock.create()
commands_lock_url = sa.url_of(commands_lock)


with open("bmc_DM_default_config.json") as f:
    bmc_dm_config = json.load(f)

# commands_dict = {}
# commands_url_dict = {}
# dm_command_dict = {}
# commands_lock_dict = {}
# commands_lock_url_dict = {}
# dm_config_list = []
# for beam in range( len( bmc_dm_config )):
#     commands_dict[beam] = sa.region.host.open_or_create(f'beam{beam}_commands', shape=[dm_command_size], dtype=np.double)
#     commands_url_dict[beam]  = sa.url_of(commands_dict[beam])
#     dm_command_dict[beam] = ba.Command.create(ba.Cmd.pause)
#     commands_lock_dict[beam] = ba.SpinLock.create()
#     commands_lock_url_dict[beam] = sa.url_of(commands_lock_dict[beam])
    
    
dm_config_list = []
dm_command_list = []
for beam, key in enumerate( bmc_dm_config ):

    cmd = ba.Command.create(ba.Cmd.pause) 
    tmp_config = {
        'beam': f'{key}',
        'component': 'dm',
        'type': 'fake', #'bmc',
        'config': bmc_dm_config[key], 
        'io': {
            'commands': commands_url.geturl(),
        },
        'sync': {
            'wait': commands_lock_url.geturl(),
            'idx':beam
        },
        'command': sa.url_of(cmd).geturl(),
    }
    dm_command_list.append( cmd )
    dm_config_list.append(tmp_config)





###############
# RTC
###############
rtc_command = ba.Command.create(ba.Cmd.pause)



fake_rtc_config = {
    'factor' : 0,
    'offset': 0.01, # number of random frame rotating to be copied in shm
}

rtc_config = {
    'component': 'rtc',
    'type': 'fake',
    'config': fake_rtc_config,
    'io': {
        'frame': frame_url.geturl(),
        'commands': commands_url.geturl(),
    },
    'sync': {
        'wait': frame_lock_url.geturl(),
        'notify': commands_lock_url.geturl(),
    },
    'command': sa.url_of(rtc_command).geturl(),
}


baldr_config_file = open("baldr_config.json", "+w")

json.dump(
    [cam_config]+\
    [rtc_config]+\
    dm_config_list
, baldr_config_file)

baldr_config_file.close()

class clean_exit:
    def __del__(self):
        print("killing all")
        cam_command.exit()
        rtc_command.exit()
        for i in range(len( dm_command_list )):
            dm_command_list[i].exit()

        frame_lock.unlock()
        commands_lock.unlock()


# Will request all component to exit
_ = clean_exit()


"""

#run everything
cam_command.run()
rtc_command.run()
for d in dm_command_list:
    d.run()

# check frames are written to SHM 
frame
# check commands are written to SHM and consitent with RTC config 
# (note we only have one command that is compied to all DMs for this test)
commands 


# check spin locks 
# pause everything 
cam_command.pause()
rtc_command.pause()
for d in dm_command_list:
    d.pause()

# check they are all in paused state
for a in [ cam_command, rtc_command] + dm_command_list:
    print( a.recv() )

# check lock states 
for a  in [frame_lock, commands_lock] :
    print( a.value(0) )
# camera is unlocked but command is locked. this could be because 
DM has recieved the last command and locked the command state, but the
rtc is not ready to unlock it.

# if we step (1 iteration of) the rtc then the rtc will take the camera 
frame and lock it 
rtc_command.step()
for a in [ frame_lock, commands_lock] :
    print( a.value(0) )

# NOW THEY SHOULD BE BOTH LOCKED
for d in dm_command_list:
    d.step()




# unlock commands 
commands_lock.unlock()
    
# do one step 

dm_command_list[0]

# run 
commands
# start writing the SHM 
rtc_command.run() 

# pause the FLI camera (stop writing to SHM and pause camera)
rtc_command.run() 

# current state  
cam_command.recv()

# can also use send command to run it 
cam_command.send( ba.Cmd.run )


# lock and unlock 
# unlock

# look at state for one of the (currently 6) lock indicies
# we have 6 for cases where one module may want to unlock many locks (e.g 1 rtc 4 dms)
frame_lock.value(5) # index 5

# try lock one # lock needs to pass an index 
frame_lock.try_lock(0)

# still reads unlocked? 
frame_lock.value(0)

# try on command 
commands_lock.try_lock(0)

# check the state 
commands_lock.value(0)

# unlock commands  ( no index since it unlocks all of them)
commands_lock.unlock()


# locking causes complete freeze (bug)
commands_lock.try_lock(0)

"""