from baldr import _baldr as ba
from baldr import sardine as sa
import numpy as np
import json

global DM_COMMAND_SIZE 
DM_COMMAND_SIZE = 140



with open("rtc_config_jsons/bmc_DM_default_config.json") as f:
    bmc_dm_config = json.load(f)
    


# for now we configure 1 RTC per DM .. ensure this is the case
#assert len( bmc_dm_config ) == len( rtc_config )


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
        'beam': f'{key}',
        'component': 'dm',
        'type': 'bmc',
        'config': bmc_dm_config[key], 
        'io': {
            'commands': commands_url_dict[beam].geturl(),
        },
        'sync':{
            'wait': commands_lock_url_dict[beam].geturl(),
        },
        'command': sa.url_of(dm_command_dict[beam]).geturl(),
    }
    dm_config_list.append(tmp_config)


baldr_config_file = open("just_dms_config.json", "+w")

json.dump( dm_config_list, baldr_config_file )

baldr_config_file.close()

class clean_exit:
    def __del__(self):
        print("killing all")
        for beam in dm_command_dict:
            dm_command_dict[beam].exit()
            commands_lock_dict[beam].unlock()


        


# Will request all component to exit
_ = clean_exit()


"""
# Note memory is written in cd /dev/shm with name 
# parsed in sa.region.host.open_or_create(f'beam{beam}_commands', shape=[DM_COMMAND_SIZE], dtype=np.double)

# opening in another process first get the url 
import json
from baldr import _baldr as ba
from baldr import sardine as sa
import numpy as np

with open('just_dms_config.json', 'r') as f:
     dm_config_dict = json.load(f)

cmd_dict = {}
for c in  dm_config_dict:
    cmd_dict[c['beam']]  = sa.from_url(np.ndarray, c['io']['commands'])



# an example to write to shared memory directly 
import mmap
import os
import numpy as np

# Define the path and size for the shared memory file
shm_path = "/dev/shm/beam0_commands"
array_shape = (140,) #0.2 * np.ones(140) # Example array shape
array_dtype = np.double  # Example dtype
shm_size = np.prod(array_shape) * np.dtype(array_dtype).itemsize  # Calculate required size

# Create a sample NumPy array
np_array = 0.35*np.ones(*array_shape).astype(array_dtype) #np.random.rand(*array_shape).astype(array_dtype)

# Write the array to shared memory
with open(shm_path, "wb") as f:
    f.truncate(shm_size)  # Allocate the file with the defined size

# Memory-map the file and write the array data
with open(shm_path, "r+b") as f:
    with mmap.mmap(f.fileno(), shm_size, access=mmap.ACCESS_WRITE) as mm:
        # Write the NumPy array to shared memory
        mm.write(np_array.tobytes())

# To read the data back into a NumPy array:
with open(shm_path, "r+b") as f:
    with mmap.mmap(f.fileno(), shm_size, access=mmap.ACCESS_READ) as mm:
        # Read the data back into a NumPy array
        np_array_from_shm = np.frombuffer(mm, dtype=array_dtype).reshape(array_shape)

# Verify the data matches
print(np.allclose(np_array, np_array_from_shm))  # Should print True if the data matches


"""
# 
# commands_dict[beam]
# frame
# cam_command.run()

# for i in rtc_command_dict:
#     rtc_command_dict[i].pause()

# for i in rtc_command_dict:
#     rtc_command_dict[i].run()
    
# for i in dm_command_dict:
#     dm_command_dict[i].pause()
    
# for i in dm_command_dict:
#     dm_command_dict[i].run()