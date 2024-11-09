from baldr import _baldr as ba
from baldr import sardine as sa
import numpy as np
import json
import subprocess

with open("rtc_config_jsons/cred1_camera_default_config.json") as f:
    cred1_config = json.load(f)

frame_size_h = 320
frame_size_w = 256
cam_command = ba.Command.create(ba.Cmd.pause)
frame = sa.region.host.open_or_create('frames', shape=[frame_size_h, frame_size_w], dtype=np.uint16)
frame_url = sa.url_of(frame)

frame_lock = ba.SpinLock.create()
frame_lock_url = sa.url_of(frame_lock)

cam_config = {
    'component': 'camera',
    'type': 'fli',
    'config':  cred1_config,
    'io': {
        'frame': frame_url.geturl(),
    },
    'sync': {
        'notify': frame_lock_url.geturl(),
    },
    'command': sa.url_of(cam_command).geturl(),
}

baldr_config_file = open("baldr_config.json", "+w")

json.dump([cam_config], baldr_config_file)

baldr_config_file.close()


# Run server with config file create above
#process = subprocess.Popen(["build/Release/baldr_main", "--config", "baldr_config.json"]\
#                            ,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#print("Process started with PID:", process.pid)

print('start running camera')
cam_command.run()



class clean_exit:
    def __del__(self):
        print("killing all")
        cam_command.exit()
        frame_lock.unlock()
# Will request all component to exit
_ = clean_exit()


## Test access in new script 
"""
# first get the actual frame url from the config file or
in the interactive python shell used to create config
In [8]: frame_url.geturl()
Out[8]: 'host://frames/0/163840?dtc=1&bits=16&extent=%5B320,256%5D'

# then new terminal in python
from baldr import _baldr as ba
from baldr import sardine as sa
import numpy as np

u = '<url of frame>'
frame = sa.from_url(np.ndarray, u)


"""