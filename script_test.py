from baldr import _baldr as ba
import sardine as sa
import numpy as np

import json

frame_size = 128

commands_size = 64

frame = sa.region.host.open_or_create('frames', shape=[frame_size], dtype=np.uint16)
commands = sa.region.host.open_or_create('commands', shape=[commands_size], dtype=np.double)

frame_url = sa.url_of(frame)
commands_url = sa.url_of(commands)

frame_mutex = sa.sync.Mutex.create()
commands_mutex = sa.sync.Mutex.create()

frame_mutex_url = sa.url_of(frame_mutex)
commands_mutex_url = sa.url_of(commands_mutex)

command = ba.Command.create()

command_url = sa.url_of(command)

fake_cam_config = {
    'size' : frame_size,
    'number': 2, # number of random frame rotating to be copied in shm
    'latency': 1000, # latency in μsec
}

cam_config = {
    'component': 'camera',
    'type': 'fake',
    'config': fake_cam_config,
    'io': {
        'frame': frame_url,
    },
    'sync': {
        'notify': frame_mutex_url,
    },
    'command': command_url,
}

camera_config_file = open("camera_config.json")

json.dumps(cam_config, camera_config_file)

# c = ba.Camera('fake', {'size':frame_size, 'number':2, 'latency':1000}, frame_url, frame_mutex_url)
