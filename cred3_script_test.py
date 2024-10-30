from baldr import _baldr as ba
from baldr import sardine as sa
import numpy as np

import json



# fake_cam_config = {
#     'size' : frame_size*frame_size,
#     'number': 100, # number of random frame rotating to be copied in shm
#     'latency': 1000, # latency in μsec
# }

cred3_cam_config = {
        "camera_index":0,
        "det_dit" : 0.0016,
        "det_fps" : 600.0,
        "det_gain" : "medium",
        "det_crop_enabled" : False,
        "det_tag_enabled" : False,
        "det_cropping_rows" : "0-511",
        "det_cropping_cols" : "0-639",
        "image_height" : 512,
        "image_width" : 640,
        "full_image_length" : 327680,
}

frame_size_h = cred3_cam_config["image_height"]
frame_size_w =  cred3_cam_config["image_width"]
commands_size = 140 #frame_size_h * frame_size_w

frame = sa.region.host.open_or_create('frames', shape=[frame_size_h, frame_size_w], dtype=np.uint16)
commands = sa.region.host.open_or_create('commands', shape=[commands_size], dtype=np.double)

frame_url = sa.url_of(frame)
commands_url = sa.url_of(commands)

cam_command = ba.Command.create(ba.Cmd.pause)
rtc_command = ba.Command.create(ba.Cmd.pause)
dm_command = ba.Command.create(ba.Cmd.pause)

frame_lock = ba.SpinLock.create()
commands_lock = ba.SpinLock.create()

frame_lock_url = sa.url_of(frame_lock)
commands_lock_url = sa.url_of(commands_lock)



cam_config = {
    'component': 'camera',
    'type': 'fli',
    'config': cred3_cam_config,
    'io': {
        'frame': frame_url.geturl(),
    },
    'sync': {
        'notify': frame_lock_url.geturl(),
    },
    'command': sa.url_of(cam_command).geturl(),
}


fake_rtc_config = {
    'factor' : 2.,
    'offset': 1., # number of random frame rotating to be copied in shm
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

dm_config = {
    'component': 'dm',
    'type': 'fake',
    'config': {}, # fake DM does not take anything
    'io': {
        'commands': commands_url.geturl(),
    },
    'sync': {
        'wait': commands_lock_url.geturl(),
    },
    'command': sa.url_of(dm_command).geturl(),
}

baldr_config_file = open("baldr_config.json", "+w")

json.dump([
    cam_config,
    rtc_config,
    dm_config
], baldr_config_file)

baldr_config_file.close()

class clean_exit:
    def __del__(self):
        print("killing all")
        cam_command.exit()
        rtc_command.exit()
        dm_command.exit()

        frame_lock.unlock()
        commands_lock.unlock()


# Will request all component to exit
_ = clean_exit()
