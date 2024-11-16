import baldr as ba
import sardine as sa
import numpy as np

import json

frame_size = 128

commands_size = frame_size*frame_size

frame = sa.host.open_or_create('frames', shape=[frame_size, frame_size], dtype=np.uint16)
# frame = sa.milk.host.open_or_create('frames', [frame_size, frame_size], 1, dtype=np.uint16)
commands = sa.host.open_or_create('commands', shape=[140], dtype=np.double)

frame_url = sa.url_of(frame)
commands_url = sa.url_of(commands)

cim = ba.ComponentInfoManager.get()

cam = cim.create_component_info("cam")
rtc = cim.create_component_info("rtc")
dm = cim.create_component_info("dm")

frame_lock = ba.SpinLock.create()
commands_lock = ba.SpinLock.create()

frame_lock_url = sa.url_of(frame_lock)
commands_lock_url = sa.url_of(commands_lock)

fake_cam_config = {
    'size' : frame_size*frame_size,
    'number': 100, # number of random frame rotating to be copied in shm
    'latency': 1000000, # latency in μsec
}


cam_config = {
    'type': 'camera:fake',
    'config': fake_cam_config,
    'io': {
        'frame': frame_url.geturl(),
    },
    'sync': {
        'notify': frame_lock_url.geturl(),
        'idx': 0,
    },
    'cinfo': sa.url_of(cam).geturl(),
    'async': True
}


fake_rtc_config = {
    'factor' : 2.,
    'offset': 1., # number of random frame rotating to be copied in shm
}

rtc_config = {
    'type': 'rtc:fake',
    'config': fake_rtc_config,
    'io': {
        'frame': frame_url.geturl(),
        'commands': commands_url.geturl(),
    },
    'sync': {
        'wait': frame_lock_url.geturl(),
        'notify': commands_lock_url.geturl(),
    },
    'cinfo': sa.url_of(rtc).geturl(),
}

dm_config = {
    'type': 'dm:fake',
    'config': {}, # fake DM does not take anything
    'io': {
        'commands': commands_url.geturl(),
    },
    'sync': {
        'wait': commands_lock_url.geturl(),
    },
    'cinfo': sa.url_of(dm).geturl(),
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
        cam.exit()
        rtc.exit()
        dm.exit()

        frame_lock.unlock()
        commands_lock.unlock()


# Will request all component to exit
_ = clean_exit()
