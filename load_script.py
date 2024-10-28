from baldr import _baldr as ba
from baldr import sardine as sa
import numpy as np

import json

camera_config_file = open("baldr_config.json", "r")

config = json.load(camera_config_file)

cam_config = config[0]
rtc_config = config[1]
dm_config = config[2]

fake_cam_config = cam_config['config']

frame = sa.from_url(np.ndarray, rtc_config['io']['frame'])
commands = sa.from_url(np.ndarray, rtc_config['io']['commands'])

cam_command = sa.from_url(ba.Command, cam_config['command'])
rtc_command = sa.from_url(ba.Command, rtc_config['command'])
dm_command = sa.from_url(ba.Command, dm_config['command'])

frame_lock = sa.from_url(ba.SpinLock, rtc_config['sync']['wait'])
commands_lock = sa.from_url(ba.SpinLock, rtc_config['sync']['notify'])

def run():
    cam_command.run()
    rtc_command.run()
    dm_command.run()

def pause():
    cam_command.pause()
    rtc_command.pause()
    dm_command.pause()

def step():
    cam_command.step()
    rtc_command.step()
    dm_command.step()



# You can now control the camera behavior from here.
# This script can be exited and re-run.


# Alternatively, you can instantiate any components locally with:
# TODO: need to expose component API. For now, we can only invoke them with: <component>(). (i.e.: `cam()` or `rtc()` )

# cam = ba.Camera.init(cam_config)
# rtc = ba.RTC.init(rtc_config)
# dm = ba.DM.init(dm_config)
