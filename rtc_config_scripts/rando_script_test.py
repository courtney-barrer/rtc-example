from baldr import _baldr as ba
from baldr import sardine as sa
import numpy as np

import json

# Hopefully don't need these locks to configure the DM!!
# commands_lock = ba.SpinLock.create()
# commands_lock_url = sa.url_of(commands_lock)

# dm_cmd = sa.region.host.open_or_create('commands', shape=[140], dtype=np.double)
# commands_url = sa.url_of(dm_cmd)
# cmd_obj = ba.Command.create(ba.Cmd.pause) 
# dm_server_config = {
#     'beam': '1', #f'{dmid}',
#     'component': 'dm',
#     'type': 'fake', #'bmc',
#     'config': {}, # bmc_dm_config[dmid], 
#     'io': {
#         'commands': commands_url.geturl(),
#     },
#     'sync':{
#         'wait': commands_lock_url.geturl(),
#         #'idx': 0,
#     },
#     'command': sa.url_of(cmd_obj).geturl(),
# }

# baldr_config_file = open("baldr_config.json", "+w")

# json.dump([dm_server_config], baldr_config_file)

# dm_server_config.close()

## Start DM server 
# cmdtmp = ["build/Release/baldr_main", "--config", "baldr_config.json"]
# process = subprocess.Popen(cmdtmp,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# with open('baldr_config.json', 'r') as file:
#    data = json.load(file)


#frame_size = 128

beam = "1"

with open("bmc_DM_default_config.json") as f:
    bmc_dm_config = json.load(f)

#frame = sa.region.host.open_or_create('frames', shape=[frame_size, frame_size], dtype=np.uint16)
commands = sa.region.host.open_or_create('commands', shape=[140], dtype=np.double)

#frame_url = sa.url_of(frame)
commands_url = sa.url_of(commands)

#cam_command = ba.Command.create(ba.Cmd.pause)
#rtc_command = ba.Command.create(ba.Cmd.pause)
dm_command = ba.Command.create(ba.Cmd.pause)

#frame_lock = ba.SpinLock.create()
commands_lock = ba.SpinLock.create()

#frame_lock_url = sa.url_of(frame_lock)
commands_lock_url = sa.url_of(commands_lock)

dm_config = {
    'component': 'dm',
    'type': 'bmc',#'fake',
    'config': bmc_dm_config[beam], #{}, # fake DM does not take anything
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
    #cam_config,
    #rtc_config,
    dm_config
], baldr_config_file)

baldr_config_file.close()

class clean_exit:
    def __del__(self):
        print("killing all")
        #cam_command.exit()
        #rtc_command.exit()
        dm_command.exit()

        #frame_lock.unlock()
        commands_lock.unlock()

# Will request all component to exit
_ = clean_exit()
