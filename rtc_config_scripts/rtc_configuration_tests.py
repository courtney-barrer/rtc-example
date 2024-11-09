


import json
import numpy as np
from astropy.io import fits

def fits_to_json(fits_file, output_json):
    # Open the FITS file
    with fits.open(fits_file) as hdul:
        data_dict = {}
        
        for hdu in hdul:
            # Check if HDU has an EXTNAME, skip if not
            if 'EXTNAME' in hdu.header:
                extname = hdu.header['EXTNAME']
                
                # Convert the data to a JSON-compatible format
                if isinstance(hdu.data, np.ndarray):
                    # Convert numpy array to a list
                    data_dict[extname] = hdu.data.tolist()
                elif np.isscalar(hdu.data):
                    # Scalar values (numbers or strings)
                    data_dict[extname] = hdu.data
                elif hdu.data is None:
                    # Handle empty data by storing None
                    data_dict[extname] = None
                else:
                    # Unexpected data type
                    raise TypeError(f"Unhandled data type for EXTNAME '{extname}'")
        
        # Write the dictionary to a JSON file
        with open(output_json, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)

# Example usage:
# fits_to_json("example.fits", "output.json")

# fits_file = 'RECONSTRUCTORS_zonal_0.04pokeamp_in-out_pokes_map_DIT-0.0049_gain_high_17-09-2024T20.06.24.fits'
# fits_to_json(fits_file, output_json):




img_h = 40
img_w = 60
pupil_pixels_length = 50
dm_size = 140 
n_modes_IM = 100 
num_TT_modes  = 2
num_HO_modes = 20

rtc_config_dict = {

  "regions": {
    "pupil_pixels": list( range(  pupil_pixels_length ) ),
    "secondary_pixels": [1,2],
    "outside_pixels": [1,2]
  },
  "reco": {
    "dm_flat": list( 0.5 * np.ones( 140 ) ),
    "IM": list( np.ones( [n_modes_IM, pupil_pixels_length] ).reshape(-1) ),
    "CM": list( np.ones( [ pupil_pixels_length, n_modes_IM] ).reshape(-1) ), 
    "R_TT": list(np.ones( [ dm_size, pupil_pixels_length] ).reshape(-1)),
    "R_HO": list(np.ones( [ dm_size, pupil_pixels_length] ).reshape(-1)),
    "I2M": list(np.ones( [ n_modes_IM, pupil_pixels_length] ).reshape(-1)),
    "I2M_TT": list(np.ones( [ num_TT_modes, pupil_pixels_length] ).reshape(-1)),
    "I2M_HO": list(np.ones( [ num_HO_modes, pupil_pixels_length] ).reshape(-1)),
    "M2C": list(np.ones( [ dm_size , n_modes_IM] ).reshape(-1)),
    "M2C_TT": list(np.ones( [ dm_size ,num_TT_modes] ).reshape(-1)),
    "M2C_HO": list(np.ones( [ dm_size , num_HO_modes] ).reshape(-1)),
    "bias": list(np.ones( [ img_h, img_w] ).reshape(-1)),
    "I0": list(np.ones( [  img_h, img_w] ).reshape(-1)),
    "flux_norm": 1.0
  },
  "pid": {
    "pid_kp": list( np.zeros( num_TT_modes ) ),
    "pid_ki": list( np.zeros( num_TT_modes ) ),
    "pid_kd": list( np.zeros( num_TT_modes ) ),
    "pid_lower_limit": list( -100 * np.ones( num_TT_modes ) ),
    "pid_upper_limit": list( 100 * np.ones( num_TT_modes ) ),
    "pid_setpoint": list( np.zeros( num_TT_modes ) )
  },
  "leakyInt": {
    "leaky_rho": list( np.zeros( num_HO_modes ) ),
    "leaky_kp":list( np.zeros( num_HO_modes ) ),
    "leaky_lower_limit": list( -100 * np.ones( num_HO_modes ) ),
    "leaky_upper_limit": list( 100 * np.ones( num_HO_modes ) )
  }
}



from baldr import _baldr as ba
from baldr import sardine as sa
import numpy as np

import json

frame_size = 128


frame = sa.region.host.open_or_create('frames', shape=[frame_size, frame_size], dtype=np.uint16)
commands = sa.region.host.open_or_create('commands', shape=[140], dtype=np.double)

frame_url = sa.url_of(frame)
commands_url = sa.url_of(commands)

cam_command = ba.Command.create(ba.Cmd.pause)
rtc_command = ba.Command.create(ba.Cmd.pause)
dm_command = ba.Command.create(ba.Cmd.pause)

frame_lock = ba.SpinLock.create()
commands_lock = ba.SpinLock.create()

frame_lock_url = sa.url_of(frame_lock)
commands_lock_url = sa.url_of(commands_lock)

fake_cam_config = {
    'size' : frame_size*frame_size,
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
        'idx': 0,
    },
    'command': sa.url_of(cam_command).geturl(),
}



rtc_config = {
    'component': 'rtc',
    'type': 'ben',
    'config': rtc_config_dict,
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


#cam_command.run()
#rtc_command.recv()
#rtc_command.run()
#