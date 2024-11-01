


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

fits_file = 'RECONSTRUCTORS_zonal_0.04pokeamp_in-out_pokes_map_DIT-0.0049_gain_high_17-09-2024T20.06.24.fits'
fits_to_json(fits_file, output_json):





{
  "regions": {
    "pupil_pixels": [100, 101, 102, 103],
    "secondary_pixels": [200, 201],
    "outside_pixels": [300, 301, 302]
  },
  "reco": {
    "dm_flat": [0.0, 0.1, 0.2],
    "IM": [1.0, 0.5, -0.2],
    "CM": [0.5, -0.1, 0.3],
    "R_TT": [0.1, 0.2],
    "R_HO": [0.05, -0.05],
    "I2M": [0.3, 0.7],
    "I2M_TT": [0.6, -0.1],
    "I2M_HO": [0.4, 0.2],
    "M2C": [0.1, 0.9],
    "M2C_TT": [0.5, 0.5],
    "M2C_HO": [0.2, -0.3],
    "bias": [10, 20, 30],
    "I0": [1.0, 1.2, 1.1],
    "flux_norm": 0.85
  },
  "pid": {
    "pid_kp": [1.0, 0.8],
    "pid_ki": [0.05, 0.02],
    "pid_kd": [0.1, 0.05],
    "pid_lower_limit": [-1.0, -1.0],
    "pid_upper_limit": [1.0, 1.0],
    "pid_setpoint": [0.0, 0.1]
  },
  "leakyInt": {
    "leaky_rho": [0.98, 0.95],
    "leaky_kp": [1.0, 1.2],
    "leaky_lower_limit": [-0.5, -0.5],
    "leaky_upper_limit": [0.5, 0.5]
  }
}



img_h = 40
img_w = 60
pupil_pixels_length = 50
dm_size = 140 
n_modes_IM = 100 
num_TT_modes  = 2
num_HO_modes = 20

{

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
