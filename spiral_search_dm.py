

import sys
import glob
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(1,'/opt/Boston Micromachines/lib/Python3/site-packages/')
sys.path.append('pyBaldr/' )
from pyBaldr import utilities as util
import bmc

data_path = 'data/'
fig_path = 'data/'
DM_serial_number = '17DW019#122'# Syd = '17DW019#122', ANU = '17DW019#053'

dm = bmc.BmcDm()

dm_err_flag  = dm.open_dm(DM_serial_number)

DMshapes_path='DMShapes/'

files = glob.glob(DMshapes_path+'*.csv')

print(files)

# Some basic DM shapes 
flatdm = pd.read_csv('DMShapes/17DW019#122_FLAT_MAP_COMMANDS.csv', header=None)[0].values
crosshair = pd.read_csv('DMShapes/Crosshair140.csv', header=None)[0].values

fourier_basis = util.construct_command_basis( basis='fourier', number_of_modes = 40, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)

#dm.send_data(flatdm + fourier_basis[:,0])
# have a look at the tip mode 
plt.figure(); plt.imshow( util.get_DM_command_in_2D( fourier_basis[:,0] ) ); plt.colorbar(); plt.savefig(fig_path + 'delme.png')

# check normalization 
for i in range(fourier_basis.shape[1]):
    if abs( 1 - np.sum( fourier_basis[:,i]**2 ) ) > 0.01:
        print(f'normalization issue mode {i}:  <M_{i}|M_{i}> = {np.sum( fourier_basis[:,i]**2 )} != 1')

TT_spiral_coefficients = util.spiral_search_TT_coefficients(dr = 1e-2, dtheta = np.deg2rad( 10 ), aoi_tp = np.deg2rad( 0 ), aoi_tt= np.deg2rad( 0 ), num_points= 100, r0=0, theta0=0)
# have a look 
plt.figure(); plt.plot( [a[0] for a in TT_spiral_coefficients], [a[1] for a in TT_spiral_coefficients],'x'); plt.savefig(fig_path + 'delme.png')

# check we using the correct tip/tilt 
# (sometimes indexing can be funny.. Sorry )
plt.figure(); plt.imshow( util.get_DM_command_in_2D(fourier_basis[:,0]) ); plt.savefig(fig_path+'delme.png')
plt.figure(); plt.imshow( util.get_DM_command_in_2D(fourier_basis[:,5]) ); plt.savefig(fig_path+'delme.png')
tip = fourier_basis[:,0]
tilt = fourier_basis[:,5]  

# spiral the DM on tip/tilt in Fourier basis 
for i, (a_tp, a_tt) in enumerate( TT_spiral_coefficients ):
    print( i )
    time.sleep( 0.5 )
    dm.send_data(flatdm + a_tp * tip + a_tt * tilt)


#i=95 ; dm.send_data(flatdm + TT_spiral_coefficients[i][0] * fourier_basis[:,0] + TT_spiral_coefficients[i][1] * fourier_basis[:,1])

# flatten again 
dm.send_data(flatdm )

# set r0 !=0 and dr = 0 for circle search 
TT_circle_coefficients = util.spiral_search_TT_coefficients( dr=0, dtheta = np.deg2rad( 10 ), aoi_tp=0, aoi_tt=0, num_points = 36, r0=0.2, theta0=0)

plt.figure(); plt.plot( [a[0] for a in TT_circle_coefficients], [a[1] for a in TT_circle_coefficients],'x'); plt.savefig(fig_path + 'delme.png')

for i, (a_tp, a_tt) in enumerate( TT_circle_coefficients ):
    print( i )
    if i>0:
        plt.figure(); plt.plot( [a[0] for a in TT_circle_coefficients[:i]], [a[1] for a in TT_circle_coefficients[:i]],'x'); plt.xlim([-0.5, 0.5] ); plt.ylim([-0.5,0.5]); plt.savefig(fig_path + 'delme.png')
    time.sleep( 0.5 )
    #dm.send_data(flatdm + a_tp * fourier_basis[:,0] + a_tt * fourier_basis[:,1])
    dm.send_data(0.5 + a_tp * tip + a_tt * tilt)
    #dm.send_data(flatdm )

dm.send_data( flatdm )

# try just the four quadrants 
dm.send_data(0.5  * np.ones( 140 ) )
dm.send_data( flatdm )
dm.send_data(0.5 + 0.4 * tip)
dm.send_data(0.5 - 0.4 * tip)
dm.send_data(0.5 + 0.4 * tilt)
dm.send_data(0.5 - 0.4 * tilt)

plt.figure(); plt.imshow( util.get_DM_command_in_2D(fourier_basis[:,0]) ); plt.savefig(fig_path+'delme.png')



# flatten again 
dm.send_data(flatdm )

# send a zonal poke 
zonal_cmds = np.eye(140)
amp = 0.03
for i, c in enumerate( zonal_cmds ):
    for j in [1,2]:
        time.sleep( 0.5 )
        dm.send_data(flatdm + (-1)**j * amp * c)


# dither a single actuator
zonal_cmds = np.eye(140)
amp = 0.03
act = 65 
for j in np.arange(100):
    time.sleep( 1 )
    dm.send_data(flatdm + (-1)**j * amp * zonal_cmds[act])

