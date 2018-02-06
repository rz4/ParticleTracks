'''
display_data.py
Updated: 2/6/18

This script is used to visualize the toy particle data.

'''
import sys; sys.path.insert(0, '../')
from data_processing.SFCMapper import SFCMapper
from data_processing.toy3d import generate_data

# Parameters
shape = (1,64,64,64)
num_seed_layers = 1
avg_bkg_tracks = 0
noise_prob = 0.01

################################################################################

if __name__ == "__main__":

    # Initialize SFCMapper

    # Generate_data
    events, sig_tracks, sig_params = generate_data(shape, num_seed_layers,
                                                    avg_bkg_tracks, noise_prob)
    print(events.shape)
