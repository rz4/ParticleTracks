'''
train_network_3d.py
Updated: 3/16/17

This script is used to train CNNs on toy 3D particle tracking data.

'''
import sys; sys.path.insert(0, '../')
#from data_visualization.display_data import display_3d_track

# Data Processing
import numpy as np
from data_processing.toy3d import generate_data
import matplotlib.pyplot as plt
from tqdm import tqdm

# Neural Network Packages
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from keras import backend as K

# Data Parameters
shape = (5000,16,16,16)
num_seed_layers = 3
avg_bkg_tracks = 3
noise_prob = 0.00

# Network Parameters
epochs = 2
batch_size = 10

################################################################################

def base_autoencoder():
    '''
    '''
    input_img = Input(shape=(16, 16, 16, 1))

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling3D((2, 2, 2), padding='same', data_format='channels_last')(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((2, 2, 2), padding='same', data_format='channels_last')(x)

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((2, 2, 2))(x)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['accuracy'])

    return autoencoder

if __name__ == '__main__':

    # Generate data
    events, sig_tracks, sig_params = generate_data(shape, num_seed_layers,
                                        avg_bkg_tracks, noise_prob, True)

    # Define network
    autoencoder = base_autoencoder()

    # Train Network
    for j in range(epochs):

        train_status = []
        batch_x = []
        batch_y = []
        for i in tqdm(range(len(events))):
            event_3d = events[i]
            sig_track_3d = sig_tracks[i]
            event_3d = np.expand_dims(event_3d, axis=-1)
            sig_track_3d = np.expand_dims(sig_track_3d, axis=-1)
            batch_x.append(event_3d)
            batch_y.append(sig_track_3d)
            if len(batch_x) == batch_size or i+1 == len(events):
                batch_x = np.array(batch_x)
                batch_y = np.array(batch_y)
                output = autoencoder.train_on_batch(batch_x, batch_y)
                batch_x = []
                batch_y = []
                train_status.append(output)

        # Calculate training loss and accuracy
        train_status = np.array(train_status)
        train_loss = np.average(train_status[:,0])
        train_acc = np.average(train_status[:,1])
        print('Train Loss ->',train_loss)
        print('Train Accuracy ->',train_acc,'\n')
