'''
train_network_2d.py
Updated: 3/16/17

This script is used to train CNNs on toy 3D particle tracking data mapped into
2D.

'''
import sys; sys.path.insert(0, '../')

# Data processing
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_processing.SFCMapper import SFCMapper
from data_processing.toy3d import generate_data

# Neural Network Packages
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

# Data Parameters
shape = (50000,16,16,16)
num_seed_layers = 3
avg_bkg_tracks = 5
noise_prob = 0.01

# Network Parameters
epochs = 2
batch_size = 50

################################################################################

def base_autoencoder():
    '''
    Method defines 2D CNN autoencoder used to run segmentation on toy particle
    tracking data.

    '''
    input_img = Input(shape=(64, 64, 1))

    x = Conv2D(64, (5, 5), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (5, 5), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy',
                        metrics=['accuracy'])

    return autoencoder


if __name__ == '__main__':

    # Initialize SFCMapper
    mapper = SFCMapper(16)

    # Generate_data
    events, sig_tracks, sig_params = generate_data(shape, num_seed_layers,
                                        avg_bkg_tracks, noise_prob, True)

    # Define network
    autoencoder = base_autoencoder() # Change for different nets

    # Train network
    for j in range(epochs):

        train_status = []
        batch_x = []
        batch_y = []
        for i in tqdm(range(len(events))):
            event_2d = mapper.map_3d_to_2d(events[i])
            sig_track_2d = mapper.map_3d_to_2d(sig_tracks[i])
            event_2d = np.expand_dims(event_2d, axis=-1)
            sig_track_2d = np.expand_dims(sig_track_2d, axis=-1)
            #event_2d = np.expand_dims(event_2d, axis=0)
            #sig_track_2d = np.expand_dims(sig_track_2d, axis=0)
            batch_x.append(event_2d)
            batch_y.append(sig_track_2d)
            if len(batch_x) == batch_size or i+1 == len(batch_x):
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

    # Run inference and display
    event = mapper.map_3d_to_2d(events[0])
    plt.imshow(event, interpolation='nearest')
    plt.show()

    event = np.expand_dims(event, axis=-1)
    event = np.expand_dims(event, axis=0)
    result = autoencoder.predict(event)
    plt.imshow(result[0][:,:,0], interpolation='nearest')
    plt.show()

    sig_track_2d = mapper.map_3d_to_2d(sig_tracks[0])
    plt.imshow(sig_track_2d, interpolation='nearest')
    plt.show()
