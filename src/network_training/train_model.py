'''
train_model.py
Updated: 3/16/17

This script is used to train CNNs on toy 3D particle track data.

'''
import sys; sys.path.insert(0, '../')
import os
import numpy as np
from tqdm import tqdm
from data_processing.toy3d import generate_data

# Networks
from networks.u_net_3d import unet_model_3d

# Data Parameters
shape = (25000,20,20,20)
num_seed_layers = 4
avg_bkg_tracks = 20
noise_prob = 0.01
seed = 1234

# Network Parameters
epochs = 20
batch_size = 10
model = unet_model_3d((1,20,20,20))
model_folder = '../../models/u_net_3d_v2'
train_set = 24000

################################################################################

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.exists(model_folder): os.mkdir(model_folder)

    # Generate data
    events, sig_tracks, sig_params = generate_data(shape, num_seed_layers,
                                        avg_bkg_tracks, noise_prob, True, seed)

    # Train Network
    history = []
    best_val_loss = None
    for j in range(epochs):

        train_status = []
        batch_x = []
        batch_y = []
        for i in tqdm(range(len(events[:train_set]))):
            event_3d = events[i]
            sig_track_3d = sig_tracks[i]
            event_3d = np.expand_dims(event_3d, axis=0)
            sig_track_3d = np.expand_dims(sig_track_3d, axis=0)
            batch_x.append(event_3d)
            batch_y.append(sig_track_3d)
            if len(batch_x) == batch_size or i+1 == len(events[:train_set]):
                batch_x = np.array(batch_x)
                batch_y = np.array(batch_y)
                output = model.train_on_batch(batch_x, batch_y)
                batch_x = []
                batch_y = []
                train_status.append(output)

        # Calculate training loss and accuracy
        train_status = np.array(train_status)
        train_loss = np.average(train_status[:,0])
        train_acc = np.average(train_status[:,1])
        print('Train Loss ->',train_loss)
        print('Train Accuracy ->',train_acc,'\n')

        val_status = []
        batch_x = []
        batch_y = []
        for i in tqdm(range(len(events[train_set:]))):
            event_3d = events[i+train_set]
            sig_track_3d = sig_tracks[i+train_set]
            event_3d = np.expand_dims(event_3d, axis=0)
            sig_track_3d = np.expand_dims(sig_track_3d, axis=0)
            batch_x.append(event_3d)
            batch_y.append(sig_track_3d)
            if len(batch_x) == batch_size or i+1 == len(events[train_set:]):
                batch_x = np.array(batch_x)
                batch_y = np.array(batch_y)
                output = model.test_on_batch(batch_x, batch_y)
                batch_x = []
                batch_y = []
                val_status.append(output)

        # Calculate training loss and accuracy
        val_status = np.array(val_status)
        val_loss = np.average(val_status[:,0])
        val_acc = np.average(val_status[:,1])
        print('Val Loss ->',val_loss)
        print('Val Accuracy ->',val_acc,'\n')

        history.append([j, train_loss, train_acc, val_loss, val_acc])
        if best_val_loss == None or val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save weights of model
            model.save_weights(model_folder + 'best_weights.hdf5')

    # Save training history to csv file
    history = np.array(history)
    np.savetxt(model_folder + 'training_results.csv', history, fmt= '%1.3f', delimiter=', ',
               header='LABELS: epoch, loss, acc, val_loss, val_acc')
