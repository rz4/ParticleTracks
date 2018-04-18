'''
vis_results.py
Updated: 3/16/17

'''
import sys; sys.path.insert(0, '../')
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_processing.toy3d import generate_data

# Networks
from networks.u_net_3d import unet_model_3d

# Data Parameters
shape = (100,20,20,20)
num_seed_layers = 4
avg_bkg_tracks = 10
noise_prob = 0.01
seed = 1234
event_nb = 1

# Network Parameters
epochs = 10
batch_size = 10
model = unet_model_3d((1,20,20,20))
model_folder = '../../models/u_net_3d/'
threshold = 0.05

################################################################################

def intersect(i, j):
    '''
    '''
    aset = set([tuple(x) for x in i])
    bset = set([tuple(x) for x in j])
    return np.array([x for x in aset & bset])

def remove_intersect(i, intersect):
    '''
    '''
    ii = []
    for j in i:
        flag = True
        for x in intersect:
            if np.array_equal(x, j): flag = False
        if flag: ii.append(j)
    return np.array(ii)

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Generate data
    events, sig_tracks, sig_params = generate_data(shape, num_seed_layers,
                                        avg_bkg_tracks, noise_prob, True, seed)

    # Load Model
    model.summary()
    model.load_weights(model_folder+'best_weights.hdf5')

    # Infer sig track from event
    event = np.expand_dims(events[event_nb], axis=0)
    sig_track = np.expand_dims(sig_tracks[event_nb], axis=0)
    infered_sig = model.predict_on_batch(np.array([event,]))[0]

    xs, ys, zs = np.where(sig_track[0,:,:,:] > 0.0)
    sig_cord = np.stack((xs, ys, zs), axis=-1)
    p0 = len(sig_cord)
    xi, yi, zi = np.where(infered_sig[0,:,:,:] > threshold)
    infer_cord = np.stack((xi, yi, zi), axis=-1)
    p1 = len(infer_cord)
    xe, ye, ze = np.where(event[0,:,:,:] > 0.0)
    event_cord = np.stack((xe, ye, ze), axis=-1)
    inter_ = intersect(event_cord, np.concatenate([sig_cord, infer_cord]))
    event_cord = remove_intersect(event_cord, inter_)
    infer_sig_cord = intersect(infer_cord, sig_cord)
    p2 = len(infer_sig_cord)
    sig_cord = remove_intersect(sig_cord, infer_sig_cord)
    infer_cord = remove_intersect(infer_cord, infer_sig_cord)

    precision = p2 / p1
    recall = p2 / p0
    f = (2 * (precision*recall)) / (precision + recall)
    print("F-Score:", f)

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if len(sig_cord) > 0:
        ax.scatter(sig_cord[:,0], sig_cord[:,1], sig_cord[:,2], c=(0.0,0.0,1.0), marker='o')
    if len(infer_cord) > 0:
        ax.scatter(infer_cord[:,0], infer_cord[:,1], infer_cord[:,2], c=(1.0,0.0,0.0), marker='o')
    if len(infer_sig_cord) > 0:
        ax.scatter(infer_sig_cord[:,0], infer_sig_cord[:,1], infer_sig_cord[:,2], c=(0.0,1.0,0.0), marker='o')
    if len(event_cord) > 0:
        ax.scatter(event_cord[:,0], event_cord[:,1], event_cord[:,2], c=(0.0,1.0,1.0), marker='o')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_zlim(0, 20)
    plt.show()
