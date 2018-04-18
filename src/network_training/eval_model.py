'''
eval_model.py
Updated: 3/16/17

'''
import sys; sys.path.insert(0, '../')
import os
import numpy as np
from tqdm import tqdm
from data_processing.toy3d import generate_data

# Networks
from networks.u_net_3d import unet_model_3d

# Data Parameters
shape = (2000,20,20,20)
num_seed_layers = 4
avg_bkg_tracks = 10
noise_prob = 0.01
seed = 9876

# Network Parameters
epochs = 10
batch_size = 10
model = unet_model_3d((1,20,20,20))
model_folder = '../../models/u_net_3d_v2/'
threshold = 0.5

################################################################################

def intersect(i, j):
    '''
    '''
    aset = set([tuple(x) for x in i])
    bset = set([tuple(x) for x in j])
    return np.array([x for x in aset & bset])

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
    scores = []
    for event_nb in tqdm(range(len(events))):
        event = np.expand_dims(events[event_nb], axis=0)
        sig_track = np.expand_dims(sig_tracks[event_nb], axis=0)
        infered_sig = model.predict_on_batch(np.array([event,]))[0]

        xs, ys, zs = np.where(sig_track[0,:,:,:] > 0.0)
        sig_cord = np.stack((xs, ys, zs), axis=-1)
        p0 = len(sig_cord)
        xi, yi, zi = np.where(infered_sig[0,:,:,:] > threshold)
        infer_cord = np.stack((xi, yi, zi), axis=-1)
        p1 = len(infer_cord)
        infer_sig_cord = intersect(infer_cord, sig_cord)
        p2 = len(infer_sig_cord)

        precision = p2 / p1
        recall = p2 / p0
        f = (2 * (precision*recall)) / (precision + recall)
        scores.append(f)

    scores = np.array(scores)
    avg_f = np.average(scores)
    max_f = np.max(scores)
    min_f = np.min(scores)
    print("Average F-Score:", avg_f)
