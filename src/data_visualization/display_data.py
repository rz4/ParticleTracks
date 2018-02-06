'''
display_data.py
Updated: 2/6/18

This script is used to visualize the toy particle data.

'''
from mayavi import mlab
import sys; sys.path.insert(0, '../')
import vtk
import os
import numpy as np
from tvtk.api import tvtk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tvtk.common import configure_input_data
from data_processing.SFCMapper import SFCMapper
from data_processing.toy3d import generate_data

# Parameters
shape = (1,64,64,64)
num_seed_layers = 5
avg_bkg_tracks = 5
noise_prob = 0.01

################################################################################

def display_3d_track(event, sig_track):
    '''
    Method displays 3d array.

    Param:
        array_3d - np.array

    '''
    v = mlab.figure(bgcolor=(1.0,1.0,1.0))

    # Coordinate Information
    xx, yy, zz = np.where(event > 0.0)

    # Generate Voxels For Protein
    append_filter = vtk.vtkAppendPolyData()
    for i in range(len(xx)):
        input1 = vtk.vtkPolyData()
        voxel_source = vtk.vtkCubeSource()
        voxel_source.SetCenter(xx[i],yy[i],zz[i])
        voxel_source.SetXLength(1)
        voxel_source.SetYLength(1)
        voxel_source.SetZLength(1)
        voxel_source.Update()
        input1.ShallowCopy(voxel_source.GetOutput())
        append_filter.AddInputData(input1)
    append_filter.Update()

    #  Remove Any Duplicate Points.
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputConnection(append_filter.GetOutputPort())
    clean_filter.Update()

    # Render Voxels
    pd = tvtk.to_tvtk(clean_filter.GetOutput())
    cube_mapper = tvtk.PolyDataMapper()
    configure_input_data(cube_mapper, pd)
    p = tvtk.Property(opacity=1.0, color=(1.0,0.0,0.0))
    cube_actor = tvtk.Actor(mapper=cube_mapper, property=p)
    v.scene.add_actor(cube_actor)

    # Coordinate Information
    xx, yy, zz = np.where(sig_track > 0.0)

    # Generate Voxels For Protein
    append_filter = vtk.vtkAppendPolyData()
    for i in range(len(xx)):
        input1 = vtk.vtkPolyData()
        voxel_source = vtk.vtkCubeSource()
        voxel_source.SetCenter(xx[i],yy[i],zz[i])
        voxel_source.SetXLength(1)
        voxel_source.SetYLength(1)
        voxel_source.SetZLength(1)
        voxel_source.Update()
        input1.ShallowCopy(voxel_source.GetOutput())
        append_filter.AddInputData(input1)
    append_filter.Update()

    #  Remove Any Duplicate Points.
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputConnection(append_filter.GetOutputPort())
    clean_filter.Update()

    # Render Voxels
    pd = tvtk.to_tvtk(clean_filter.GetOutput())
    cube_mapper = tvtk.PolyDataMapper()
    configure_input_data(cube_mapper, pd)
    p = tvtk.Property(opacity=1.0, color=(0.0,1.0,0.0))
    cube_actor = tvtk.Actor(mapper=cube_mapper, property=p)
    v.scene.add_actor(cube_actor)

    mlab.show()

if __name__ == "__main__":

    # Initialize SFCMapper
    mapper = SFCMapper(64)

    # Generate_data
    events, sig_tracks, sig_params = generate_data(shape, num_seed_layers,
                                            avg_bkg_tracks, noise_prob, False)

    # Display 3D
    display_3d_track(events[0], sig_tracks[0])

    # Map 3D to 2D
    event_2d = mapper.map_3d_to_2d(events[0])
    sig_tracks_2d = mapper.map_3d_to_2d(sig_tracks[0])

    # Display 2D
    cmap = ListedColormap([[0,0,0,0.0], (1.0,0.0,0.0)])
    plt.imshow(event_2d, cmap=cmap, interpolation='nearest')
    cmap = ListedColormap([[0,0,0,0], (0.0,0.0,1.0)])
    plt.imshow(sig_tracks_2d, cmap=cmap, interpolation='nearest')
    plt.show()
