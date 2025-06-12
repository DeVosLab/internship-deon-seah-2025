from argparse import ArgumentParser
from pathlib import Path
import colorcet as cc
import seaborn as sns
import napari
import numpy as np
import pandas as pd
import h5py

# Load image file and preprocess image from the HDF5 file
def preprocess_image(h5_path, memb_size, extracted_slices, sample):
    if Path(h5_path).is_file():
        f = h5py.File(h5_path, 'r')

    elif '%d' in h5_path:
        memb_size_bytes = memb_size * 1024**3
        f = h5py.File(h5_path, 'r', driver='family', memb_size=memb_size_bytes)
    
    else:
        raise ValueError(f'Provided path is neither a .h5 file or file family: {Path(h5_path)}')
    
    with f:

        # visualise only the extracted slices
        if extracted_slices:
            group = f[sample]

            img = group['img'][()]
            z = group['z'][()]

            z_counts = np.bincount(z, minlength=img.shape[0])
            z_centre = np.argmax(z_counts)

            half = 15 // 2
            z_start = max(0, z_centre - half)
            z_end = min(img.shape[0], z_start + 15)
            z_start = max(0, z_end - 15)
            img = img[z_start:z_end, :, :, :]

            print(f'Visualising slices {z_start}:{z_end}')

        # visualise the entire image
        else:
            img = f[sample]['img'][:]

    img = np.transpose(
        img,
        (1, 0, 2, 3)) # transposes image shape (prev. z, c, y, x, now c, z, y, x)

    return img

# Read labels and coords
def read_labels_coords(labels_path):
    data = pd.read_csv(labels_path)

    x = data['x'][:]
    y = data['y'][:]
    z = data['z'][:]
    points = np.stack([z, y, x], axis=1)

    true_label = data['true_labels'][:]
    pred_prob = data['predicted_probability'][:]
    pred_class = data['predicted_class'][:]

    return points, pred_prob, pred_class, true_label

# Visualisation in napari
def main(**kwargs):
    assert kwargs['h5_path'] is not None
    assert kwargs['labels_path'] is not None
    assert kwargs['sample'] is not None
    assert kwargs['voxelsize'] is not None

    h5_path = kwargs['h5_path']
    memb_size = kwargs['memb_size']
    labels_path = kwargs['labels_path']
    sample = kwargs['sample']
    do_out_of_slice = kwargs['do_out_of_slice']
    hide_all_points = kwargs['hide_all_points']
    point_size = kwargs['point_size']
    voxsize = kwargs['voxelsize']
    extracted_slices = kwargs['extracted_slices']

    image = preprocess_image(h5_path, memb_size, extracted_slices, sample)
    points, pred_prob, pred_class, true_label = read_labels_coords(labels_path)

    # Initialise napari
    viewer = napari.Viewer()
    
    # Add image layer; one per channel (2)
    viewer.add_image(
        image,
        channel_axis=0,
        colormap=['gray','magenta'],
        name=f'{sample}',
        scale=voxsize)

    # Add points layer for predicted class
    unique_class = sorted(set(pred_class))

    for each_class in unique_class:
        cmap = sns.color_palette(cc.glasbey, n_colors=len(unique_class))
        cluster_colours = {cluster: cmap[i % len(cmap)]
                        for i, cluster in enumerate(unique_class)}
        cluster_mask = pred_class == each_class
        cluster_points = points[cluster_mask]
        viewer.add_points(
            data=cluster_points,
            size=point_size,
            name=f'predicted class {each_class}',
            face_color=[cluster_colours[each_class]] * len(cluster_points),
            ndim=3,
            out_of_slice_display=True if do_out_of_slice else False,
            symbol='o',
            visible=False if hide_all_points else True,
            scale=voxsize
            )
        
    # Add points layer for ground truth labelling
    unique_clusters = sorted(set(true_label))

    for cluster in unique_clusters:
        cmap = sns.color_palette(cc.glasbey, n_colors=len(unique_clusters))
        cluster_colours = {cluster: cmap[i % len(cmap)]
                        for i, cluster in enumerate(unique_clusters)}
        cluster_mask = true_label == cluster
        cluster_points = points[cluster_mask]
        viewer.add_points(
            data=cluster_points,
            size=point_size,
            name=f'true class {cluster}',
            face_color=[cluster_colours[cluster]] * len(cluster_points),
            ndim=3,
            out_of_slice_display=True if do_out_of_slice else False,
            symbol='o',
            visible=False if hide_all_points else True,
            scale=voxsize
            )
    
    # Add points layer for predicted probabilities
    viewer.add_points(
        data=points,
        size=point_size,
        name='predicted probabilities',
        features={'pred_prob': pred_prob},
        face_color='pred_prob',
        face_colormap='inferno',
        ndim=3,
        out_of_slice_display=True if do_out_of_slice else False,
        symbol='o',
        visible=False if hide_all_points else True,
        scale=voxsize
        )

    napari.run()

def parse_args():
    parser = ArgumentParser(description='Visualisation of Sample with Labelled Points in napari')
    parser.add_argument('--h5_path', type=str, default=None,
                        help='Path to HDF5 file')
    parser.add_argument('--memb_size', type=float, default=3.5,
                        help='Specify size of each .h5 file family member')
    parser.add_argument('--labels_path', type=str, default=None,
                        help='Path to files containing labels and coordinates')
    parser.add_argument('--sample', type=str, default=None,
                        help='Specific sample to visualise in napari')
    parser.add_argument('--do_out_of_slice', action='store_true',
                        help='Toggle out_of_slice_display for all points layers')
    parser.add_argument('--hide_all_points', action='store_true',
                        help='Hides all points layers, can be manually unhidden')
    parser.add_argument('--point_size', type=int, default=10,
                        help='Specify size of each point in points layer')
    parser.add_argument('--voxelsize', type=float, nargs='+', default=None,
                        help='Defines the voxel size of the image in Z, Y, X')
    parser.add_argument('--extracted_slices', action='store_true',
                        help='Only visualises extracted slices')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))