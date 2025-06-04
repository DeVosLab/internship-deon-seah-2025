from argparse import ArgumentParser
from pathlib import Path
import os
import re
import colorcet as cc
import seaborn as sns
import napari
import numpy as np
import pandas as pd
import h5py

"""
o Load in sample image (.h5)
o Read in labels and coords of this sample
o Using different colours, render the points on the image channels
o One image layer per channel
o One points layer per cluster
"""

# Load image file and preprocess image from the HDF5 file
def preprocess_image(h5_path, memb_size, sample):
    if Path(h5_path).is_file():
        f = h5py.File(h5_path, 'r')

    elif '%d' in h5_path:
        memb_size_bytes = memb_size * 1024**3
        first_file = h5_path.replace('%d', '0')
        if not Path(first_file).exists():
            raise FileNotFoundError(f'Expected first file in family not found: {Path(first_file)}')
        f = h5py.File(h5_path, 'r', driver='family', memb_size=memb_size_bytes)
    
    else:
        raise ValueError(f'Provided path is neither a .h5 file or file family: {Path(h5_path)}')
    
    with f:
        img = f[sample]['img'][:]

    img = np.transpose(
        img,
        (1, 0, 2, 3)) # transposes image shape (prev. z, c, y, x, now c, z, y, x)

    return img

# Read labels and coords
def read_labels_coords(labels_path, sample):
    sample_labels_files = sorted([f for f in Path(labels_path).glob(f'*_{sample}_*') if f.suffix == '.csv'])
    channel_labels = {}
    for filename in sample_labels_files:
        channel = re.search(r'channel_(\d+)', str(filename))
        channel = int(channel.group(1))
        file_path = os.path.join(labels_path, filename)
        data = pd.read_csv(file_path)
        channel_labels[channel] = data
    return channel_labels

# Visualisation in napari
def main(**kwargs):
    assert kwargs['h5_path'] is not None
    assert kwargs['labels_path'] is not None
    assert kwargs['sample'] is not None

    h5_path = kwargs['h5_path']
    memb_size = kwargs['memb_size']
    labels_path = kwargs['labels_path']
    sample = kwargs['sample']
    do_out_of_slice = kwargs['do_out_of_slice']
    hide_all_points = kwargs['hide_all_points']

    image = preprocess_image(h5_path, memb_size, sample)
    channel_labels = read_labels_coords(labels_path, sample)

    # Initialise napari
    viewer = napari.Viewer()
    
    # Add image layer; one per channel (2)
    viewer.add_image(
        image,
        channel_axis=0,
        colormap=['green','magenta'],
        name=f'{sample}',
        scale=(5, 0.64, 0.64))

    # Add points layer; one per cluster (35)
    for channel, coords_label in channel_labels.items():
        points = coords_label[['z', 'y', 'x']].values
        labels = coords_label['label'].values

        unique_clusters = sorted(set(labels))
        for i, cluster in enumerate(unique_clusters):
            cmap = sns.color_palette(cc.glasbey, n_colors=len(unique_clusters))
            cluster_colours = {cluster: cmap[i % len(cmap)]
                            for i, cluster in enumerate(unique_clusters)}
            cluster_mask = labels == cluster
            cluster_points = points[cluster_mask]
            viewer.add_points(
                data=cluster_points,
                size=10,
                name=f'channel {channel}, cluster {cluster}',
                face_color=[cluster_colours[cluster]] * len(cluster_points),
                ndim=3,
                out_of_slice_display=True if do_out_of_slice else False,
                symbol='o',
                visible=False if hide_all_points else True,
                scale=(5, 0.64, 0.64)
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
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))