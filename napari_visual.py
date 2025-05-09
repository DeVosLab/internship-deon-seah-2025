from argparse import ArgumentParser
from pathlib import Path
import os
import re
import colorcet as cc
import seaborn as sns
import napari
import tifffile
import numpy as np
import pandas as pd

"""
o Load in sample image (.tiff)
o Read in labels and coords of this sample
o Using different colours, render the points on the image channels
o One image layer per channel
o One points layer per cluster
"""

# Load image file and preprocess image
def preprocess_image(img_path, sample):
    sample = 'r01c14'
    tif_files = [f for f in Path(img_path).glob('*.tif') if f.is_file()]
    img_file = next((f for f in tif_files if f.stem == sample), None)

    image = tifffile.imread(str(img_file))
    padded_image = np.pad(
        image,
        pad_width=((0, 0), (0, 0), (80, 80), (80, 80)),
        mode='constant',
        constant_values=0)
    img = np.transpose(
        padded_image,
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
    assert kwargs['img_path'] is not None
    assert kwargs['labels_path'] is not None
    assert kwargs['sample'] is not None

    img_path = kwargs['img_path']
    labels_path = kwargs['labels_path']
    sample = kwargs['sample']

    image = preprocess_image(img_path, sample)
    channel_labels = read_labels_coords(labels_path, sample)

    # Initialise napari
    viewer = napari.Viewer()

    # Add image layer; one per channel (2)
    viewer.add_image(
        image,
        channel_axis=0,
        colormap=['green','magenta'],
        name=f'{sample}')

    # Add points layer; one per cluster (35)
    for channel, coords_label in channel_labels.items():
        points = coords_label.iloc[:, 0:3].to_numpy()
        labels = coords_label.iloc[:, -1].to_numpy()

        unique_clusters = sorted(set(labels))
        for cluster in unique_clusters:
            if cluster == -1:
                continue
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
                out_of_slice_display=True,
                symbol='o',
                visible=False)

    napari.run()

def parse_args():
    parser = ArgumentParser(description='Visualisation of Sample with Labelled Points in napari')
    parser.add_argument('--img_path', type=str, default=None,
                        help='Path to image files')
    parser.add_argument('--labels_path', type=str, default=None,
                        help='Path to files containing labels and coordinates')
    parser.add_argument('--sample', type=str, default=None,
                        help='Specific sample to visualise in napari')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))