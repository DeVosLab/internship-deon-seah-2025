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

# Load image file and preprocess image from the HDF5 file
def preprocess_image(h5_path, memb_size, extracted_slices, sample, do_mip):
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
    
    if do_mip:
        img = np.max(img, axis=1) # applies max projection on z, now c, y, x

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
    do_mip = kwargs['do_mip']

    image = preprocess_image(h5_path, memb_size, extracted_slices, sample, do_mip)
    channel_labels = read_labels_coords(labels_path, sample)

    # Initialise napari
    viewer = napari.Viewer()
    
    # Add image layer; one per channel (2)
    viewer.add_image(
        image,
        channel_axis=0,
        colormap=['gray','magenta'],
        name=f'{sample}',
        scale=voxsize[1:] if do_mip else voxsize
        )

    # Add points layer; one per cluster (35)
    for channel, coords_label in channel_labels.items():
        points = coords_label[['z', 'y', 'x']].values
        labels = coords_label['label'].values

        unique_clusters = sorted(set(labels))

        for cluster in unique_clusters:
            cmap = sns.color_palette(cc.glasbey, n_colors=len(unique_clusters))
            cluster_colours = {cluster: cmap[i % len(cmap)]
                            for i, cluster in enumerate(unique_clusters)}
            cluster_mask = labels == cluster
            cluster_points = points[cluster_mask]
            viewer.add_points(
                data=cluster_points[:, 1:] if do_mip else cluster_points,
                size=point_size,
                name=f'channel {channel}, cluster {cluster}',
                face_color=[cluster_colours[cluster]] * len(cluster_points),
                ndim=2 if do_mip else 3,
                out_of_slice_display=True if do_out_of_slice else False,
                symbol='o',
                visible=False if hide_all_points else True,
                scale=voxsize[1:] if do_mip else voxsize
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
    parser.add_argument('--do_mip', action='store_true',
                        help='Performs maximum intensity projection along z-axis')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))