from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import umap
import hdbscan
import warnings

warnings.filterwarnings('ignore')

"""
o Find a sample with clear positive and negative cells
o Read in feature data of this sample
o Transform coordinates to polar coordinates centred around centre of sample
o Split features of first and second channel
o Perform PCA on each channel (two or three components)
o Create a scatter plot of PCAs where points are coloured as function of:
  1. marker intensity
  2. polar coordinates
o Perform UMAP for better visualisation
o Provide option to perform HDBScan
o Add output path to save plots and labels with coords to folder.
"""

# Find a sample with clear positive and negative cells
  # sample = 'r01c14'

# Read in feature data of sample
def read_data(input_path):
    data = pd.read_parquet(input_path,
                           engine='fastparquet')
    return data

def get_coords_features(input_path, sample):
    data = read_data(input_path)
    sample_match = data[data['filename_img'].apply(lambda p: Path(p).stem == sample)]
    if sample_match.empty:
        raise ValueError(f'No matching rows found for sample: {sample}')
    data = sample_match
    coords = data[['z', 'y', 'x']].to_numpy()
    features = data.iloc[:, -2048:].to_numpy()
    return data, coords, features

# Transform coordinates to polar coordinates centred around centre of sample
def convert_coords(input_path, sample, coords_type):
    data, coords, _ = get_coords_features(input_path, sample)
    centre = coords.mean(axis=0) # compute centre
    dcoords = coords - centre    # dist. from 0 along z, y, x axes
    dz, dy, dx = dcoords.T

    if coords_type == 'polar' or coords_type == 'cylindrical':
        theta = np.arctan2(dy, dx)  # theta angle
        data['theta_angle'] = theta

        if coords_type == 'polar':
            r = np.sqrt(dz**2 + dy**2 + dx**2) # radial distance (polar)
            phi = np.arccos(dz / r)            # phi angle
            data['radial_distance'] = r
            data['phi_angle'] = phi

        else:
            r = np.sqrt(dy**2 + dx**2) # radial distance (cyl)
            data['radial_distance'] = r

    else:
        pass
    return data

# Split features of first and second channel and perform PCA on each channel (two or three components)
def perform_pca_umap(input_path, sample, method):
    if method == 'PCA' or method == 'UMAP':
        _, _, features = get_coords_features(input_path, sample)
        features_ch0 = features[:, 1024:]
        pca0 = PCA(n_components=3).fit_transform(features_ch0)
        features_ch1 = features[:, -1024:]
        pca1 = PCA(n_components=3).fit_transform(features_ch1)
        if method == 'UMAP':
            ch0 = umap.UMAP(n_components=3, random_state=42).fit_transform(pca0)
            ch1 = umap.UMAP(n_components=3, random_state=42).fit_transform(pca1)
        else:
            ch0 = pca0
            ch1 = pca1
    return ch0, ch1

# With the option to perform HDBScan, cluster using HDBScan
def perform_hdbscan(input_path, sample, method, clustering=False):
    if clustering:
        ch0, ch1 = perform_pca_umap(input_path, sample, method)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        labels0 = clusterer.fit(ch0)
        labels1 = clusterer.fit(ch1)
        return labels0, labels1
    else:
        return None, None

# Normalising colourmaps for radial distance, theta and phi angle plots
def normalise_cmaps(input_path, sample, coords_type):
    data = convert_coords(input_path, sample, coords_type)
    norm_mi0 = Normalize(
        vmin=data['intensities_0'].quantile(0.01),
        vmax=data['intensities_0'].quantile(0.99))
    norm_mi1 = Normalize(
        vmin=data['intensities_1'].quantile(0.1),
        vmax=data['intensities_1'].quantile(0.9))
    norm_theta = None
    norm_phi = None

    if coords_type == 'polar' or coords_type == 'cylindrical':
        norm_theta = Normalize(
            vmin=data['theta_angle'].min(),
            vmax=data['theta_angle'].max())

        if coords_type == 'polar':
            norm_phi = Normalize(
                vmin=data['phi_angle'].min(),
                vmax=data['phi_angle'].max())
            
    return norm_mi0, norm_mi1, norm_theta, norm_phi

# Creating scatter plots
def main(**kwargs):
    assert kwargs['input_path'] is not None
    assert kwargs['output_path'] is not None
    assert kwargs['sample'] is not None

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    input_path = kwargs['input_path']
    output_path = kwargs['output_path']
    Path(output_path).mkdir(exist_ok=True, parents=True)
    sample = kwargs['sample']
    method = kwargs['method']
    coords_type = kwargs['coords_type']
    clustering = kwargs['clustering']

    # defining params
    if method != 'cartesian':
        print(f'Converting Cartesian coordinates to {coords_type} coordinates...')
    data = convert_coords(input_path, sample, coords_type)
    print(f'Performing {method} on {sample} features...')
    components = perform_pca_umap(input_path, sample, method) # ch0, ch1
    
    plots_dir = Path(output_path).joinpath('plots')
    plots_dir.mkdir(exist_ok=True, parents=True) 

    if coords_type == 'polar':
        # defining more params for polar
        colour_fields = ['intensities_0', 'radial_distance', 'theta_angle', 'phi_angle',
                         'intensities_1', 'radial_distance', 'theta_angle', 'phi_angle']
        norm_mi0, norm_mi1, norm_theta, norm_phi = normalise_cmaps(input_path, sample, coords_type)
        norms = [norm_mi0, None, norm_theta, norm_phi,
                 norm_mi1, None, norm_theta, norm_phi]
        
        for channel, features in enumerate(components):
            num_plots = 4
            fig = plt.figure(figsize=(20, 5))
            for i in range(num_plots):
                ax = fig.add_subplot(1, num_plots, i+1,
                                     projection='3d')
                scatter = ax.scatter(
                    features[:, 0],
                    features[:, 1],
                    features[:, 2],
                    c=data[colour_fields[i]] if channel == 0 else data[colour_fields[i+4]],
                    cmap='inferno' if i == 0 or i == 1 else 'RdBu',
                    norm=norms[i] if channel == 0 else norms[i+4],
                    s=3)
                fig.colorbar(scatter, ax=ax, label=colour_fields[i])
                ax.set_title(f'{method} of Channel {channel} - Coloured by {colour_fields[i] if channel == 0 else colour_fields[i+4]}')
            fig_path = f'{plots_dir}\\{time_stamp}_{sample}_channel_{channel}_{coords_type}_{method}_plot.png'
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f'Saved plotted features for: {sample}, Channel {channel}')
            plt.tight_layout()
    
    elif coords_type == 'cylindrical':
        # defining more params for cylindrical
        colour_fields = ['intensities_0', 'radial_distance', 'theta_angle', 'z',
                         'intensities_1', 'radial_distance', 'theta_angle', 'z']
        norm_mi0, norm_mi1, norm_theta, norm_phi = normalise_cmaps(input_path, sample, coords_type)
        norms = [norm_mi0, None, norm_theta, None,
                 norm_mi1, None, norm_theta, None]
        
        for channel, features in enumerate(components):
            num_plots = 4
            fig = plt.figure(figsize=(20, 5))
            for i in range(num_plots):
                ax = fig.add_subplot(1, num_plots, i+1,
                                     projection='3d')
                scatter = ax.scatter(
                    features[:, 0],
                    features[:, 1],
                    features[:, 2],
                    c=data[colour_fields[i]] if channel == 0 else data[colour_fields[i+4]],
                    cmap='inferno' if i == 0 or i == 1 else 'RdBu',
                    norm=norms[i] if channel == 0 else norms[i+4],
                    s=3)
                fig.colorbar(scatter, ax=ax, label=colour_fields[i])
                ax.set_title(f'{method} of Channel {channel} - Coloured by {colour_fields[i] if channel == 0 else colour_fields[i+4]}')
            fig_path = f'{plots_dir}\\{time_stamp}_{sample}_channel_{channel}_{coords_type}_{method}_plot.png'
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f'Saved plotted features for: {sample}, Channel {channel}')
            plt.tight_layout()

    else: # cartesian
        # defining more params for cartesian
        colour_fields = ['intensities_0', 'x', 'y', 'z',
                         'intensities_1', 'x', 'y', 'z']
        norm_mi0, norm_mi1, norm_theta, norm_phi = normalise_cmaps(input_path, sample, coords_type)
        norms = [norm_mi0, None, None, None,
                 norm_mi1, None, None, None]
        
        for channel, features in enumerate(components):
            num_plots = 4
            fig = plt.figure(figsize=(20, 5))
            for i in range(num_plots):
                ax = fig.add_subplot(1, num_plots, i+1,
                                     projection='3d')
                scatter = ax.scatter(
                    features[:, 0],
                    features[:, 1],
                    features[:, 2],
                    c=data[colour_fields[i]] if channel == 0 else data[colour_fields[i+4]],
                    cmap='inferno' if i == 0 or i == 1 else 'RdBu',
                    norm=norms[i] if channel == 0 else norms[i+4],
                    s=3)
                fig.colorbar(scatter, ax=ax, label=colour_fields[i])
                ax.set_title(f'{method} of Channel {channel} - Coloured by {colour_fields[i] if channel == 0 else colour_fields[i+4]}')
            fig_path = f'{plots_dir}\\{time_stamp}_{sample}_channel_{channel}_{coords_type}_{method}_plot.png'
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f'Saved plotted features for: {sample}, Channel {channel}')
            plt.tight_layout()

    if clustering:
        print(f'Performing HDBSCAN clustering...')
        labels = perform_hdbscan(input_path, sample, method, clustering) # labels0, labels1
        num_plots = 2
        fig = plt.figure(figsize=(10, 5))

        for channel, components_channel in enumerate(components):
            ax = fig.add_subplot(1, num_plots, channel+1,
                                 projection='3d')
            scatter = ax.scatter(
                components_channel[:, 0],
                components_channel[:, 1],
                components_channel[:, 2],
                c=labels[channel].labels_,
                cmap='cet_glasbey',
                s=3)
            plt.legend()
            ax.set_title(f'HDBSCAN Clustering of Channel {channel}')
            fig_path = f'{plots_dir}\\{time_stamp}_{sample}_{method}_HDBSCAN_clustering.png'
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.tight_layout()

        # Saving labels and coordinates to .csv file
        print('Saving labels and coordinates...')

        labels0, labels1 = labels
        labels_dir = Path(output_path).joinpath('labels')
        labels_dir.mkdir(exist_ok=True, parents=True) 


        labels0 = labels0.labels_
        df0 = pd.DataFrame(data.iloc[:, 4:7], columns=['z', 'y', 'x'])
        df0['label'] = labels0
        df0.to_csv(f'{labels_dir}\\{time_stamp}_{sample}_labels_with_coords_channel_0.csv', index=False)

        labels1 = labels1.labels_
        df1 = pd.DataFrame(data.iloc[:, 4:7], columns=['z', 'y', 'x'])
        df1['label'] = labels1
        df1.to_csv(f'{labels_dir}\\{time_stamp}_{sample}_labels_with_coords_channel_1.csv', index=False)
        
    else:
        pass

    plt.show()
    print('Done!')
    
def parse_args():
    parser = ArgumentParser(description='3D Visualisation of PCA/UMAP')
    parser.add_argument('--input_path', type=str, default=None,
                        help='Path to input file with features')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to output folder')
    parser.add_argument('--sample', type=str, default=None,
                        help='Specific sample to create plots for')
    parser.add_argument('--method', type=str, default='PCA',
                        help='Dimension reduction method to use')
    parser.add_argument('--coords_type', type=str, default='cartesian', choices=['polar', 'cylindrical', 'cartesian'],
                        help='Type of coordinates to colour points by')
    parser.add_argument('--clustering', action='store_true',
                        help='Cluster with HDBScan')
    args = parser.parse_args()

    if not args.method.isupper():
        args.method = args.method.upper()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))