from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import colorcet
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
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
o Provide option to visualise on clustermap
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

# Split features of the different channels
def split_features(input_path, sample):
    _, _, features = get_coords_features(input_path, sample)
    _, total_features = features.shape
    num_channels = total_features // 1024

    channel_features = [
        features[:, i*1024:(i+1)*1024]
        for i in range(num_channels)]
    
    return num_channels, channel_features
    
# Perform PCA or UMAP on each channel (two or three components)
def perform_pca_umap(input_path, sample, method):
    _, channel_features = split_features(input_path, sample)

    if method == 'PCA':
        reduced_ft = []
        for i, ft in enumerate(channel_features):
            scaler = StandardScaler()
            ft_scaled = scaler.fit_transform(ft)
            pca = PCA(
                n_components=3,
                random_state=42)
            reduced = pca.fit_transform(ft_scaled)
            reduced_ft.append(reduced)

    elif method == 'UMAP':
        reduced_ft = []
        for i, ft in enumerate(channel_features):
            scaler = StandardScaler()
            ft_scaled = scaler.fit_transform(ft)
            umapper = umap.UMAP(n_components=3, random_state=42)
            reduced = umapper.fit_transform(ft_scaled)
            reduced_ft.append(reduced)

    return reduced_ft

# With the option to perform HDBScan, cluster using HDBScan
def perform_hdbscan(input_path, sample, method, cluster_reduced, do_dendro, clustering):
    if clustering:
        if cluster_reduced:
            channel_ft = perform_pca_umap(input_path, sample, method)
        else:
            _, channel_ft = split_features(input_path, sample)

        clusterers = []
        labels = []
        for i, ft in enumerate(channel_ft):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
            label = clusterer.fit(ft)
            labels.append(label)
            clusterers.append(clusterer)

        return clusterers, labels
    
    else:
        return None

# Normalising colourmaps for radial distance, theta and phi angle plots
def normalise_cmaps(input_path, sample, coords_type):
    data = convert_coords(input_path, sample, coords_type)
    norm_mi0 = Normalize(
        vmin=data['intensity_0'].quantile(0.01),
        vmax=data['intensity_0'].quantile(0.99))
    norm_mi1 = Normalize(
        vmin=data['intensity_1'].quantile(0.01),
        vmax=data['intensity_1'].quantile(0.99))
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
    cluster_reduced = kwargs['cluster_reduced']
    clustermap = kwargs['clustermap']
    map_reduced = kwargs['map_reduced']
    do_dendro = kwargs['do_dendro']

    # defining params
    if coords_type != 'cartesian':
        print(f'Converting Cartesian coordinates to {coords_type} coordinates...')
    data = convert_coords(input_path, sample, coords_type)
    print(f'Performing {method} on {sample} features...')
    components = perform_pca_umap(input_path, sample, method)
    
    plots_dir = Path(output_path).joinpath(f'plots\\{sample}')
    plots_dir.mkdir(exist_ok=True, parents=True) 

    # defining more params for polar
    if coords_type == 'polar':
        colour_fields = ['intensity_0', 'radial_distance', 'theta_angle', 'phi_angle',
                         'intensity_1', 'radial_distance', 'theta_angle', 'phi_angle']
        norm_mi0, norm_mi1, norm_theta, norm_phi = normalise_cmaps(input_path, sample, coords_type)
        norms = [norm_mi0, None, norm_theta, norm_phi,
                 norm_mi1, None, norm_theta, norm_phi]
    
    # defining more params for cylindrical
    elif coords_type == 'cylindrical':
        colour_fields = ['intensity_0', 'radial_distance', 'theta_angle', 'z',
                         'intensity_1', 'radial_distance', 'theta_angle', 'z']
        norm_mi0, norm_mi1, norm_theta, norm_phi = normalise_cmaps(input_path, sample, coords_type)
        norms = [norm_mi0, None, norm_theta, None,
                 norm_mi1, None, norm_theta, None]

    # defining more params for cartesian
    else: # cartesian coords
        colour_fields = ['intensity_0', 'x', 'y', 'z',
                         'intensity_1', 'x', 'y', 'z']
        norm_mi0, norm_mi1, norm_theta, norm_phi = normalise_cmaps(input_path, sample, coords_type)
        norms = [norm_mi0, None, None, None,
                 norm_mi1, None, None, None]
        
    # for loops to plot features for four variables and for all channels
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

    # perform HDBSCAN clustering
    if clustering:
        print(f'Performing HDBSCAN clustering...')
        clusterers, labels = perform_hdbscan(input_path, sample, method, cluster_reduced, do_dendro, clustering)
        num_plots, _ = split_features(input_path, sample)
        fig = plt.figure(figsize=(num_plots * 5, 5))

        # plot features for each channel, coloured by HDBSCAN labels
        for channel, channel_components in enumerate(components):
            ax = fig.add_subplot(1, num_plots, channel+1,
                                 projection='3d')
            cluster_labels = labels[channel].labels_

            scatter = ax.scatter(
                channel_components[:, 0],
                channel_components[:, 1],
                channel_components[:, 2],
                c=cluster_labels,
                cmap='cet_glasbey',
                s=3)
            fig.colorbar(scatter, ax=ax, label='Cluster labels')
            ax.set_title(f'HDBSCAN Clustering of Channel {channel}')
            fig_path = f'{plots_dir}\\{time_stamp}_{sample}_{method}_HDBSCAN_clustering.png'
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.tight_layout()

            # save labels, coordinates, and components to .csv file
            labels_dir = Path(output_path).joinpath(f'labels\\{sample}')
            labels_dir.mkdir(exist_ok=True, parents=True)

            df = pd.DataFrame(data.iloc[:, 4:7], columns=['z', 'y', 'x'])
            df['label'] = cluster_labels
            df['cp1'] = channel_components[:, 0]
            df['cp2'] = channel_components[:, 1]
            df['cp3'] = channel_components[:, 2]
            df.to_csv(f'{labels_dir}\\{time_stamp}_{sample}_labels_with_coords_channel_{channel}.csv', index=False)
            print(f'Saved labels and coordinates for channel {channel}')

        # plot condensed cluster trees
        if do_dendro:
            for i, clusterer in enumerate(clusterers):
                print(f'Plotting condensed cluster tree for channel {i}...')
                plt.figure(figsize=(8, 6))
                clusterer.condensed_tree_.plot(select_clusters=True)
                plt.title(f'Condensed Cluster Tree - Channel {i}')
                fig_path = f'{plots_dir}\\{time_stamp}_{sample}_HDBSCAN_cluster_tree_channel_{channel}.png'
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.tight_layout()

    else:
        pass

    # visualise features against channel labels in a clustermap
    if clustermap:
        print(f'Plotting clustermaps...')
        
        if map_reduced:
            channel_features = components
        else:
            _, channel_features = split_features(input_path, sample)

        for channel, channel_components in enumerate(channel_features):
            cluster_data = pd.DataFrame(channel_components)
            g = sns.clustermap(cluster_data,
                               method='ward',
                               metric='euclidean',
                               row_cluster=True,
                               col_cluster=False)
            g.figure.suptitle(f'Channel {channel} Clustermap',
                              y=1.02)
            fig_path = f'{plots_dir}\\{time_stamp}_{sample}_{method}_clustermap_channel_{channel}.png'
            g.figure.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f'Saved clustermap for channel {channel}')

            if do_dendro:
                z = linkage(cluster_data,
                            method='ward',
                            metric='euclidean')
                plt.figure(figsize=(6, 4))
                dendrogram(z)
                plt.title(f'Row dendrogram for channel {channel}')
                fig_path = f'{plots_dir}\\{time_stamp}_{sample}_clustermap_cluster_tree_channel_{channel}.png'
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.tight_layout()
    
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
                        help='Dimensionality reduction method to use')
    parser.add_argument('--coords_type', type=str, default='cartesian',
                        choices=['polar', 'cylindrical', 'cartesian'],
                        help='Type of coordinates to colour points by')
    parser.add_argument('--clustering', action='store_true',
                        help='Cluster with HDBSCAN')
    parser.add_argument('--cluster_reduced', action='store_true',
                        help='Perform HDBSCAN clustering on dimensionality-reduced data (after PCA/UMAP)')
    parser.add_argument('--do_dendro', action='store_true',
                        help='Plots a dendrogram based on the internally constructed cluster hierarchy tree.')
    parser.add_argument('--clustermap', action='store_true',
                        help='Plots a hierarchically-clustered heatmap')
    parser.add_argument('--map_reduced', action='store_true',
                        help='Cluster on (raw) features or dimensionality-reduced features')
    args = parser.parse_args()

    if not args.method.isupper():
        args.method = args.method.upper()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))