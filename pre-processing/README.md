﻿# Dimensionality Reduction and Spatial Clustering for Image Embeddings

## `slice_extraction_parquet`: Extracting dense slices from image
This script filters the input Parquet file of cell embeddings to retain 15 z-slices, centred on the densest region (i.e., the slice with the most detected points). It was designed to reduce the dataset size while preserving biologically relevant areas, making downstream analyses more efficient and focused.

```bash
python -m slice_extraction_parquet \
--input_path /path/to/fille_with_features.parquet \
--output_path /path/to/output_folder \
```
Input:
- One Parquet file that contains the cell embeddings and metadata.

Parameters:
- `--input_path`: Path to Parquet file containing cell embeddings and metadata.
- `--output_path`: Path to folder where the new, filtered Parquet file will be directed to.

Output:
- One Parquet file, filtered for only 15 slices per sample image, retaining the cell embeddings and metadata for each row.


## `pca_umap_3d`: Performing dimensionality reduction and visualising as scatter plots
This script takes a Parquet file containing the cell embeddings and generates 3D scatter plots for the sample that the user specifies. Depending on flags that the user defines on the command line, this script can perform PCA or UMAP, HDBSCAN clustering, and/or clustermap generation, producing scatter plots for each channel. Points will be coloured by the marker intensity, spatial coordinates (polar, cylindrical, or cartesian), octant regions, and optionally, cluster labels. This offers a rich visual insight into the correlation between spatial and phenotypic cell variation.

![2025-06-10_15-01-58_3d_5d_LBT123_NES_NLS_CT_merged_15_channel_1_polar_UMAP_plot](https://github.com/user-attachments/assets/17f02f47-333e-4502-ba8e-d268861f8b6d)
*Figure showing the collection of plots on Channel 1 generated using UMAP, coloured by marker intensity, polar coordinates, octant regions, and three levels of clustering from creating the cluster map.*

![image](https://github.com/user-attachments/assets/17387132-bb25-4e81-8c39-6d94889a2eb2)
*Figure showing the cluster maps generated for both Channels 0 and 1, at the third level, with eight clusters.*

```bash
python -m pca_umap_3d \
--input_path /path/to/file_with_features.parquet \
--output_path /path/to/output_folder \
--sample 3d_sample_001 \
--method umap \
--coords_type polar \
```
Input:
- One Parquet file that contains the cell embeddings and metadata.

Parameters:
- `--input_path`: Path to Parquet file containing cell embeddings and metadata.
- `--output_path`: Path to folder where scatter plots (and labels with coordinates) will be directed to.
- `--sample`: Specify sample to perform dimensionality reduction (and clustering) on.
- `--method`: Dimensionality reduction method to use: PCA or UMAP, default = PCA.
- `--coords_type`: Type of coordinates to colour points by: Cartesian, polar (spherical), or cylindrical, default = polar.
- `--do_hdbscan`: (Optional) If flagged, HDBSCAN clustering will be performed, additional scatter plots will be created per channel, and a CSV file containing coordinates of the points and their cluster labels will be saved into the output folder.
- `--hdbscan_reduced`: (Optional) If flagged, dimensionality-reduced features will be used for HDBSCAN clustering instead of the raw embeddings from the Parquet file.
- `--clustermap`: (Optional) If flagged, a hierarchically-clustered heatmap will be created, and a CSV file containing coorodinates of the points and their cluster labels will be saved into the output folder.
- `--n_levels`: Level of depth for clustering (n levels result in 2^n clusters), default = 3.
- `--map_reduced`: (Optional) If flagged, dimensionality-reduced features will be used to create cluster map instead of the raw embeddings from the Parquet file.

Output - one output folder, containing two daughter folders:
- The `/labels` folder will only be created if `--do_hdbscan` or `--clustermap` was flagged. It will generate as many CSV files as there are channels for each method of clustering. Each CSV file contains the coordinates (`'z', 'y', 'x'`) of the points and their respective cluster labels, generated from HDBSCAN clustering or cluster map creation. If `--clustermap` was flagged, sub-folders will be created, each corresponding to the level of clustering performed, in them would contain the CSV files generated.
- The `/plots` folder will contain PNG files of the plots that the script created. There will be at least as many plots as there are channels. If `--clustering` was flagged, HDBSCAN clustering will be performed and another scatter plot will be created and saved into this folder.


## `napari_points_labels`: Visualising clusters in napari
This script extracts sample images from an HDF5 file (or file family) and overlays cluster-labelled points onto the organoid in napari. The cluster labels are read in from the CSV files, which are contained in the folder specified by the user. These CSV files were generated from creating cluster maps when running `pca_umap_3d` and flagging `--clustermap`. Each channel's points are coloured by the cluster labels, allowing spatial inspection of the clusters. This visual context is essential for identifying the positive cluster to guide downstream class label prediction.

![image](https://github.com/user-attachments/assets/02729e9e-a84b-4deb-994e-040fbd61e342)
*Figure showing the visualisation of a cerebral organoid from the HDF5 file, under maximum intensity projection along the z-axis, with the points coloured by 3 levels of clustering, or 8 clusters. The red cluster distinctly follows where the tumour is.*

```bash
python -m napari_points_predicted \
--h5_path /path/to/hdf5_file[_family_%d].h5 \
--labels_path /path/to/labels_folder \
--sample 3d_sample_001 \
--voxelsize 3.695 0.3594 0.3594 \
```
Input:
- One HDF5 file or file family.
- The `/labels/sample` folder from the last step, containing the CSV files with the coordinates and cluster labels of the points for each channel.

Parameters:
- `--h5_path`: Path to HDF5 file or file family member (replace `_0.h5` with `_%d.h5`).
- `--memb_size`: If input is a HDF5 file family, specify the size of each file family member.
- `--labels_path`: Path to CSV file containing point coordinates, ground truth labels, predicted class labels, and predicted probabilities, from MLP prediction.
- `--sample`: Specify sample to visualise, important for selecting correct image from HDF5 file, and for ensuring that the correct labels CSV file are read.
- `--do_out_of_slice`: (Optional) If flagged, out of slice display will be toggled on for all points layers, can be toggled off manually later.
- `--hide_all_points`: (Optional) If flagged, all points layers visibility will be toggled off, can be toggled on manually later.
- `--point_size`: Specify the size of the points, default = 10.
- `--voxelsize`: Specify the voxel size of the points, default = [1, 1, 1].
- `--extract_slices`: (Optional) If flagged, only the 15 point-densest slices will be extracted and visualised.
- `--do_mip`: (Optional) If flagged, maximum intensity projection will be applied along the z-axis, converting the 3D image into 2D.

Output:
- No output folder will be created, but napari will be launched with the points layers rendered over the image layers.
