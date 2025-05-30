# Dimensionality Reduction and Spatial Clustering for Image Embeddings

## `pca_umap_3d`: Performing dimensionality reduction and visualising as scatter plots
The user will have to specify the sample that the dimensionality reduction and visualisation will be performed on. This script will then perform dimensionality reduction on cell embedding features from the input PARQUET file for each channel, using the chosen method (PCA or UMAP). Scatter plots will thereafter be created, coloured by the marker intensities and coordinates. By default, polar coordinates will be used, but the user can also define the type of coordinates to be cylindrical or Cartesian. These figures will be saved as PNG files into the output folder.

Additionally, the user can also choose to perform HDBSCAN clustering, which will be performed on the dimensionally reduced features. More scatter plots comparing the channels will be created, and labels generated from the clustering will be saved to CSV files, into the output folder, along with the coordinates extracted from the PARQUET file. The user will also have the option to create a hierarchically-clustered heatmap that clusters the points based on the specified channel number.

```bash
python -m pca_umap_3d \
--input_path /path/to/file_with_features \
--output_path /path/to/output_folder \
--sample r01c01 \
--method umap \
--coords_type polar \
```
Input:
- One PARQUET file that contains the cell embeddings and metadata.

Parameters:
- `--input_path`: Path to PARQUET file containing cell embeddings and metadata.
- `--output_path`: Path to folder where scatter plots (and labels with coordinates) will be directed to.
- `--sample`: e.g. r01c01, Specific sample to perform dimensionality reduction on.
- `--method`: Dimensionality reduction method to use: pca or umap.
- `--coords_type`: Type of coordinates to colour points by: cartesian, polar, or cylindrical.
- `--clustering`: (optional) If flagged, HDBSCAN will be performed, scatter plots will be created, and labels with coordinates will be saved into the output folder.
- `--cluster_reduced`: If flagged, dimensionality-reduced features will be used for HDBSCAN clustering.
- `--do_dendro`: If flagged, a dendrogram based on the internally constructed cluster hierarchy tree will be plotted.
- `--clustermap`: (optional) If flagged, a hierarchically-clustered heatmap will be created.
- `--n_levels`: Level of depth for clustering (higher n_levels will result in more clusters).
- `--map_reduced`: If flagged, dimensionality-reduced features will be used to create cluster map.

Output - one output folder, containing two daughter folders:
- The `labels` folder will only be created if `--clustering` was flagged. It will contain as many CSV files as there are channels. Each CSV file contains the coordinates (`'z, y, x'`) of the points and their respective labels, generated from HDBSCAN clustering.
- The `plots` folder will contain PNG files of the plots that the script created. There will be at least as many plots as there are channels. If `--clustering` was flagged, HDBSCAN will be performed and another scatter plot will be created and saved into this folder.


## `napari_points_labels`: Visualising in napari
This script will extract the TIF file corresponding to the specified sample from the input images folder and open each channel as an individual image layer in napari. Points will also be rendered in napari, based on the coordinates specified in the labels CSV files for each channel, extracted from the input labels folder. These labels will be coloured by clusters, which is a result of the HDBSCAN clustering from the previous step.

```bash
python -m napari_visual \
--h5_path /path/to/image_folder \
--labels_path /path/to/labels_folder \
--sample r01c01
```
Input:
- Either a singular HDF5 file or a HDF5 file family with `%d` replacing the member number.
- The `labels` folder from the last step, containing the CSV files with the coordinates and labels of the points for each channel.

Parameters:
- `--h5_path`: Path to folder containing image files.
- `--memb_size`: Only if a HDF5 file family was used, specify the size of the family members.
- `--labels_path`: Path to folder with files containing labels and coordinates.
- `--sample`: e.g. r01c01, Specific sample to visualise in napari.
- `--do_out_of_slice`: (optional) If flagged, out of slice display will be toggled for all points layers upon launch.
- `--hide_all_points`: (optional) If flagged, all points layers will be hidden upon launch.

Output:
- No output folder will be created, but napari will be launched with the points rendered over the images.