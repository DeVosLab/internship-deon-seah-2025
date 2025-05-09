# Dimensionality Reduction and Spatial Clustering for Image Embeddings
```bash
python -m pca_umap_3d \
--input_path /path/to/file_with_features \
--output_path /path/to/output_folder \
--sample r01c01 \
--method umap \
--coords_type polar \
--clustering
```

Parameters:
- `--input_path`: Path to PARQUET file containing cell embeddings and metadata.
- `--output_path`: Path to folder where scatter plots (and labels with coordinates) will be directed to.
- `--sample`: e.g. r01c01, Specific sample to perform dimensionality reduction on.
- `--method`: Dimensionality reduction method to use: pca or umap.
- `--coords_type`: Type of coordinates to colour points by: cartesian, polar, or cylindrical.
- `--clustering`: (optional) If flagged, HDBSCAN will be performed, a scatter plot will be created, and labels with coordinates will be saved into the output folder.

```bash
python -m napari_visual \
--img_path /path/to/image_folder \
--labels_path /path/to/labels_folder \
--sample r01c01
```

Parameters:
- `--img_path`: Path to folder containing image files.
- `--labels_path`: Path to folder with files containing labels and coordinates.
- `--sample`: e.g. r01c01, Specific sample to visualise in napari.