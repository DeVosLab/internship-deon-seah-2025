from pathlib import Path
import numpy as np
import h5py

"""
o find z-slice with most number of points
o take consecutive slices before and after that slice, so that the total is 10um thick
o all slices lie within valid z-range
"""

def extract_slices(img, points):
    z_coords = points[:, 0].astype(int)
    z_counts = np.bincount(z_coords, minlength=img.shape[0])
    z_centre = np.argmax(z_counts)

    half = 15 // 2
    z_start = max(0, z_centre - half)
    z_end = z_start + 15
    if z_end > img.shape[0]:
        z_end = img.shape[0]
        z_start = max(0, z_end - 15)
    subvol = img[z_start:z_end, :, :, :]
    return subvol, (z_start, z_end)

input_path = 'D:/3D_data/cerebral_dataset/output/hdf5/gt_channel_0/patches/cerebral_organoids_0_%d.h5'
output_path = Path('D:/3D_data/cerebral_dataset/output/hdf5/gt_channel_0/patches/filtered')
output_path = str(output_path.joinpath('cerebral_organoids_0_filtered_%d.h5'))

# Read in the data
print('Reading in the dataset...')
with h5py.File(input_path, 'r', driver='family', memb_size=3.5 * 1024**3) as f_in, \
     h5py.File(output_path, 'w', driver='family', memb_size=3.5 * 1024**3) as f_out:
    
    for sample_name in f_in:
        print(f'Extracting slices for {sample_name}')
        group = f_in[sample_name]
        group_out = f_out.create_group(sample_name)

        img = group['img'][()]
        z = group['z'][()]
        y = group['y'][()]
        x = group['x'][()]
        points = np.stack([z, y, x], axis=1)

        subvol, (z_start, z_end) = extract_slices(img, points)

        print(f'Transferring information for extracted slices: {z_start}:{z_end}')
        group_out.create_dataset('img', data=subvol, compression='gzip')

        mask = (z >= z_start) & (z < z_end)
        z_filtered = z[mask] - z_start
        y_filtered = y[mask]
        x_filtered = x[mask]

        group_out.create_dataset('z', data=z_filtered, compression='gzip')
        group_out.create_dataset('y', data=y_filtered, compression='gzip')
        group_out.create_dataset('x', data=x_filtered, compression='gzip')

        for key in group:
            if key in ['img', 'z', 'y', 'x']:
                continue
            data = group[key][()]
            if len(data) == len(z):
                group_out.create_dataset(key, data=data[mask], compression='gzip')
            else:
                group_out.create_dataset(key, data=data, compression='gzip')

        for attr in group.attrs:
            group_out.attrs[attr] = group.attrs[attr]

    for attr in f_in.attrs:
        f_out.attrs[attr] = f_in.attrs[attr]
