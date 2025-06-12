from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import pandas as pd

def main(**kwargs):

    assert kwargs['input_path'] is not None
    assert kwargs['output_path'] is not None

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    input_path = kwargs['input_path']
    input_stem = Path(input_path).stem
    output_path = kwargs['output_path']
    Path(output_path).mkdir(exist_ok=True, parents=True)

    # Read in the data
    print('Reading in the dataset...')
    data = pd.read_parquet(input_path, engine='fastparquet')

    filtered_chunks = []

    for name, group in data.groupby('filename_img'):
        print(f'Extracting slices for {name}')
        z_counts = group['z'].value_counts().sort_index()
        z_centre = z_counts.idxmax()

        half = 15 // 2
        z_vals = z_counts.index
        z_start = max(z_vals.min(), z_centre - half)
        z_end = z_start + 15

        z = data['z'][:]
        mask = (z >= z_start) & (z < z_end)
        z_filtered = z[mask] - z_start

        print(f'Keeping slices {z_start}:{z_end}')
        z_range = list(range(z_start, z_end + 1))[:15]
        filtered = group[group['z'].isin(z_range)]
        filtered.loc[:, 'z'] = z_filtered

        filtered_chunks.append(filtered)

    # Concatenate all chunks together
    filtered_data = pd.concat(filtered_chunks)

    # Save output .parquet file
    out_dir = Path(output_path)
    out_path = f'{out_dir}/{time_stamp}_{input_stem}_filtered.parquet'
    filtered_data.to_parquet(out_path, index=False)

    print('Done!')

def parse_args():
    parser = ArgumentParser(description='Extracting the densest slices of a 3D image')
    parser.add_argument('--input_path', type=str, default=None,
                        help='Path to input file with features')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to output folder')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))