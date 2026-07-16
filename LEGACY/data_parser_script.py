import numpy as np
import xarray as xr
import os
from pathlib import Path

def parse_dataset(path: str):
    _data = np.loadtxt(path, unpack=True)
    wave_number, _counts = _data[-2], _data[-1]
    num_coords = len(_data) - 2
    coords = [_data[idx] for idx in range(num_coords)]
    unique_coords = [np.unique(coord, return_inverse=True) for coord in coords]
    unique_coords += [np.unique(wave_number, return_inverse=True)]
    idxs, values = [coord[1] for coord in unique_coords], [coord[0] for coord in unique_coords]
    counts_shape = tuple([len(coord[0]) for coord in unique_coords])
    counts = np.zeros(counts_shape, dtype=_counts.dtype)

    for z in zip(*idxs, _counts):
        counts[tuple(z[:-1])] = z[-1]

    dimension_names = [f'X_{i}' for i in range(len(values) - 1)] + ['wave_number']
    counts = xr.DataArray(counts, coords=values, dims=dimension_names)
    counts.wave_number.attrs['units'] = 'cm^-1'
    return counts

if __name__ == "__main__":
    # Define directories

    datadir = Path(__file__).parent / "data"
    material = "buffer+graphene/stdrecipe_1" # Specify your folder
    raw_data_dir = datadir / 'raw_data' / material
    parsed_data_dir = datadir / 'parsed_data_spatial' / material

    # Ensure parsed_data directory exists
    Path(parsed_data_dir).mkdir(exist_ok=True)

    # Traverse raw_data directory
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, raw_data_dir)
                parts = relative_path.split(os.sep)
                # if len(parts) == 1:
                #     # File is in the main folder
                #     system_type = "main"
                #     subdirs = []
                # else:
                system_type = parts[0]
                subdirs = parts[1:-1]

                # Parse the dataset
                dataset = parse_dataset(file_path)
                dataset.attrs['system_type'] = system_type
                
                # Generate output directory
                output_dir = os.path.join(parsed_data_dir, system_type, *subdirs)
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate filename with dimensions AND original name
                original_name = os.path.splitext(file)[0]
                dims = dataset.dims[:-1]  # Exclude wave_number
                dim_lengths = [str(len(dataset[dim])) for dim in dims]
                dims_part = 'x'.join(dim_lengths)
                fname = f"{system_type}_{dims_part}_{original_name}.nc"
                
                # Save to NetCDF
                output_path = os.path.join(output_dir, fname)
                dataset.to_netcdf(output_path, engine="netcdf4")
                print(f"Saved parsed dataset to: {output_path}")