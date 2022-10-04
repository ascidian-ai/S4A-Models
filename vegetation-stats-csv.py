from multiprocessing import Pool, cpu_count, freeze_support
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union
import netCDF4
from pycocotools.coco import COCO
from tqdm import tqdm

import time
from datetime import datetime
from utils.settings.config import CROP_ENCODING, LINEAR_ENCODER
from utils.tools import NVDI, kNVDI

def starttimer():
    start = datetime.now()
    return start

def endtimer(start):
    end = datetime.now()
    durn = end - start
    return end, durn

RANDOM_SEED = 16
IMG_SIZE = 366

np.random.seed(RANDOM_SEED)

def extract_metrics(
        mode,
        crop_encoding,
        file: Union[str, Path],
        verbose: bool = True,
        freq: str = '1MS',
        save_path: Union[str, Path] = 'datasets/indices'
) -> bool:
    if verbose:
        st = starttimer()
        print(f'Working : {file} | {st.strftime("%Y-%m-%d %H:%M:%S")}')

    file = Path(file)
    save_path = Path(save_path)

    dump_path = save_path / 'temp' / (file.stem + '_indices.csv.gz')

    if dump_path.exists():
        print(f'Exists: {file}')
        return True

    # Load netcdf as xarray
    netcdf = netCDF4.Dataset(file, 'r')

    # Create Date Range filter
    year = netcdf.patch_year
    date_range = pd.date_range(start=f'{year}-01-01', end=f'{int(year) + 1}-01-01', freq=freq)

    csv_data = []

    # RED Band
    B04 = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf['B04']))
    B04 = B04.groupby_bins(
        'time',
        bins=date_range,
        right=True,
        include_lowest=False,
        labels=date_range[:-1]
    ).median(dim='time')
    B04 = B04.resample(time_bins=freq).median(dim='time_bins')
    B04 = B04.interpolate_na(dim='time_bins', method='linear', fill_value='extrapolate')

    # RED Band
    B08 = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf['B08']))
    B08 = B08.groupby_bins(
        'time',
        bins=date_range,
        right=True,
        include_lowest=False,
        labels=date_range[:-1]
    ).median(dim='time')
    B08 = B08.resample(time_bins=freq).median(dim='time_bins')
    B08 = B08.interpolate_na(dim='time_bins', method='linear', fill_value='extrapolate')

    ndvi = NVDI(B04, B08)
    kndvi = kNVDI(B04, B08)

    parcels = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf['parcels'])).to_array().squeeze().values
    labels = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf['labels'])).to_array().squeeze().values

    # iterate month by month ???????????????????????????????????????????????
    for parcel in np.unique(parcels):
        # Mask of this unique parcel
        mask = (parcels == parcel)
        label_code = int(np.unique(labels[mask])[0])
        label_name = crop_encoding[label_code]
        csv_band = {
            'mode': mode,
            'country_code': netcdf.patch_country_code,
            'year' : netcdf.patch_year,
            'month': 0,
            'tile_id': netcdf.patch_tile,
            'patch_id': netcdf.patch_name,
            'parcel_id': int(parcel),
            'label_code': label_code,
            'label_name': label_name,
            'counts': int(np.count_nonzero(mask))
        }

        # How to Mask ?????????????????????????
        #parcel_ndvi = ndvi[mask]
        #parcel_kndvi = kndvi[mask]

        # Calculate average of mean, stdev and var for ndvi and kndvi
        #csv_band[f'NDVI'] = parcel_ndvi
        #csv_band[f'kNDVI'] = parcel_kndvi
        csv_band[f'{parcel_ndvi}_{interval:02d}_mean'] = float(np.nanmean(parcel_ndvi))
        csv_band[f'{parcel_ndvi}_{interval:02d}_std'] = float(np.nanstd(parcel_ndvi))

        csv_data.append(csv_band)

    # Concat to dataframe and write to disk
    pd.DataFrame(csv_data).to_csv(
        dump_path,
        compression='gzip',
        index=False
    )

    if verbose:
        et, durn = endtimer(st)
        print(f'Finished: {str(file)} | Duration: {durn.total_seconds()}s')


    return True


def main():

    # Fork is faster and more memory efficient for our task, default for UNIX, but making sure
    #assert platform.system().lower() == 'linux', f'This system is using fork() as a method for multiprocessing,' \
    #                                             f'fork is only available in UNIX systems. You are running on: ' \
    #                                             f'"{platform.system()}.'

    # Parse cli argumnets
    args = parse_args()

    # If a value not between 1-max_cores is given, change it to use max_cores-2
    if args.num_processes not in range(1, cpu_count() + 1):
        args.num_processes = cpu_count()

    # Keep min between max available cores and parsed ones
    # Always leave 2 cores out for synchronization
    args.num_processes = min(args.num_processes, cpu_count() - 2)

    # Training, Validation or Test Data set
    mode = args.mode

    # Convert to Pathlib objects
    coco_path = Path(args.coco_path)
    coco_root = Path(args.coco_root)
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    # temp path
    Path(save_path / 'temp').mkdir(exist_ok=True, parents=True)

    time_start = time.time()

    def log_result(result):
        # This is called whenever my_func() returns a result.
        # result_list is modified only by the main process, not the pool workers.
        result_list.append(result)

    def errorhandler(exc):
        print('Exception:', exc)

    if args.num_processes > 1:
        freeze_support()
        # Run in parallel
        pool = Pool(processes=args.num_processes, maxtasksperchild=1)

    # Create Crop Encodings
    crop_encoding_all = {v: k for k, v in CROP_ENCODING.items()}
    crop_encoding = {k: crop_encoding_all[k] for k in LINEAR_ENCODER.keys() if k != 0}
    crop_encoding[0] = 'Background/Other'
    crop_encoding_all[0] = 'Background/Other'

    total_files = 0
    result_list = []

    coco = COCO(coco_path)

    patches = sorted([patch['file_name'] for patch in list(coco.imgs.values())])

    for patch in patches:

        patch_path = coco_root / patch

        if args.num_processes > 1:
            # Launch n processes, where n (args.num_processes)
            _ = pool.apply_async(extract_metrics,
                                 args=( mode, crop_encoding_all, patch_path, args.verbose, '1MS', save_path),
                                 callback=log_result,
                                 error_callback=errorhandler
                                 )
        else:
            result = extract_metrics( mode=mode, crop_encoding=crop_encoding_all,
                                      file=patch_path, verbose=args.verbose, freq='1MS', save_path=save_path)
            result_list.append(result)

        total_files += 1

    if args.num_processes > 1:
        pool.close()
        pool.join()

    print(f'Successfully completed {sum(result_list)}/{total_files} file(s).')

    csvs = [file for file in save_path.rglob('temp/*patch*.gz')]

    # Open dfs
    dfs = [pd.read_csv(csv) for csv in tqdm(csvs, ncols=75, desc='Merging CSV(s).')]
    # Concat and write
    pd.concat(dfs, ignore_index=True).to_csv(save_path / (coco_path.stem + '_indices.csv.gz'),
                                             index=False, compression='gzip')

    # Delete
    _ = [csv.unlink() for csv in tqdm(csvs, ncols=75, desc='Deleting..')]

    # Remove dir
    Path(save_path / 'temp').rmdir()

    print(f'\nDone. Time Elapsed: {(time.time() - time_start) / 60:0.2f} min(s).\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str, default='exp1_patches2000_strat_coco_val.json')
    parser.add_argument('--coco_root', type=str, default='coco_files/')
    parser.add_argument('--save_path', type=str, default='data/oad/')
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--mode', type=str, default='test', required=True)
    parser.add_argument('--verbose', action='store_true', default=True, required=False)

    return parser.parse_args()


if __name__ == '__main__':
    main()
