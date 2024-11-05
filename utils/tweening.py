"""
This python module contains a set of helper functions to generate a 
tweening dataset, based on data already downloaded. 
"""

########################################################################
# Import libraries
########################################################################
import os
import glob
import shutil
import json
import re
import functools
import sys
import time
from tqdm.notebook import trange, tqdm

import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rx
import rasterio as rs
import math

from rasterio.enums import Resampling
from shapely.geometry import box
from shapely.ops import transform
from shapely.geometry import mapping
import geopandas as gpd
from fiona.crs import from_epsg
from pyproj import CRS

from shapely.geometry import mapping
from datetime import date, datetime, timedelta
from datetime import timedelta

########################################################################
# Functions
########################################################################
def create_data(path):
    
    tif_fps = sorted(glob.glob(os.path.join(path, '*.tif')))
    dataset_name = []

    for tfs in tif_fps:
        print(f"Processing file: {tfs}")
        ds = xr.open_dataset(tfs)
        date = (os.path.basename(tfs).split(".")[2:4])
        date = ''.join(date)
        print(date)
        ds = ds.assign_coords(time=date)
        dataset_name.append(ds)
    dataset_name = xr.concat(dataset_name, dim='time')
    return dataset_name

def read_raster_data(input_pattern, band_index, timestamps):
    raster_files = sorted(glob.glob(input_pattern))
    data_arrays = []
    
    for i, raster_path in enumerate(raster_files):
        try:
            with rs.open(raster_path) as src:
                raster = src.read(band_index)  # Read the specified band
                data_array = xr.DataArray(
                    data=raster,
                    dims=["y", "x"],
                    coords={"y": range(raster.shape[0]), "x": range(raster.shape[1])}
                )
                data_arrays.append(data_array.expand_dims(time=[timestamps[i]]))  # Add time dimension
        except Exception as e:
            print(f"Error reading {raster_path}: {e}")

    return xr.concat(data_arrays, dim='time')

def process_target_band(target_tile_path):
    """
    Processes lst bands for grid and crs
    Args:
        target_tile_path (str): Path to directory with geotiff target files
    """
        # load target and update meta
    with rs.open(target_tile_path) as target:
        meta = target.meta
        array = target.read(1)

        # Create grid
        target_tile = rx.open_rasterio(target_tile_path)
        grid_out = target_tile.squeeze().drop("band")
        grid_out.attrs["spatial_ref"] = grid_out["spatial_ref"].attrs
        grid_out = grid_out.drop("spatial_ref")

        # Extract crs
        crs = target_tile.rio.crs

        return [array], grid_out, meta, crs

def process_hls_bands(hls_bands):
    """
    Processes hls bands by reading band data as array
    Args:
        hls_bands (list): list of HLS bands
    """

    hls_bands_list = []
    for layer in hls_bands:
        with rs.open(layer) as band:
            array = band.read(1)
            hls_bands_list.append(array)

    return hls_bands_list

def duplicate_hls_bands(save_file, hls_date, tweening_period, data_directory, city_iso, tile_id):
    """
    Duplicates geotiff consisting of stacked HLS bands for every hour between tweening period (days)
    Args:
        save_file (str): path to initial HLS stacked geotiff corresponding to T000000
        tweening_period (int): number of days to create tweening inputs for
        data_directory (str): path to save tweening inputs
        city_iso (str): city-name_iso
        tile_id (str): tile_id
    """
    
    
    with rs.open(save_file) as src:
        num_bands = min(6, src.count)  # ensure 6 hls bands and 4 hls indices
        image = np.array([src.read(b + 1).astype(np.float32) for b in range(num_bands)])
        profile = src.profile
        
        profile.update(count=num_bands)

        year = int(hls_date[0:4])
        month = int(hls_date[4:6])
        day = int(hls_date[6:8])
        base_time = "000000"
        num_files = [range(24*tweening_period)]
        with tqdm(total=len(num_files), leave=True) as pbar:
            for x in num_files:
                for months in range(month, month+1):
                    for days in range(day, day + tweening_period):
                        date = datetime(year, month, days)  
                        time = base_time[0:4]
                        for hour in range(24):  
                            tif = f'{data_directory}{city_iso}.{tile_id}.{date.strftime("%Y%m%d")}.T{hour:02d}{time}.input_file.tif'
                            #print(f'Processing {tif}')  
                            with rs.open(tif, 'w', **profile) as dst:
                                for b in range(num_bands):
                                    dst.write(image[b], b + 1)
                            pbar.update(1)
                               
                            

def stacking(processed_bands, grid, crs):
    """
    Stacks HLS bands 
    Args:
        processed_bands (list): list of data arrays from stacked HLS bands
        grid (xr data array): array of target lst grid
        crs (str): crs of target lst   
    """
       
    multidim_array = np.array(processed_bands)
    
    multi_band_da = xr.DataArray(
        multidim_array,
        coords={
            "band": np.arange(1, len(processed_bands) + 1),
            "y": grid.y.values,
            "x": grid.x.values,
        },
        dims=["band", "y", "x"],
    )
    
    multi_band_da.rio.write_crs(crs, inplace=True)

    return multi_band_da

def files_extractor(folder_path: str, outputs=False):
    """
    Extracts files corresponding to city-name_iso 
    Args:
        folder_path (str): path to hls data 
         
    """

    extensions_to_find = [".nc", ".tif"]
    if outputs == False:
        all_files = []
        for extention in extensions_to_find:
            source_file_pattern = f"{folder_path}/**/*{extention}"
            files_with_ext = glob.glob(source_file_pattern, recursive=True)
            all_files.extend(files_with_ext)

    else:
        all_files = []
        for extention in extensions_to_find:
            source_file_pattern = f"{folder_path}/*{extention}"
            files_with_ext = glob.glob(source_file_pattern)
            all_files.extend(files_with_ext)

    city_names = []
    for file_path in all_files:
        file_name = file_path.split("/")[-1]
        city = file_name.split(".")[0]
        city_names.append(city)

    unique_names = list(set(city_names))
    unique_names.sort()
    all_files.sort()

    return unique_names, all_files

def filter_city(city_name, lst):
    """
    Filters files based on city-name_iso
    Args:
        city_name (str): city-name_iso
        lst (str): path to lst file
         
    """
    filtered_files = [
        file_path
        for file_path in lst
        if city_name == file_path.split("/")[-1].split(".")[0]
    ]

    return filtered_files

def filter_year(year, files):
    """
    Filters files based on year to reduce data in memory
    Args:
        year (int): year from date
        files (str): path to all files
         
    """

    filtered_files = [
        file_path
        for file_path in files
        if year == file_path.split("/")[-1].split(".")[1]
    ]
    return filtered_files

def convert_time(city):
    """
    Retreives time shift from UTC to local for city of interest
    Args:
        city (str): city-name_iso
         
    """
    global_cities_db_filepath = "../assets/databases/global_cities_database.csv"  # add to main payload checks above
    time_zone_csv = pd.read_csv(global_cities_db_filepath)
    city_name = city.split("_")[0]
    city_iso = city.split("_")[1]
    filtered_df_t = time_zone_csv[(time_zone_csv["CITY_NAME"] == city_name) & (time_zone_csv["COUNTRY_ISO"] == city_iso)]
    time_shift = filtered_df_t["UTC_TIMESHIFT"].values[0]
    return time_shift

def load_era5(era5_files, city_iso):
    """
    Loads ERA5 t2m files into xarray data array in expected format, performs time shift on ERA5 data (UTC to local)
    Args:
        era5_files (list): list of ERA5 netcdf files for city and year of interest
        city_iso (str): city-name_iso
         
    """
    timeshift = convert_time(city_iso)
    # Get variables
    list_sub_dict = [path.split("/")[5] for path in era5_files]
    vars = np.unique(list_sub_dict)

    # Run concating of variables
    merged_ds = []
    var_keys = []
    for var in vars:
        var_paths = [era5_path for era5_path in era5_files if var in era5_path]
        concat_ds = []
        for ds in var_paths:
            dataset = xr.open_dataset(ds)

            #change name of time dim
            if "valid_time" in list(dataset.dims):
                dataset =  dataset.rename({'valid_time': 'time'})
            else:
                dataset
            
            # Remove unnesacarry dims
            if "expver" in list(dataset.dims):
                dataset = dataset.reduce(np.nanmean, dim="expver", keep_attrs=True)
            else:
                dataset

            # Remove unnecessary coords
            if "expver" in list(dataset.coords):
                dataset = dataset.drop_vars("expver")
            else:
                dataset

            if "number" in list(dataset.coords):
                dataset = dataset.drop_vars("number")
            else:
                dataset

            if "surface" in list(dataset.coords):
                dataset = dataset.drop_vars("surface")
            else:
                dataset
                
            # Check dataset for inf and replace with nan
            dataset = dataset.where(dataset != np.isinf, np.nan)
            dataset = dataset.rio.write_crs("EPSG:4326")

            dataset["longitude"] = dataset["longitude"].astype(np.float32, keep_attrs=True)
            dataset["latitude"] = dataset["latitude"].astype(np.float32, keep_attrs=True)

            key = list(dataset.keys())

            concat_ds.append(dataset)
            var_keys.append(key)

        var_ds = xr.concat(concat_ds, dim="time")
        merged_ds.append(var_ds)

    # Extract bounds
    min_lon = merged_ds[0]["longitude"].min().values
    max_lon = merged_ds[0]["longitude"].max().values
    min_lat = merged_ds[0]["latitude"].min().values
    max_lat = merged_ds[0]["latitude"].max().values

    # Merged
    merged_ds[-1] = merged_ds[-1].rio.clip_box(
        minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat
    )
    merged_ds[-1] = xr.align(merged_ds[0], merged_ds[-1], join="override")[1]
    main_ds = xr.merge(merged_ds)

    # Do timeshift to local time
    if timeshift >= 0:
        main_ds_shifted = main_ds.shift(time=math.ceil(timeshift))
        main_ds_shifted = main_ds_shifted.dropna(dim="time", how="all")
    elif timeshift < 0:
        main_ds_shifted = main_ds.shift(time=math.floor(timeshift))
        main_ds_shifted = main_ds_shifted.dropna(dim="time", how="all")

    # Assign CRS
    main_ds_shifted = main_ds_shifted.rio.write_crs("EPSG:4326")
    # da_variable.rio.set_spatial_dims("x", "y", inplace=True)

    # Get unique keys
    unique_var_keys = list(set([x[0] for x in var_keys]))
    unique_var_keys = sorted(unique_var_keys)
    return main_ds_shifted, unique_var_keys

def process_era5_bands(era5_ds, var_keys, grid, crs, date, time):
    """
    Produces ERA5 stacked bands (arrays) for each statistic
    Args:
        era5_ds (xr dataset): xr dataset of t2m files loaded in previous function
        var_keys (list): city-list of variables in era5 files (if others are downloaded, e.g. skin_temperature)
        grid (xr data array): array of target lst grid
        crs (str): crs of target lst  
        date (str)
        time (str)         
    """
    # Reproject data to confirm with grid
    ds_proj = era5_ds.rio.reproject(crs, nodata=np.nan)

    # Crop data according to grid bounds
    clipped_ds = ds_proj.rio.clip_box(
        minx=grid.rio.bounds()[0] - 10000,
        miny=grid.rio.bounds()[1] - 10000,
        maxx=grid.rio.bounds()[2] + 10000,
        maxy=grid.rio.bounds()[3] + 10000,
    )
    # Convert date-time to local
    datetime_string = date + time
    datetime_obj = pd.to_datetime(date + time)

    # Set time parameters
    sunrise_time = pd.to_datetime(date + "T060000")
    sunset_time = pd.to_datetime(date + "T180000")
    prev_sunset_time = pd.to_datetime(date + "T180000") - pd.DateOffset(days=1)
    foll_sunrise_time =  pd.to_datetime(date + "T060000") + pd.DateOffset(days=1)
    delta_times = [
        (datetime_obj - pd.DateOffset(hours=2)),
        (datetime_obj + pd.DateOffset(hours=2)),
    ]

    # Run main layer extraction loop
    era5_bands = []
    for var in var_keys:
        # 2m temperature
        if var == 't2m':

            # Extract var array
            var_da = clipped_ds[var]
            
            # Convert units to degrees celcius
            var_da = var_da - 273.15

            # Aquisition time
            aquisition_time = var_da.sel(time=datetime_obj, method="pad").rename(new_name_or_name_dict="t2m_aquisition")
            
            # Aquisition delta 2 hours before, two hours after
            aquisition_delta_sample = var_da.sel(time=slice(delta_times[0], delta_times[1]))
            aquisition_delta = (aquisition_delta_sample[-1] - aquisition_delta_sample[0]).rename(new_name_or_name_dict="t2m_delta")

            # Daytime max
            daytime_sample = var_da.sel(time=slice(sunrise_time, sunset_time))
            daytime_max = daytime_sample.max('time').rename(new_name_or_name_dict="t2m_max_day")

            # Previous night min
            prev_night_sample = var_da.sel(time=slice(prev_sunset_time, sunrise_time))
            prev_night_min = prev_night_sample.min('time').rename(new_name_or_name_dict="t2m_min_prev")

            # Add to list of bands
            era5_bands.extend([aquisition_time, aquisition_delta, daytime_max, prev_night_min])

        #else:
            #print('not_implemented')

    # Regrid layers
    era5_regrid_bands = []
    for band in [aquisition_time, aquisition_delta, daytime_max, prev_night_min]:
        spatial_avg = float(band.mean().values)
        new_array = np.full(
            (grid.data.shape[0], grid.data.shape[1]), spatial_avg
        )
        era5_regrid = grid.copy()
        era5_regrid.data = new_array
        era5_regrid_bands.append(era5_regrid.data)

    return era5_regrid_bands

def process_aux_bands(city, grid, crs, date, time):
    """
    Produces auxiliary feature bands of cos and sin time of day and day of year -- these are not used in training/inference
    Args:
        city (str): city-name_iso
        grid (xr data array): array of target lst grid
        crs (str): crs of target lst  
        date (str)
        time (str)         
    """
    # Convert time to local time
    datetime_string = date + time
    datetime_obj = pd.to_datetime(datetime_string)
    

    #### COS, SIN time features
    # arrays of 24 hours, 366 days
    time_of_d = np.linspace(0, 2*math.pi, 24)
    day_of_y  = np.linspace(0, 2*math.pi, 366)  

    # doy for acquisition
    doy = [int(date[:4]), int(date[4:6]), int(date[6:])]
    day_of_year = int(format(datetime(doy[0],doy[1],doy[2]), '%j'))
    if day_of_year >= 366:
        day_of_year = 0

    # toy for acquisition - Assume time on image is local time
    time_of_day = str(time) #changed from str(time.time())
    
    hour = int(time_of_day[1:2]) ### changed from 0:2
    mins = int(time_of_day[3:5])

    if mins >=30:
        hour += 1
    hour_of_day = hour

    if hour_of_day >= 24:
        hour_of_day = 0 
        
    # COS, SIN time of day
    cos_tod = np.cos(time_of_d[hour_of_day])
    sin_tod = np.sin(time_of_d[hour_of_day])
    
    # COS, SIN day of year
    cos_doy = np.cos(day_of_y[day_of_year])
    sin_doy = np.sin(day_of_y[day_of_year])

    aux_bands = []
    for band in [cos_tod, sin_tod, cos_doy, sin_doy]:
        new_array = np.full((grid.data.shape[0], grid.data.shape[1]), band)
        aux_regrid = grid.copy()
        aux_regrid.data = new_array
        aux_bands.append(aux_regrid.data)

    return aux_bands


def add_era5_stack(one_file, city_iso, grid_out, crs, era5_4city):
    """
    Computes and stacks ERA5 and aux feature bands to each stacked HLS file for each hour
    Args:
        one_file (str): path to one HLS only stacked hourly input
        city_iso (str): city-name_iso
        grid (xr data array): array of target lst grid
        crs (str): crs of target lst  
        era5_4city (list): list of ERA5 files for city 
              
    """
    date = one_file.split("/")[-1].split(".")[2]
    time = one_file.split("/")[-1].split(".")[3]
    city = one_file.split("/")[-1].split(".")[0]

    # load and collate ERA5 data
    era5s = filter_year(year=date[0:4], files = era5_4city)
    era5_ds, keys = load_era5(era5s, city_iso)
    era5_bands = process_era5_bands(era5_ds, keys, grid_out, crs, date, time)
    
    # add aux features 
    aux_bands = process_aux_bands(city, grid_out, crs, date, time)

    #print(era5_regrid_bands)
    output_filename = one_file.replace("input_file", "inputs")

    # Read hls_bands
    hls_bands = []
    with rs.open(one_file) as src:
        array1 = src.read(1)
        hls_bands.append(array1)
        array2 = src.read(2)
        hls_bands.append(array2)
        array3 = src.read(3)
        hls_bands.append(array3)
        array4 = src.read(4)
        hls_bands.append(array4)
        array5 = src.read(5)
        hls_bands.append(array5)
        array6 = src.read(6)
        hls_bands.append(array6)
       
    all_bands = hls_bands + era5_bands +  aux_bands

    multidim_array = np.array(all_bands)
    multi_band_da = xr.DataArray(
            multidim_array,
            coords={
                "band": np.arange(1, len(all_bands) + 1),
                "y": grid_out.y.values,
                "x": grid_out.x.values,
            },
            dims=["band", "y", "x"],
        )
    multi_band_da.rio.write_crs(crs, inplace=True)

    multi_band_da.rio.to_raster(output_filename, dtype="float32", driver="COG")
