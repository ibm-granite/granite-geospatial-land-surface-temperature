"""
This python module contains a set of classes with helper functions to execute
the main preprocessing workflow for generating a multicity training dataset for
the UHI model fine-tuning. For more information, refer to the README.
"""

########################################################################
# Import libraries
########################################################################
import sys
import warnings
import traceback
import logging

logging.basicConfig(
    # stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
)

import multiprocessing
from multiprocessing import Pool, Manager

import os
import glob
import shutil
import json
import re
import functools

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

from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore", module="rioxarray")
logging.getLogger("rioxarray").setLevel(logging.CRITICAL + 1)
logging.getLogger("rasterio").setLevel(logging.CRITICAL + 1)

########################################################################
# Classes
########################################################################


class utils:
    """
    Set of methods to assist the main preprocessing workflow.

    ...

    Methods
    -------
    get_city_bbox(city_name, buffer, csv_path)
        Get bounding boxes for cities from pre-loaded cities database

    files_extractor(folder_path: str, outputs=False)
        Extract files from downloaded folders

    load_era5(era5_files, timeshift)
        Load ERA5 hourly 2m_temperature downloaded datasets and adjust for 
        local time zone

    filter_images(city_name, tile_id, date, time, lst)
        Filter hls images corresponding to lst tiles

    filter_city(city_name, lst)
        Filter downloaded datasets for corresponding city
    
    extract_info(file_path)
        Extract city_name, tile_id, date and time from filename

    """

    @staticmethod
    def get_city_bbox(city_name, buffer, csv_path):

        # open csv
        cities_df = pd.read_csv(csv_path)

        # city_bbox = []

        city = city_name.split("_")[0]
        city_iso = city_name.split("_")[1]

        filtered_df = cities_df[
            (cities_df["CITY_NAME"] == city) & (cities_df["COUNTRY_ISO"] == city_iso)
        ]

        if len(filtered_df) == 1:

            city_bbox = {
                "name": city_name,
                "maxx": filtered_df["BBX_XMAX"].values[0],
                "maxy": filtered_df["BBX_YMAX"].values[0],
                "minx": filtered_df["BBX_XMIN"].values[0],
                "miny": filtered_df["BBX_YMIN"].values[0],
            }

            # Convert bbox to polygon
            bbox_geom = box(
                filtered_df["BBX_XMIN"].values[0],
                filtered_df["BBX_YMIN"].values[0],
                filtered_df["BBX_XMAX"].values[0],
                filtered_df["BBX_YMAX"].values[0],
            )

            # Add buffer
            bbox_geom_buffered = bbox_geom.buffer(buffer, join_style=2)
            # Extract total bounds with buffer
            city_bbox_buffered = {
                "name": city_name,
                "maxx": bbox_geom_buffered.bounds[2],
                "maxy": bbox_geom_buffered.bounds[3],
                "minx": bbox_geom_buffered.bounds[0],
                "miny": bbox_geom_buffered.bounds[1],
            }

        else:
            logging.error(
                "City name is not found in list. Please check csv file for correct city name and ISO."
            )

        return city_bbox_buffered

    @staticmethod
    def files_extractor(folder_path: str, outputs=False):

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

    @staticmethod
    def load_era5(era5_files, timeshift):

        # Get variables
        list_sub_dict = [path.split("/")[4] for path in era5_files]
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

    @staticmethod
    def filter_images(city_name, tile_id, date, time, lst):

        search_key = ".".join([city_name, tile_id, date, time])
        filtered_files = [file_path for file_path in lst if search_key in file_path]

        # Sort list !!!
        filtered_files = np.sort(filtered_files)
        return filtered_files

    @staticmethod
    def filter_city(city_name, lst):

        filtered_files = [
            file_path
            for file_path in lst
            if city_name == file_path.split("/")[-1].split(".")[0]
        ]

        return filtered_files

    @staticmethod
    def extract_info(file_path):

        tile_id = file_path.split("/")[-1].split(".")[1]
        date = file_path.split("/")[-1].split(".")[2]
        time = file_path.split("/")[-1].split(".")[3]

        return tile_id, date, time


class hls_preprocess:
    """
    Set of methods to preprocess hls data.
    ...

    Methods
    -------
    compute_lst(nir, therm1, therm2, red)
        Calculate LST for given HLS bands B04, B10, B11 and B05 
        - Relevant literature: 
            Du C, Ren H, Qin Q, Meng J, Zhao S. A Practical Split-Window Algorithm for 
            Estimating Land Surface Temperature from Landsat 8 Data. Remote Sensing. 
            2015; 7(1):647-665. https://doi.org/10.3390/rs70100647
   
   scaling(band)
        Scale HLS bands 

    preprocess_hls(list_cities, bbox_buffer, csv_path, percent, hls_downloads, 
                    hls_save_dir, n_threads, hls_scale)
        Run preprocessing for every acquisition

    hls_band_process(working_path, percent, city, miny, maxy, minx, maxx, 
                    hls_save_dir, hls_scale, lock)
        Crop, filter, write CRS and calculate LST for each HLS band

    """

    @staticmethod
    def compute_lst(nir, therm1, therm2, red):
        # check scale factor, scale if unscaled (!=1)
        if nir.attrs["scale_factor"] != 1:
            nir_scaled = hls_preprocess.scaling(nir)
            therm1_scaled = hls_preprocess.scaling(therm1)
            therm2_scaled = hls_preprocess.scaling(therm2)
            red_scaled = hls_preprocess.scaling(red)

        else:
            nir_scaled = nir
            therm1_scaled = therm1
            therm2_scaled = therm2
            red_scaled = red

        lst = red_scaled.copy()
        NDVI = (nir_scaled - red_scaled) / ((nir_scaled + red_scaled) + 1e-10)
        pv = np.square(
            np.divide(
                (np.subtract(NDVI, np.min(NDVI))),
                (np.subtract(np.max(NDVI), np.min(NDVI))),
            )
        )

        emissivity_10 = (0.0015 * pv) + 0.9848
        emissivity_11 = (0.0011 * pv) + 0.9885

        tb_10 = therm2_scaled
        tb_11 = therm1_scaled

        mean_e = (
            emissivity_10 + emissivity_11
        ) / 2  # Use emissivity_10 twice as emissivity_11 is not defined
        diff_e = (
            emissivity_10 - emissivity_11
        )  # Similarly, emissivity_11 is not defined
        diff_tb = tb_10 - tb_11

        lst_data = (
            tb_10
            + (1.387 * diff_tb)
            + (0.183 * (diff_tb**2))
            - 0.268
            + ((54.3 - (2.238 * 0.013)) * (1 - mean_e))
            + ((-129.2 + (16.4 * 0.013)) * diff_e)
        )
        lst.data = lst_data
        lst.data = xr.where(lst.data != np.inf, lst.data, np.nan, keep_attrs=True)
        lst.data = xr.where(lst.data <= -273.15, -9999.0, lst.data, keep_attrs=True)
        lst.data = xr.where(lst.data > 100, -9999.0, lst.data, keep_attrs=True)
        lst.attrs["long_name"] = "LST"
        lst.attrs["scale_factor"] = 1
        return lst

    @staticmethod
    def scaling(band):
        scale_factor = band.attrs["scale_factor"]
        band_out = band.copy()
        band_out.data = band.data * scale_factor
        band_out.data = xr.where(
            band_out.data == (scale_factor * -9999.0),
            -9999.0,
            band_out.data,
            keep_attrs=True,
        )
        band_out.attrs["scale_factor"] = 1
        return band_out


    @staticmethod
    def preprocess_hls(list_cities, bbox_buffer, csv_path, percent, hls_downloads, hls_save_dir, n_threads, hls_scale):

        manager = Manager()
        lock = manager.Lock()

        processing_params = []

        # get bbox for city
        for city_iso in list_cities:
            city_bbox_buff = utils.get_city_bbox(city_iso, bbox_buffer, csv_path)

            miny = city_bbox_buff["miny"]
            maxy = city_bbox_buff["maxy"]
            minx = city_bbox_buff["minx"]
            maxx = city_bbox_buff["maxx"]
            city = city_bbox_buff["name"]

            processing_dir = os.path.join(hls_downloads, city)

            try:
                assert os.path.exists(
                    processing_dir
                ), f"Directory is empty. Please download data for {city}"
            except AssertionError as e:
                raise e

            for subdir in os.listdir(processing_dir):
                results = checks.check_bands_exist(os.path.join(processing_dir, subdir))

                if results == True:
                    working_path = os.path.join(processing_dir, subdir)

                # ADD parameters and subdirs to tasks for MP
                processing_params.append(
                    (
                        working_path,
                        percent,
                        city,
                        miny,
                        maxy,
                        minx,
                        maxx,
                        hls_save_dir,
                        hls_scale,
                        lock,
                    )
                )

        #### PREPROCESS WITH MULTIPROCESSING
        logging.info("Executing multiprocessing for HLS preprocessing tasks")

        with Pool(processes=n_threads) as pool:
            results = pool.starmap(hls_preprocess.hls_band_process, processing_params)
            pool.close()
            pool.join()
        logging.info(f"HLS preprocessing complete for {city}!")

    @staticmethod
    def hls_band_process(working_path, percent, city, miny, maxy, minx, maxx, hls_save_dir, hls_scale, lock):

        for file in os.listdir(working_path):
            sub = working_path.split("/")[-1]

            blue_path = os.path.join(working_path, "{}.B02.tif")
            green_path = os.path.join(working_path, "{}.B03.tif")
            red_path = os.path.join(working_path, "{}.B04.tif")
            nir_path = os.path.join(working_path, "{}.B05.tif")
            swir1_path = os.path.join(working_path, "{}.B06.tif")
            swir2_path = os.path.join(working_path, "{}.B07.tif")
            ten_path = os.path.join(working_path, "{}.B10.tif")
            eleven_path = os.path.join(working_path, "{}.B11.tif")
            fm_path = os.path.join(working_path, "{}.Fmask.tif")

            redband = red_path.format(sub, sub)
            blueband = blue_path.format(sub, sub)
            greenband = green_path.format(sub, sub)
            nirband = nir_path.format(sub, sub)
            temp10 = ten_path.format(sub, sub)
            temp11 = eleven_path.format(sub, sub)
            fmask = fm_path.format(sub, sub)
            swir1 = swir1_path.format(sub, sub)
            swir2 = swir2_path.format(sub, sub)

            # OPEN BANDS
            B04 = rx.open_rasterio(redband).squeeze("band", drop=True)
            B04.attrs["scale_factor"] = 0.0001

            B03 = rx.open_rasterio(greenband).squeeze("band", drop=True)
            B03.attrs["scale_factor"] = 0.0001

            B02 = rx.open_rasterio(blueband).squeeze("band", drop=True)
            B02.attrs["scale_factor"] = 0.0001

            B05 = rx.open_rasterio(nirband).squeeze("band", drop=True)
            B05.attrs["scale_factor"] = 0.0001

            tempImage10 = rx.open_rasterio(temp10).squeeze("band", drop=True)
            tempImage10.attrs["scale_factor"] = 0.01

            tempImage11 = rx.open_rasterio(temp11).squeeze("band", drop=True)
            tempImage11.attrs["scale_factor"] = 0.01

            B06 = rx.open_rasterio(swir1).squeeze("band", drop=True)
            B06.attrs["scale_factor"] = 0.0001

            B07 = rx.open_rasterio(swir2).squeeze("band", drop=True)
            B07.attrs["scale_factor"] = 0.0001

            fmask = rx.open_rasterio(fmask).squeeze("band", drop=True)

            cc = float(B04.attrs["cloud_coverage"])

            # FILTER BAND PROPERTIES FOR CLOUD COVER
            if cc <= percent:
                # CROP
                bbox = box(minx, miny, maxx, maxy)

                geo = gpd.GeoDataFrame(
                    {"geometry": bbox}, index=[0], crs=CRS.from_epsg(4326)
                )

                # UTM
                shapeUTM = geo.to_crs(
                    B04.spatial_ref.crs_wkt
                )  # Take the CRS from the band

                try:
                    B04_cropped = B04.rio.clip(
                        shapeUTM.geometry.values, shapeUTM.crs, all_touched=True
                    )
                    x = np.unique(
                        B04_cropped
                    )  # get number of unique values in cropped band

                except:
                    logging.error(
                        f"No overlap with bounding box found for {file}. Trying the next file"
                    )  # need to write to erraneous HLS files log

                else:
                    if (
                        len(x) == 1
                    ):  # only 1 value (-9999.0) if no data in the cropped image
                        logging.error(
                            f"No valid pixels in cropped file: {file}"
                        )  # need to write to erraneous HLS files log
                        continue
                    else:
                        B02_cropped = B02.rio.clip(
                            shapeUTM.geometry.values, shapeUTM.crs, all_touched=True
                        )
                        B03_cropped = B03.rio.clip(
                            shapeUTM.geometry.values, shapeUTM.crs, all_touched=True
                        )
                        B04_cropped = B04.rio.clip(
                            shapeUTM.geometry.values, shapeUTM.crs, all_touched=True
                        )
                        B05_cropped = B05.rio.clip(
                            shapeUTM.geometry.values, shapeUTM.crs, all_touched=True
                        )
                        B06_cropped = B06.rio.clip(
                            shapeUTM.geometry.values, shapeUTM.crs, all_touched=True
                        )
                        B07_cropped = B07.rio.clip(
                            shapeUTM.geometry.values, shapeUTM.crs, all_touched=True
                        )
                        B10_cropped = tempImage10.rio.clip(
                            shapeUTM.geometry.values, shapeUTM.crs, all_touched=True
                        )
                        B11_cropped = tempImage11.rio.clip(
                            shapeUTM.geometry.values, shapeUTM.crs, all_touched=True
                        )
                        fmask_cropped = fmask.rio.clip(
                            shapeUTM.geometry.values, shapeUTM.crs, all_touched=True
                        )

                        if hls_scale == True:
                            # SCALE
                            B02_cropped_scaled = hls_preprocess.scaling(B02_cropped)
                            B03_cropped_scaled = hls_preprocess.scaling(B03_cropped)
                            B04_cropped_scaled = hls_preprocess.scaling(B04_cropped)
                            B05_cropped_scaled = hls_preprocess.scaling(B05_cropped)
                            B06_cropped_scaled = hls_preprocess.scaling(B06_cropped)
                            B07_cropped_scaled = hls_preprocess.scaling(B07_cropped)
                            B10_cropped_scaled = hls_preprocess.scaling(B10_cropped)
                            B11_cropped_scaled = hls_preprocess.scaling(B11_cropped)

                        elif hls_scale == False:  # Keep unscaled, pass cropped bands
                            B02_cropped_scaled = B02_cropped
                            B03_cropped_scaled = B03_cropped
                            B04_cropped_scaled = B04_cropped
                            B05_cropped_scaled = B05_cropped
                            B06_cropped_scaled = B06_cropped
                            B07_cropped_scaled = B07_cropped
                            B10_cropped_scaled = B10_cropped
                            B11_cropped_scaled = B11_cropped

                        # CALC LST
                        lst = hls_preprocess.compute_lst(
                            B05_cropped_scaled,
                            B11_cropped_scaled,
                            B10_cropped_scaled,
                            B04_cropped_scaled,
                        )

                        
                        bands = [
                            B02_cropped_scaled,
                            B03_cropped_scaled,
                            B04_cropped_scaled,
                            B05_cropped_scaled,
                            B06_cropped_scaled,
                            B07_cropped_scaled,
                            fmask_cropped,
                            lst,
                        ]

                        # write out lst and band images
                        for band, band_name in zip(
                            bands,
                            [
                                "B02",
                                "B03",
                                "B04",
                                "B05",
                                "B06",
                                "B07",
                                "fmask",
                                "lst",
                            ],
                        ):

                            tile = sub.split(".")[2]
                            year = int(sub.split(".")[3][0:4])
                            doy = int(sub.split(".")[3][4:7])
                            time = "T" + str(sub.split(".")[3].split("T")[1])
                            date = (datetime(year, 1, 1) + timedelta(doy - 1)).strftime(
                                "%Y%m%d"
                            )

                            # Check folders or create
                            lst_path = os.path.join(hls_save_dir, "target-lst")
                            if os.path.exists(lst_path) == False:
                                os.mkdir(lst_path)

                            fmask_path = os.path.join(hls_save_dir, "fmask")
                            if os.path.exists(fmask_path) == False:
                                os.mkdir(fmask_path)

                        
                            hls_bands_path = os.path.join(hls_save_dir, "hls-bands")
                            if os.path.exists(hls_bands_path) == False:
                                os.mkdir(hls_bands_path)


                            if band_name == "lst":
                                data_path = lst_path

                            elif band_name == "fmask":
                                data_path = fmask_path

                            else:
                                data_path = hls_bands_path

                            # second check: if calculated lst/bands have only invalid pixels (-9999.0), don't write out bands
                            if np.nansum(band.data) == 0.0:
                                # print(
                                #    f"File: {tile}.{year}{doy}{time} was entirely invalid pixels - will not be exported."
                                # )
                                continue

                            band.rio.to_raster(
                                raster_path=data_path
                                + "/"
                                + f"{city}.{tile}.{date}.{time}.{band_name}.tif",
                                driver="COG",
                            )



class parallel_patching:
    """
    Set of methods to perform the stacking and patching workflow. This class includes the
    multiprocessing implementation for stacking and patching.

    ...

    Methods
    ----------
    process_target_band(target_tile_path)
        Extracts grid and CRS from target LST files
    process_hls_bands(hls_bands)
        Reads in HLS bands as arrays
    process_era5_bands(self, era5_ds, var_keys, grid, crs, date, time)
        Processes ERA5 dataset for ERA5 statistic layers
    process_aux_bands(self, grid, crs, date, time)
        Creates auxiliary feature bands which represent cos and sin functions of the day of year and time of day
    process_fmask(fmask_path)
        Reads fmask HLS image as an array
    stacking(processed_bands, grid, crs)
        Stacks HLS, ERA5 and aux feature arrays as xarray data array
    interpolator(multi_da, interp=None)
        Interpolates over NaNs on stacked inputs
    masking(multi_band_da, fmask_array, country_border: gpd.GeoDataFrame, crs, interp=None)
        Performs negative pixel and ocean masking on stacked data array
    filtering(patch, perc_allowed_nans)
        Filters patches based on specified allowed percentage of NaNs
    patching(self, bands_ds, x_size, y_size, perc_allowed_nans, patched_inputs_dir, patched_targets_dir, city_name, tile_id, date, time, crs) 
        Performs patching on stacked data arrat based on patch size specified
    stack2patch(self, target_tile_path, hls_band_paths, era5_ds, var_keys, fmask_path, info, lock)
        Processed individual inputs and targets to be stacked
    run_process(self)
        Runs complete workflow for stacking and patching 

    """

    def __init__(
        self,
        city_name,
        timeshift,
        sat_inputs,
        climate_inputs,
        targets,
        ocean_mask,
        fmasks,
        interp_method,
        patch_size_x,
        patch_size_y,
        perc_nans,
        patchedinputs_dir,
        patchedtargets_dir,
        n_threads,
        output_type
    ):

        self.city = city_name
        self.timeshift = timeshift
        self.sat_inputs = sat_inputs
        self.era5_inputs = climate_inputs
        self.targets = targets
        self.ocean_mask = ocean_mask
        self.fmasks = fmasks
        self.interp_method = interp_method
        self.x_size = patch_size_x
        self.y_size = patch_size_y
        self.perc_allowed_nans = perc_nans
        self.patched_inputs_dir = patchedinputs_dir
        self.patched_targets_dir = patchedtargets_dir
        self.set_threads = n_threads
        self.output = output_type

    @staticmethod
    def process_target_band(target_tile_path):

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

    @staticmethod
    def process_hls_bands(hls_bands):

        hls_bands_list = []
        for layer in hls_bands:
            with rs.open(layer) as band:
                array = band.read(1)
                hls_bands_list.append(array)

        return hls_bands_list
 
    @staticmethod   
    def process_era5_bands(self, era5_ds, var_keys, grid, crs, date, time):

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
        time_shift = self.timeshift
        time_shift_secs = time_shift * 3600
        
        new_time = datetime_obj + timedelta(seconds = time_shift_secs)
        # Shift era5 data to local time - already shifted in load
       
        sunrise_time = pd.to_datetime(date + "T060000")
        sunset_time = pd.to_datetime(date + "T180000")
        prev_sunset_time = pd.to_datetime(date + "T180000") - pd.DateOffset(days=1)
        foll_sunrise_time =  pd.to_datetime(date + "T060000") + pd.DateOffset(days=1)
        delta_times = [
            (new_time - pd.DateOffset(hours=2)),
            (new_time + pd.DateOffset(hours=2)),
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
                aquisition_time = var_da.sel(time=new_time, method="pad").rename(new_name_or_name_dict="t2m_aquisition")
                
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

            else:
                print('ERA5 2m_temperature data not implemented')

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

    @staticmethod 
    def process_aux_bands(self, grid, crs, date, time):
        # Convert time to local time
        datetime_string = date + time
        datetime_obj = pd.to_datetime(datetime_string)
        time_shift = self.timeshift
        time_shift_secs = time_shift * 3600
        new_time = datetime_obj + timedelta(seconds = time_shift_secs) #local time

        #### COS, SIN time features
        # arrays of 24 hours, 366 days
        time_of_d = np.linspace(0, 2*math.pi, 24)
        day_of_y  = np.linspace(0, 2*math.pi, 366)  

        # doy for acquisition
        doy = [int(date[:4]), int(date[4:6]), int(date[6:])]
        day_of_year = int(format(datetime(doy[0],doy[1],doy[2]), '%j'))

        if day_of_year >= 366:
            day_of_year = 0

        # toy for acquisition
        time_of_day = str(new_time.time())
        hour = int(time_of_day[0:2])
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

    @staticmethod
    def process_fmask(fmask_path):

        with rs.open(fmask_path) as fmask:
            meta = fmask.meta
            fmask_array = fmask.read(1)

        return fmask_array

    @staticmethod
    def stacking(processed_bands, grid, crs):
       
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


    @staticmethod
    def interpolator(multi_da, interp=None):

        # Check if interp selected
        if interp != False:

            # Check if gap argument given
            isgap = interp.split("_")[-1]
            if isgap != "gap" and isgap != "mean":

                # Use interpolation schemes to fill small nan patches
                interp_method = interp.split("_")[0]
                multi_da_filled = multi_da.interpolate_na(dim="x", method=interp_method)
                multi_da_filled = multi_da_filled.interpolate_na(
                    dim="y", method=interp_method, use_coordinate=False
                )

            elif isgap == "gap":

                # Use interpolation schemes to fill small nan patches
                interp_method = interp.split("_")[0]
                multi_da_filled = multi_da.interpolate_na(
                    dim="x", method=interp_method, max_gap=20
                )
                multi_da_filled = multi_da_filled.interpolate_na(
                    dim="y",
                    method=interp_method,
                    use_coordinate=False,
                    max_gap=20,
                )

            elif interp == "mean":

                # Use tile average to fill nan patches
                bands_avg_filled = []
                for band in multi_da:
                    band_avg = band.mean().values
                    band_filled = band.fillna(band_avg)
                    bands_avg_filled.append(band_filled)
                multi_da_filled = xr.concat(bands_avg_filled, dim="band")

        else:

            # Copy nan stacked data for no interp
            multi_da_filled = multi_da.copy()

        return multi_da_filled

    @staticmethod
    def masking(
        multi_band_da, fmask_array, country_border: gpd.GeoDataFrame, crs, interp=None
    ):

        # Mask out negative reflectance values in HLS bands with np.nan
        hls_bands = multi_band_da.sel(band=slice(1, 6))
        hls_bands = hls_bands.where(hls_bands >= 0)
        
        # Mask out LST values below absolute zero with np.nan
        target_band = multi_band_da.sel(band=15, method="nearest")
        target_band_neg_da = target_band.where(target_band >= -237.15)
        
        # Extract era5 bands
        era5_bands = multi_band_da.sel(band=slice(7, 10))

        # Extract aux bands
        aux_bands = multi_band_da.sel(band=slice(11, 14))

        # Concat
        multi_band_da_neg = xr.concat(
            [hls_bands, era5_bands, aux_bands, target_band_neg_da], dim="band"
        )
       
        # Create union mask from hls and lst bands
        data_isnan = np.isnan(multi_band_da_neg.data)
        data_mask = functools.reduce(lambda x, y: x | y, data_isnan)
        # all_mask = hls_mask | np.isnan(target_band_neg_da)
        all_mask = 1 - data_mask

        # Mask out all bands according to union mask
        # multi_band_da_neg = xr.concat(
        #    [hls_bands, era5_bands, target_band_neg_da], dim="band"
        # )
        multi_band_da_masked = multi_band_da_neg.where(all_mask, np.nan)

        # interpolate missing data in inputs only
        filled_inputs = parallel_patching.interpolator(
            multi_band_da_masked.sel(band=slice(1, 14)), interp=interp
        )
        filled_data = xr.concat(
            [filled_inputs, multi_band_da_masked.isel(band=-1)], dim="band"
        )

        # Mask out ocean bodies
        ocean_masked = filled_data.rio.clip(
            country_border.geometry.values, crs=crs, all_touched=True, drop=False
        )

        # # Set missing data to required no_data value (-9999->targets, np.nan->inputs)
        # ocean_masked[-1] = ocean_masked[-1].fillna(-9999)

        return ocean_masked

    @staticmethod
    def filtering(patch, perc_allowed_nans):

        # Check for clouds in patch
        bands_clouds = []
        for band in patch[:-1]:
            is_cloud = (band == -9999).any()
            bands_clouds.append(is_cloud.data)

        any_clouds = any(bands_clouds)

        # Check patch bands for nans
        bands_perc_nan = []
        for band in patch[:-1]:
            perc_nan = (
                np.count_nonzero(np.isnan(band)) / (band.shape[0] * band.shape[1])
            ) * 100
            bands_perc_nan.append(perc_nan)

        greater_than = [perc > perc_allowed_nans for perc in bands_perc_nan]
        any_nan = any(greater_than)

        return any_nan, any_clouds

    @staticmethod
    def patching(
        self,
        bands_ds,
        x_size,
        y_size,
        perc_allowed_nans,
        patched_inputs_dir,
        patched_targets_dir,
        city_name,
        tile_id,
        date,
        time,
        crs,
    ):

        # generate image segments
        ty, tx = x_size, y_size
        num_image_cols = bands_ds.x.size // tx
        num_image_rows = bands_ds.y.size // ty
        tot_tiles = 0

        
        for i in range(num_image_rows):
            for j in range(num_image_cols):

                bands_patch = bands_ds.isel(
                    {"x": slice(j * tx, (j + 1) * tx), "y": slice(i * ty, (i + 1) * ty)}
                )
                #print("bands_patch size", bands_patch.shape)
                # Exlude patch based on clouds and nan thresholds
                filter_nan, filter_clouds = parallel_patching.filtering(
                    patch=bands_patch, perc_allowed_nans=perc_allowed_nans
                )

                if filter_clouds == True:
                    continue
                elif filter_nan == True:
                    continue
                else:
                    # Split bands into input and output

                     # Convert date-time to local

                    datetime_string = date + time
                    datetime_obj = pd.to_datetime(datetime_string)
                    time_shift = self.timeshift
                    time_shift_secs = time_shift * 3600
                    new_time = datetime_obj + timedelta(seconds = time_shift_secs)
                    time_updated = "T" + time.replace(time, (str(new_time.time()).replace(":", "")))


                    input_bands_fn = os.path.join(
                        patched_inputs_dir,
                        f"{city_name}.{tile_id}.index_{num_image_rows*i+j}"
                        + f"_{x_size}.{date}.{time_updated}.inputs.tif",
                    )
                    input_bands = bands_patch.sel(band=slice(1, 14))
                    #print("input_bands size", input_bands.shape)
                    input_bands.rio.write_crs(crs, inplace=True)
                    input_bands.rio.write_nodata(np.nan, inplace=True)
                    input_bands.rio.to_raster(input_bands_fn, dtype="float32", driver = "COG")

                    # write bands_tile to geotiff
                    output_bands_fn = os.path.join(
                        patched_targets_dir,
                        f"{city_name}.{tile_id}.index_{num_image_rows*i+j}"
                        + f"_{x_size}.{date}.{time_updated}.lst.tif",
                    )
                    output_bands = bands_patch.sel(band=15, method="nearest")
                    output_bands.rio.write_crs(crs, inplace=True)
                    # output_bands.rio.write_nodata(-9999.0, inplace=True)
                    output_bands.rio.write_nodata(-9999.0, inplace=True)
                    output_bands.rio.to_raster(output_bands_fn, dtype="float32", driver = "COG")

                    tot_tiles += 1  # This number is not accurately captured includes patches with no data

        return tot_tiles
    
    @staticmethod
    def stack2patch(
        self,
        target_tile_path,
        hls_band_paths,
        era5_ds,
        var_keys,
        fmask_path,
        info,
        lock,
    ):

        current_process = multiprocessing.Process().name
        task_name = target_tile_path.split("/")[-1]

        try:

            # Load and process target data
            target_band, grid, meta, crs = parallel_patching.process_target_band(
                target_tile_path
            )
            
            # Load and process hls bands
            hls_bands = parallel_patching.process_hls_bands(hls_band_paths)
           
            # Process era5 data
            era5_bands = parallel_patching.process_era5_bands(self,
                era5_ds, var_keys, grid, crs, date=info[2], time=info[3]
            )

            aux_bands = parallel_patching.process_aux_bands(self, grid, crs, date=info[2], time=info[3])

            # Stack bands into multidim xarray dataarray
            n_bands = 15

            processed_bands = hls_bands + era5_bands + aux_bands + target_band
           
            assert (
                len(processed_bands) == n_bands
            ), f"{current_process}: {task_name} --> ERROR: number of bands for stacking. Please check."
            stacked_tile_da = parallel_patching.stacking(
                processed_bands=processed_bands, grid=grid, crs=crs
            )
            
            # Load fmask
            fmask = parallel_patching.process_fmask(fmask_path)

            # Load ocean mask
            ocean_mask_polygon = self.ocean_mask.to_crs(crs)

            masked_tile_da = parallel_patching.masking(
                stacked_tile_da,
                fmask_array=fmask.copy(),
                country_border=ocean_mask_polygon,
                crs=crs,
                interp=self.interp_method,
            )

            # FLAG for stacked-tiles or patches
            if self.output == "stacked-tiles":
                # generate image segments
                ty, tx = masked_tile_da.y.size , masked_tile_da.x.size
                num_image_cols = masked_tile_da.x.size // tx
                num_image_rows = masked_tile_da.y.size // ty
                tot_tiles = 0

                
                for i in range(num_image_rows):
                    for j in range(num_image_cols):

                        bands_patch = masked_tile_da.isel(
                            {"x": slice(j * tx, (j + 1) * tx), "y": slice(i * ty, (i + 1) * ty)}
                        )
                        
                        # Exlude patch based on clouds and nan thresholds
                        filter_nan, filter_clouds = parallel_patching.filtering(
                            patch=bands_patch, perc_allowed_nans=self.perc_allowed_nans
                        )
                        if filter_clouds == True:
                            continue
                        elif filter_nan == True:
                            continue
                        else:

                            datetime_string = info[2] + info[3]
                            datetime_obj = pd.to_datetime(datetime_string)
                            time_shift = self.timeshift
                            time_shift_secs = time_shift * 3600
                            new_time = datetime_obj + timedelta(seconds = time_shift_secs)
                            time_updated = "T" + info[3].replace(info[3], (str(new_time.time()).replace(":", "")))
                            
                            ### WRITE OUT STACKED TILE
                            stacked_tile_dir = os.path.join(os.path.dirname(self.patched_inputs_dir), "stacked-tiles")
                            
                            if os.path.exists(stacked_tile_dir) == False:
                                os.mkdir(stacked_tile_dir)


                            masked_tile_da_fn = os.path.join(
                                    stacked_tile_dir,
                                    f"{info[0]}.{info[1]}"
                                    + f".{info[2]}.{time_updated}.inputs.tif",
                                )
                            stacked_input_bands = masked_tile_da.sel(band=slice(1, 14))
                            stacked_input_bands.rio.write_crs(crs, inplace=True)
                            stacked_input_bands.rio.write_nodata(np.nan, inplace=True)
                            stacked_input_bands.rio.to_raster(masked_tile_da_fn, dtype="float32", driver="COG")

                            # write lst_tile to geotiff
                            lst_tile_dir = os.path.join(os.path.dirname(self.patched_targets_dir), "processed-lst")
                            if os.path.exists(lst_tile_dir) == False:
                                os.mkdir(lst_tile_dir)
                            
                            stacked_output_bands_fn = os.path.join(
                                lst_tile_dir, 
                                f"{info[0]}.{info[1]}"
                                + f".{info[2]}.{time_updated}.lst.tif",
                            )
                            lst_output_bands = masked_tile_da.sel(band=15, method="nearest")
                            lst_output_bands.rio.write_crs(crs, inplace=True)
                            lst_output_bands.rio.write_nodata(np.nan, inplace=True)
                            lst_output_bands.rio.to_raster(stacked_output_bands_fn, dtype="float32", driver="COG")

            elif self.output == "stacked-patches":
                # Running splitting of masked data
                tot_patches = parallel_patching.patching(self,
                    masked_tile_da,
                    x_size=self.x_size,
                    y_size=self.y_size,
                    perc_allowed_nans=self.perc_allowed_nans,
                    patched_inputs_dir=self.patched_inputs_dir,
                    patched_targets_dir=self.patched_targets_dir,
                    city_name=info[0],
                    tile_id=info[1],
                    date=info[2],
                    time=info[3],
                    crs=crs,
                )

                assert (
                    tot_patches != 0
                ), f"{current_process}: {task_name} --> WARNING: No patches meet cloud cover and nan threholds."

        except Exception as e:
            with lock:
                logging.error(
                        f'{current_process}: {task_name} --> ERROR: {e}, in {traceback.format_exc().split(",")[1]}'
                    )

            return "{}: Failed with error --> {} in {}".format(
                task_name, e, traceback.format_exc().split(",")[1]
                )
    

    #@classmethod
    def run_process(self):

        # Set multiprocessing manager
        manager = Manager()
        lock = manager.Lock()

        try:
            # Load ERA5 array for city
            era5_files = self.era5_inputs
            
            timeshift = self.timeshift
            era5_ds, keys = utils.load_era5(era5_files, timeshift)
           
            assert era5_ds != None, "Please check era5 files."

            # Loop through target tiles and create list of args
            mp_args = []
            logging.info(
                f"Creating tasks for {len(self.targets)} target tiles for: {self.city}"
            )
            for target_tile_path in self.targets:

                # Extracting naming info
                tile_id, date, time = utils.extract_info(target_tile_path)
                
                info = [self.city, tile_id, date, time]

                # HLS inputs for tile
                hls_band_paths = utils.filter_images(
                    city_name=self.city,
                    tile_id=tile_id,
                    date=date,
                    time=time,
                    lst=self.sat_inputs,
                )

                # Fmask for tile
                fmask_path = utils.filter_images(
                    city_name=self.city,
                    tile_id=tile_id,
                    date=date,
                    time=time,
                    lst=self.fmasks,
                )


                mp_args.append(
                    (
                        self,
                        target_tile_path,
                        hls_band_paths,
                        era5_ds,
                        keys,
                        fmask_path[0],
                        info,
                        lock,
                    )
                )
                
            # Run assemble function on parallel threads
            logging.info(
                f"Executing multiprocessing for {len(mp_args)} tasks for: {self.city}"
            )
            with Pool(processes=self.set_threads) as pool:
                results = pool.starmap(parallel_patching.stack2patch, mp_args)
                pool.close()
                pool.join()

            return results

        except Exception as e:
            logging.error(f'ERROR: {e}, in {traceback.format_exc().split(",")[1]}')
            return [
                "{}: Failed with error --> {} in {}".format(
                    self.city, e, traceback.format_exc().split(",")[1]
                )
            ]


class checks:
    """
    Set of methods to perform checks on the output stacked tile/patch inputs and targets. 
    Verifies that inputs have corresponding targets and inputs have the correct number 
    of bands.

    ...

    Methods
    ----------
    check_patch(inputs_filepath, target_dir, lock)
        Checks number of bands in every patch
    check_match(inputs_list, targets_list)
        Checks that inputs have corresponding targets
    check_tiles(inputs_filepath_tiles, target_dir_tiles, lock)
        Checks number of bands in stacked tiles
    check_match_tile(inputs_list_tile, targets_list_tile)
        Checks stacked tile inputs have corresponding tile targets
    check_bands_exist(subdir)
        Ensures every HLS acuisition has all necessary bands
    run_checks(self)
        Runs checks for all patches using multiprocessing
    run_checks_tiles(self)
        Runs checks for all tiles using multiprocessing
    """

    def __init__(self, patchedinputs_dir, patchedtargets_dir, n_threads):

        self.patched_inputs_dir = patchedinputs_dir
        self.patched_targets_dir = patchedtargets_dir
        self.set_threads = n_threads

    @staticmethod
    def check_patch(inputs_filepath, target_dir, lock):

        missing_bands_errors = []
        dtype_errors = []
        stacked_patch = xr.open_dataset(inputs_filepath)
        if stacked_patch.band.count() < 14:  # Change based on V3 bands in stack
            missing_bands_errors.append(inputs_filepath)
        if stacked_patch.dtypes["band_data"] != "float32":
            dtype_errors.append(inputs_filepath)

        # Combine lists and delete
        combined_list = list(set(missing_bands_errors + dtype_errors))
        if len(combined_list) != 0:
            for inputs_filepath in combined_list:
                os.remove(
                    inputs_filepath
                )  # Must delete targets also for inputs that have less bands
            for inputs_filepath in combined_list:
                filename = (
                    ".".join(inputs_filepath.split("/")[-1].split(".")[:-2])
                    + ".lst.tif"
                )
                target_file = os.path.join(target_dir, filename)
                os.remove(target_file)

        return missing_bands_errors, dtype_errors
    
    @staticmethod
    def check_match(inputs_list, targets_list):

        inputs_basename = [
            ".".join(os.path.basename(patch).split(".")[0:5]) for patch in inputs_list
        ]
        targets_basename = [
            ".".join(os.path.basename(patch).split(".")[0:5]) for patch in targets_list
        ]

        # Find images in target that are not in inputs
        targets_not = sorted(set(targets_basename).difference(inputs_basename))
        if len(targets_not) == 0:
            targets_not_paths = []
        else:
            targets_not_paths = []
            for file in targets_not:
                fullpath_target_not = [
                    filepath for filepath in targets_list if file in filepath
                ][0]
                targets_not_paths.append(fullpath_target_not)
                os.remove(fullpath_target_not)

        # find images in inputs that are not in target
        inputs_not = sorted(set(inputs_basename).difference(targets_basename))
        if len(inputs_not) == 0:
            inputs_not_paths = []
        else:
            inputs_not_paths = []
            for file in inputs_not:
                fullpath_inputs_not = [
                    filepath for filepath in inputs_list if file in filepath
                ][0]
                inputs_not_paths.append(fullpath_inputs_not)
                os.remove(fullpath_inputs_not)

        return targets_not_paths, inputs_not_paths
    
    #STACKED TILES
    @staticmethod
    def check_tiles(inputs_filepath_tiles, target_dir_tiles, lock):

        missing_bands_errors_tiles = []
        dtype_errors_tiles = []
        stacked_tile = xr.open_dataset(inputs_filepath_tiles)
        if stacked_tile.band.count() < 14:  # Change based on V3 bands in stack
            missing_bands_errors_tiles.append(inputs_filepath_tiles)
        if stacked_tile.dtypes["band_data"] != "float32":
            dtype_errors_tiles.append(inputs_filepath_tiles)

        # Combine lists and delete
        combined_list_tiles = list(set(missing_bands_errors_tiles + dtype_errors_tiles))
        if len(combined_list_tiles) != 0:
            for inputs_filepath_tiles in combined_list_tiles:
                os.remove(
                    inputs_filepath_tiles
                )  # Must delete targets also for inputs that have less bands
            for inputs_filepath_tiles in combined_list_tiles:
                filename = (
                    ".".join(inputs_filepath_tiles.split("/")[-1].split(".")[:-2])
                    + ".lst.tif"
                )
                target_file = os.path.join(target_dir_tiles, filename)
                os.remove(target_file)

        return missing_bands_errors_tiles, dtype_errors_tiles

    @staticmethod
    def check_match_tile(inputs_list_tile, targets_list_tile):

        inputs_basename_tiles = [
            ".".join(os.path.basename(tile).split(".")[0:4]) for tile in inputs_list_tile
        ]
        targets_basename_tiles = [
            ".".join(os.path.basename(tile).split(".")[0:4]) for tile in targets_list_tile
        ]

        # Find images in target that are not in inputs
        targets_not_tiles = sorted(set(targets_basename_tiles).difference(inputs_basename_tiles))
        if len(targets_not_tiles) == 0:
            targets_not_paths_tiles = []
        else:
            targets_not_paths_tiles = []
            for file in targets_not_tiles:
                fullpath_target_not_tile = [
                    filepath for filepath in targets_list_tile if file in filepath
                ][0]
                targets_not_paths_tiles.append(fullpath_target_not_tile)
                os.remove(fullpath_target_not_tile)

        # find images in inputs that are not in target
        inputs_not_tiles = sorted(set(inputs_basename_tiles).difference(targets_basename_tiles))
        if len(inputs_not_tiles) == 0:
            inputs_not_paths_tiles = []
        else:
            inputs_not_paths_tiles = []
            for file in inputs_not_tiles:
                fullpath_inputs_not_tiles = [
                    filepath for filepath in inputs_list_tile if file in filepath
                ][0]
                inputs_not_paths_tiles.append(fullpath_inputs_not_tiles)
                os.remove(fullpath_inputs_not_tiles)

        return targets_not_paths_tiles, inputs_not_paths_tiles

    @staticmethod
    def check_bands_exist(subdir):
        l30_bands = [
            "HLS.L30.*.*.v2.0.B02.tif",
            "HLS.L30.*.*.v2.0.B03.tif",
            "HLS.L30.*.*.v2.0.B04.tif",
            "HLS.L30.*.*.v2.0.B05.tif",
            "HLS.L30.*.*.v2.0.B06.tif",
            "HLS.L30.*.*.v2.0.B07.tif",
            "HLS.L30.*.*.v2.0.B09.tif",
            "HLS.L30.*.*.v2.0.B10.tif",
            "HLS.L30.*.*.v2.0.B11.tif",
            "HLS.L30.*.*.v2.0.Fmask.tif",
        ]
        list_files = os.listdir(subdir)
        len_list_files = []

        for f in l30_bands:
            obj = [b for b in list_files if b.split(".")[-2] == f.split(".")[-2]]
            if len(obj) == 1:
                len_list_files.append(obj[0])

        if len(len_list_files) == 10:
            return True
        else:
            return False

    def run_checks(self):

        # initialize multiprocessing manager
        manager = Manager()
        lock = manager.Lock()

        # Running missing inputs/targets checks
        logging.info(f"Checking for missmatching files...")
        all_inputs = glob.glob(os.path.join(self.patched_inputs_dir, "*.tif"))
        all_targets = glob.glob(os.path.join(self.patched_targets_dir, "*.tif"))
        missing_targets, missing_inputs = checks.check_match(all_inputs, all_targets)

        # check missing bands and dtype errors
        logging.info(f"Checking for input file errors...")
        logging.info(f"Intializing and running multiprocessing...")
        all_inputs = glob.glob(os.path.join(self.patched_inputs_dir, "*.tif"))
        mp_args = []
        for input in all_inputs:

            mp_args.append((input, self.patched_targets_dir, lock))

        with Pool(processes=self.set_threads) as pool:
            band_checks_results = pool.starmap(checks.check_patch, mp_args)
            pool.close()
            pool.join()

        # Unpack band check errors
        dtype_errors = []
        missing_band_errors = []
        for result in band_checks_results:
            if len(result[0]) != 0:
                missing_band_errors.append(result[0][0])
            if len(result[1]) != 0:
                dtype_errors.append(result[1][0])

        return missing_inputs, missing_targets, missing_band_errors, dtype_errors
    
    def run_checks_tiles(self):

        # initialize multiprocessing manager
        manager = Manager()
        lock = manager.Lock()

        # Running missing inputs/targets checks
        logging.info(f"Checking for missmatching tiles...")
        all_inputs_tiles = glob.glob(os.path.join(self.patched_inputs_dir, "stacked_tiles", "*.tif"))
        all_targets_tiles = glob.glob(os.path.join(self.patched_targets_dir,"processed_lst", "*.tif"))
        missing_targets_tiles, missing_inputs_tiles = checks.check_match_tile(all_inputs_tiles, all_targets_tiles)

        # check missing bands and dtype errors
        logging.info(f"Checking for input tile errors...")
        logging.info(f"Intializing and running multiprocessing...")
        all_inputs_tiles = glob.glob(os.path.join(self.patched_inputs_dir, "stacked_tiles", "*.tif"))
        mp_args = []
        for input in all_inputs_tiles:

            mp_args.append((input, self.patched_targets_dir, lock))

        with Pool(processes=self.set_threads) as pool:
            band_checks_results_tiles = pool.starmap(checks.check_tiles, mp_args)
            pool.close()
            pool.join()

        # Unpack band check errors
        dtype_errors_tiles = []
        missing_band_errors_tiles = []
        for result in band_checks_results_tiles:
            if len(result[0]) != 0:
                missing_band_errors_tiles.append(result[0][0])
            if len(result[1]) != 0:
                dtype_errors_tiles.append(result[1][0])

        return missing_inputs_tiles, missing_targets_tiles, missing_band_errors_tiles, dtype_errors_tiles


class splitting:

    """
    Set of methods to perform train-test-val splitting of patches, which can be used 
    for fine-tuning purposes. 
    ...

    Methods
    ----------
    create_df(target_list, inputs_list)
        Checks dataframe of all input patches and corresponding target patches
    split(self, df)
        Splits the dataframe using scikit learn train_test_split()
    move_split(input_src, input_dst, target_src, target_dst, lock)
        Moves the patches to the corresponding folders based on the split
    split_move_check(self, dst_root)
        Checks that data moved into split folders have no missing patches
    check_bands_exist(subdir)
        Ensures every HLS acuisition has all necessary bands
    run_splitting(self)
        Runs splitting of patches using multiprocessing
    """

    def __init__(
        self,
        patchedinputs,
        patchedtargets,
        destination_root,
        n_threads=12,
    ) -> None:

        self.patched_inputs = patchedinputs
        self.patched_targets = patchedtargets
        self.destination_root = destination_root
        self.set_threads = n_threads
        self.holdout_cities = None
        self.holdout_datasets = None
        self.missing_train_x = None
        self.missing_train_y = None
        self.missing_val_x = None
        self.missing_val_y = None
        self.missing_test_x = None
        self.missing_test_y = None

    def create_df(target_list, inputs_list):

        # Create master df
        df = pd.DataFrame(target_list, columns=["targetpath"])
        df["target_filename"] = df["targetpath"].str.split("/").str[-1]
        df["name"] = df["target_filename"].str.split(".").str[:-2].apply(".".join)
        df["city"] = df["target_filename"].str.split(".").str[0]
        df["tile_id"] = df["target_filename"].str.split(".").str[1]
        df["date"] = df["target_filename"].str.split(".").str[3]
        df["year"] = df["date"].str[:4]
        df["month"] = df["date"].str[4:6]
        df["class_citymonth"] = df[["city", "month"]].agg(".".join, axis=1)
        df["class_citytileyearmonth"] = df[["city", "tile_id", "year", "month"]].agg(
            ".".join, axis=1
        )
        df["class_citytilemonth"] = df[["city", "tile_id", "month"]].agg(
            ".".join, axis=1
        )

        # create inputs df
        df_inputs = pd.DataFrame(inputs_list, columns=["inputpath"])
        df_inputs["inputs_filename"] = df_inputs["inputpath"].str.split("/").str[-1]
        df_inputs["name"] = (
            df_inputs["inputs_filename"].str.split(".").str[:-2].apply(".".join)
        )

        # join tables based on name
        full_df = df_inputs.merge(df, left_on="name", right_on="name")

        return full_df   

    def split(self, df):

        # Split train and test
        train, test = train_test_split(
            df, test_size=0.1, random_state=1
        )
        # Split train and validation
        train, val = train_test_split(
            train, test_size=0.2, random_state=1
        )

        # Add hold out data to test set
        # test_sets = [test] + self.holdout_datasets
        test = pd.concat([test, self.holdout_datasets])

        # Order datasets
        train = train.sort_values(by=["name"]).reset_index().drop("index", axis=1)
        val = val.sort_values(by=["name"]).reset_index().drop("index", axis=1)
        test = test.sort_values(by=["name"]).reset_index().drop("index", axis=1)

        return train, val, test

    def move_split(input_src, input_dst, target_src, target_dst, lock):

        # Move target
        shutil.copyfile(target_src, target_dst)
        # Move inputs
        shutil.copyfile(input_src, input_dst)

        return input_dst, target_dst

    def split_move_check(self, dst_root):

        # Check train
        search_train_x = os.path.join(dst_root, "train", "inputs", "*.tif")
        moved_train_x = glob.glob(search_train_x)
        search_train_y = os.path.join(dst_root, "train", "targets", "*.tif")
        moved_train_y = glob.glob(search_train_y)
        missing_train_x, missing_train_y = checks.check_match(
            moved_train_x, moved_train_y
        )

        self.missing_train_x = missing_train_x
        self.missing_train_y = missing_train_y

        # Check val
        search_val_x = os.path.join(dst_root, "val", "inputs", "*.tif")
        moved_val_x = glob.glob(search_val_x)
        search_val_y = os.path.join(dst_root, "val", "targets", "*.tif")
        moved_val_y = glob.glob(search_val_y)
        missing_val_x, missing_val_y = checks.check_match(moved_val_x, moved_val_y)

        self.missing_val_x = missing_val_x
        self.missing_val_y = missing_val_y

        # Check test
        search_test_x = os.path.join(dst_root, "test", "inputs", "*.tif")
        moved_test_x = glob.glob(search_test_x)
        search_test_y = os.path.join(dst_root, "test", "targets", "*.tif")
        moved_test_y = glob.glob(search_test_y)
        missing_test_x, missing_test_y = checks.check_match(moved_test_x, moved_test_y)

        self.missing_test_x = missing_test_x
        self.missing_test_y = missing_test_y

        # check folder final
        moved_train_y = glob.glob(search_train_y)
        moved_val_y = glob.glob(search_val_y)
        moved_test_y = glob.glob(search_test_y)

        return moved_train_y, moved_val_y, moved_test_y

    def run_splitting(self):

        # Initialize multiprocessing manager
        manager = Manager()
        lock = manager.Lock()

        # Create dataframe of inputs and outputs
        logging.info(
            f"Creating dataframe of inputs and corresponding targets...."
        )  # Add error handling here to continue or cut the pipeline. If no data in folders this will error out.
        all_examples_df = splitting.create_df(
            target_list=self.patched_targets, inputs_list=self.patched_inputs
        )

        # Split sample into train, validationa and test
        logging.info(f"Splitting data into train, validation, and test sets...")
        train, val, test = splitting.split(self, df=all_examples_df)

        # Finally datasets individually
        logging.info(f"Initializing multiprocessing for moving split data ...")
        keys = ["train", "val", "test"]
        move_mpargs = []
        for key, dataset in enumerate([train, val, test]):
            dst_path = os.path.join(self.destination_root, keys[key])
            logging.info(f"{keys[key]} destination folder set to: {dst_path}")
            for label, row in dataset.iterrows():

                # Move target
                target_dst_path = os.path.join(
                    dst_path, "targets", row["target_filename"]
                )
                target_src_path = row["targetpath"]

                # Move inputs
                inputs_dst_path = os.path.join(
                    dst_path, "inputs", row["inputs_filename"]
                )
                input_src_path = row["inputpath"]

                move_mpargs.append(
                    (
                        input_src_path,
                        inputs_dst_path,
                        target_src_path,
                        target_dst_path,
                        lock,
                    )
                )

        logging.info(f"Moving data...")
        with Pool(processes=self.set_threads) as pool:
            results = pool.starmap(splitting.move_split, move_mpargs)
            pool.close()
            pool.join()

        # Check move
        logging.info(f"Checking errors and returning final datasets...")
        train, val, test = splitting.split_move_check(
            self,
            dst_root=self.destination_root,
        )

        return train, val, test


class calc_stats:
    """
    Method to calculate means and standard deviations of patches after splitting. 
    Writes these out as a json file in the train-test-val folder. 
    This is a required input for the config file when performing fine-tuning.
    ...

    Methods
    ----------
    cal_mean_std(self)
        Calculates the mean and standard deviation for patches
       
    """

    def __init__(self, inputs, destination_root) -> None:

        self.train_inputs = inputs
        self.destination_root = destination_root

    def cal_mean_std(self):

        # Calculate population mean
        logging.info("Calculating means...")
        pop_mean = 0.0
        mean = []
        for tfp in self.train_inputs:
            ds = xr.open_dataset(tfp)
            date = os.path.basename(tfp).split(".")[3]
            ds = ds.assign_coords(time=date, inplace=True)
            pop_mean += ds.mean(dim=["x", "y"], skipna=True)
            mean_ = ds.mean(dim=["x", "y"], skipna=True)
            mean.append(mean_)
        pop_mean /= len(self.train_inputs)

        # Calculated stds from pop mean
        logging.info("Calculating standard deviations...")
        std_ = 0.0

        for m in mean:
            std_ += (m - pop_mean) ** 2
        std_ /= len(self.train_inputs)

        stds = std_["band_data"].to_numpy()
        std = np.sqrt(stds)

        # write means and stds to JSON
        save_file = os.path.join(self.destination_root, "img_norm_cfg.json")
        logging.info(
            f"Writing mean and std for each band to the following file: {save_file}"
        )
        img_norm_cfg = {
            "means": pop_mean["band_data"].values.tolist(),
            "stds": std.tolist(),
        }

        jsonString = json.dumps(img_norm_cfg)
        jsonFile = open(save_file, "w")
        jsonFile.write(jsonString)
        jsonFile.close()

        return img_norm_cfg
