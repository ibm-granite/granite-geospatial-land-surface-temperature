"""
This python module contains a set helper functions with helper functions to execute
the main download script for downloading HLS L30 and ERA5 am_temperature datasets 
for given cities of interest. 
"""
########################################################################
# Import libraries
########################################################################

from rasterio.enums import Resampling
from shapely.geometry import box
from shapely.ops import transform
from shapely.geometry import mapping
import geopandas as gpd
from datetime import datetime
import itertools
from collections import defaultdict
import pandas as pd
import numpy as np
import pyproj
import pystac
import pystac_client
import json
import os
import requests
import shutil
import argparse
import cdsapi
import zipfile
import calendar
import multiprocessing
from multiprocessing import Pool, Manager

import logging
import warnings
import traceback
import sys
from datetime import date, datetime, timedelta
from datetime import timedelta

now = datetime.now()
dt_string = now.strftime("%Y%m%d%H%M%S")

logging.basicConfig(
    # stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

########################################################################
# Utils
########################################################################
class utils:
    """
    Set of methods to run main downloads script.

    ...

    Methods
    -------
    get_city_bbox(city_name, buffer, csv_path)
        Get bounding boxes for cities from pre-loaded cities database

    download_hls(list_cities, bbox_buffer, csv_path, percent, HLS_COLLECTION_IDS,  date_range, hls_save_dir)
        Query NASA STAC for HLS L30 acquisitions which meet cloud cover percentage and period of interest

    load_era5(era5_files, timeshift)
        Load ERA5 hourly 2m_temperature downloaded datasets and adjust for local time zone

    download_url(urls, save_paths, lock)
        Download list of URLs returned from NASA STAC query

    download_era5(era5_download_path, list_cities, bbox_buffer, csv_path, era5_years, var)
        Create arguments for multiprocessing - list of cities with bounding boxes and periods of interest 
    
    cds_download(city, city_lats, city_lons, era5_download_path, var, year, lock)
        Downloads list of arguments from above

    """
    @staticmethod
    def get_city_bbox(city_name, buffer, csv_path):
    
        # open csv
        cities_df = pd.read_csv(csv_path)

        #city_bbox = []

        city = city_name.split('_')[0]
        city_iso = city_name.split('_')[1]
        
      
        filtered_df = cities_df[(cities_df["CITY_NAME"] == city) & (cities_df['COUNTRY_ISO']==city_iso)]

        if len(filtered_df) == 1:

            city_bbox = {'name':city_name,
                        'maxx': filtered_df['BBX_XMAX'].values[0],
                        'maxy': filtered_df['BBX_YMAX'].values[0],
                        'minx': filtered_df['BBX_XMIN'].values[0],
                        'miny': filtered_df['BBX_YMIN'].values[0]
                        }
        
            # Convert bbox to polygon
            bbox_geom = box(
                filtered_df['BBX_XMIN'].values[0], 
                filtered_df['BBX_YMIN'].values[0], 
                filtered_df['BBX_XMAX'].values[0], 
                filtered_df['BBX_YMAX'].values[0]
                )
    
            # Add buffer
            bbox_geom_buffered = bbox_geom.buffer(buffer, join_style=2)
            # Extract total bounds with buffer
            city_bbox_buffered = {
                'name':city_name,
                'maxx': bbox_geom_buffered.bounds[2],
                'maxy': bbox_geom_buffered.bounds[3],
                'minx': bbox_geom_buffered.bounds[0],
                'miny': bbox_geom_buffered.bounds[1]
                }

        else:
            logging.warning(f'City name: {city_name} is not found in the global list. Please check csv file for correct city name and ISO, edit in payload.')

        return city_bbox_buffered


    @staticmethod
    def download_hls(list_cities, bbox_buffer, csv_path, percent, HLS_COLLECTION_IDS,  date_range, hls_save_dir):
       
        num_threads = 10
        manager = Manager()
        lock = manager.Lock()
        
        ####### QUERY DETAILS

        CMR_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"

        catalog = pystac_client.Client.open(CMR_STAC_URL)
        l30_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B09', 'B10', 'B11', 'Fmask']


        ####    
        for city_iso in list_cities:
            try:
                utils.get_city_bbox(city_iso, bbox_buffer, csv_path)
            except:
                logging.warning(f'Error: Invalid city-name_ISO for {city_iso}. Please edit the config file.')

            city_bbox_buff = utils.get_city_bbox(city_iso, bbox_buffer, csv_path)
            miny = city_bbox_buff['miny']
            maxy = city_bbox_buff['maxy']
            minx = city_bbox_buff['minx']
            maxx = city_bbox_buff['maxx']
            city = city_bbox_buff['name']

            all_files = [] #list for the city

            # Get city bbox for query roi
            CRS_STRING = "epsg:4326"
            EPSG = pyproj.CRS.from_string(CRS_STRING).to_epsg()
            cc = float(percent)

            # bounding box that surrounds the Little Triangle
            AOI = box(float(minx), float(miny), float(maxx), float(maxy)) # (x_min, y_min, x_max, y_max)

            # STAC items store bounding box info in epsg:4326
            transformer_4326 = pyproj.Transformer.from_crs(
                crs_from=CRS_STRING,
                crs_to="epsg:4326",
                always_xy=True)

            roi = transform(transformer_4326.transform, AOI).bounds
            

            #### QUERY STAC
            hls_history_search = catalog.search(
                collections = HLS_COLLECTION_IDS,
                bbox = roi, 
                datetime = date_range)

            all_items = [] # all query results

            for page in hls_history_search.pages():
                all_items.extend(page.items)

            collection_history = {collection: defaultdict(list) for collection in HLS_COLLECTION_IDS}
            
            save_path = os.path.join(hls_save_dir, city)
            # create city folder for downloads
            if os.path.exists(save_path):
                logging.info('HLS city folder exists, saving files.')
            else:
                os.mkdir(save_path)
                logging.info('Created HLS folder for city, saving files.')
                
            logging.info(f"Saving HLS files to {save_path}")

            for a in all_items:
                # filter on cloud cover
                if a.properties['eo:cloud_cover'] <= cc:
                    all_bands = l30_bands
                    for x in a.assets:
                        if any(b==x for b in all_bands):
                            all_files.append( (a.assets[x].href, os.path.join(save_path, a.assets[x].href.split('/')[5]), lock) )
            
            #Check there is data from query
            if len(all_files) == 0:
                logging.info(f'No HLS files found for {city} over {date_range} with {cc} % cloud cover. Exiting.')
                print(f'No HLS files found for {city} over {date_range} with {cc} % cloud cover. Exiting.')
                break
            else:
                logging.info(f'Found {len(all_files)} HLS band images for {city} with {cc} % cloud cover, preparing to download.')
                print(f'Found {len(all_files)} HLS band images for {city} with {cc} % cloud cover, preparing to download.')

            # create subdirectories for acquisitions
            unique_names = []
            for i in all_files:
                y = (i[0].split('/')[5])
                unique_names.append(y)

            dir_names = np.unique(unique_names)

            for f in dir_names:
                if os.path.exists(os.path.join(save_path, f)):
                    #print('Subdir exists, checking next subdir.')
                    continue
                else:
                    os.mkdir(os.path.join(save_path, f))

            #### DOWNLOAD  WITH MULTIPROCESSING #####
            logging.info(f"Executing multiprocessing for {len(all_files)} HLS tasks for: {city}")
             
            with Pool(processes=num_threads) as pool: 
                results = pool.starmap(utils.download_url, all_files)
                pool.close()
                pool.join()
            logging.info(f"HLS downloads complete for {city}!")
            
        #return list of files to be downloaded

            ##### DATA CHECK #####
            logging.info(f"Checking for failed HLS downloads...")
            missing_files = []
            for x in all_files:
                single_file = x[0].split('/')[6]

                #Check
                if not os.path.isfile(os.path.join(x[1], single_file)):
                    missing_files.append(single_file)
            
            if len(missing_files) > 0:
                
                logging.info(f"A total of {len(missing_files)} files were unable to download.")
                with open("logs/failed_downloads-{dt_string}.log", "w") as failed_downloads:
                    failed_downloads.write("\nThe following HLS files could not be downloaded:")
                    for line in missing_files:
                        failed_downloads.write("%s\n" % line)
                                       

    @staticmethod
    def download_url(urls, save_paths, lock):

        if os.path.isfile(os.path.join(save_paths, urls.split('/')[6])):
            logging.info(f"File {os.path.join(save_paths, urls.split('/')[6])} exists. Skipping.")

        else:
            try:         
                os.system(f"wget --tries=30 -P {save_paths} {urls}")
                #logging.info(f"successfully downloaded {urls.split('/')[6]}.")
            except:
                logging.error(f"Error: file {urls} could not be downloaded.")
                

    @staticmethod
    def download_era5(era5_download_path, list_cities, bbox_buffer, csv_path, era5_years, var):

        num_threads = 10
        manager = Manager()
        lock = manager.Lock()

        # get bbox for city
        for city_iso in list_cities:
            city_bbox_buff = utils.get_city_bbox(city_iso, bbox_buffer, csv_path)
        
            miny = city_bbox_buff['miny']
            maxy = city_bbox_buff['maxy']
            minx = city_bbox_buff['minx']
            maxx = city_bbox_buff['maxx']
            city = city_bbox_buff['name']

            city_lats = [maxy, miny] 
            city_lons = [minx, maxx]
    
            params = []

            for year in era5_years:
                for v in var:
                    # Check folders
                    if os.path.exists(os.path.join(era5_download_path, v)):
                        logging.info('climate variable directory exists')
                        #continue
                    else: 
                        os.mkdir(os.path.join(era5_download_path, v))
                        logging.info('climate variable directory does not exist, creating.')

                    # Check file   
                    if os.path.isfile(os.path.join(era5_download_path, v, f"{city}.{year}.nc")):
                        logging.info(f"ERA5 file for {city}.{year} exists. Skipping this download.")
                    else: 
                        params.append((city, city_lats, city_lons, era5_download_path, v, year, lock))
                        
            #### DOWNLOAD  WITH MULTIPROCESSING #####
            logging.info(f"Executing multiprocessing for {len(params)} ERA5 tasks for: {city}")
            
            with Pool(processes=num_threads) as pool:
                    results = pool.starmap(utils.cds_download, params)
                    pool.close()
                    pool.join()
            logging.info(f"ERA5 downloads complete for {city}")

            #### DATA CHECK #####
            logging.info(f"Checking for failed ERA5 downloads...")
            missing_era5_files = []
            for x in params:
                single_year_era5_file = f"{x[0]}.{x[5]}.nc"
                #Check
                if not os.path.isfile(os.path.join(x[3], x[4], single_year_era5_file)):
                    missing_era5_files.append(single_year_era5_file)
            
            if len(missing_era5_files) > 0:
                logging.info(f"A total of {len(missing_era5_files)} files were unable to download.")
                with open("logs/failed_downloads.log", "w") as failed_er_downloads:
                    failed_er_downloads.write("\nThe following ERA5 files could not be downloaded:")
                    for entry in missing_era5_files:
                        failed_er_downloads.write("%s\n" % entry)

    @staticmethod
    def cds_download(city, city_lats, city_lons, era5_download_path, var, year, lock):

        c = cdsapi.Client()
        file_name = os.path.join(era5_download_path, var, f"{city}.{year}.nc")
    
        try:
            dataset = "reanalysis-era5-land"
            request = {'variable': [var],   
                        'year': year,
                        'month': ['01', '02', '03', '04', '05','06', '07', '08', '09', '10', '11', '12'],
                        'day': [
                                '01', '02', '03',
                                '04', '05', '06',
                                '07', '08', '09',
                                '10', '11', '12',
                               '13', '14', '15',
                                '16', '17', '18',
                                '19', '20', '21',
                               '22', '23', '24',
                                '25', '26', '27',
                                '28', '29', '30',
                                '31'],
                        'time': ['00:00', '01:00', '02:00',
                                    '03:00', '04:00', '05:00',
                                    '06:00', '07:00', '08:00',
                                    '09:00', '10:00', '11:00',
                                    '12:00', '13:00', '14:00',
                                   '15:00', '16:00', '17:00',
                                   '18:00', '19:00', '20:00',
                                   '21:00', '22:00', '23:00'],
                        'data_format': 'netcdf',
                            #bbox passed as a list [max_lat, min_lon, min_lat, max_lon]
                        'area': [
                                city_lats[0], city_lons[0], city_lats[1], city_lons[1]]
                        }
            c.retrieve(dataset, request, file_name).download()
        except:
            logging.info(f"Error: Unable to download ERA5 data for {city}: {year}: {var}. Please try again.")
    