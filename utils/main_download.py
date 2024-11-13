"""
This python script initiates the main download workflow for HLS and ERA5 2m_temperature 
datasets based on the parameters specified in the download config file. 

"""

########################################################################
# Import libraries
########################################################################
import json
import os
import argparse
import sys
import logging
from datetime import datetime, date, timedelta
import pandas as pd

# ------------- Set up logging ------------------------
now = datetime.now()
dt_string = now.strftime("%Y%m%d%H%M%S")

# Check and create logs folder
if os.path.exists("logs/") == False:
    os.mkdir("logs/")

logging.basicConfig(
    filename=f"logs/download-log-{dt_string}.log",
    filemode="w",
    # stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

from download_functions import utils


########################################################################
# Main function
########################################################################

def main():

    """
    Main function

    ...

    Arguments
    ---------
    --configpath : str
        Path to the json config file, containing download parameters.
        Must contain list of city names. 
    """

     ########################################################################
    # VERIFY PAYLOAD AND PASS ARGUMENTS
    ########################################################################

    logging.info("Saving logs to logs/\n")

    # Pass arguments from payload
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configpath", type=str, dest="configpath", required=True, help="path to the json config")
    args = parser.parse_args()
    configpath = args.configpath
    
    try:
        assert os.path.isfile(configpath), f"Error: File does not exist: {configpath}. Please provide a valid file path to the config file"
    except AssertionError as e:
        raise e

    with open(configpath, "r") as json_file_config:
        input_params = json.load(json_file_config)
        workflow_options = input_params["workflow_options"]
        data_collections = ["HLS_landsat", "ERA5_land"] # select both data collections
        percent = workflow_options["hls_cloud_percent"]
        start_date = workflow_options["start_date"]
        end_date = workflow_options["end_date"]
        list_cities = workflow_options["city_iso_names"]
        bbox_buffer = workflow_options["city_bbox_buffer"]
        var = ["2m_temperature"]
        save_dir = workflow_options["data_dir"]
        csv_path = "../assets/databases/global_cities_database.csv"

        #### Check config file for errors #####
        if os.path.exists(save_dir) == False:
                logging.warning(
                    f"Warning! Path for saving data does not exist, creating it..."
                )
                os.makedirs(save_dir)

        #### Create directories for data #####

        hls_save_dir = os.path.join(save_dir, "hls")
        if os.path.exists(hls_save_dir) == False:
            os.mkdir(os.path.join(save_dir, "hls"))

        era5_download_path = os.path.join(save_dir, "era5")
        if os.path.exists(era5_download_path) == False:
            os.mkdir(os.path.join(save_dir, "era5"))

        #### Run downloads ####
        
        for x in data_collections:
            if x == "HLS_landsat": 
                ### DOWNLOAD HLS
                HLS_COLLECTION_IDS = "HLSL30.v2.0"
                date_range = start_date + "T00:00:00Z/" + end_date + "T23:59:59Z"
                logging.info(f"Starting HLS downloads...")
                utils.download_hls(list_cities, bbox_buffer, csv_path, percent, HLS_COLLECTION_IDS,  date_range, hls_save_dir)
                logging.info('Completed downloading HLS files.')

                print(f"Completed downloading HLS files.")

            elif x == "ERA5_land":
                ### DOWNLOAD ERA5
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                era5_years = sorted(list(set(pd.date_range(start_date,end_date,freq='d').strftime('%Y').tolist())))
                era5_months = sorted(list(set(pd.date_range(start_date,end_date,freq='d').strftime('%m').tolist())))
                logging.info(f"Starting ERA5 Land downloads...")
                utils.download_era5(era5_download_path, list_cities, bbox_buffer, csv_path, era5_years, era5_months, var)
                logging.info('Completed downloading ERA5 Land files.')
                print(f"Completed downloading ERA5 Land files.")

            else:
                logging.warning('Data collection not specified.')
        
        print(f"Workflow for {configpath} is complete.")

if __name__ == "__main__":
    main()