"""
This python script initiates the main preprocessing workflow for preprocessing 
of data for the UHI model training. Specifically this scripts will take the raw
hls bands, calculate lst, and then stack and patch the bands with calculated era5 
bands. This script carries out the data checks, splitting of data into train
validation, and test sets for fine-tuning purposes. It also performs the calculations
of means and standard for the train set, and finally runs the normalization. 
For more information, refer to the README.
"""

########################################################################
# Import libraries & setup environment
########################################################################
import os
import logging
from datetime import datetime

# ------------- Setting environmental variables ------------------------

os.environ["NUMEXPR_MAX_THREADS"] = "16"
os.environ["GDAL_MAX_BAND_COUNT"] = "100000"
os.environ["RUN_DATE_TIME"] = datetime.now().strftime("%Y%m%d%H%M%S")
now = datetime.now()
dt_string = now.strftime("%Y%m%d%H%M%S")

# Check and create logs folder
if os.path.exists("logs/") == False:
    os.mkdir("logs/")

logging.basicConfig(
    filename=f"logs/main-processing-log-{dt_string}.log",
    filemode="w",
    # stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
)

logging.getLogger("rioxarray").setLevel(logging.CRITICAL + 1)
logging.getLogger("rasterio").setLevel(logging.CRITICAL + 1)

import argparse
import json
import pandas as pd
import geopandas as gpd

from preprocessing import utils
from preprocessing import parallel_patching
from preprocessing import hls_preprocess
from preprocessing import checks
from preprocessing import splitting
from preprocessing import calc_stats



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
        Path to the json config file, containing workflow parameters.
        Must contain list of city names. See official readme for appropriate
        payload.

    """

    ########################################################################
    # VERIFY PAYLOAD AND PASS ARGUMENTS
    ########################################################################

    logging.info("Saving logs to logs/\n")

    # Pass arguments from payload
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configpath",
        type=str,
        dest="configpath",
        required=True,
        help="path to the payload json containing workflow options",
    )
    args = parser.parse_args()
    configpath = args.configpath

    try:
        assert os.path.isfile(configpath), f"Error! File does not exist: {configpath}. Please provide a valid file path to the config file"
    except AssertionError as e:
        raise e

    logging.info(f"Loading config file: {configpath}")
    # Load payload and check required options
    try:
        with open(configpath, "r") as f:

            # Load json payload
            input_params = json.load(f)

            # Extract workflow type
            workflow_type = input_params["workflow_type"]

            # Extract city options
            city_bbox_buffer = input_params["workflow_options"]["cities"]["buffer"]
            hls_save_dir = input_params["workflow_options"]["cities"]["data_save_dir"]
            city_names = input_params["workflow_options"]["cities"]["city_names"]
            
            # Compute options
            n_threads = input_params["workflow_options"]["compute"]["max_threads"]

    except (ValueError, KeyError) as e:
        logging.error(f"Error! Failed to decode {configpath}, please verify JSON file.")
        raise e

    ########################################################################
    # LOAD GLOBAL ASSETS
    ########################################################################

    logging.info("Loading global assets...")

    # Load global city database
    global_cities_db_filepath = "../assets/databases/global_cities_database.csv"  # add to main payload checks above
    global_cities_db = pd.read_csv(global_cities_db_filepath)

    # Load ocean mask database
    ocean_mask_filepath = "../assets/databases/global_cities_oceanmask.shp"
    ocean_mask_db = gpd.read_file(ocean_mask_filepath)

    logging.info("Global assets loaded.\n")

    ########################################################################
    # RUN WORKFLOWS
    ########################################################################

    # ----------------- MAIN HLS PROCESSINGS AND LST CALC ------------------

    logging.info("Starting HLS preprocessing...")

    # Extract HLS options
    hls_options = input_params["workflow_options"]["hls_processing"]
    hls_downloads = hls_options["hls_downloads"]
    cloud_percent = hls_options["cloud_percent"]
    hls_scale = hls_options["scale_hls_bands"]
    

    if os.path.exists(hls_downloads) == False:  # add to main payload checks above
        logging.error(
            f"ERROR! Directory for HLS downloads is invalid. Please verify directory and edit payload."
        )

    if os.path.exists(hls_save_dir) == False:  # add to main payload checks above
        logging.error(
            f"ERROR! Directory for saving processed data is invalid. Creating directory."
        )
        os.mkdir(hls_save_dir)

    hls_preprocess.preprocess_hls(
        city_names,
        city_bbox_buffer,
        global_cities_db_filepath,
        cloud_percent,
        hls_downloads,
        hls_save_dir,
        n_threads,
        hls_scale,
    )

    logging.info("Completed preprocessing of HLS files.\n")

    # ---------------- MAIN STACKING AND PATCHING PROCESS --------------------

    logging.info("Starting main stacking and patching workflow...")

    try:

        # Extract options for stacking and patching
        patching_options = input_params["workflow_options"]["stack_patch"]
        era5_dir = patching_options["era5_dir"]
        hls_dir = hls_save_dir
        patched_dir = hls_save_dir
        interp_method = patching_options["interpolation"]
        patch_sizes = patching_options["patch_sizes"]
        percentage_nans = patching_options["perc_nan"]
        output_type = patching_options["output_files"]

        # Check if data folder exist else create
        data_dirs_exists = list(
            os.path.exists(dir) for dir in [era5_dir, hls_dir]
        )
        if False in data_dirs_exists:
            logging.error(
                f"ERROR! 1 or more of the directories does not exist.\
                        Please verify directory and files exist."
            )
        else:
            logging.info(
                f"Fetching data in the following folders for stacking and patching:\
                        \nERA5 VARS.: {era5_dir}\
                        \nHLS PROC. OUTPUTS: {hls_dir}"
            )

        # Test for folders
        lst_dir = os.path.join(hls_dir, "target-lst")
        hls_bands_dir = os.path.join(hls_dir, "hls-bands")
        fmask_dir = os.path.join(hls_dir, "fmask")

        hls_dirs_exists = list(
            os.path.exists(dir)
            for dir in [lst_dir, hls_bands_dir, fmask_dir]
        )

        if False in hls_dirs_exists:
            logging.error(
                f"ERROR! 1 or more of the HLS processed directories does not exist.\
                        Please verify directory and files exist."
            )
            raise Exception
        else:
            logging.info(
                f"Fetching data in the following folders for stacking and patching:\
                        \nLST TARGETS.: {lst_dir}\
                        \nHLS BANDS: {hls_bands_dir}\
                        \nHLS FMASK:{fmask_dir}"
            )

        logging.warning(f"Warning! Creating directors for saving patches...")
        patched_inputs_dir = os.path.join(patched_dir, "patched-inputs")
        patched_targets_dir = os.path.join(patched_dir, "patched-targets")
        try:
            os.makedirs(patched_inputs_dir)
        except OSError as error:
            logging.warning(f"Warning! {patched_inputs_dir} already exist.")
        try:
            os.makedirs(patched_targets_dir)
        except OSError as error:
            logging.warning(f"Warning! {patched_targets_dir} already exist.")

        logging.info(
            f"Saving patches to:\
                    \nINPUTS: {patched_inputs_dir}\
                    \nTARGETS: {patched_targets_dir}"
        )

    except (ValueError, KeyError) as e:
        logging.error(
            f"Error! Failed to decode {configpath}, please verify JSON file."
        )
        raise e

    # Log configurations
    logging.info(
        f"Setting patch sizes to: {patch_sizes[0]} - x; {patch_sizes[1]} - y for all cities."
    )
    logging.info(f"Setting tile interpolation to: {interp_method} for all cities.")
    logging.info(
        f"Setting allowed nans in patches to: {percentage_nans} for all cities."
    )
    logging.info(f"Fetching data...")

    # Extract all files and cities for target data
    target_cities, all_targets = utils.files_extractor(lst_dir, outputs=True)
    logging.info(
        f"Found {len(all_targets)} target tiles, accross these cities: {target_cities}"
    )

    # Extract all files and cities for hls data
    hls_cities, all_hls_inputs = utils.files_extractor(hls_bands_dir, outputs=True)
    logging.info(
        f"Found {len(all_hls_inputs)} hls input tiles, accross these cities: {hls_cities}"
    )

    # ERA5 inputs
    era5_cities, all_era5_inputs = utils.files_extractor(era5_dir)
    logging.info(
        f"Found {len(all_era5_inputs)} era5 datasets, accross these cities: {era5_cities}"
    )

    # Fmask inputs
    fmask_cities, all_fmask_inputs = utils.files_extractor(fmask_dir, outputs=True)
    logging.info(
        f"Found {len(all_fmask_inputs)} fmasks, accross these cities: {fmask_cities}"
    )

    # Loop of each city and perform preprocessing steps
    logging.info(f"Processing the following cities only: {city_names}\n")
    completed_cities = []
    for city in city_names:
        logging.info(f"Started processing for {city}...")
        logging.info(f"Extracting files for {city}...")

        # List of all target tiles for city
        targets4city = utils.filter_city(city_name=city, lst=all_targets)
        
        # List of all hls for city
        hls4city = utils.filter_city(city_name=city, lst=all_hls_inputs)

        # List of all era5 files for city
        era5_4city = utils.filter_city(city_name=city, lst=all_era5_inputs)
        
        # List of all fmasks for city
        fmasks4city = utils.filter_city(city_name=city, lst=all_fmask_inputs)

        # Extract timeshift for local time conversion
        city_n = city.split("_")[0]
        city_iso = city.split("_")[1]
        filtered_df_t = global_cities_db[(global_cities_db["CITY_NAME"] == city_n) & (global_cities_db["COUNTRY_ISO"] == city_iso)]
        timeshift4city = filtered_df_t["UTC_TIMESHIFT"].values[0]
        
        # Extract coastline boundary for country
        city_iso = city.split("_")[-1]
        ocean_mask_df = ocean_mask_db[ocean_mask_db["iso"] == city_iso]

        # Initialize multiprocessing class and run
        logging.info(
            f"Initializing stacking and patching multiprocessing workflow ..."
        )
        patch_city = parallel_patching(
            city_name=city,
            timeshift=timeshift4city,
            sat_inputs=hls4city,
            climate_inputs=era5_4city,
            targets=targets4city,
            ocean_mask=ocean_mask_df,
            fmasks=fmasks4city,
            interp_method=interp_method,
            patch_size_x=patch_sizes[0],
            patch_size_y=patch_sizes[1],
            perc_nans=percentage_nans,
            patchedinputs_dir=patched_inputs_dir,
            patchedtargets_dir=patched_targets_dir,
            n_threads=n_threads,
            output_type=output_type
        )
        results = patch_city.run_process()

        # Log errors
        errors = []
        for result in results:
            if result is not None:
                errors.append(result)

        logging.info(
            f"A total of {len(errors)} files could not be processed for {city}. See error logs for details."
        )

        logging.info("Writing errored tiles to log file.")
        with open(
            f"logs/tile-errors-{city}-{os.environ['RUN_DATE_TIME']}.log", "a"
        ) as error_file:
            error_file.write(
                f"\nThe following tiles for {city} had processing errors:"
            )
            for line in errors:
                error_file.write("%s\n" % line)

        logging.info(f"Completed processing of {city}.")
        completed_cities.append(city)

        logging.info(f"Following cities are complete: {completed_cities}.\n")

    logging.info(
        f"Stacking and patching workflow complete for payload {configpath}.\n"
    )

    # ---------------- MAIN DATA PROCESSING CHECKS --------------------

    logging.info(f"Starting main data checks workflow...")

    try:

        # Extract options for checks
        patched_inputs_dir = patched_inputs_dir
        patched_targets_dir = patched_targets_dir

        # Check if data folder exist else create
        data_dirs_exists = list(
            os.path.exists(dir) for dir in [patched_inputs_dir, patched_targets_dir]
        )
        if False in data_dirs_exists:
            logging.error(
                f"ERROR! 1 or more of the directories does not exist.\
                        Please verify directory and files exist."
            )
            raise Exception
        else:
            logging.info(
                f"Fetching data in the following folders for data checks:\
                        \nPATCHED INPUT DIR.: {patched_inputs_dir}\
                        \nPATCHED TARGET DIR.: {patched_targets_dir}"
            )

    except (Exception, ValueError, KeyError) as e:
        logging.error(
            f"Error! Failed to retrieve patching directories."
        )
        raise e

    # All patched inputs
    input_patch_cities, all_patched_inputs = utils.files_extractor(
        patched_inputs_dir, outputs=True
    )
    logging.info(
        f"Found {len(all_patched_inputs)} input patches, accross these cities: {input_patch_cities}"
    )

    # All patched targets
    target_patch_cities, all_patched_targets = utils.files_extractor(
        patched_targets_dir, outputs=True
    )
    logging.info(
        f"Found {len(all_patched_targets)} target patches, accross these cities: {target_patch_cities}"
    )

    # Run checks - patches
    if output_type == "stacked-patches":

        all_checks = checks(
            patchedinputs_dir=patched_inputs_dir,
            patchedtargets_dir=patched_targets_dir,
            n_threads=n_threads,
        )
        missing_inputs, missing_targets, missing_band_errors, dtype_errors = (
            all_checks.run_checks()
        )

        logging.warning(
            f"WARNING: There are {len(missing_targets)} target images with no corresponding input image."
        )
        logging.warning(
            f"WARNING: There are {len(missing_inputs)} input images with no corresponding target image."
        )
        logging.warning(
            f"WARNING: A total of {len(missing_band_errors)} files hand missing bands. Files have been deleted."
        )
        logging.warning(
            f"WARNING: A total of {len(dtype_errors)} files had Dtype errors. Files have been deleted."
        )

        # Final totals after checks
        # All patched inputs
        final_input_patch_cities, final_all_patched_inputs = utils.files_extractor(
            patched_inputs_dir, outputs=True
        )
        logging.info(
            f"Final {len(final_all_patched_inputs)} input patches, accross these cities: {final_input_patch_cities}"
        )

        # All patched targets
        final_target_patch_cities, final_all_patched_targets = utils.files_extractor(
            patched_targets_dir, outputs=True
        )
        logging.info(
            f"Final {len(final_all_patched_targets)} target patches, accross these cities: {final_target_patch_cities}\n"
        )

        with open(
            f"logs/patch-errors-{os.environ['RUN_DATE_TIME']}.log", "a+"
        ) as tile_file:
            tile_file.write("\nThe following inputs had no corresponding target:\n")
            for line in missing_inputs:
                tile_file.write("%s\n" % line)
            tile_file.write("\nThe following targets had no corresponding inputs:\n")
            for line in missing_targets:
                tile_file.write("%s\n" % line)
            tile_file.write("\nThe following files had missing bands:\n")
            for line in missing_band_errors:
                tile_file.write("%s\n" % line)
            tile_file.write("\nThe following files had this wrong dtype:\n")
            for line in dtype_errors:
                tile_file.write("%s\n" % line)

    # Run checks on full tiles
    elif output_type == "stacked-tiles":

        all_checks_tiles = checks(
            patchedinputs_dir=patched_inputs_dir,
            patchedtargets_dir=patched_targets_dir,
            n_threads=n_threads,
        )
        missing_inputs_tiles, missing_targets_tiles, missing_band_errors_tiles, dtype_errors_tiles = (
            all_checks_tiles.run_checks_tiles()
        )

        logging.warning(
            f"WARNING: There are {len(missing_targets_tiles)} target tiles with no corresponding input tile."
        )
        logging.warning(
            f"WARNING: There are {len(missing_inputs_tiles)} input tiles with no corresponding target tile."
        )
        logging.warning(
            f"WARNING: A total of {len(missing_band_errors_tiles)} tiles hand missing bands. Tiles have been deleted."
        )
        logging.warning(
            f"WARNING: A total of {len(dtype_errors_tiles)} tiles had Dtype errors. Tiles have been deleted."
        )

        # Final totals after checks
        # All patched inputs
        final_input_tile_cities, final_all_tile_inputs = utils.files_extractor(
            os.path.join(patched_inputs_dir, "stacked_tiles"), outputs=True
        )
        logging.info(
            f"Final {len(final_all_tile_inputs)} input tiles, accross these cities: {final_input_tile_cities}"
        )

        # All patched targets
        final_target_tile_cities, final_all_tile_targets = utils.files_extractor(
            os.path.join(patched_targets_dir, "processed_lst"), outputs=True
        )
        logging.info(
            f"Final {len(final_all_tile_targets)} target tiles, accross these cities: {final_target_tile_cities}\n"
        )

        with open(
            f"logs/stacked-tile-errors-{os.environ['RUN_DATE_TIME']}.log", "a+"
        ) as tile_file:
            tile_file.write("\nThe following input tiles had no corresponding target:\n")
            for line in missing_inputs_tiles:
                tile_file.write("%s\n" % line)
            tile_file.write("\nThe following targets tiles had no corresponding inputs:\n")
            for line in missing_targets_tiles:
                tile_file.write("%s\n" % line)
            tile_file.write("\nThe following tiles had missing bands:\n")
            for line in missing_band_errors_tiles:
                tile_file.write("%s\n" % line)
            tile_file.write("\nThe following tiles had this wrong dtype:\n")
            for line in dtype_errors_tiles:
                tile_file.write("%s\n" % line)

    logging.info(
        f"Checks complete. See log file for files with errors. Please NOTE erronoues files have been deleted."
    )

    # ---------------- MAIN DATA SPLIT PROCESS --------------------
    
    if output_type == "stacked-patches":

        logging.info(f"Starting main train-val-test splitting on dataset...")

        try:

            # Extract options for splitting
            splitting_dir = os.path.join(hls_save_dir, "train-val-test")
            if os.path.exists(splitting_dir) == False:
                logging.info("Creating train-val-test splitting dir")
                os.mkdir(splitting_dir)

            # Check for train, val, test inputs and targets dirs
            train_inputs_dir = os.path.join(splitting_dir , "train", "inputs")
            if not os.path.exists(train_inputs_dir):
                os.makedirs(train_inputs_dir)

            train_targets_dir = os.path.join(splitting_dir , "train", "targets")
            if not os.path.exists(train_targets_dir):
                os.makedirs(train_targets_dir)

            test_inputs_dir = os.path.join(splitting_dir , "test", "inputs")
            if not os.path.exists(test_inputs_dir):
                os.makedirs(test_inputs_dir)

            test_targets_dir = os.path.join(splitting_dir , "test", "targets")
            if not os.path.exists(test_targets_dir):
                os.makedirs(test_targets_dir)

            val_inputs_dir = os.path.join(splitting_dir , "val", "inputs")
            if not os.path.exists(val_inputs_dir):
                os.makedirs(val_inputs_dir)

            val_targets_dir = os.path.join(splitting_dir , "val", "targets")
            if not os.path.exists(val_targets_dir):
                os.makedirs(val_targets_dir)

            patched_inputs_dir = patched_inputs_dir
            patched_targets_dir = patched_targets_dir
            trainvaltest_dir = splitting_dir
            

            # Check if data folder exist else create
            data_dirs_exists = list(
                os.path.exists(dir) for dir in [x[0] for x in os.walk(trainvaltest_dir)]
            )
            if False in data_dirs_exists:
                logging.error(
                    f"ERROR! 1 or more of the directories does not exist.\
                            Please verify directory and files exist."
                )
                raise Exception
            else:
                logging.info(
                    f"Fetching data in the following folders for splitting:\
                            \nPATCHED INPUT DIR.: {patched_inputs_dir}\
                            \nPATCHED TARGET DIR.: {patched_targets_dir}"
                )

        except (Exception, ValueError, KeyError) as e:
            logging.error(
                f"Error! Failed to decode {configpath}, please verify JSON file."
            )
            raise e
        

        # All patched inputs
        input_patch_cities, all_patched_inputs = utils.files_extractor(
            patched_inputs_dir, outputs=True
        )
        logging.info(
            f"Found {len(all_patched_inputs)} input patches, accross these cities: {input_patch_cities}"
        )

        # All patched targets
        target_patch_cities, all_patched_targets = utils.files_extractor(
            patched_targets_dir, outputs=True
        )
        logging.info(
            f"Found {len(all_patched_targets)} target patches, accross these cities: {target_patch_cities}"
        )

        # Run splitting
        split_data = splitting(
            patchedinputs=all_patched_inputs,
            patchedtargets=all_patched_targets,
            destination_root=trainvaltest_dir,
            n_threads=n_threads,
        )
        train, val, test = split_data.run_splitting()

        logging.info(f"Main train-val-test splitting workflow complete.")
        logging.info(
            f"Splitting results: Training --> {len(train)}, Validation --> {len(val)}, Test --> {len(test)}."
        )
        
        #    ---------------- MAIN MEANS & STANDARDS CALC --------------------

        logging.info(
            f"Starting means and standard deviations calculations for training data..."
        )

        try:
            # Extract options for splitting
            trainvaltest_dir = splitting_dir

            # Check if data folder exist else create
            data_dirs_exists = list(
                os.path.exists(dir) for dir in [x[0] for x in os.walk(trainvaltest_dir)]
            )
            if False in data_dirs_exists:
                logging.error(
                    f"ERROR! 1 or more of the directories does not exist.\
                            Please verify directory and files exist."
                )
                raise Exception
            else:
                logging.info(
                    f"Fetching data in the following folder for splitting:\
                            \nSPLIT DIR.: {trainvaltest_dir}" #\
                            #\nPATCHED TARGET DIR.: {os.path.join(trainvaltest_dir, "patched-targets")}"
                )

        except (Exception, ValueError, KeyError) as e:
            logging.error(
                f"Error! Failed to retrieve splitting directory"
            )
            raise e

        # All patched targets
        train_inputs_path = os.path.join(trainvaltest_dir, "train", "inputs")
        train_inputs_cities, all_train_inputs = utils.files_extractor(
            train_inputs_path, outputs=True
        )
        logging.info(
            f"Found {len(all_train_inputs)} training input patches, accross these cities: {train_inputs_cities}"
        )

        train_calc = calc_stats(
            inputs=all_train_inputs, destination_root=trainvaltest_dir
        )
        stats_results = train_calc.cal_mean_std()

        logging.info(f"Finished calculating means and standard deviations.")
        logging.info(
            f"These are the means and standard deviations for the training set:\n{stats_results}\n"
        )

        #    ---------------- MAIN AUX feature extraction --------------------

        #    ---------------- END OF WORKFLOWS --------------------

    
        logging.info(f"Workflow {workflow_type} not implemented, please verify json")

        #     ########################################################################
        #     # END OF MAIN WORKFLOW
        #     ########################################################################

        logging.info(f"Workflow for {configpath} to produce stacked patches for inference or fine-tuning is complete.")
        print(f"Workflow for {configpath} to produce stacked patches for inference or fine-tuning is complete.")

    else:
        logging.info(f"Workflow for {configpath} to produce stacked tiles for inference is complete.")
        print(f"Workflow for {configpath} to produce stacked tiles for inference is complete.")

if __name__ == "__main__":
    main()
