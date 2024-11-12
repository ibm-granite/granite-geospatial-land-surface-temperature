import os
import glob
import rioxarray
import shutil
import rasterio

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

from matplotlib import cm
from matplotlib.colors import Normalize 
from matplotlib.animation import FuncAnimation
from IPython.display import Image, display
from datetime import datetime
from tqdm import tqdm

matplotlib.rcParams['agg.path.chunksize'] = 10000


def mae(pred, truth):

    """
    Function for calculating the mean absolute error (MAE) between the prediction and target.
    Args:
        pred: xarray.Dataset
            An xarray dataset of the model prediction.
        truth: xarray.Dataset
            An xarray dataset of the ground truth.
    """

    abs_diff = abs(pred-truth)
    mae = abs_diff.mean(dim=['x','y'], skipna=True)['band_data'].values[0]
    return mae

def norm(arr1):

    """
    Function for normalizing an array using histogram normaliztion.
    Args:
        arr1: numpy.ndarray()
        An array that contains un-normalized data.

    Returns:
        The normalized array.
    
    If np.nanmax(arr1) - np.nanmin(arr1) is 0:
        Returns an array of zeros
    """
    if np.nanmax(arr1) - np.nanmin(arr1) == 0: 
      return np.zeros_like(arr1)  # Avoid division by zero
    return (arr1 - np.nanmedian(arr1)) / np.nanstd(arr1)

def stack_rgb(stacked_inputs_path):
    """
    Extract, normalize and stack HLS bands corresponding to RGB image from stacked input file 
    Args:
        stacked_inputs_path (str): Path to stacked input file        
    """
    with rasterio.open(stacked_inputs_path) as src:
        blue_band = src.read(1)
        green_band = src.read(2)
        red_band = src.read(3)
        # normalize these bands
        red_normalized = norm(red_band) 
        green_normalized = norm(green_band) 
        blue_normalized = norm(blue_band) 
        rgb_normalized = np.dstack(((red_normalized + 0.3), (green_normalized +0.3), (blue_normalized +0.3)))
        rgb_normalized = xr.DataArray(rgb_normalized)
    
    return rgb_normalized


def calculate_metrics(target_fps, predict_fps):

    """
    Function for calculating some metrics given targets and predictions
    Args:
        target_fps: List of paths to LST ground-truth files.
        predict_fps: List of paths to predicted LST prediction.
    """

    metrics = []
    for tfp, pfp in tqdm(zip(target_fps, predict_fps), total=len(target_fps)):

        if not (os.path.isfile(tfp) and os.path.isfile(pfp)):
            print(f"Skipping:  Either one of these files missing: \n{tfp}\n{pfp}")
            continue

        # read in target data
        ds_target = xr.open_dataset(tfp)
        ds_target = ds_target.where(ds_target['band_data'] != -9999., np.nan)
        ds_pred = xr.open_dataset(pfp)
        ds_pred = ds_pred.where(~np.isnan(ds_target['band_data']), np.nan)

        # calculate MAE
        mae_val = mae(ds_pred, ds_target)

        if "index" in os.path.basename(tfp):
            city_country, _, _, datestamp, timestamp = os.path.basename(tfp).split(".")[0:5]
        else:
            city_country, _, datestamp, timestamp = os.path.basename(tfp).split(".")[0:4]

        timestamp = datetime.strptime(datestamp+timestamp, "%Y%m%dT%H%M%S")
        metrics.append((tfp, pfp, city_country, timestamp, mae_val))

    df = pd.DataFrame(metrics, columns=["target_fp", "prediction_fp", "city_country", "timestamp", "mae"])

    return df

def plot_box_plot(targets_path_patch, predictions_path_patch, results_path, save_plot):

    """Plotting routine
    Args:
        targets_path_patch (str): Path to directory with geotiff target patches.
        predictions_path_patch (str): Path to directory where the geotiff predictions are saved.
        result_path (str): Path to directory where the box plot must be saved.
        save_plot (bool): "True" will display and save the generated plot. "False" will only display the plot.
    """

    assert targets_path_patch.exists(), f"Folder not found: {str(targets_path_patch)}" 
    assert predictions_path_patch.exists(), f"Folder not found: {str(predictions_path_patch)}" 
    assert results_path.exists(), f"Folder not found: {str(results_path)}" 

    # gather target file paths and corresponding predictions file paths
    target_fps = glob.glob(os.path.join(targets_path_patch, "*.tif"))
    target_fps_names = [".".join(os.path.basename(x).split(".")[0:5]) for x in target_fps]
    predict_fps = [os.path.join(predictions_path_patch, f"{x}.inputs_pred.tif") for x in target_fps_names]

    # generate metrics df
    metrics_df = calculate_metrics(target_fps, predict_fps)

    means = np.squeeze(metrics_df.groupby('city_country').mean(numeric_only=True).values)
    stds = np.squeeze(metrics_df.groupby('city_country').std(numeric_only=True).values)
    names = list(np.squeeze(metrics_df.groupby('city_country').mean(numeric_only=True).index.values))
    xs = np.arange(len(means))

    plt.figure(figsize=(8, 6))
    plt.bar(xs, means, color='lightblue')
    plt.grid(True, alpha=0.5, zorder=0)
    plt.title('Box plots for unseen timestamps and cities')
    plt.xlabel('City')
    plt.ylabel('Mean absolute error ($^\circ$C)')
    plt.xticks(rotation=45)
    plt.errorbar(xs, means, stds, fmt='.', color='0.5', elinewidth=1.5, capthick=1.5, errorevery=1, ms=5, capsize=4)
    #plt.gca().set_xticklabels(names)
    plt.gca().set_xticks(np.arange(len(names)), labels=names)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.show()

    return metrics_df


def plot_results(tile_name, inputs_path, targets_path, predictions_path):
    """Plotting routine
        Args:
            tile_name:  Name of the tile (excluding suffix such as ".stacked_inputs.tif", ".lst.tif", and ".nputs_pred.tif")
            inputs_path: Path to the folder containing input images (suffix: ".inputs.tif")
            targets_path: Path to the folder containing ground-truth LST observations (suffix: ".lst.tif")
            predictions_path: Path to the folder containing the LST predictions (suffix: ".inputs_pred.tif")
    """
    
    input_path = inputs_path.joinpath(f"{tile_name}.inputs.tif")
    target_path = targets_path.joinpath(f"{tile_name}.lst.tif")
    pred_path = predictions_path.joinpath(f"{tile_name}.inputs_pred.tif")

    assert input_path.exists(), f"File not found: {str(input_path)}"
    assert target_path.exists(), f"File not found: {str(target_path)}"
    assert pred_path.exists(), f"File not found: {str(pred_path)}"

    # Read inputs
    input_bands = []
    with rasterio.open(input_path) as src:
        input_bands.append(norm(src.read(3))+0.3) #red
        input_bands.append(norm(src.read(2))+0.3) #green
        input_bands.append(norm(src.read(1))+0.3) #blue
    rgb_normalized = xr.DataArray(np.dstack((input_bands[0], input_bands[1], input_bands[2])))

    # Read target and prediction files
    ds_lst_target = xr.open_dataset(target_path)
    ds_lst_target = ds_lst_target.where(ds_lst_target['band_data'] != -9999., np.nan)
    ds_lst_pred = xr.open_dataset(pred_path)
    ds_lst_pred = ds_lst_pred.where(~np.isnan(ds_lst_target['band_data']), np.nan)
    ds_error =  ds_lst_pred - ds_lst_target

    min_target = ds_lst_target.quantile(0.05)['band_data'].values
    max_target = ds_lst_pred.quantile(0.95)['band_data'].values
    min_error = ds_error.quantile(0.005)['band_data'].values
    max_error = ds_error.quantile(0.995)['band_data'].values
    error_max_size = max(abs(min_error), max_error)
    min_error, max_error = -abs(error_max_size), error_max_size
    data_normalizer = Normalize(min_target, max_target)
    error_normalizer = Normalize(min_error, max_error)

    # calculate error metrics
    mae_val = round(mae(ds_lst_pred, ds_lst_target), 3)

    # Plotting
    nrows, ncols = 1, 5
    axw, axh = 4, 4
    wsp, hsp = 0.03, 0.03

    fig = plt.figure(figsize=(axw*ncols, axh*nrows))
    gs = gridspec.GridSpec(nrows, ncols, wspace=wsp, hspace=hsp, width_ratios=[1,1,1,1,0.75])

    #Plot the RGB image
    ax_rgb = plt.subplot(gs[0, 0])
    plt.imshow(rgb_normalized, aspect="auto")
    ax_rgb.set_title("RGB Color Composite", fontsize=10)
    ax_rgb.get_xaxis().set_ticklabels([])
    ax_rgb.get_yaxis().set_ticklabels([])
    
    #Plot the target/ground truth
    ax_gt = plt.subplot(gs[0, 1])
    target_handle = ds_lst_target['band_data'].plot(ax=ax_gt, cmap='jet', norm=data_normalizer, vmin=min_target, vmax=max_target, add_colorbar=False)
    ax_gt.set_title("LST Observation", fontsize=10)
    ax_gt.set(ylabel=None, xlabel=None)
    ax_gt.get_xaxis().set_ticklabels([])
    ax_gt.get_yaxis().set_ticklabels([])

    #Plot the prediction
    ax_pred = plt.subplot(gs[0, 2])
    ds_lst_pred['band_data'].plot(ax=ax_pred, cmap='jet', norm=data_normalizer, vmin=min_target, vmax=max_target, extend='both', add_colorbar=False)
    ax_pred.set_title(f"LST Prediction", fontsize=10)
    ax_pred.set(ylabel=None, xlabel=None)
    ax_pred.get_xaxis().set_ticklabels([])
    ax_pred.get_yaxis().set_ticklabels([])

    #Plot the error (with the MAE in the the Title) 
    ax_err = plt.subplot(gs[0, 3])
    error_handle = ds_error['band_data'].plot(ax=ax_err, cmap='bwr', norm=error_normalizer, vmin=min_error, vmax=max_error, add_colorbar=False)
    ax_err.set_title(f"Error (MAE: {mae_val:.3f}$^\circ$C)", fontsize=10)
    ax_err.set(ylabel=None, xlabel=None)
    ax_err.get_xaxis().set_ticklabels([])
    ax_err.get_yaxis().set_ticklabels([])

    # colorbars
    ax_cbar = plt.subplot(gs[0, 4])
    cbar_lst = plt.colorbar(target_handle, ax=ax_cbar, orientation='vertical', location='left', extend='both', pad=0.25, label="Temperature ($^\circ$C)")
    cbar_err = plt.colorbar(error_handle, ax=ax_cbar, orientation='vertical', location='left', extend='both', pad=0.25, label="Error ($^\circ$C)")
    cbar_lst.ax.yaxis.set_ticks_position('right')
    cbar_err.ax.yaxis.set_ticks_position('right')
    ax_cbar.axis('off')

    # title
    city_name = input_path.stem.split(".")[0]
    acq_date = input_path.stem.split(".")[2]
    acq_time = input_path.stem.split(".")[3]
    timestamp = datetime.strptime(acq_date+acq_time, "%Y%m%dT%H%M%S")
    city_name = (city_name.split("_")[0]).upper() + f" ({(city_name.split('_')[1]).upper()})"
    fig.suptitle(f"City: {city_name},  Date & Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n", y=1.05, fontsize=12)

    plt.show()

def plot_rgb_lst_distribution_scatter(patches_tiles, target_patches_path, inference_path, result_path, input_path, save_plot):
    """Plotting routine
        Args:
            patches_tiles (list): List of input patches or tiles to generate comparison plots for.
            target_patches_path (str): Path to directory with geotiff target patches.
            inference_path (str): Path to directory where the geotiff predictions are saved.
            result_path (str): Path to directory where the result/comparison plots must be saved.
            input_path (str): Path to directory with geotiff input patches.
            save_plot (bool): "True" will display and save the generated plots. "False" will only display the plots.s
    """
    
    # inputs to parse to the main method
    input_patches_paths = []

    for pt in patches_tiles:
        patches_ = os.path.join(input_path, pt)
        input_patches_paths.append(patches_)

    # predictions to parse to the main method
    pred_patches_paths = []

    for inp in patches_tiles:
        inp_ = os.path.basename(inp)
        inp_ = inp_.replace("inputs", "inputs_pred")
        inp_ = os.path.join(inference_path, inp_)
        pred_patches_paths.append(inp_)

    # targets to parse to the main method
    target_patches_paths = []

    for tar in patches_tiles:
        patches_ = os.path.join(target_patches_path, tar)
        patches_ = patches_.replace("inputs", "lst")
        target_patches_paths.append(patches_)
    target_patches_paths

    #Create Plots Save Directory
    if os.path.exists(result_path):
        print("Results directory exits!")
    else: 
        print("Creating Results directory...")
        os.makedirs(result_path) 
    #Subdirectory to store comparison plots
    comp_plots_path = os.path.join(result_path, 'comparison_plots')
    if os.path.exists(comp_plots_path):
        print("Comparison plots directory exits!")
    else: 
        print("Creating comparison plots directory...")
        os.makedirs(comp_plots_path)

    ncols = 6
    nexp = 1
    axw, axh = 4, 4
    wsp, hsp = 0.03, 0.03
    title_x_pos = 0.5
    title_y_pos = 1.0
    
    for inp, tar, pred in zip(input_patches_paths, target_patches_paths, pred_patches_paths):

        fig = plt.figure(figsize=(axw*ncols, axh*nexp))
        gs = gridspec.GridSpec(nexp+1, ncols, height_ratios=[2,0.5], width_ratios=[1,1,1,1,1,1], wspace=wsp, hspace=hsp)

        #Open target and prediction as datasets
        ds_target = xr.open_dataset(tar)
        ds_target = ds_target.where(ds_target['band_data'] != -9999., np.nan)
        ds_pred = xr.open_dataset(pred)
        ds_pred = ds_pred.where(~np.isnan(ds_target['band_data']), np.nan)
        #Compute the error between the prediction and target
        error =  ds_pred - ds_target

        #Create the RGB object
        hls_bands = []
        with rasterio.open(inp) as src:
            red_band = src.read(3)
            hls_bands.append(red_band)
            green_band = src.read(2)
            hls_bands.append(green_band)
            blue_band = src.read(1)
            hls_bands.append(blue_band)
        red_normalized = norm(red_band)
        green_normalized = norm(green_band)
        blue_normalized = norm(blue_band)
        rgb_normalized = np.dstack((red_normalized+0.3, green_normalized+0.3, blue_normalized+0.3))
        rgb_normalized = xr.DataArray(rgb_normalized)

        mae_val = mae(ds_pred, ds_target)
        mae_val = round(mae_val, 3)

        #Min and max for predictions and errors
        min_p = np.nanmin(np.squeeze(ds_pred['band_data'].values))
        max_p = np.nanmax(np.squeeze(ds_pred['band_data'].values))
        min_e = np.nanmin(np.squeeze(error['band_data'].values))
        max_e = np.nanmax(np.squeeze(error['band_data'].values))
        error_abs = max(abs(min_e), max_e)
        error_min, error_max  = -abs(error_abs), error_abs

        #min and max for ground truth 
        gt_min = np.nanmin(np.squeeze(ds_target['band_data'].values))
        gt_max = np.nanmax(np.squeeze(ds_target['band_data'].values))
        
        #min and max across predictions and ground truth
        vmin, vmax = np.minimum(gt_min,min_p), np.maximum(gt_max, max_p)
        
        expi = 0 
        
        #Plot the RGB image
        ax_rgb = plt.subplot(gs[expi, 0])
        plt.imshow(rgb_normalized)
        ax_rgb.set_title("RGB", x=title_x_pos, y=title_y_pos)
        ax_rgb.get_xaxis().set_ticklabels([])
        ax_rgb.get_yaxis().set_ticklabels([])

        #Plot the target/ground truth
        ax_gt = plt.subplot(gs[expi, 1])
        target_handle = ds_target['band_data'].plot(ax=ax_gt, cmap='jet', vmin=vmin, vmax=vmax, add_colorbar=False)
        ax_gt.set_title("Ground Truth", x=title_x_pos, y=title_y_pos)
        ax_gt.set(ylabel=None)
        ax_gt.set(xlabel=None)
        ax_gt.get_xaxis().set_ticklabels([])
        ax_gt.get_yaxis().set_ticklabels([])

        #Plot the prediction
        ax_pred = plt.subplot(gs[expi, 2])
        ds_pred['band_data'].plot(ax=ax_pred, cmap='jet', vmin=vmin, vmax=vmax, add_colorbar=False)
        ax_pred.set_title(f"Prediction", x=title_x_pos, y=title_y_pos)
        ax_pred.set(ylabel=None)
        ax_pred.set(xlabel=None)
        ax_pred .get_xaxis().set_ticklabels([])
        ax_pred .get_yaxis().set_ticklabels([])

        #Plot the error (with the MAE in the the Title) 
        ax_err = plt.subplot(gs[expi, 3])
        error_handle = error['band_data'].plot(ax=ax_err, cmap='bwr', vmin=error_min, vmax=error_max, add_colorbar=False)
        ax_err.set_title(f"Error (MAE: {mae_val})", x=title_x_pos, y=title_y_pos)
        ax_err.set(ylabel=None)
        ax_err.set(xlabel=None)
        ax_err.get_xaxis().set_ticklabels([])
        ax_err.get_yaxis().set_ticklabels([])

        #Plot the prediction and target histograms
        ax_hist = plt.subplot(gs[expi, 4])
        ds_target['band_data'].plot.hist(bins=30, range=[vmin, vmax], label='Ground Truth')
        ds_pred['band_data'].plot.hist(bins=30, range=[vmin, vmax], alpha=0.5, label='Prediction')
        ax_hist.set_title(f"Histogram", x=title_x_pos, y=title_y_pos)
        ax_hist.legend(prop={'size': 10})
        ax_hist.set(xlabel=r'LST [$\degree$ C]')
        #ax_hist.get_xaxis().set_ticklabels([])
        ax_hist.get_yaxis().set_ticklabels([])

        #Flatten the target and prediction dtatsets into arrays
        target_line = np.squeeze(ds_target['band_data'].values.reshape(-1))
        pred_line = np.squeeze(ds_pred['band_data'].values.reshape(-1))

        #Plot the scatter density plot
        ax_sd = plt.subplot(gs[expi, 5])
        ax_sd.scatter(target_line, pred_line, s=5)
        ax_sd.plot([vmin, vmax], [vmin, vmax], 'r-')
        ax_sd.set_xlim(vmin, vmax)
        ax_sd.set_ylim(vmin, vmax)
        ax_sd.yaxis.tick_right()
        ax_sd.yaxis.set_label_position("right")
        ax_sd.set_title(f"Scatter Plot", x=title_x_pos, y=title_y_pos)
        ax_sd.set(xlabel=r'LST Obs [$\degree$ C]')
        ax_sd.set(ylabel=r'LST Pred [$\degree$ C]')

        #Add a Title for the entire plot - effectively the patch information
        fig.suptitle(f"City: {(os.path.basename(inp)).split('.')[0]}, Tile ID: {(os.path.basename(inp)).split('.')[1]}, Patch Index: {(os.path.basename(inp)).split('.')[2]}, Date: {(os.path.basename(inp)).split('.')[3]}, Time: {(os.path.basename(inp)).split('.')[4]}", y = 1.1)
        # add colorbar for taget and prediction
        ax_cbar = plt.subplot(gs[expi+1, 1:3])
        #gs = gridspec.GridSpec(nexp+1, ncols, height_ratios=[1,0.5], width_ratios=[1,1,1,1,1])
        fig.colorbar(target_handle, ax=ax_cbar, use_gridspec=False, fraction=0.5, shrink=0.5, orientation='horizontal')
        ax_cbar.axis('off')

        #add colorbar for error map
        ax_cbar_err = plt.subplot(gs[expi+1, 3])
        #gs = gridspec.GridSpec(nexp+1, ncols, height_ratios=[1,0.5], width_ratios=[1,1,1,1,1])
        fig.colorbar(error_handle, ax=ax_cbar_err, use_gridspec=False, fraction=0.5, orientation='horizontal')
        ax_cbar_err.axis('off')

        #Save the image
        if save_plot == True:
            fname = os.path.join(comp_plots_path, (os.path.splitext(os.path.basename(inp))[0:5][0]) + '_comp_plot_enhance.png')
            print(f"Saving plot ..... {fname}")
            fig.savefig(fname, dpi=400, bbox_inches='tight')
        else:
            continue

    #Display the image
    fig.show()


def load_lst(pred_file):
    """Loads lst prediction files and returns the prediction as a Dataset
        Args:
            pred_file (str): Full path to the geotiff of the lst prediction 
    """
    try:
        with rasterio.open(pred_file) as src:
            lst_data = src.read(1)
            if lst_data is None or np.all(np.isnan(lst_data)):
                print(f"No valid data in {pred_file}")
                return np.zeros((src.height, src.width))  # Return a blank frame
            return lst_data
    except Exception as e:
        print(f"Error loading {pred_file}: {e}")
        return None

def plot_rgb_and_lst(input_file, pred_files, minmax, output_file):
    """Plotting routine
        Args:
            input_file (list): List of input files (geotiffs) to generate animation.
            pred_files (list): List of prediction files (geotiffs) to generate animation.
            minmax (float):  Calculate the temporal min and max of the predictions using the analyze_rasters method
            output_file (str): Animation output filename. Must use the .gif extension. 
    """
    # Create a GridSpec for the layout
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])  # Three columns: RGB, LST, Colorbar

    # Plot RGB tile
    ax_rgb = fig.add_subplot(gs[0, 0])
    if input_file:
        rgb = stack_rgb(input_file)
        ax_rgb.imshow(rgb, aspect='equal', interpolation='none')
        ax_rgb.set_title("RGB Tile", fontdict={'color': 'black'})
        ax_rgb.set_ylabel(f"RGB plot: {os.path.basename(input_file)}", fontsize=12)
        ax_rgb.axis('off')
    else:
        print("No input files found.")

    # Animation of predictions
    ax_lst = fig.add_subplot(gs[0, 1])
    ax_lst.set_title("LST Animation", fontdict={'color': 'black'})

    if pred_files:
        initial_lst = load_lst(pred_files[0])
        img = ax_lst.imshow(initial_lst, norm=Normalize(minmax.global_min, minmax.global_max), cmap='jet', alpha=1)
        ax_lst.axis('off')

        # Create colorbar in its own subplot
        ax_cbar = fig.add_subplot(gs[0, 2])
        cbar = plt.colorbar(img, cax=ax_cbar)  # Assign colorbar to the specified axis
        cbar.set_label('LST ($^{0}C$)', rotation=0, labelpad=20)

        # Animation of the predicted LST
        def update(frame):
            lst = load_lst(pred_files[frame])  # Load LST
            if lst is not None:
                img.set_array(lst)
                ax_lst.set_title(f"LST {os.path.basename(pred_files[frame]).split('.')[2]}.{os.path.basename(pred_files[frame]).split('.')[3]}")
            return img,

        ani = FuncAnimation(fig, update, frames=len(pred_files), interval=500, blit=False, repeat=True)

        # Save the animation
        ani.save(output_file, writer='pillow', fps=1)
        print(f"Animation saved as {output_file}")
    else:
        print("No prediction files found.")

    plt.tight_layout() 

class RasterStats:
    """Class to hold global minimum and maximum raster statistics."""
    def __init__(self, global_min, global_max):
        self.global_min = global_min
        self.global_max = global_max

def get_rasters(pred_files):
    """Get raster data from a list of predicted files.
        Args:
            pred_files (list): List of prediction files (geotiffs) to generate animation.
    """
    band_data = [] 
    for gtfp in pred_files:
        try:
            with rasterio.open(gtfp) as src:
                for band_num in range(1, src.count + 1):
                    band_data.append(src.read(band_num))
        except Exception as e:
            print(f"Error reading {gtfp}: {e}")
    return np.array(band_data)

def calc_stats(band_data_all):
    """Calculate the global minimum and maximum based on 5th and 95th percentiles.
    Args:
        band_data_all: numpy.ndarray()
            An array that contains the data from all the prediction rasters

    """
    if len(band_data_all.shape) == 3:
        band_data_all = np.expand_dims(band_data_all, axis=0)

    all_data_combined = band_data_all.reshape(-1)

    global_min = np.nanpercentile(all_data_combined, 5)
    global_max = np.nanpercentile(all_data_combined, 95)

    return global_min, global_max

def analyze_rasters(inference_path):
    """function to analyze rasters in the given inference path.
    Args:
        inference_path (str): Path to directory where the geotiff predictions are saved.
    """
    pred_files = glob.glob(os.path.join(inference_path, "*_pred.tif"))
    pred_files = sorted(pred_files)  # Sort by date

    if pred_files:
        raster_data = get_rasters(pred_files)
        print("Raster data shape:", raster_data.shape)

        global_min, global_max = calc_stats(raster_data)
        print("5th Percentile (Global Minimum Value):", global_min)
        print("95th Percentile (Global Maximum Value):", global_max)
        
        return RasterStats(global_min, global_max)
    else:
        print("No predicted files found.")
        return None
 

def plot_preprocessed_images(input_file, target_file):
    """Plotting RGB image from stacked inputs generated
    Args:
        input_file (str): Path to stacked input file 
        target_file (str): Path to processed target file
    """
# Plot RGB from stacked inputs

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    
    rgb = stack_rgb(input_file)
    ax_rgb = plt.subplot(1, 2, 1)
    plt.imshow(rgb, alpha=1, aspect='auto', interpolation='none')
    ax_rgb.set_title("RGB tile", x=0.5, y=1.0, fontdict={'color': 'black'})
    ax_rgb.set_ylabel(f"RGB plot: {input_file.split('/')[-1].split('.')[0] +'.'+ input_file.split('/')[-1].split('.')[2] +'.'+ input_file.split('/')[-1].split('.')[3]}")
    ax_rgb.get_xaxis().set_ticklabels([])
    ax_rgb.get_yaxis().set_ticklabels([]);
    
    # Plot lst from processed targets
    lst_band = xr.open_dataset(target_file)
    lst_band = lst_band.rio.reproject("EPSG:4326")
    lst_band['LST (Degrees Celsius)'] = lst_band['band_data']
    lst_band = lst_band.drop_vars(['band_data'])
    
    ax_lst = plt.subplot(1, 2, 2)
    lst_band['LST (Degrees Celsius)'].plot(ax=ax_lst, cmap='jet', add_colorbar=True, robust=True)
    ax_lst.set_title("LST tile", x=0.5, y=1.0, fontdict={'color': 'black'});
    fig.tight_layout()
    fig.show()
