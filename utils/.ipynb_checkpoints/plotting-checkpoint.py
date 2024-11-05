
import pandas as pd
import matplotlib.pyplot as plt 
import os
import glob
import xarray as xr
import rioxarray
import numpy as np
import shutil
import pandas as pd
import pylab as plb
from collections import Counter
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import pylab as plb
from matplotlib import cm
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
from IPython.display import Image
import matplotlib
import rasterio

def mae(pred, truth):
    abs_diff = abs(pred-truth)
    mae = abs_diff.mean(dim=['x','y'], skipna=True)
    return mae

def norm(array):
    return (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
    #return (array - np.nanmean(array)) / (np.nanstd(array)).astype(int)

def max_norm(array):
    return (array) / (np.nanmax(array))
    #return (array - np.nanmean(array)) / (np.nanstd(array)).astype(int)

def min_max(exp, pred_list, ds_pred, patch_exp_pred_paths):
    min_ = []
    max_ = []
    
    for patch_exp_pred_path in patch_exp_pred_paths:
        min_e = np.nanmin(np.squeeze(ds_pred['band_data'].values))
        min_.append(min_e)
        max_e = np.nanmax(np.squeeze(ds_pred['band_data'].values))
        max_.append(max_e)

    return min_, max_e

def stack_rgb(stacked_inputs_path):
    with rasterio.open(stacked_inputs_path) as src:
        blue_band = src.read(1)
        green_band = src.read(2)
        red_band = src.read(3)
        # normalize these bands
        red_normalized = norm(red_band)
        green_normalized = norm(green_band)
        blue_normalized = norm(blue_band)
        rgb_normalized = np.dstack((red_normalized, green_normalized, blue_normalized))
        rgb_normalized = xr.DataArray(rgb_normalized)
    
    return rgb_normalized

def unique_cities(path):
    cities = []

    patches = sorted(glob.glob(os.path.join(path, '*.tif')))

    for tp in patches[:]:
        city = os.path.basename(tp.split('.')[0])
        cities.append(city)
    
    unique_cities = np.unique(np.array(cities))
    unique_cities = unique_cities.tolist()
    
    return unique_cities

def pred_patches(target_patches_path):

    test_patches = sorted(glob.glob(os.path.join(target_patches_path, '*.tif')))
    pred_p = []

    for pred in test_patches:
            pred_ = os.path.basename(pred)
            pred_ = pred_.replace("lst", "inputs_pred")
            pred_p.append(pred_)
    
    return pred_p

def city_list(target_patches_path, city):

    patches = pred_patches(target_patches_path)
    result = []

    for i in patches[:]:
        if i.split(".")[0] == city:
           result.append(i)
    return result

def target_list(target_patches_path, city):

    targ_list = []
    
    for tf in city_list(target_patches_path, city):
    
        print
        target_file = tf.replace("inputs_pred", "lst")
        
        targ_list.append(target_file)
    
    return targ_list

def city_mae(inference_path, target_patches_path, city):
    
    preds = city_list(target_patches_path, city)
    targets = target_list(target_patches_path, city)
    
    mae_city = []
    
    for idx, pred in enumerate(preds):

        try:
            os.path.isfile(os.path.join(inference_path, pred))
            ds_target = xr.open_dataset(os.path.join(target_patches_path, targets[idx]))
            ds_target = ds_target.where(ds_target['band_data'] != -9999., np.nan)
            ds_pred = xr.open_dataset(os.path.join(inference_path, pred))
            ds_pred = ds_pred.where(~np.isnan(ds_target['band_data']), np.nan)
            
            #Calculate MAE:
            ds_mae = mae(ds_pred, ds_target)
            mae_val = ds_mae.to_array()
            mae_val = round(mae_val.item(), 3)
            mae_city.append(mae_val)
        except: 
            print(f"Corresponding prediction for target file {targets[idx]} does not exist. Skipping...")
    
    mae_city = np.asarray(mae_city)
    
    return mae_city

def all_cities_mae(inference_path, target_patches_path):

    city_maes = []
    test_cities = unique_cities(target_patches_path)

    for cities in test_cities:
        np_city = city_mae(inference_path, target_patches_path, cities)
        city_maes.append(np_city)
    return city_maes

def plot_box_plot(target_patches_path, inference_path, result_path, save_plot):

    """Plotting routine

    Args:
        target_patches_path (str): Path to directory with geotiff target patches.
        inference_path (str): Path to directory where the geotiff predictions are saved.
        result_path (str): Path to directory where the box plot must be saved.
        save_plot (bool): "True" will display and save the generated plot. "False" will only display the plot.
    """

    #Create Plots Save Directory
    if os.path.exists(result_path):
        print("Results directory exits!")
        print('\n')
    else: 
        print("Creating Results directory...")
        print('\n')
        os.makedirs(result_path) 
    #Subdirectory to store comparison plots
    comp_plots_path = os.path.join(result_path, 'comparison_plots')
    if os.path.exists(comp_plots_path):
        print("Comparison plots directory exits!")
        print('\n')
    else: 
        print("Creating comparison plots directory...")
        print('\n')
        os.makedirs(comp_plots_path)

    labels = unique_cities(target_patches_path)
    data = all_cities_mae(inference_path, target_patches_path)

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, patch_artist=True, labels=labels)
    plt.grid(True, alpha=0.5)
    plt.title(f"Box plots for unseen timestamps and cities")
    plt.xlabel('city')
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.ylabel(r'MAE [LST $^{o}C$]')
    if save_plot == True:
        plt.savefig(os.path.join(comp_plots_path, f"boxplot.png"), dpi=400)
        plt.show()
    else:
        plt.show()

def plot_violin_plot(target_patches_path, inference_path, result_path, save_plot):

    """Plotting routine

    Args:
        target_patches_path (str): Path to directory with geotiff target patches.
        inference_path (str): Path to directory where the geotiff predictions are saved.
        result_path (str): Path to directory where the box plot must be saved.
        save_plot (bool): "True" will display and save the generated plot. "False" will only display the plot.
    """

    #Create Plots Save Directory
    if os.path.exists(result_path):
        print("Results directory exits!")
        print('\n')
    else: 
        print("Creating Results directory...")
        print('\n')
        os.makedirs(result_path) 
    #Subdirectory to store comparison plots
    comp_plots_path = os.path.join(result_path, 'comparison_plots')
    if os.path.exists(comp_plots_path):
        print("Comparison plots directory exits!")
        print('\n')
    else: 
        print("Creating comparison plots directory...")
        print('\n')
        os.makedirs(comp_plots_path)

    labels = unique_cities(target_patches_path)
    labels.insert(0, '')
    data = all_cities_mae(inference_path, target_patches_path)

    plt.figure(figsize=(8, 6))
    plt.violinplot(data, showmeans=True)
    plt.grid(True, alpha=0.5)
    plt.title(f"Violin plots for unseen timestamps and cities")
    plt.xlabel('city')
    plt.gca().set_xticks(np.arange(len(labels)), labels=labels)
    #plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.ylabel(r'MAE [LST $^{o}C$]')
    if save_plot == True:
        plt.savefig(os.path.join(comp_plots_path, f"violinplot.png"), dpi=400)
        plt.show()
    else:
        plt.show()

def plot_rgb_lst_distribution_scatter(patches_tiles, target_patches_path, inference_path, result_path, input_path, save_plot):
    """Plotting routine

        Args:
            patches_tiles (list): List of input patches or tiles to generate comaprison plots for.
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
        print('\n')
    else: 
        print("Creating Results directory...")
        print('\n')
        os.makedirs(result_path) 
    #Subdirectory to store comparison plots
    comp_plots_path = os.path.join(result_path, 'comparison_plots')
    if os.path.exists(comp_plots_path):
        print("Comparison plots directory exits!")
        print('\n')
    else: 
        print("Creating comparison plots directory...")
        print('\n')
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

        red_band[red_band==-9999.] = np.nan
        green_band[green_band==-9999.] = np.nan
        blue_band[blue_band==-9999.] = np.nan

        red_normalized = max_norm(red_band)
        green_normalized = max_norm(green_band)
        blue_normalized = max_norm(blue_band)
        rgb_normalized = np.dstack((red_normalized, green_normalized, blue_normalized))
        rgb_normalized = xr.DataArray(rgb_normalized)

        ds_mae = mae(ds_pred, ds_target)
        mae_val = ds_mae.to_array()
        mae_val = round(mae_val.item(), 3)

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
        gs = gridspec.GridSpec(nexp+1, ncols, width_ratios=[1,1,1,1,1,1])
        fig.colorbar(target_handle, ax=ax_cbar, use_gridspec=True, orientation='horizontal')
        ax_cbar.axis('off')

        #add colorbar for error map
        ax_cbar_err = plt.subplot(gs[expi+1, 3])

        fig.colorbar(error_handle, ax=ax_cbar_err, use_gridspec=True, orientation='horizontal')
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

def plot_preprocessed_images(input_file, target_file):
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