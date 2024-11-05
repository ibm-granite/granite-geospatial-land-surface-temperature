---
license: apache-2.0
---

#  Model Card for granite-geospatial-land-surface-temperature

<p align="center" width="100%">
<img src="https://github.com/ibm-granite/granite-geospatial-land-surface-temperature/blob/main/assets/images/Johannesburg_summer_lst_animation.gif?raw=true" width="800">
</p>

The granite-geospatial-land-surface-temperature model is a fine-tuned geospatial foundation model for predicting the land surface temperature (LST) using satellite imagery along with climate statistics. Excessive urban heat has been shown to have adverse effects across a range of dimensions, including increased energy demand, severe heat stress on human and non-human populations, and worse air and water quality. As global cities become more populous with increasing rates of urbanization, it is crucial to model and understand urban temperature dynamics and its impacts. Characterizing and mitigating Urban Heat Island (UHI) effects is dependent on the availability of high-resolution (spatial and temporal) LST data. This model is fine-tuned using a combination of Harmonised Landsat Sentinel-2 [(HLS L30)](https://hls.gsfc.nasa.gov/products-description/l30/) and ECMWF Reanalysis v5 [(ERA5-Land)](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview) $2 m$ near-surface air temperature ($T_{2m}$) datasets across 28 global cities from varying hydroclimatic zones for the period 2013-2023. 

<p align="center" width="100%">
<img src="https://github.com/ibm-granite/granite-geospatial-land-surface-temperature/blob/main/assets/images/cities_map2.png?raw=true" width="800">
</p>

## How to Get Started with the Model

This model was trained using [Terratorch](https://github.com/IBM/terratorch).

We make the weights as well as the configuration file that defines it available.

You can use it easily with Terratorch through:

```python
from terratorch.cli_tools import LightningInferenceModel

ckpt_path = hf_hub_download(repo_id="ibm-granite/granite-geospatial-land-surface-temperature", filename="LST_model.ckpt")
config_path = hf_hub_download(repo_id="ibm-granite/granite-geospatial-land-surface-temperature", filename="config.yaml")

model = LightningInferenceModel.from_config(config_path, ckpt_path)

inference_results, input_file_names = model.inference_on_dir(<input_directory>)
```

For more details, check out the tutorials below which guide the user through the three functionalities:

1. Check out the [Getting Started Notebook!](https://github.com/ibm-granite/granite-geospatial-land-surface-temperature/blob/main/notebooks/1_getting_started.ipynb).

2. For data download and data pre-processing to create your own dataset check out the [Download Notebook!](https://github.com/ibm-granite/granite-geospatial-land-surface-temperature/blob/main/notebooks/2_download_data.ipynb) and the [Preprocessing Notebook!](https://github.com/ibm-granite/granite-geospatial-land-surface-temperature/blob/main/notebooks/3_preprocess_data.ipynb).

3. For Tweening (Temporal Gap-Filling) check out the [Introuction to LST Tweening Notebook!](https://github.com/ibm-granite/granite-geospatial-land-surface-temperature/blob/main/notebooks/4_introduction_to_LST_Tweening.ipynb) for a tutorial on how to implement Tweening and the [Tweening Data Preparation Notebook!](https://github.com/ibm-granite/granite-geospatial-land-surface-temperature/blob/main/notebooks/5_tweening_data_preparation.ipynb) for a tutorial on preparing the data for Tweening.

## Model Description

The granite-geospatial-land-surface-temperature model is a geospatial foundation model that has been fine-tuned using HLS L30 and ERA5-Land data to predict LST at a high spatial resolution ($30 m$) and high temporal frequency (hourly). The fine-tuned granite-geospatial-land-surface-temperature model incorporates a Shifted Windowing (SWIN) Transformer architecture and leverages the IBM Earth Observation Foundation Model, “Prithvi-SWIN-L” as the base foundation model. For fine-tuning, we used a SWIN backbone with unfrozen pre-trained weights for the encoder and a decoder that comprised of a Unified Perceptual Parsing for Scene Understanding (UperNet) regression head with an auxiliary 1-layer Convolution regression head and a Linear final activation layer. 

More details on the base foundation model can be found in this [paper](https://arxiv.org/abs/2310.18660)

## Model Application

**Temporal Gap Filling (Tweening):** <br>
We present an application of the granite-geospatial-land-surface-temperature model for temporal gap filling (“Tweening” or in betweening). This approach attempts to solve for the temporal limitations in LST observations by synthesizing hourly inputs of stacked HLS and ERA5 temperature statistics. 

For more details on this approach, refer to:
- [Introduction to LST Tweening](https://github.com/ibm-granite/granite-geospatial-land-surface-temperature/blob/main/notebooks/4_introduction_to_LST_Tweening.ipynb)

## Model Releases (along with the branch name where the models are stored):

- **tag v1 —** - 05/11/2024

- Stay tuned for more models!
 
### Model Sources

- **Repository:** https://github.com/ibm-granite/granite-geospatial-land-surface-temperature
- **Paper (UHI):** https://ieeexplore.ieee.org/document/10641750 - we have since extended on this approach by training on multiple cites to downscale to $30 m$ resolution LST. We have also included functionality for temporal gap filling, "Tweening". 
- **Paper (foundation model):** https://arxiv.org/abs/2310.18660 

### External Blogs
- https://www.ibm.com/blog/ai-battle-extreme-heat/

## Training Data

The model was trained on a collection of HLS and ERA5 datasets acquired for the period 2013-2023:
- Harmonized Landsat-Sentinel 2 (HLS) L30: https://hls.gsfc.nasa.gov/products-description/l30/
    - Citation and Attribution: Masek, J., J. Ju, J. Roger, S. Skakun, E. Vermote, M. Claverie, J. Dungan, Z. Yin, B. Freitag, C. Justice.  HLS Sentinel-2 Multi-spectral Instrument Surface Reflectance Daily Global 30m v2.0. 2021, distributed by NASA EOSDIS Land Processes Distributed Active Archive Center, https://doi.org/10.5067/HLS/HLSS30.002. (Accessed on 24-OCT-2024).

- ERA5 Land 2m_temperature: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
    - Citation and Attribution: Muñoz Sabater, J. (2019): ERA5-Land hourly data from 1950 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.e2161bac (Accessed on 24-OCT-2024).

For fine-tuning, the model requires stacked input patches of size 224 x 224, which consists of the 6 HLS band layers [B02-B07] and an ERA5 2m temperature layer. We filter through HLS acquisitions based on cloud cover, crop the stacked inputs to the corresponding city bounding box and process the inputs for a specified percentage of invalid pixels across patches. Output patches are written out with a coordinate reference system (CRS) matching the UTM projection of the city and a timestamp converted from UTC to local time. LST targets are processed from HLS bands following a split-window algorithm, these are then processed to obtain target patches. 

For more details on the download and preprocessing pipelines used to produce the fine-tuning and inference datasets, please refer to:

- [Download](https://github.com/ibm-granite/granite-geospatial-land-surface-temperature/blob/main/notebooks/2_download_data.ipynb)

- [Preprocessing](https://github.com/ibm-granite/granite-geospatial-land-surface-temperature/blob/main/notebooks/3_preprocess_data.ipynb)

## Model Card Authors

Muaaz Bhamjee, Zaheed Gaffoor, Tamara Govindasamy, Craig Mahlasi, Etienne Vos, Mangaliso Mngomezulu, Gciniwe Baloyi, Sibusisiwe Makhanya

## IBM Public Repository Disclosure: 

All content in this repository including code has been provided by IBM under the associated 
open source software license and IBM is under no obligation to provide enhancements, 
updates, or support. IBM developers produced this code as an 
open source project (not as an IBM product), and IBM makes no assertions as to 
the level of quality nor security, and will not be maintaining this code going forward.