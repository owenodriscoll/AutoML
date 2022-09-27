#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:50:01 2022

@author: owen
"""
# %reset -f

import xsar
from matplotlib import pyplot as plt  
import xarray as xr
import numpy as np
from scipy import signal, ndimage
from tqdm.auto import tqdm
import os
import scipy.ndimage
import glob
import xrft
import pandas as pd

from scipy.ndimage.filters import maximum_filter, minimum_filter
from sklearn.metrics import r2_score

path = '/home/owen/Documents/python scripts'
os.chdir(path)
import equations as eq
import importlib
importlib.reload(eq)

from scipy.spatial import distance
#%%

# path = '/home/owen/Documents/sentinel/wv_test_images_cells'  #wv_test_images_old  wv_test_images_cells
path = '/home/owen/Documents/buoy_data/cells_4_scattering_transform'
os.chdir(path)
sar = []
filenames = glob.glob(path + '/*.nc')
title_list = []
for file in filenames:
    data = xr.open_dataset(file)
    sar.append(data)
    title_list.append(data.name)

#%%

sar_list = []

for i in tqdm(range(500, 1050)):
# for i in tqdm(idx_0 + idx_40 +  idx_80):
# for i in tqdm(df_analysis[df_analysis['RSE_wdir'] > 85].index.values):      # plot those with large errors
# for i in tqdm(df_analysis[(df_analysis.wdir_estimate_in_image_frame%180 <10) | \
#     (df_analysis.wdir_estimate_in_image_frame%180 > 170)].index.values):      # plot those with spectral peak near azimuth
    sar_ds = sar[i]
    
    freq_max = 1 / 600
    freq_min = 1 / 3000
    interpolation = 'linear'
    save_directory = None #'/home/owen/Documents/buoy_data/images/spectral_analysis_31_05/'
    dissip_rate = 1
    label = sar_ds.label  # whether its a Micro Convective Cell or Wind Streaks
    Zi = float(sar_ds.pblh_era5) # !!! CHANGE such that the value from ERA5 pblh is used 
    wdir_era5 = float(sar_ds.wdir_era5) 
    
    # prepare dataset by detrending, filling nans and converting units/coordinates
    sar_ds = eq.ds_prepare(sar_ds)
    
    if np.isnan(sar_ds.sigma0_detrend).sum() != 0:
        print("detrended nanfilled field still contains NaN's, scene skipped")
    
    # calculate cartesian spectrum (of detrended sigma0 field) and coarsely interpolate to polar spectrum 
    PolarSpec_pre, fx_pre, fy_pre = eq.ds_to_polar(sar_ds, interpolation = interpolation)
    
    # use coarse polar spectrum to estiamte orientation of energy and compute wind field
    sar_ds, sum_theta, angle_pre_conversion, energy_dir, wdir_estimate = eq.ds_windfield(sar_ds, PolarSpec_pre, wdir_era5, freq_max = freq_max, freq_min = freq_min)
    
    sar_list.append(sar_ds)



#%% Calculate distances and differences between points on a wind field


epsilon_list = []

# for i in tqdm(range(50)):
for i in tqdm(range(len(sar_list))):
# for i in tqdm([ 10,  23,  30,  39,  57,  80,  86, 153, 162, 165, 173, 215, 225,
#         228, 236, 239, 258, 259, 270, 282, 310, 316, 326, 443, 472, 474,
#         510, 522]):
# for i in tqdm([ 10,  23,  30,  57,  86, 236, 239, 522]):
    
    

# for i in tqdm([10, 22, 45]):
    n = 50
    idx_sar = i
    # sar_list[sar_image].sigma0.plot()
    
    # test = sar_list[sar_image].isel(atrack_m=np.random.randint(0, sar_list[sar_image].atrack_m.size, n),
    #                         xtrack_m=np.random.randint(0, sar_list[sar_image].xtrack_m.size, n))
    sar_image = sar_list[idx_sar].sigma0[:50, :50]
    image_subsample = sar_image.isel(
        atrack_m=np.array([int(x) for x in np.linspace(0, sar_image.atrack_m.size-1, 50)]),
        xtrack_m=np.array([int(x) for x in np.linspace(0, sar_image.atrack_m.size-1, sar_image.xtrack_m.shape[0])])
        )
    
    
    image_subsample_dataframe = image_subsample.to_dataframe()
    list_of_list = np.array(list(image_subsample_dataframe.index))
    coordinates = [tuple(x) for x in list_of_list]
    
    windfield_distance_grid = distance.cdist(coordinates, coordinates, metric='euclidean')
    
    # find diagonal indexes and lower triangle for removal
    low_tri_idx = np.tril_indices(len(windfield_distance_grid))
    
    # set to undesireable indexes to 0
    windfield_distance_grid[low_tri_idx] = 0
    
    # ravel grid and remove all points with 0 distance
    windfield_distance_ravel = np.ravel(windfield_distance_grid)
    windfield_distance = np.delete(windfield_distance_ravel, windfield_distance_ravel == 0)
    
    # -- Repeat but for wind field differencees using same indexes to delete
    windfield = np.array(image_subsample_dataframe.sigma0)
    
    # Calculate absolute differences between each element 
    windfield_difference_grid = windfield[:,None] - windfield
    
    # ravel grid and remove all points with 0 distance
    windfield_difference_ravel = np.ravel(windfield_difference_grid)
    windfield_difference = np.delete(windfield_difference_ravel, windfield_distance_ravel == 0)
    
    df_test = pd.DataFrame()
    df_test['windfield_difference'] = np.ravel(windfield_difference)
    df_test['windfield_distance'] = np.ravel(windfield_distance)
    
    
    # #%%
    
    res = 100
    distances = np.arange(25,30025,res)
    func = [(r'$\overline{(U_1 - U_2)^2}$', lambda x: np.mean(x**2))]
    # -- Group on wind-field DISTANCE in 100 meter bins between 0 and 30km and calculate average wind-field DIFFERENCE
    df_grouped = df_test.groupby(pd.cut(df_test['windfield_distance'], distances))['windfield_difference']
    df_grouped_mean = df_grouped.agg(func)
    S2 = np.squeeze(df_grouped_mean.values)
    r = distances[1:] - res/2 # bin centerpoints
    
    # S = 2*epsilon**(2/3) * r**(2/3) --> epsilon = (S / r**(2/3) / 2)**(3/2)  Kaimal et al (1972)
    epsilon = (S2 / r**(2/3) / 2)**(3/2)
    df_grouped_mean['epsilon'] = epsilon
    df_grouped_mean['bin_center'] = 1/r
    # epsilon_list.append(df_grouped_mean)
    
    plt.plot(1/r, epsilon, linewidth = 0.5, c = 'k'); plt.yscale('log'); plt.xscale('log'); 
    plt.ylabel(r'$\epsilon\ [m^2\ s^{-3}]$ ');plt.xlabel(r'$\xi\ [m^{-1}]$')
    plt.xlim([1/np.max(distances), 1e-2]);# plt.ylim([1e-6, 5e-3]);
    
    func = [(r'$\overline{(U_1 - U_2)^2}$', lambda x: np.mean(x**3))]
    df_grouped = df_test.groupby(pd.cut(df_test['windfield_distance'], distances))['windfield_difference']
    df_grouped_mean = df_grouped.agg(func)
    S3 = np.squeeze(df_grouped_mean.values)
    
    epsilon = -S3 * 5 / 4 / r 
    
    # plt.plot(1/r, epsilon, linewidth = 0.5, c = 'r'); plt.yscale('log'); plt.xscale('log'); 
    # plt.ylabel(r'$\epsilon\ [m^2\ s^{-3}]$ ');plt.xlabel(r'$\xi\ [m^{-1}]$')
    # plt.xlim([1/np.max(distances), 1e-2]); plt.ylim([1e-6, 7e-3]);


    # windfield = sar_image.values
    # max_iter = np.shape(windfield)[0]
    
    # epsilon_S2 = np.zeros(max_iter)
    # epsilon_S3 = np.zeros(max_iter)

    # for j in range(1, max_iter):
    #     nr = j
    #     r = nr * 100
    #     S2 = np.nanmean((windfield[:,nr:]-windfield[:,:-nr])**2)
    #     C2 = 2
    #     epsilon_S2[j] = (S2 / C2 / r**(2/3))**(3/2)
        
    #     S3 = np.nanmean((windfield[:,nr:]-windfield[:,:-nr])**3)
    #     epsilon_S3[j] = S3 * -5 / 4 / r 

    # plt.plot(1/np.arange(100, max_iter*100, 100), epsilon_S2[1:], '--k'); plt.yscale('log'); plt.xscale('log');
    # plt.plot(1/np.arange(100, max_iter*100, 100), epsilon_S3[1:], '--r'); plt.yscale('log'); plt.xscale('log');


plt.figure()
min_diff = []
# for i in range(len(epsilon_list)):
for i in [ 10,  23,  30,  39,  57,  80,  86, 153, 162, 165, 173, 215, 225,
        228, 236, 239, 258, 259, 270, 282, 310, 316, 326, 443, 472, 474,
        510, 522]:
    
    # calculate bin centers of gradient
    diff_bins = 1/(1/epsilon_list[i].bin_center.values[:-1] + np.diff(1/epsilon_list[i].bin_center.values)[0]/2)
    
    # find index of bin center below 1000m
    bin_pre_1000 = np.argmin(1/diff_bins < 1000)
    
    # calculate gradient
    diff = abs(np.diff(np.log10(epsilon_list[i].epsilon.values)))
    
    # calculate average strcture function gradient for wavelengths shorter than 1000m
    min_diff.append(np.mean(diff[:bin_pre_1000]))
    
    # plot gradients
    plt.plot(diff_bins, abs(np.diff(np.log10(epsilon_list[i].epsilon.values))))
    plt.xscale('log'); plt.yscale('log'); 
    
np.argwhere(min_diff< np.array([0.04])).T[0]
    
# test for non-gaussian before applying third order


for i in tqdm([i + 500 for i in [ 10,  23,  30,  57,  86, 236, 239, 522]]):    
    sar_ds = sar[i]
    
    freq_max = 1 / 600
    freq_min = 1 / 3000
    interpolation = 'linear'
    save_directory = None #'/home/owen/Documents/buoy_data/images/spectral_analysis_31_05/'
    dissip_rate = 1
    label = sar_ds.label  # whether its a Micro Convective Cell or Wind Streaks
    Zi = float(sar_ds.pblh_era5) # !!! CHANGE such that the value from ERA5 pblh is used 
    wdir_era5 = float(sar_ds.wdir_era5) 
    
    # prepare dataset by detrending, filling nans and converting units/coordinates
    sar_ds = eq.ds_prepare(sar_ds)
    
    if np.isnan(sar_ds.sigma0_detrend).sum() != 0:
        print("detrended nanfilled field still contains NaN's, scene skipped")
    
    # calculate cartesian spectrum (of detrended sigma0 field) and coarsely interpolate to polar spectrum 
    PolarSpec_pre, fx_pre, fy_pre = eq.ds_to_polar(sar_ds, interpolation = interpolation)
    
    # use coarse polar spectrum to estiamte orientation of energy and compute wind field
    sar_ds, sum_theta, angle_pre_conversion, energy_dir, wdir_estimate = eq.ds_windfield(sar_ds, PolarSpec_pre, wdir_era5, freq_max = freq_max, freq_min = freq_min)
    
    plt.figure()
    sar_ds.windfield.plot()
    print(sar_ds.attrs['L_era5'])


#%%  open LES wind field

path = '/home/owen/Documents/data/surface_wind_eureca.nc'

LES = xr.open_dataset(path)

n = 100
points_y = np.random.randint(0, LES.ym.size, n)
points_x = np.random.randint(0, LES.xm.size, n)

epsilon_list = []

for i in tqdm(range(len(LES.time))):
    idx_time = i
    u = np.array(LES.isel(time = idx_time).u)+2  # -8    +2
    v = np.array(LES.isel(time = idx_time).v)-15  # + 0   -15
    xm = np.array(LES.isel(time = idx_time).xm)
    ym = np.array(LES.isel(time = idx_time).ym)
    
    LES['windfield'] = (('ym', 'xm'), np.sqrt(v**2 + u**2))
    test = LES['windfield'].isel(ym=points_y, xm=points_x)
    
    test_dataframe = test.to_dataframe()
    list_of_list = np.array(list(test_dataframe.index))
    coordinates = [tuple(x) for x in list_of_list]
    
    windfield_distance_grid = distance.cdist(coordinates, coordinates, metric='euclidean')
    
    # find diagonal indexes and lower triangle for removal
    low_tri_idx = np.tril_indices(len(windfield_distance_grid))
    
    # set to undesireable indexes to 0
    windfield_distance_grid[low_tri_idx] = 0
    
    # ravel grid and remove all points with 0 distance
    windfield_distance_ravel = np.ravel(windfield_distance_grid)
    windfield_distance = np.delete(windfield_distance_ravel, windfield_distance_ravel == 0)
    
    # -- Repeat but for wind field differencees using same indexes to delete
    windfield = np.array(test_dataframe.windfield)
    
    # Calculate absolute differences between each element 
    windfield_difference_grid = windfield[:,None] - windfield
    
    # ravel grid and remove all points with 0 distance
    windfield_difference_ravel = np.ravel(windfield_difference_grid)
    windfield_difference = np.delete(windfield_difference_ravel, windfield_distance_ravel == 0)
    
    df_test = pd.DataFrame()
    df_test['windfield_difference'] = np.ravel(windfield_difference)
    df_test['windfield_distance'] = np.ravel(windfield_distance)
    
    
    # #%%
     
    
    
    ##
    res = 100
    distances = np.arange(25,30025,res)
    func = [(r'$\overline{(U_1 - U_2)^2}$', lambda x: np.mean(x**2))]
    # -- Group on wind-field DISTANCE in 100 meter bins between 0 and 30km and calculate average wind-field DIFFERENCE
    df_grouped = df_test.groupby(pd.cut(df_test['windfield_distance'], distances))['windfield_difference']
    df_grouped_mean = df_grouped.agg(func)
    S2 = np.squeeze(df_grouped_mean.values)
    df_grouped_mean
    r = distances[1:] - res/2 # bin centerpoints
    
    # S = 2*epsilon**(2/3) * r**(2/3) --> epsilon = (S / r**(2/3) / 2)**(3/2)
    epsilon = (S2 / r**(2/3) / 2)**(3/2)
    
    plt.plot(1/r, epsilon, linewidth = 0.5, c = 'k'); plt.yscale('log'); plt.xscale('log'); 
    plt.ylabel(r'$\epsilon\ [m^2\ s^{-3}]$ ');plt.xlabel(r'$\xi\ [m^{-1}]$')
    plt.xlim([1/np.max(distances), 1e-2]); plt.ylim([1e-6, 5e-3]);
    
    func = [(r'$\overline{(U_1 - U_2)^2}$', lambda x: np.mean(x**3))]
    df_grouped = df_test.groupby(pd.cut(df_test['windfield_distance'], distances))['windfield_difference']
    df_grouped_mean = df_grouped.agg(func)
    S3 = np.squeeze(df_grouped_mean.values)
    
    epsilon = -S3 * 5 / 4 / r 
    
    # plt.plot(1/r, epsilon, linewidth = 0.5, c = 'r'); plt.yscale('log'); plt.xscale('log'); 
    plt.ylabel(r'$\epsilon\ [m^2\ s^{-3}]$ ');plt.xlabel(r'$\xi\ [m^{-1}]$')
    
    
    # NON ISOTROPIC!
    
    windfield = LES['windfield'].values
    max_iter = np.shape(windfield)[0]
    
    epsilon_S2 = np.zeros(max_iter)
    epsilon_S3 = np.zeros(max_iter)

    for j in range(1, max_iter):
        nr = j
        r = nr * 100
        wind_diff = windfield[:, nr:]-windfield[:, :-nr]
        S2 = np.nanmean(wind_diff**2)
        C2 = 2
        epsilon_S2[j] = (S2 / C2 / r**(2/3))**(3/2)
        
        S3 = np.nanmean(wind_diff**3)
        epsilon_S3[j] = S3 * -5 / 4 / r 

    plt.plot(1/np.arange(100, max_iter*100, 100), epsilon_S2[1:], '--k'); plt.yscale('log'); plt.xscale('log');
    # plt.plot(1/np.arange(100, max_iter*100, 100), epsilon_S3[1:], '--r'); plt.yscale('log'); plt.xscale('log');
    
    plt.xlim([1/np.max(distances), 1e-2]); plt.ylim([1e-6, 5e-3]);



