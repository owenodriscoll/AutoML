#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:10:18 2022

@author: owen
"""

# %reset -f

import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
import os, glob

path = '/home/owen/Documents/python scripts'
os.chdir(path)
import equations_atmos as eqa
import equations as eq
import importlib
importlib.reload(eqa)


#%% perform calculations on WV's imagettes

directory = '/home/owen/Documents/sentinel/wv_images_rolls_wdirwspd_33/'

filenames = glob.glob(directory + '*.nc')

idxes = [371, 321, 302, 176, 139, 122, 121, 112, 110, 72, 49, 20, 15, 346, 304, 175, 134, 120, 96, 81] # very good spectra and swell masked spectra
# idxes = [49,  51,  59,  64,  69,  98, 113, 117, 124, 145, 173, 176, 182, 188, 206, 209, 219, 226, 245, 265, 283, 321, 324, 350, 363, 376, 380, 385, 387]  # window effect < 0.8
# idxes = [36, 45, 85, 146, 166, 168, 175, 205, 220, 244, 252, 254, 298, 319, 375] #var_cartesian > 5
# idxes = [15, 110, 176, 302, 96, 371]
# idxes = [112, 129, 174, 189, 233, 242, 257, 295] # nice +1 prior to -5/3
# 96, 371, good example of swell highjacking spectrum
# 110, 176, 302 nice microscale, 15 nice mesoscale

# -- prepare storage for in loop
df_scenes = []
rows = []

for i in tqdm(idxes):
# for i in tqdm(range(100,400)):
    # 
    try:
        # i = 2
        
        # -- open file
        file = filenames[i]
        sar_ds = xr.open_dataset(file)

        # -- input parameters
        freq_max = 1 / 600  # bandpass upper limit in frequency, i.e. freq_max of 1 / 600 discards wavelengths shorter than 600 m
        freq_min = 1 / 3000  # bandpass lower limit in frequency 
        freq_lower_lim = 1/300
        interpolation = 'linear'  # cartesian to polar interpolation method, 'linear', 'nearest'
        save_directory = None # directory to created image
        label = 'Wind Streaks'  # whether its a 'Micro Convective Cell' or 'Wind Streaks'
        wdir_ref = None # a-priory wind direction, e.g. from ERA5. If no value given its not used
        wdir_ambiguity = sar_ds.attrs['wdir_era5'] # assumed general wind direction used to resolve ambiguity, if not known leave at 0
        Zi_era5 = sar_ds.attrs['pblh_era5'] # era5 boundary layer height, if not known leave at 800
        dissip_rate = 1 # dimensionless energy dissipation rate (semi constant in the mixed layer at 0.5-1.0, varies much more in the mixed layer)
        z = 10  # measurement height
        
        # -- prepare dataset by applying coordinate conversions and adding additional detrended arrays
        sar_ds = eqa.ds_prepare(sar_ds)
        
        # -- compute cartesian spectrum of sigma0 field
        cartesian_spectrum_sigma0, var_cartesian_sigma0, var_beyond_nyquist_sigma0 = eqa.ds_cartesian_spectrum(sar_ds, smoothing = True, parameter = 'sigma0_detrend', scaling  = 'density', detrend=None, window = None, window_correction = None)
            
        # -- interpolate cartesian sigma0 spectrum to polar spectrum
        max_f = cartesian_spectrum_sigma0.f_range.max().values*1
        PolarSpec_sigma_0 = eqa.polar_interpolation(cartesian_spectrum_sigma0, max_f, x_coord = 'f_range', y_coord = 'f_azimuth', Nt = 720, Nf = 600, interpolation = 'linear')
                
        # -- use polar spectrum to compute wind direction and corresponding wind field from sigma0
        sar_ds, sum_theta, angle_pre_conversion, energy_dir, energy_dir_range, wdir =  eqa.ds_windfield(sar_ds, PolarSpec_sigma_0, wdir_ambiguity = wdir_ambiguity, wdir_ref = None, label = label, freq_max = freq_max, freq_min = freq_min)
        
        # -- compute cartesian windfield spectrum 
        cartesian_spectrum, var_cartesian, var_beyond_nyquist = eqa.ds_cartesian_spectrum(sar_ds, smoothing = False, parameter = 'windfield', scaling  = 'density', detrend='constant', window = 'hann', window_correction = 'True')

        # -- average and interpolate polar spectra of tiles to a single polar spectra representative of the entire field
        # -- windfield
        da_polar_mean, da_polar_plot_mean = eqa.tiled_spectra(ds = sar_ds, parameter = 'windfield', tiles = 2)
        # -- sigma0
        da_polar_sigma_mean, _ = eqa.tiled_spectra(ds = sar_ds, parameter = 'sigma0', tiles = 2)
                
        # -- calculate parameters from averaged spectra
        var_windfield = sar_ds['windfield'].var().values*1
        theta_spacing = da_polar_mean.theta.spacing * np.pi / 180 # in radian
        frequency_spacing = da_polar_mean.f.spacing
        
        # -- compute statistics from averaged spectra
        beam1, beam2, var_windfield, beams, var_bandpass, var_highpass, var_lowpass, var_bandpass_beam, var_polar, var_beam, \
        polar_effect, window_effect, low_pass_frac, high_pass_frac, bandpass_frac, frac_beam, density_beam, density_bandpass, density_beam_bandpass, freq_25, freq_50, freq_75, \
        angle_diff_max_min_75_25, angle_diff_max_min_75_50, angle_diff_max_min_50_25, angle_diff_min_theta_75_of_min_freq, angle_diff_min_theta_50_of_min_freq, angle_diff_min_theta_25_of_min_freq,\
        angle_diff_theta_25_of_min_freq_theta_75_of_min_freq, angle_diff_theta_50_of_min_freq_theta_75_of_min_freq, angle_diff_theta_75_of_min_freq_theta_50_of_min_freq, \
        angle_diff_theta_75_of_min_freq_theta_25_of_max_freq, angle_diff_theta_75_of_min_freq_theta_50_of_max_freq, angle_diff_theta_50_of_min_freq_theta_25_of_max_freq, \
        angle_diff_theta_25_of_max_freq_theta_75_of_max_freq, angle_diff_theta_50_of_max_freq_theta_75_of_max_freq, angle_diff_theta_25_of_max_freq_theta_50_of_max_freq \
         = eqa.spectral_calculations(da_polar_mean, theta_spacing, frequency_spacing, var_windfield, var_cartesian, var_beyond_nyquist, angle_pre_conversion = angle_pre_conversion, freq_max = 1 / 600, freq_min = 1 / 3000)
        
        # ----------- Testing-------------
        # -- ORIGINAL with spectra computed as average from several tiles
        spectrum_bandpass_beam = beam2.sel(f = slice(freq_min, freq_max)) * 2 # (times two because there are two beams --> average both)
        bandpass_PSD = spectrum_bandpass_beam*spectrum_bandpass_beam.f
        
        # -- spectra computed from un-tiled image
        da_polar_mean_nonTiled, _ = eqa.tiled_spectra(ds = sar_ds, parameter = 'windfield', tiles = 1)
        beam1_untiled, beam2_untiled, _, beams, var_bandpass, var_highpass, var_lowpass, var_bandpass_beam, var_polar, var_beam, \
        polar_effect, window_effect, low_pass_frac, high_pass_frac, bandpass_frac, frac_beam, density_beam, density_bandpass, density_beam_bandpass, freq_25, freq_50, freq_75, \
        angle_diff_max_min_75_25, angle_diff_max_min_75_50, angle_diff_max_min_50_25, angle_diff_min_theta_75_of_min_freq, angle_diff_min_theta_50_of_min_freq, angle_diff_min_theta_25_of_min_freq,\
        angle_diff_theta_25_of_min_freq_theta_75_of_min_freq, angle_diff_theta_50_of_min_freq_theta_75_of_min_freq, angle_diff_theta_75_of_min_freq_theta_50_of_min_freq, \
        angle_diff_theta_75_of_min_freq_theta_25_of_max_freq, angle_diff_theta_75_of_min_freq_theta_50_of_max_freq, angle_diff_theta_50_of_min_freq_theta_25_of_max_freq, \
        angle_diff_theta_25_of_max_freq_theta_75_of_max_freq, angle_diff_theta_50_of_max_freq_theta_75_of_max_freq, angle_diff_theta_25_of_max_freq_theta_50_of_max_freq \
         = eqa.spectral_calculations(da_polar_mean_nonTiled, theta_spacing, frequency_spacing, var_windfield, var_cartesian, var_beyond_nyquist, angle_pre_conversion = angle_pre_conversion, freq_max = 1 / 600, freq_min = 1 / 3000)
        spectrum_bandpass_beam_unTiled = beam2.sel(f = slice(freq_min, freq_max)) * 2 # (times two because there are two beams --> average both)
        bandpass_PSD_unTiled = spectrum_bandpass_beam_unTiled*spectrum_bandpass_beam_unTiled.f
        
        plt.figure()
        bandpass_PSD_unTiled.sum(dim='f').plot(yscale = 'log')
        bandpass_PSD.sum(dim='f').plot(yscale = 'log')
        
        plt.figure()
        bandpass_PSD_unTiled.sum(dim='theta').plot(yscale = 'log')
        bandpass_PSD.sum(dim='theta').plot(yscale = 'log')
        
        # --------------------------------
    
        # -- compute friction velocity of wind field in loop 1
        U_n = sar_ds['windfield'].median().values*1
        u_star, z_0, Cdn = eqa.loop1(U_n = U_n, z = z)
        
        # -- compute atomospheric parameters from wind field in loop 2
        smoothing = False
        sigma_u, L, B, w_star, w_star_normalised_deviation, corr_fact, H, Zi_estimate, valley_estimate, idx_inertial_min, idx_inertial_max, PSD= \
            eqa.loop2B(U_n=U_n, u_star=u_star, z_0=z_0, Zi=Zi_era5, Cdn=Cdn, PolarSpec=da_polar_mean, label=label,
                        z = z, dissip_rate = dissip_rate, freq_max = freq_max, freq_min = freq_min, freq_lower_lim=freq_lower_lim, smoothing = smoothing)
        
        # -- condense spectra into a singular PSD value --> calculate normalise with frequency, normalise with 
        _, S_windfield_xi_mean, S_windfield_xi_std_norm = eqa.da_PSD(da_polar_mean, idx_inertial_max = idx_inertial_max, idx_inertial_min = idx_inertial_min)
        _, S_sigma0_xi_mean, S_sigma0_xi_std_norm = eqa.da_PSD(da_polar_sigma_mean, idx_inertial_max = idx_inertial_max, idx_inertial_min = idx_inertial_min)
        
        # compute spatial wavelengths of detected peaks
        x_axis = da_polar_mean.f.values
        lambda_peak = 1/x_axis[idx_inertial_max]
        lambda_valley = 1/x_axis[idx_inertial_min]
        
        # store found data
        row = [file, L, sar_ds.attrs['L_era5'], sar_ds.attrs['L'], u_star, sar_ds.attrs['friction_velocity'], wdir, sar_ds.attrs['wdir_estimate'], 
               sar_ds.attrs['wdir_era5'], w_star, w_star_normalised_deviation, Zi_estimate, lambda_peak, lambda_valley, window_effect, var_cartesian, 
               S_windfield_xi_mean, S_windfield_xi_std_norm, S_sigma0_xi_mean, S_sigma0_xi_std_norm]
        rows.append(row)
        
        ##########################################
        #### Energy spectrum contour analysis ####
        ##########################################
    
        speccc = da_polar_plot_mean.sel(f = slice(freq_min, 200)) * da_polar_plot_mean.f
        speccc_ref = da_polar_plot_mean.sel(f = slice(freq_min, freq_max)) * da_polar_plot_mean.f
        cumsum_scaled = (speccc.cumsum(dim = 'f') / speccc_ref.sum() * len(speccc.theta) )
        contours = cumsum_scaled.rolling(theta=1).mean().interpolate_na(dim = 'theta', method= 'linear', fill_value= 'extrapolate')

        percentile_25 = (cumsum_scaled>0.25).argmax(dim = 'f')
        percentile_50 = (cumsum_scaled>0.5).argmax(dim = 'f')
        percentile_75 = (cumsum_scaled>0.75).argmax(dim = 'f')

        mean_25th, median_25th, std_25th, mad_25th = eqa.contourStats(percentile_25)
        mean_50th, median_50th, std_50th, mad_50th = eqa.contourStats(percentile_50)
        mean_75th, median_75th, std_75th, mad_75th = eqa.contourStats(percentile_75)
        
        
        atrack_mean_25th, xtrack_mean_25th = speccc.freq_atrack_m[percentile_25, :], speccc.freq_xtrack_m[percentile_25, :]
        atrack_mean_50th, xtrack_mean_50th = speccc.freq_atrack_m[percentile_50, :], speccc.freq_xtrack_m[percentile_50, :]
        atrack_mean_75th, xtrack_mean_75th = speccc.freq_atrack_m[percentile_75, :], speccc.freq_xtrack_m[percentile_75, :]

        
        
        #####################
        ## Plot spectrum ####
        #####################
        
        if i%1 == np.nan:
            # plot spectrum
            fig = plt.figure(figsize = (24,6))
            
            plt.subplot(1, 3, 1)
            
            hex_list = ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c']
            hex_list = ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494']
            hex_list = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac']
            
            cmap_windfield = eq.get_continuous_cmap(hex_list)
            cbar_kwargs = {'orientation':'vertical','shrink':0.99, 'aspect':40,'label':'Wind speed [$m\ s^{-1}$]'} #, 'anchor' :(1.0, 10.5)}
            sar_ds.windfield.plot(robust = True, cmap = cmap_windfield, cbar_kwargs= cbar_kwargs)
            plt.title('Wind field')
            plt.ylabel(r'Azimuth [$m$]')
            plt.xlabel(r'Range [$m$]')
            
            # ------------------ 2D SPECTRUM ------------------
            plt.subplot(1, 3, 2)
            
            ds_2Dpsd_plot = da_polar_plot_mean
            theta = ds_2Dpsd_plot.theta
            f = ds_2Dpsd_plot.f
            fx = f*np.cos(np.deg2rad(theta))
            fy = f*np.sin(np.deg2rad(theta))
            d_theta_plot = ds_2Dpsd_plot.theta.spacing * np.pi / 180
            var2D = np.log10(ds_2Dpsd_plot*ds_2Dpsd_plot.f* np.prod([ds_2Dpsd_plot.f.spacing, d_theta_plot]))
            vmin = np.nanpercentile(var2D, 50)
            vmax = np.nanpercentile(var2D, 95)
            
            # -- determine colors
            c = var2D
            hex_list = ['#edf8fb', '#b3cde3', '#8c96c6', '#8856a7', '#810f7c']
            cmap_PSD = eq.get_continuous_cmap(hex_list)
            plt.scatter(fx, fy, vmin = vmin, s = 2.5, c = c, vmax = vmax, cmap = cmap_PSD, lw = 0,) # alpha=0.05)
            # plt.colorbar()
            
            plt.title(r'$\mathrm{log}_{10}\left(\mathrm{PSD}\right)$')
            plt.ylabel(r'$\xi$ Azimuth [$m^{-1}$]')
            plt.xlabel(r'$\xi$ Range [$m^{-1}$]')
            
            # -- add bandpass radii, beams and energy dir
            idx_beam1 = (beam1.f == max(beam1.f)) | (beam1.theta == max(beam1.theta)) | (beam1.theta == min(beam1.theta))
            idx_beam2 = (beam2.f == max(beam2.f)) | (beam2.theta == max(beam2.theta)) | (beam2.theta == min(beam2.theta))
            plt.scatter(beam2.freq_xtrack_m.values[idx_beam2], beam2.freq_atrack_m.values[idx_beam2], c = 'chocolate', s = 1, alpha = 0.7)
            plt.scatter(beam1.freq_xtrack_m.values[idx_beam1], beam1.freq_atrack_m.values[idx_beam1], c = 'chocolate', s = 1, alpha = 0.7)
            dx, dy = beam1.freq_atrack_m[-1, beam1.shape[1]//2], beam1.freq_xtrack_m[-1, beam1.shape[1]//2]
            plt.arrow(0, 0, dy, dx, length_includes_head = True, width = 0.0001, color = 'chocolate'); 
            plt.arrow(0, 0, -dy, -dx, length_includes_head = True, width = 0.0001, color = 'chocolate'); 
            eq.circ([200, 1 / freq_max, 1 / freq_min], ['navy', 'navy', 'navy', ], True, linewidth = [3, 3, 3], linestyle = ['-.', '--', '-'])
            
            # -- plot percentiles
            plt.plot(atrack_mean_25th, xtrack_mean_25th, linewidth = 2, linestyle = '-', c = 'black', alpha = 1)
            plt.plot(atrack_mean_50th, xtrack_mean_50th, linewidth = 2, linestyle = '-.', c = 'black', alpha = 1)
            plt.plot(atrack_mean_75th, xtrack_mean_75th, linewidth = 2, linestyle = ':', c = 'black', alpha = 1)
            
            # -- add legend
            legend_elements = [Line2D([0], [0], linestyle='-.', color='navy', label='200m'),
                                Line2D([0], [0], linestyle='--', color='navy', label='600m'),
                                Line2D([0], [0], linestyle='-', color='navy', label='3000m'),
                                Line2D([0], [0], linestyle='-', color='black', label=r'25$^{th}$', markerfacecolor='black', markersize=15),
                                Line2D([0], [0], linestyle='-.', color='black', label=r'50$^{th}$', markerfacecolor='black', markersize=15),
                                Line2D([0], [0], linestyle=':', color='black', label=r'75$^{th}$', markerfacecolor='black', markersize=15),
                                Line2D([0], [0], marker=r'$\leftarrow$', color='w', label='Energy dir.', markerfacecolor='chocolate', markersize=20),
                                Line2D([0], [0], linestyle = '-', color ='chocolate', label="Beam")
                                ]
            # Create the figure
            plt.legend(handles=legend_elements, loc='best', ncol = 1, framealpha =0.99, edgecolor = 'black')
            
            # ----------------- 1D SPECTRUM ------------------- 
            plt.subplot(1, 3, 3)
            
            ds_psd_plot = da_polar_mean# beams
            d_theta = ds_psd_plot.theta.spacing * np.pi / 180
            PSD_plot = ((ds_psd_plot*ds_psd_plot.f).sum(dim = 'theta')) * np.prod([ds_psd_plot.f.spacing, d_theta]) / ds_psd_plot.f.spacing
            # ((ds_polar_mean*ds_polar_mean.f).sum(dim = 'theta')) * np.prod([ds_polar_mean.f.spacing, d_theta]) / ds_polar_mean.f.spacing
            
            # plot complete spectrum in grey
            plt.plot(PSD_plot.f[3:], PSD_plot[3:], c = 'gray', marker='o', markersize=1,  linewidth=0)
            
            # highlight what is believed to be within th einertial subrange
            plt.plot(PSD_plot.isel(f = slice(idx_inertial_max, idx_inertial_min)).f, PSD_plot.isel(f = slice(idx_inertial_max, idx_inertial_min)), c = 'black', marker='o', markersize=1,  linewidth=0)
            ylims = fig.get_axes()[0].get_ylim()
            plt.vlines(x=[freq_max, freq_min], ymin=ylims[0], ymax=ylims[1], colors = ['navy', 'navy'], linestyles = ['--', '-'])
            
            ######## plot -5/3 powerlaw #######
            # select frequency axis
            x_axis_plotting = PSD_plot.f.values
            # select point on frequency axis deemed to be within intertial subrange
            axis_kolmogorov = x_axis_plotting[idx_inertial_max:idx_inertial_min]
            # find value of highest frequency point on frequency axis
            value_kolmogorov = PSD_plot[idx_inertial_min].values
            # some things to prepare the amplitude of the powerlaw slope
            a = 1/x_axis_plotting[idx_inertial_max:idx_inertial_min]**(5/3)
            kolmogorov = value_kolmogorov * a/ (min(a))
            # plot results
            plt.plot(axis_kolmogorov, kolmogorov,'C3--',  linewidth=3)
            
            ######## plot +1 powerlaw #######
            # select frequency axis
            x_axis_plotting = PSD_plot.f.values
            # select point on frequency axis deemed to be within intertial subrange
            idx_3000 = np.argmin(abs(1 / x_axis - 3000))
            axis_plusOne = x_axis_plotting[idx_3000:idx_inertial_max]
            # find value of highest frequency point on frequency axis
            value_plusOne = PSD_plot[idx_inertial_max].values
            # some things to prepare the amplitude of the powerlaw slope
            a = 1/x_axis_plotting[idx_3000:idx_inertial_max]**(-1)
            plusOne = value_plusOne * a/ (max(a))
            # plot results
            plt.plot(axis_plusOne, plusOne,'C3--',  linewidth=3)
            
            plt.title('Sum of all angles')
            plt.yscale('log')
            plt.xscale('log')
            plt.xlim(1e-04, 9e-03)
            plt.ylim(5e-00, 5e03)
            plt.ylabel(r'$S(\xi)$ [$m^{1}s^{-2}$]')
            _ = plt.xlabel(r'$\xi$ [$m^{-1}$]')
            
            _ = plt.suptitle(i)

        
        pass
    except Exception as e:
        print(e)
        pass
    
    
columns = ['file', 'L', 'L_era5', 'L_pre', 'u_star','u_star_pre', 'wdir', 'wdir_pre', 'wdir_era5', 'w_star', 
           'w_star_normalised_deviation', 'Zi_estimate', 'lambda_peak', 'lambda_valley', 'window_effect', 'var_cartesian', 
           'S_U_xi', 'S_U_xi_std', 'S_sigmau_xi', 'S_sigmau_xi_std_norm']

df_results = pd.DataFrame(rows, columns = columns)
df_results['y_test'] = abs(df_results['L_era5'].values)
df_results['y_pred'] = abs(df_results['L_pre'].values)
df_results['y_ML'] = abs(df_results['L'].values)
df_results['abs_error_prev'] =  abs(np.log10(df_results.y_test) - np.log10(df_results.y_ML))
df_results['abs_error_new'] =  abs(np.log10(df_results.y_test) - np.log10(df_results.y_ML))
df_results['inertial_length'] = df_results.lambda_peak - df_results.lambda_valley

df_filt = df_results[(df_results.w_star_normalised_deviation < 0.25) & (df_results.inertial_length > 200)]

hist_steps = 30
title = 'Correction using stacked regression from ERA5 $L$ (rolls)'
x_axis_title = r"|Obukhov length| validation (ERA5)"
_ = eq.plot_envelope(df_filt, hist_steps, title = title, x_axis_title = x_axis_title, alpha = 0.5)


#%% Save or reload

# -- store as csv
# name = 'test_smoothed_spectrum4'
# directory_save = '/home/owen/Documents/data/wv_scenes_dataframes/'
# df_results.to_csv(directory_save + name + '.csv', index = False)

# -- load csv
df = pd.read_csv(directory_save + name + '.csv')
df_results = df#[df.w_star_normalised_deviation < 0.25]

#%%

from sklearn.metrics import median_absolute_error, r2_score
L_est = 'y_pred'
L_val = 'y_test'
L_new = 'y_ML'#'y_pred'  y_ML
# L_val = 'L_era5'
# L_val = 'L_buoy'

df_filt = df_filt #df_results#[(df_results.w_star_normalised_deviation < 0.25) & (df_results.inertial_length > 200) & (df_results.inertial_length > 200)]

# -- calculate performance with respect to validation
print('r2 new: %1.3f' % r2_score(np.log10(abs(df_filt[L_val])), np.log10(abs(df_filt[L_new])))) 
print('MAE new: %1.3f' % median_absolute_error(np.log10(abs(df_filt[L_val])), np.log10(abs(df_filt[L_new]))))

# -- calculate previous performance with respect to validation
print('r2 original: %1.3f' % r2_score(np.log10(abs(df_filt[L_val])), np.log10(abs(df_filt[L_est])))) 
print('MAE original: %1.3f' % median_absolute_error(np.log10(abs(df_filt[L_val])), np.log10(abs(df_filt[L_est]))))

# -- calculate errors for both new and previous methods
df_filt['abs_error_new'] = abs((np.log10(abs(df_filt[L_val])) -  np.log10(abs(df_filt[L_new]))).values)
df_filt['abs_error_pre'] = abs((np.log10(abs(df_filt[L_val])) -  np.log10(abs(df_filt[L_est]))).values)

# -- function to randomly sample with replacement 
def bootstrapping(df, samples = 10000, col_names = ['bootstrap_median', 'bootstrap_mean']):
    
    # -- prepare empty lists to store data to
    bootstrap_median = [] 
    bootstrap_mean = []
    
    # -- calculate statistics on new dataset and store
    for i in range(samples):
        
        # -- randomly sample with replacement from dataframe
        df_rand_samp = df.sample(n=len(df), replace = True)
        
        # -- store data
        bootstrap_median.append(df_rand_samp.median())
        bootstrap_mean.append(df_rand_samp.mean())
        
    data = list(zip(bootstrap_median, bootstrap_mean))
    df_bootstrap = pd.DataFrame(data, columns = col_names)
    return df_bootstrap
    
bootstrap_new = bootstrapping(df_filt['abs_error_new'], samples = 10000, col_names = ['bstrp_median_new', 'bstrp_mean_new'])
bootstrap_prev = bootstrapping(df_filt['abs_error_prev'], samples = 10000, col_names = ['bstrp_median_prev', 'bstrp_mean_prev'])
# (bootstrap_new - bootstrap_prev).hist(bins = 50)

bootstrap_merged = bootstrap_new.merge(bootstrap_prev, left_index = True, right_index = True)
bootstrap_merged.boxplot(column = ['bstrp_median_new', 'bstrp_median_prev', 'bstrp_mean_new', 'bstrp_mean_prev'], notch = True, figsize = (8, 5), whis = (0.05, 90))
# plt.ylim([0,1])
plt.title(r"Expected distributions $\mathrm{log}_{10}(\epsilon)$")



# __________________________________________________________________________________________________
# null hypothesis: no difference between reference errors and new errors
# thus, if we shuffle the errors (i.e. combine reference and new errors and randomly select half)
# N times, we can determine what the distribution of errors for the null hypothesis would be
# and thus what the odds are that median(reference errors) occur

column_concat = list(df_filt['abs_error_new'].values) + list(df_filt['abs_error_pre'].values)
df_new_and_pre = pd.DataFrame(column_concat)

diffs_median = []
samples = 10000
for i in tqdm(range(samples)):
    
    # -- sample the combined dataframe
    df_sampled = df_new_and_pre.sample(n=len(df_filt), replace = False)
    
    # -- calculate metric for sample
    diff_median = df_sampled.median().values[0]*1
    
    # -- store metric
    diffs_median.append(diff_median)

# calculate actual difference between new method and reference method
value_test = (df_filt['abs_error_new']).median()
print(" \n p <= 2.5%% for x <= %1.3f \n p >= 97.5%% for x >= %1.3f \n x = %1.3f"  %(np.percentile(diffs_median, 2.5), np.percentile(diffs_median, 97.5), value_test))

plt.figure(figsize = (8, 5))
heights = plt.hist(diffs_median, bins = 30, density = True)
plt.vlines(np.percentile(diffs_median, 5), 0, max(heights[0]), colors = 'k', label= r'p $\leq$ 5%')
plt.vlines(value_test, 0, max(heights[0]), colors = 'r', label= 'x')
plt.title('Distribution of null hypothesis')
plt.ylabel('Relative frequency [-]')
plt.xlabel('Median of Median Absolute Errors')
plt.legend()

#%% PLOTS AS USED IN THE LATEX DOCUMENT

fig = plt.figure(figsize = (20,4.5))

plt.subplot(1, 3, 1)

hex_list = ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c']
hex_list = ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494']
hex_list = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac']

cmap_windfield = eq.get_continuous_cmap(hex_list)
cbar_kwargs = {'orientation':'vertical','shrink':0.99, 'aspect':40,'label':'Wind speed [$m\ s^{-1}$]'} #, 'anchor' :(1.0, 10.5)}
sar_ds.windfield.plot(robust = True, cmap = cmap_windfield, cbar_kwargs= cbar_kwargs)
# plt.title('Wind field')
plt.title('')
plt.ylabel(r'Azimuth [$m$]')
plt.xlabel(r'Range [$m$]')

# ------------------ 2D SPECTRUM ------------------
fig = plt.figure(figsize = (16,4.5))
plt.subplot(1, 3, 2)

ds_2Dpsd_plot = da_polar_plot_mean
theta = ds_2Dpsd_plot.theta
f = ds_2Dpsd_plot.f
fx = f*np.cos(np.deg2rad(theta))
fy = f*np.sin(np.deg2rad(theta))
d_theta_plot = ds_2Dpsd_plot.theta.spacing * np.pi / 180
var2D = np.log10(ds_2Dpsd_plot*ds_2Dpsd_plot.f* np.prod([ds_2Dpsd_plot.f.spacing, d_theta_plot]))
vmin = np.nanpercentile(var2D, 50)
vmax = np.nanpercentile(var2D, 95)

# -- determine colors
c = var2D
hex_list = ['#edf8fb', '#b3cde3', '#8c96c6', '#8856a7', '#810f7c']
cmap_PSD = eq.get_continuous_cmap(hex_list)
plt.scatter(fx, fy, vmin = vmin, s = 2.5, c = c, vmax = vmax, cmap = cmap_PSD, lw = 0,) # alpha=0.05)
# plt.colorbar()

# plt.title(r'$\mathrm{log}_{10}\left(\mathrm{PSD}\right)$')
plt.ylabel(r'$\xi$ Azimuth [$m^{-1}$]')
plt.xlabel(r'$\xi$ Range [$m^{-1}$]')

# ------------------ 2D SPECTRUM ANNOTATIONS ------------------
fig = plt.figure(figsize = (16,4.5))
plt.subplot(1, 3, 3)
# -- add bandpass radii, beams and energy dir
idx_beam1 = (beam1.f == max(beam1.f)) | (beam1.theta == max(beam1.theta)) | (beam1.theta == min(beam1.theta))
idx_beam2 = (beam2.f == max(beam2.f)) | (beam2.theta == max(beam2.theta)) | (beam2.theta == min(beam2.theta))
plt.scatter(beam2.freq_xtrack_m.values[idx_beam2], beam2.freq_atrack_m.values[idx_beam2], c = 'chocolate', s = 1, alpha = 0.7)
plt.scatter(beam1.freq_xtrack_m.values[idx_beam1], beam1.freq_atrack_m.values[idx_beam1], c = 'chocolate', s = 1, alpha = 0.7)
dx, dy = beam1.freq_atrack_m[-1, beam1.shape[1]//2], beam1.freq_xtrack_m[-1, beam1.shape[1]//2]
plt.arrow(0, 0, dy, dx, length_includes_head = True, width = 0.0001, color = 'chocolate'); 
plt.arrow(0, 0, -dy, -dx, length_includes_head = True, width = 0.0001, color = 'chocolate'); 
eq.circ([200, 1 / freq_max, 1 / freq_min], ['navy', 'navy', 'navy', ], True, linewidth = [3, 3, 3], linestyle = ['-.', '--', '-'])

# -- plot percentiles
plt.plot(atrack_mean_25th, xtrack_mean_25th, linewidth = 2, linestyle = '-', c = 'black', alpha = 1)
plt.plot(atrack_mean_50th, xtrack_mean_50th, linewidth = 2, linestyle = '-.', c = 'black', alpha = 1)
plt.plot(atrack_mean_75th, xtrack_mean_75th, linewidth = 2, linestyle = ':', c = 'black', alpha = 1)

# -- add legend
legend_elements = [Line2D([0], [0], linestyle='-.', color='navy', label='200m'),
                   Line2D([0], [0], linestyle='--', color='navy', label='600m'),
                   Line2D([0], [0], linestyle='-', color='navy', label='3000m'),
                   Line2D([0], [0], linestyle='-', color='black', label=r'25$^{th}$', markerfacecolor='black', markersize=15),
                   Line2D([0], [0], linestyle='-.', color='black', label=r'50$^{th}$', markerfacecolor='black', markersize=15),
                   Line2D([0], [0], linestyle=':', color='black', label=r'75$^{th}$', markerfacecolor='black', markersize=15),
                   Line2D([0], [0], marker=r'$\leftarrow$', color='w', label='Energy dir.', markerfacecolor='chocolate', markersize=20),
                   Line2D([0], [0], linestyle = '-', color ='chocolate', label='Beam'),
                   ]
# Create the figure
plt.legend(handles=legend_elements, loc='best', ncol = 2, framealpha =0.99, edgecolor = 'black', borderpad = 0.1)

# plt.title(r'Annotations')
plt.ylabel(r'$\xi$ Azimuth [$m^{-1}$]')
plt.xlabel(r'$\xi$ Range [$m^{-1}$]')


# ------------------ 1D SPECTRUM ------------------
fig = plt.figure(figsize = (16,4.5))
plt.subplot(1, 3, 3)

ds_psd_plot = da_polar_mean
d_theta = ds_psd_plot.theta.spacing * np.pi / 180
PSD_plot = ((ds_psd_plot*ds_psd_plot.f).sum(dim = 'theta')) * np.prod([ds_psd_plot.f.spacing, d_theta]) / ds_psd_plot.f.spacing
# ((ds_polar_mean*ds_polar_mean.f).sum(dim = 'theta')) * np.prod([ds_polar_mean.f.spacing, d_theta]) / ds_polar_mean.f.spacing

# -- plot peak and trough
plt.scatter(x_axis_plotting[idx_inertial_max], PSD_plot.values[idx_inertial_max]/1.4, marker = "^", c = 'navy', s = 50)
plt.scatter(x_axis_plotting[idx_inertial_min], PSD_plot.values[idx_inertial_min]*1.4, marker = "v", c = 'navy', s = 50)

# plot complete spectrum in grey
plt.plot(PSD_plot.f[3:], PSD_plot[3:], c = 'gray', marker='o', markersize=2,  linewidth=0)
plt.xlim(1e-04, 5e-03)
plt.ylim(1e1, 5e03)

# highlight what is believed to be within th einertial subrange
plt.plot(PSD_plot.isel(f = slice(idx_inertial_max, idx_inertial_min)).f, PSD_plot.isel(f = slice(idx_inertial_max, idx_inertial_min)), c = 'black', marker='o', markersize=2,  linewidth=0)
ylims = fig.get_axes()[0].get_ylim()
plt.vlines(x=[freq_max, freq_min], ymin=ylims[0], ymax=ylims[1], colors = ['navy', 'navy'], linestyles = ['--', '-'], linewidth = [3, 3])

######## plot -5/3 powerlaw #######
# select frequency axis
x_axis_plotting = PSD_plot.f.values
# select point on frequency axis deemed to be within intertial subrange
axis_kolmogorov = x_axis_plotting[idx_inertial_max:idx_inertial_min]
# find value of highest frequency point on frequency axis
value_kolmogorov = PSD_plot[idx_inertial_min].values
# some things to prepare the amplitude of the powerlaw slope
a = 1/x_axis_plotting[idx_inertial_max:idx_inertial_min]**(5/3)
kolmogorov = value_kolmogorov * a/ (min(a))
# plot results
plt.plot(axis_kolmogorov, kolmogorov,'C3--',  linewidth=3)


# plt.title('Sum of all angles')
plt.yscale('log')
plt.xscale('log')


legend_elements = [Line2D([0], [0], linestyle='-', color='navy', label='3000m'),
                   Line2D([0], [0], linestyle='--', color='navy', label='600m'),
                   Line2D([0], [0], linestyle='--', color='C3', label=r'$-5/3$'),
                   Line2D([0], [0], marker='^', color='w', label='Spectral peak', markerfacecolor='navy', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label=r'S($\xi$)', markerfacecolor='grey', markersize=5),
                   Line2D([0], [0], marker='o', color='w', label='Estimated \ninertial sub.', markerfacecolor='black', markersize=5),
                   Line2D([0], [0], marker='v', color='w', label='Spectral trough', markerfacecolor='navy', markersize=10),
                   ]
plt.legend(handles=legend_elements, loc='upper left', ncol = 2, framealpha =0.99, edgecolor = 'black', borderpad = 0.2)


plt.ylabel(r'$S(\xi)$ [$m^{3}s^{-2}$]')
_ = plt.xlabel(r'$\xi$ [$m^{-1}$]')

# _ = plt.suptitle(i)
# fig.tight_layout()

# name = 'test'
# fig.savefig('/home/owen/Documents/results/paper/' + name + '.png', bbox_inches='tight', dpi=240) 