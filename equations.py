#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 07:22:33 2022

@author: owen
"""
import pandas as pd
import xarray as xr
import xsar
import xsarsea
import xrft
from scipy import ndimage
from scipy import signal
import cv2

#%% calculations

def MAD(data):
    median = np.nanmedian(data)
    MAD = np.nanmedian(abs(data - median))
    return MAD

def wdir_errors(wdir_estimate, wdir_validation):
    Error_wdir = wdir_estimate - wdir_validation
    Error_wdir = np.where(Error_wdir>=270, Error_wdir - 360, Error_wdir)
    Error_wdir = np.where(Error_wdir>=90, Error_wdir - 180, Error_wdir)
    Error_wdir = np.where(Error_wdir<=-270, Error_wdir + 360, Error_wdir)
    Error_wdir = np.where(Error_wdir<=-90, Error_wdir + 180, Error_wdir)
    return Error_wdir

    
# used for weighting w* values in inertial subrange
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and weighted standard deviation.
    """
    average = np.average(values, weights = weights)
    variance = np.average((values - average)**2, weights = weights)
    return (average, np.sqrt(variance))

#%% plot data

def plottings(image, samplerate = 50, minP = 1, maxP = 99, unit = 'metres', title= 'title', cmap = 'Greys_r'):
    if unit == 'metres':
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        plt.title(title)
        plt.ylabel('Azimuth [Km]')
        plt.xlabel('Range [Km]')
        vmin= np.nanpercentile(image,minP)
        vmax= np.nanpercentile(image,maxP)
        xend = image.shape[1]*samplerate/1000  #divide by 1000 to go from metre to km
        yend = image.shape[0]*samplerate/1000  #divide by 1000 to go from metre to km
        cbar1 = ax.imshow(image, vmin = vmin, vmax = vmax, cmap=cmap, origin = 'lower', extent=[0,xend,0,yend])
        plt.colorbar(cbar1, fraction=0.031, pad=0.05)

    if unit == 'frequency':
        from matplotlib.ticker import MaxNLocator
        pi = 3.1415926535
        yaxis = 2*pi*(1/((1/np.arange(1,np.shape(image)[0]//2+1))*(2*samplerate*np.shape(image)[0]//2)))
        xaxis = 2*pi*(1/((1/np.arange(1,np.shape(image)[1]//2+1))*(2*samplerate*np.shape(image)[1]//2)))

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.yaxis.set_major_locator(MaxNLocator(8))
        ax.xaxis.set_major_locator(MaxNLocator(8)) 
        plt.title(title)
        plt.ylabel(r'$k$ [$rad\ m^{-1}$]')
        plt.xlabel(r'$k$ [$rad\ m^{-1}$]')
        vmin=np.nanpercentile(image,minP)
        vmax= np.nanpercentile(image,maxP)
        cbar1 = ax.imshow(image, vmin = vmin, vmax = vmax, cmap=cmap, origin = 'lower', extent=[-yaxis[-1],yaxis[-1],-xaxis[-1],xaxis[-1]])
        plt.colorbar(cbar1,ax=ax, fraction=0.031, pad=0.05)
        # plt.locator_params(axis='y', nbins=9)
        # plt.locator_params(axis='x', nbins=9)
        
def plot_2DPS(PS, x_axis_samplerate, y_axis_samplerate):
    
    vmin = np.percentile(10*np.log10(PS),50)
    vmax = np.percentile(10*np.log10(PS),99.0)
    
    x = np.shape(PS)[1]
    y = np.shape(PS)[0]
    x_axis = 1/((1/np.arange(1, x // 2+1))*(2*x_axis_samplerate * x //2))
    y_axis = 1/((1/np.arange(1, y // 2+1))*(2*y_axis_samplerate * y //2))
    
    fig, (ax1) = plt.subplots(1, 1,figsize=(8,6))
    plt.title('2D PSD')
    plt.ylabel(r'$\lambda$ [$ m^{-1}$]')
    plt.xlabel(r'$\lambda$ [$ m^{-1}$]')
    cbar1 = ax1.imshow(10*np.log10(PS), vmin=vmin, cmap = 'inferno', vmax=vmax , origin = 'lower', extent=[-x_axis[-1],x_axis[-1],-y_axis[-1],y_axis[-1]])
    # import scipy.ndimage
    # cbar1 = ax1.imshow(scipy.ndimage.filters.gaussian_filter(10*np.log10(psd2D), 3, mode='constant'), vmin=vmin, vmax=vmax , origin = 'lower', extent=[-axis[-1],axis[-1],-axis[-1],axis[-1]])
    
    plt.colorbar(cbar1,ax=ax1)

#%% 

import numpy as np
from matplotlib import pyplot as plt  

def twoDPS(image, samplerate, plotting = False):
    """
    Calculate 2D power spectrum of input NRCS data 
    
    """
    
    F1 = np.fft.fft2(np.array(image))
    # low spatial frequencies are in the center of the 2D fourier transformed image.
    F2 = np.fft.fftshift( F1 )
    # Calculate a 2D power spectrum
    psd2D = np.abs( F2)**2
    
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]

    if plotting == True:
        vmin = np.percentile(10*np.log10(psd2D),50)
        vmax = np.percentile(10*np.log10(psd2D),99.0)
        pi = 3.1415926535
        axis = 2*pi*(1/((1/np.arange(1, h // 2+1))*(2*samplerate* h //2)))
        
        fig, (ax1) = plt.subplots(1, 1,figsize=(8,6))
        plt.title('Two-dimensional power spectra [dB]')
        plt.ylabel(r'$k$ [$rad\ m^{-1}$]')
        plt.xlabel(r'$k$ [$rad\ m^{-1}$]')
        cbar1 = ax1.imshow(10*np.log10(psd2D), vmin=vmin, vmax=vmax , origin = 'lower', extent=[-axis[-1],axis[-1],-axis[-1],axis[-1]])
        plt.colorbar(cbar1,ax=ax1)
    return psd2D 

#%% spectral circles


def circ(radii, circle_colors = 'w', legend = False, linewidth = 2.5, linestyle = '--'):
    
    pi = 3.1415926535
    
    def multi_circ(radius, color, legend, linewidth, linestyle):
        circx = np.cos(np.arange(0,2*pi,0.01)) * 1 / radius
        circy = np.sin(np.arange(0,2*pi,0.01)) * 1 / radius
    
        if legend == False:
            plt.plot(circx,circy, c = color, linestyle = linestyle, linewidth= linewidth)
            plt.text(np.max(circx)/1.5, np.max(circy)/1.5, str(np.round(radius,0)) +'m', color = color, fontsize = 15, weight = 'bold')
        else:
            plt.plot(circx,circy, c = color, linestyle = linestyle, linewidth= linewidth, label = str(int(radius)) +'m')
    
    if len(circle_colors) != len(radii):
        color = circle_colors[0]
        [multi_circ(radius, color, legend) for radius in radii]
        
    else:
        [multi_circ(radius, color, legend, linewidth, linestyle) for radius, color, linewidth, linestyle in zip(radii, circle_colors,linewidth, linestyle)]
    
    if legend == True:
        plt.legend(loc = 'upper right', fancybox=True, framealpha=0.5)

#%% cmod application

import sys
sys.path.insert(0, '/home/owen/Documents/python scripts')

def applyCMOD(NRCS, phi, incidence, x_samplerate, y_samplerate, iterations = 10 , CMOD5 = False, plotting = False):
    import cmod5n
    
    """
    code retrieved from 
    https://gitlab.tudelft.nl/drama/stereoid/-/blob/a8a31da38369a326a5172d76a51d73bba8bc2d58/stereoid/oceans/cmod5n.py
    
    """
    
    # estimate windspeed using CMOD function
    windspeed = cmod5n.cmod5n_inverse(NRCS, phi, incidence, CMOD5, iterations = iterations)

    # plot windspeed
    if plotting == True:
        if CMOD5 == True:
            title = 'CMOD5 windspeed [m/s]'
        else:
            title = 'CMOD5.n windspeed [m/s]'
            
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        plt.title(title)
        plt.ylabel('Azimuth [Km]')
        plt.xlabel('Range [Km]')
        vmin= np.nanpercentile(windspeed,1)
        vmax= np.nanpercentile(windspeed,99)
        xend = windspeed.shape[1]*x_samplerate/1000  #divide by 1000 to go from metre to km
        yend = windspeed.shape[0]*y_samplerate/1000  #divide by 1000 to go from metre to km
        cbar1 = ax.imshow(windspeed, vmin = vmin, vmax = vmax, cmap = 'jet', origin = 'lower', extent=[0,xend,0,yend])
        plt.colorbar(cbar1, fraction=0.031, pad=0.05)
        
    return windspeed
    
#%% 

def HammingWindow(image, plotting = False):
    
    """
    Apply Hamming window on input NRCS data
    """
    #create 1D Hamming windows
    windowx = np.hamming(image.shape[1]).T
    windowy = np.hamming(image.shape[0])
    
    #meshgrid to combine both 1-D filters into 2D filter
    windowX, windowY = np.meshgrid(windowx, windowy)
    window = windowX*windowY
    
    #plot filter
    if plotting == True:
        fig = plt.figure(figsize=(10,5))
        plt.title('Hamming window filter')
        ax = fig.add_subplot(111)
        cbar1 = ax.imshow(window)
        plt.colorbar(cbar1,ax=ax)
    return window

#%%

def lowpass(image, CutOffWavelength, samplerate, kernelSize = 51, cmap = 'Grey', plotting = False):
    
    fc = samplerate / CutOffWavelength   # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    
    N = kernelSize
    if not N % 2: N += 1  # Make sure that N is odd.

    # Compute sinc filter.
    h  = N; w  = N
    wc = w/2; hc = h/2
    Y, X = np.ogrid[-hc:hc, -wc:wc]

    # compute sinc function
    s = np.sinc(2 * fc * X) * np.sinc(2 * fc * Y)

    # Compute Hamming window.
    w = HammingWindow(s)

    # Multiply sinc filter by window.
    f = s * w
 
    # Normalize to get unity gain.
    kernel = f / np.sum(f)
    
    # kernel = np.fft.ifft2(np.fft.ifftshift((1 - abs(np.fft.fftshift(np.fft.fft2(kernel))))))  # <-- if you want to make it low pass
    
    #apply kernel and calculate spectrum of filtered image
    image_filt = signal.convolve2d(image, kernel, boundary='symm', mode='same')
    image_filt_freq = twoDPS(image_filt, samplerate)
    
    #plot frequency domain response of filter
    freq_domain = abs(np.fft.fftshift(np.fft.fft2(kernel)))
    
    if plotting == True:
        # plottings(image_filt, samplerate, unit = 'metres', title = 'low pass filtered image', cmap = cmap)
        # plottings(image_filt_freq, samplerate, unit = 'frequency', title = 'frequency response low pass filtered image', cmap = 'viridis')
        plt.figure()
        plt.plot(freq_domain[N//2])
    return image_filt, image_filt_freq, freq_domain

def bandpass(ds, field_of_interest, freq_max = 1 / 300, freq_min = 1 / 3000):
    
    """
    ideal frequency bandpass by interpolating to polar spectrum and setting all
    
    Input:
        ds: sar dataset with wind with dimensions 'atrack' and 'xtrack'
        field_of_interest: field to be bandpass filtered with dimensions  'atrack' and 'xtrack'
        freq_max: greatest frequency, i.e. low pass limit of the bandpass
        freq_min: smallest frequency, i.e. high pass limit of the bandpass

    Output:
        ifft_field_of_interest = bandpass filtered version of ds[field_of_interest]

    """
    
    
    field_of_interest2 = field_of_interest+'nanfilled'
    
    interp = ds[field_of_interest].interpolate_na(dim = 'atrack', method= 'linear', fill_value= 'extrapolate')  # interpolate nans in one axis
    ds[field_of_interest2] = interp.interpolate_na(dim = 'xtrack', method= 'linear', fill_value= 'extrapolate')  # interpolate nans in second axis
    
    fft_field_of_interest = xrft.fft(ds[field_of_interest2], detrend='constant') # calculate 2d spectrum
    # xrft automaticically adds new coordinates which are the original plus with new prefix 'freq_', thus 'atrack' becomes 'freq_atrack'
    
    freq_radial = np.sqrt(fft_field_of_interest.freq_atrack**2 + fft_field_of_interest.freq_xtrack**2) # calculate radial frequency from center
    fft_field_of_interest_bandpass = xr.where((freq_radial.values > freq_max) | (freq_radial.values < freq_min), 0, fft_field_of_interest) # bandpass by setting all frequencies outside of bandpass to zero
    
    # invert the FFT, add mean and take absolute (to convert to real only) to arrive at a bandpass filtered wind field
    ifft_field_of_interest = abs(xrft.ifft(fft_field_of_interest_bandpass) + ds[field_of_interest].mean()) # add mean as it is lost during bandpass
    
    return ifft_field_of_interest

#%%

def ds_prepare(sar_ds):
    
    """
    This function takes a sar dataset (with coordinates in pixels), fills the NaN's (where possible), detrends and adds coordinates in metres 
    
    Input: 
        sar_ds: sar dataset containing a 'sigma0' field with coordinates 'atrack' and 'xtrack'  
        
    Output:
        sar_ds: dataset updated to contain nanfilled 'sigma0_nanfill' and 'sigma0_detrend'
    """
    
    # slice such that nans are excluded from raw data
    sar_ds = sar_ds[{'atrack':slice(5,-5), 'xtrack':slice(5,-5)}]
    
    # remove Nans by interpolation
    interp_01 = sar_ds.sigma0.interpolate_na(dim = 'atrack', method= 'linear', fill_value= 'extrapolate')
    sar_ds['sigma0_nanfill'] = interp_01.interpolate_na(dim = 'xtrack', method= 'linear', fill_value= 'extrapolate')
    
    # # detrending using the xsarsea detrend (not great but fast)
    # sar_ds['sigma0_detrend'] = xsarsea.sigma0_detrend(sar_ds.sigma0_nanfill, sar_ds.incidence)
    # interp_02 = sar_ds.sigma0_detrend.interpolate_na(dim = 'atrack', method= 'linear', fill_value= 'extrapolate')
    # sar_ds['sigma0_detrend'] = interp_02.interpolate_na(dim = 'xtrack', method= 'linear', fill_value= 'extrapolate')
    
    # detrending using kernels
    pixel_size = np.mean([sar_ds.pixel_atrack_m, sar_ds.pixel_xtrack_m]) # find average pixel size in metres
    filter_size = 10000 # median filter has size of 10000 metres
    pixels = int(filter_size // pixel_size) # find number of pixels in median filter size
    if pixels%2!= 1: # has to be odd
        pixels+=1
        
    # median kernel (good but slow)
    # sigma0_trend = ndimage.median_filter(sar_ds.sigma0_nanfill, size = pixels)
    
    # gaussian kernel (ok time and performance)
    sigma0_trend = cv2.GaussianBlur(sar_ds.sigma0_nanfill.values, (pixels , pixels), 0)
    
    # remove trend, divide by and window
    sar_ds['sigma0_detrend_plotting'] = (sar_ds.sigma0_nanfill - sigma0_trend) / sigma0_trend
    sar_ds['sigma0_detrend'] = sar_ds.sigma0_detrend_plotting * HammingWindow(sar_ds.sigma0)
    
    # Convert to units in meter and calculate 2D spectra 
    sar_ds = sar_ds.assign_coords({'atrack_m': ('atrack', np.arange(0, len(sar_ds.atrack))*sar_ds.pixel_atrack_m), \
                                   'xtrack_m': ('xtrack', np.arange(0, len(sar_ds.xtrack))*sar_ds.pixel_xtrack_m)})
    sar_ds = sar_ds.swap_dims({'atrack':'atrack_m','xtrack':'xtrack_m'})
    sar_ds = sar_ds.drop(['atrack', 'xtrack'])

    
    # if image still contains NaN's throw error
    nr_nans = sum(sum(np.isnan(sar_ds['sigma0_detrend'].values)))
    
    if nr_nans != 0:
        print("Detrended scene contains " + str(nr_nans) +" NaN's")
    
    return sar_ds

def ds_to_polar(sar_ds, interpolation = 'linear'):
    
    """
    Input:
        sar_ds: sar dataset containing a 'sigma0_detrend' array with cooresponding coordinates in metres
        interpolation: string specifying method of interpolation between cartesian and polar grid. E.g. 'linear' 'nearest'
    
    Output:
        PolarSpec: dataset containing interpolated 'sigma0_detrend' on polar grid. to be used for wind direction orientation
        fx: horizontal coordinates of polar spectrum on cartesian grid
        fy: vertical coordinates of polar spectrum on cartesian grid
    """
    
    # calculate spectrum from detrended image
    # This spectrum is only used to calcualte direction so does not matter if scaling is 'spectrum' or 'density'
    # detrend such energy in DC is removed
    # xrft automatically computes frequency (1 / lambda) along axis which is why ds has to be prepared such that xtrack and atrack are in meters
    spectrum = xrft.power_spectrum(sar_ds['sigma0_detrend'], scaling  = 'density', detrend= 'constant')

    # smooth spectrum with gaussian filter, this yields more consistent estiamtion of angles
    sigma = [2,2] # arbitrarily selected
    spectrum_smoothed = ndimage.gaussian_filter(spectrum, sigma, mode='constant')
    ds_spectrum = xr.Dataset({})
    ds_spectrum['spectrum'] = spectrum
    ds_spectrum['spectrum_smoothed'] = (('freq_atrack_m', 'freq_xtrack_m'), spectrum_smoothed)
    
    # coordinate converion 
    spectrum_smoothed = ds_spectrum.spectrum_smoothed.assign_coords({'f_range':spectrum.freq_xtrack_m, 'f_azimuth': spectrum.freq_atrack_m})
    spectrum_smoothed = spectrum_smoothed.swap_dims({'freq_xtrack_m':'f_range','freq_atrack_m':'f_azimuth'})
    spectrum_smoothed.f_range.attrs.update({'spacing':spectrum_smoothed.freq_xtrack_m.spacing})
    spectrum_smoothed.f_azimuth.attrs.update({'spacing':spectrum_smoothed.freq_atrack_m.spacing})

    # number of theta's, i.e. angular resolution of polar plot (720 yields ang. res. of 0.5 deg, 360 ang. res. of 1 deg)
    Nt = 720
    theta_spacing = 360 / Nt
    theta = np.arange(0, 360, theta_spacing)
    theta = xr.DataArray(theta, dims='theta', coords={'theta':theta})

    # number of spatial frequencies, i.e. spacing between iterpolated frequencies
    Nf = 600
    fspacing = float(spectrum_smoothed.f_range.max() / Nf)
    f = np.arange(0, spectrum_smoothed.f_range.max(), fspacing)
    f = xr.DataArray(f, dims='f', coords={'f':f})

    # calculate equivalent coordinates in cartesion cgrid
    fx = f*np.cos(np.deg2rad(theta))
    fy = f*np.sin(np.deg2rad(theta))
    
    # interpolation from cartesian grid to polar coordinates
    PolarSpec = spectrum_smoothed.interp(f_range = fx, f_azimuth = fy, assume_sorted = True, kwargs= {'fill_value':None}, method= interpolation)
    PolarSpec.f.attrs.update({'spacing':fspacing})
    PolarSpec.theta.attrs.update({'spacing':theta_spacing})
    
    return PolarSpec, fx, fy

def ds_windfield(sar_ds, PolarSpec_pre, wdir_era5, freq_max = 1 / 600, freq_min = 1 / 3000):
    
    """
    This function derives the orientation of greatest energy within a bandpass of the smoothed polar spectrum
    Next the wind field is calculated using the found orientation with respect to the radar sensor (i.e. range direction) using CMOD5.N
    
    Input:
        sar_ds: sar dataset from 'ds_prepare'
        PolarSpec_pre: polar dataset from 'ds_to_polar'
        wdir_era5: a priory wind direction with which to resolve 180 degree ambiguity
        freq_max: frequency of bandpass shorter wavelength in 1/m, high frequencies can contain swell
        freq_min: frequency of bandpass upper wavelength in 1/m, low frequencies can contian mesoscale activity
        
    Output:
        sar_ds: updated with a find field with coordinates in metre
        sum_theta: sum of energy per theta within bandpass
        angle_pre_conversion: angle in polar spectrum with greatest energy 
    """

    bandpass_subset = PolarSpec_pre.sel(f = slice(freq_min, freq_max))

    # sum of all energy per theta 
    sum_theta = (bandpass_subset*bandpass_subset.f).sum(dim = 'f')  # bandpass_subset.isel(r_axis = slice(20,-1)).sum(dim = 'r_axis')

    # find peak orientation in smoothed spectrum
    angle_pre_conversion = bandpass_subset['theta'].isel({'theta':  sum_theta.argmax()}).values
    
    # convert peak to azimuth direction
    angle = ( -(angle_pre_conversion - 90) + 360) % 360

    # convert angle from w.r.t. azimuth to w.r.t. range
    energy_dir_range =  (angle  - 90 ) % 360 

    # convert angle from w.r.t. azimuth to w.r.t. North
    energy_dir =  ((sar_ds.ground_heading.mean().values  + 360 ) % 360 + angle + 360 ) % 360

    # energy direction is perpindicular to wind direction for Wind Streaks
    if sar_ds.label == 'Wind Streaks':
        offset = 90 # approximate
    if sar_ds.label == 'Micro Convective Cell':
        offset = 0 # approximate

    wdir = (energy_dir + offset + 360) % 360

    # use ERA5 widr as reference to resolve 180 degree ambiguity        
    diff = abs(wdir_era5 - wdir)
    if (diff >= 90) & (diff <= 270):
        wdir = (wdir + 180) % 360
        angle = (angle + 180) % 360
        energy_dir_range = (energy_dir_range + 180) % 360        

    # add correction for convection type and convert from azimuth to range by + 90 (radar viowing direction)
    phi = (angle + offset + 90 ) % 360 

    # calculate wind field
    windfield = applyCMOD(sar_ds.sigma0.values, phi , sar_ds.incidence.values, \
                             x_samplerate = sar_ds.pixel_xtrack_m, y_samplerate = sar_ds.pixel_atrack_m, \
                             iterations = 10, CMOD5 = False, plotting = False)
        
    sar_ds['windfield'] = (('atrack_m', 'xtrack_m'), windfield)
    
    return sar_ds, sum_theta, angle_pre_conversion, energy_dir, wdir



def ds_spectral_calculations(sar_ds, angle_pre_conversion, interpolation = 'linear', freq_max = 1 / 600, freq_min = 1 / 3000):
    
    """
    Function to calculate spectral characteristics of the windfield contained in 'sar_ds'
    
    Input:
        sar_ds: sar dataset with wind field calculated in 'ds_windfield'
        angle_pre_conversion: angle in polar spectrum with greatest energy 
        interpolation: string specifying method of interpolation between cartesian and polar grid. E.g. 'linear' 'nearest'
        freq_max: frequency of bandpass shorter wavelength in 1/m, high frequencies can contain swell
        freq_min: frequency of bandpass upper wavelength in 1/m, low frequencies can contian mesoscale activity
        
    Output:
        spectrum: cartesian spectrum
        PolarSpec: dataset containing high res. interpolated 'sigma0_detrend' density spectrum on polar grid. To be used for spectral calculations
        PolarSpec_plot: low resolution version for potting
        cartesian_nrj: variance (energy) of cartesian image. To be used to determine effect of wind correction from spectral calculation
        variance_beyond_nyquist: variance in the corners of cartesian grid that falls outisde of poalr spectrum
        beam1: datarray with polar spectrum of beam 1
        beam2: datarray with polar spectrum of beam 2
        var_windfield: np.var(windfield). Variance unadultarated by polar conversion or windows
        beams: dataset with polar information for beams centred around 'angle_pre_conversion'
        var_bandpass: variance of polar spectrum contained within bandpass 
        var_highpass: variance of polar spectrum contained within frequencies greater than maximum specified (i.e. swell) including frequencies beyon nyquist but within cartesian 
        var_lowpass: variance of polar spectrum contained within frequencies smaller than minimum specified (i.e. mesoscale)
        var_bandpass_beam: variance of polar spectrum contained within bandpass section of the beam
        var_polar: variance contained in polar spectrum. Expected to be slightly less than var_windfield due to windowing and polar interpolation
        var_beam: variance contained in beams of polar spectrum
        density_beam: relative amount of energy in beam compared to average of spectrum (e.g. 1= average 2 = twice average)
        density_bandpass: relative amount of energy in bandpass compared to average of spectrum
        density_beam_bandpass: relative amount of energy in bandpass within beam compared to average of bandpass
        fx: horizontal coordinates of polar spectrum on cartesian grid
        fy: vertical coordinates of polar spectrum on cartesian grid
        fx_plot: low resolution version for plotting
        fy_plot: low resolution version for plotting
    """
    
    # calculate power spectral density of windfield using (and correcting for) a Hanning window
    CartSpec = xrft.power_spectrum(sar_ds['windfield'], scaling  = 'density', detrend='constant', window = 'hann', window_correction = 'True')  

    # add and swap dimensions
    CartSpec = CartSpec.assign_coords({'f_range':CartSpec.freq_xtrack_m, 'f_azimuth': CartSpec.freq_atrack_m})
    CartSpec = CartSpec.swap_dims({'freq_xtrack_m':'f_range','freq_atrack_m':'f_azimuth'})
    CartSpec.f_range.attrs.update({'spacing':CartSpec.freq_xtrack_m.spacing})
    CartSpec.f_azimuth.attrs.update({'spacing':CartSpec.freq_atrack_m.spacing})

    # calculate total energy inside cartesian spectrum, dividing density spcetrum by spacing which is equal to the variance 
    cartesian_nrj = CartSpec.sum().values * np.prod([CartSpec.f_range.spacing, CartSpec.f_azimuth.spacing])

    # calculate energy that falls outside polar spectrum but within Cartesian
    x, y = np.meshgrid(CartSpec.f_range, CartSpec.f_azimuth)
    indexes_beyond_nyquist = np.where(np.sqrt(x**2 + y**2) > CartSpec.f_range.max().values, 1, 0)
    variance_beyond_nyquist = (CartSpec * indexes_beyond_nyquist).sum().values * np.prod([CartSpec.f_range.spacing, CartSpec.f_azimuth.spacing])

    #################################################################
    ######## polar conversion at low resolution for plotting ########
    #################################################################

    Nt = 720
    theta_spacing = 360 / Nt
    theta = np.arange(0, 360, theta_spacing)
    theta = xr.DataArray(theta, dims='theta', coords={'theta':theta})

    # number of spatial frequencies
    Nf= 300
    fspacing = float(CartSpec.f_range.max() / Nf)
    f = np.arange(0, CartSpec.f_range.max(), fspacing)
    f = xr.DataArray(f, dims='f', coords={'f':f})

    # calculate equivalent coordinates of polar system in cartesian system
    fx_plot = f*np.cos(np.deg2rad(theta))
    fy_plot = f*np.sin(np.deg2rad(theta))

    # interpolate from cartesian spectrum to polar spectrum
    PolarSpec_plot = CartSpec.interp(f_range = fx_plot, f_azimuth = fy_plot, assume_sorted = True, kwargs = {'fill_value':None}, method = interpolation)
    PolarSpec_plot.f.attrs.update({'spacing':fspacing})
    PolarSpec_plot.theta.attrs.update({'spacing':theta_spacing})
    
    ###########################################################
    ######## actual polar conversion and interpolation ########
    ###########################################################
    
    # number of theta's, 3600 yields angular resultion of 0.1 degree, allows for minimal loss during interpolation
    Nt = 3600
    theta_spacing = 360 / Nt
    theta = np.arange(0, 360, theta_spacing)
    theta = xr.DataArray(theta, dims='theta', coords={'theta':theta})

    # number of spatial frequencies
    Nf= 600
    fspacing = float(CartSpec.f_range.max() / Nf)
    f = np.arange(0, CartSpec.f_range.max(), fspacing)
    f = xr.DataArray(f, dims='f', coords={'f':f})

    # calculate equivalent coordinates of polar system in cartesian system
    fx = f*np.cos(np.deg2rad(theta))
    fy = f*np.sin(np.deg2rad(theta))

    # interpolate from cartesian spectrum to polar spectrum
    PolarSpec = CartSpec.interp(f_range = fx, f_azimuth = fy, assume_sorted = True, kwargs = {'fill_value':None}, method = interpolation)
    PolarSpec.f.attrs.update({'spacing':fspacing})
    PolarSpec.theta.attrs.update({'spacing':theta_spacing})

    # calculate angular spacing in radians to convert from density spectrum to energy spectrum
    d_theta = theta_spacing * np.pi / 180
    
    # calculate total energy within the polar spectrum, depending on the winddowing effect polar_nrj < cartesian_nrj
    polar_nrj = (PolarSpec*PolarSpec.f).sum().values * np.prod([PolarSpec.f.spacing, d_theta])

    # select range of angles with conditioning incase it passes the 0 or 360 degree boundary (for both sides of the spectrum)
    beam_size = 20
    slice_min = (angle_pre_conversion - beam_size + 360 ) % 360
    slice_max = (angle_pre_conversion + beam_size + 360 ) % 360
    
    angle_pre_conversion_mirror = (angle_pre_conversion + 180 ) % 360
    slice_min_mirror = (slice_min + 180 ) % 360
    slice_max_mirror = (slice_max + 180 ) % 360

    # add conditions in case the beam crosses the 0 - 360 boundary 
    # calculated for two seperate beams instead of for one multiplied times 2 in case polar spectrum is not composed of two equal halves
    # beam 1
    # if maximum crosses 360 line
    if (slice_max < angle_pre_conversion) & (slice_min < angle_pre_conversion):
        idx_beam1 = [i[0] for i in np.argwhere( (PolarSpec.theta.values >= slice_min) | (PolarSpec.theta.values <= slice_max))]
    # if minimum crosses 360 line
    elif (slice_max > angle_pre_conversion) & (slice_min > angle_pre_conversion):
        idx_beam1 = [i[0] for i in np.argwhere( (PolarSpec.theta.values >= slice_min) | (PolarSpec.theta.values <= slice_max))]
    # if neither crosses 360 line   
    elif (slice_max > angle_pre_conversion) & (slice_min < angle_pre_conversion):
        idx_beam1 = [i[0] for i in np.argwhere( (PolarSpec.theta.values >= slice_min) & (PolarSpec.theta.values <= slice_max))]
      
    # beam 2 (e.g. beam 1 but shifted 180 degree)
    # if maximum crosses 360 line   
    if (slice_max_mirror < angle_pre_conversion_mirror) & (slice_min_mirror < angle_pre_conversion_mirror):
        idx_beam2 = [i[0] for i in np.argwhere( (PolarSpec.theta.values >= slice_min_mirror) | (PolarSpec.theta.values <= slice_max_mirror))]
    # if minimum crosses 360 line
    elif (slice_max_mirror > angle_pre_conversion_mirror) & (slice_min_mirror > angle_pre_conversion_mirror):
        idx_beam2 = [i[0] for i in np.argwhere( (PolarSpec.theta.values >= slice_min_mirror) | (PolarSpec.theta.values <= slice_max_mirror))]
    # if neither crosses 360 line     
    elif (slice_max_mirror > angle_pre_conversion_mirror) & (slice_min_mirror < angle_pre_conversion_mirror):
        idx_beam2 = [i[0] for i in np.argwhere( (PolarSpec.theta.values >= slice_min_mirror) & (PolarSpec.theta.values <= slice_max_mirror))]

    # select beam subset (i.e. points within few degrees of angle of greatest variation)
    beam1 = PolarSpec[:, idx_beam1]
    beam2 = PolarSpec[:, idx_beam2]
    beams = xr.concat([beam1, beam2], "theta") # add both beams into a single dataraay

    # calculate energy in polar spectrum, converting angular spacing to radians
    d_theta = theta_spacing * np.pi / 180
    polar_nrj_beams = (beams*beams.f).sum().values * np.prod([beams.f.spacing, d_theta])

    # calculate energy within different parts of the polar spectrum
    spectrum_bandpass = PolarSpec.sel(f = slice(freq_min, freq_max))
    spectrum_highpass = PolarSpec.sel(f = slice(freq_max, 1))  # all energy in wavelengths shorter than minimum wavelength, including that which falls outside polar but still within cartesian
    spectrum_lowpass = PolarSpec.sel(f = slice(0, freq_min)) # all energy in wavelengths longer than the maximum, should be energy in mesoscale
    var_bandpass = (spectrum_bandpass*spectrum_bandpass.f).sum().values * np.prod([spectrum_bandpass.f.spacing, d_theta])
    var_highpass = (spectrum_highpass*spectrum_highpass.f).sum().values * np.prod([spectrum_highpass.f.spacing, d_theta]) + variance_beyond_nyquist
    var_lowpass = (spectrum_lowpass*spectrum_lowpass.f).sum().values * np.prod([spectrum_lowpass.f.spacing, d_theta])

    spectrum_bandpass_beam = beams.sel(f = slice(freq_min, freq_max))
    var_bandpass_beam = (spectrum_bandpass_beam*spectrum_bandpass_beam.f).sum().values * np.prod([spectrum_bandpass_beam.f.spacing, d_theta])

    var_windfield = np.var(sar_ds['windfield']).values*1 # to convert from datarray to float
    var_polar = polar_nrj
    var_beam = polar_nrj_beams
    
    polar_effect = var_polar / cartesian_nrj
    window_effect = cartesian_nrj / var_windfield
    low_pass_frac = var_lowpass / var_polar
    high_pass_frac = var_highpass / var_polar
    bandpass_frac = var_bandpass / var_polar
            
    frac_beam = var_beam / var_polar
    density_beam = frac_beam / (beam_size * 4 / 360)
    density_bandpass = var_bandpass / var_polar
    density_beam_bandpass = var_bandpass_beam / var_bandpass / (beam_size * 4 / 360)
    
    #########################################################################################
    ######## spectral calculations concerning energy distribution and peak locations ########
    #########################################################################################
    
    # create relative cumsum of lowpass spectrum with respect to bandpass (e.g. how much energy is present up to specific frequencies w.r.t. total energy in bandpass at same theta)
    PolarSpecHighpass = PolarSpec.sel(f = slice(freq_min, 1/PolarSpec.f.max())) * PolarSpec.f # 200m since at 
    PolarSpecBandpass = PolarSpec.sel(f = slice(freq_min, freq_max)) * PolarSpec.f
    cumsum_scaled = (PolarSpecHighpass.cumsum(dim = 'f') / PolarSpecBandpass.sum() * len(PolarSpecHighpass.theta) )

    ############ 2D cumsum  ###############
    percentile_25 = (cumsum_scaled>0.25).argmax(dim = 'f')
    percentile_50 = (cumsum_scaled>0.50).argmax(dim = 'f')
    percentile_75 = (cumsum_scaled>0.75).argmax(dim = 'f')
    percentile_diff_75_25 = xr.where((percentile_75 - percentile_25)<0, np.max(percentile_75) ,(percentile_75 - percentile_25))
    percentile_diff_75_50 = xr.where((percentile_75 - percentile_50)<0, np.max(percentile_75) ,(percentile_75 - percentile_50))
    percentile_diff_50_25 = xr.where((percentile_50 - percentile_25)<0, np.max(percentile_50) ,(percentile_50 - percentile_25))
    
    
    
    
    theta_diff_min_75_25 = percentile_diff_75_25.theta[percentile_diff_75_25.min().values*1].values*1 # theta at which it requires fewwest frequencies to go from 25% to 75% of energy
    theta_diff_min_75_50 = percentile_diff_75_50.theta[percentile_diff_75_50.min().values*1].values*1 # theta at which it requires fewwest frequencies to go from 25% to 75% of energy
    theta_diff_min_50_25 = percentile_diff_50_25.theta[percentile_diff_50_25.min().values*1].values*1 # theta at which it requires fewwest frequencies to go from 25% to 75% of energy
    
    theta_25_of_min_freq = percentile_25.theta[percentile_25.min().values*1].values*1
    theta_50_of_min_freq = percentile_50.theta[percentile_50.min().values*1].values*1
    theta_75_of_min_freq = percentile_75.theta[percentile_75.min().values*1].values*1
    theta_25_of_max_freq = percentile_25.theta[percentile_25.max().values*1].values*1
    theta_50_of_max_freq = percentile_50.theta[percentile_50.max().values*1].values*1
    theta_75_of_max_freq = percentile_75.theta[percentile_75.max().values*1].values*1
    
    
    # # new AngleDiffPolar updated on 12-06-2022
    # def AngleDiffPolar(angle1, angle2):
    #     # calculates the smallest angular difference taking into acount the 360 to 0 jump and keeps sign
    #     AngleDiff = (angle1 - angle2 + 360 ) % 360
    #     AngleDiffCounterClock = AngleDiff - 360
    #     AngleDiffMin = min([abs(AngleDiff), abs(AngleDiffCounterClock)])
    #     return AngleDiffMin
    
    def AngleDiffPolar(angle1, angle2):
        # calculates the smallest angular difference taking into acount the 360 to 0 jump
        # if clockwise movement from angle 1 to angle 2 is shortesdt then the angular difference is positive, else negative
        
        AngleDiff = (angle1 - angle2 + 360 ) % 360
        AngleDiffCounterClock = AngleDiff - 360
        AngleDiffMin = min([abs(AngleDiff), abs(AngleDiffCounterClock)])
    
        AngleDiff_plus1 = ((angle1 + 0.001) - angle2 + 360 ) % 360
        AngleDiffCounterClock_plus1 = AngleDiff_plus1 - 360
        AngleDiffMin_plus1 = min([abs(AngleDiff_plus1), abs(AngleDiffCounterClock_plus1)])
    
        if AngleDiffMin_plus1 > AngleDiffMin:
            AngleDiffMin *= -1
            
        return AngleDiffMin
    
    angle_diff_max_min_75_25  = AngleDiffPolar(angle_pre_conversion, theta_diff_min_75_25)
    angle_diff_min_theta_75_of_min_freq = AngleDiffPolar(theta_diff_min_75_25, theta_25_of_min_freq)
    angle_diff_theta_25_of_min_freq_theta_75_of_min_freq = AngleDiffPolar(theta_25_of_min_freq, theta_75_of_min_freq)
    angle_diff_theta_75_of_min_freq_theta_25_of_max_freq = AngleDiffPolar(theta_75_of_min_freq, theta_25_of_max_freq)
    angle_diff_theta_25_of_max_freq_theta_75_of_max_freq = AngleDiffPolar(theta_25_of_max_freq, theta_75_of_max_freq)
    
    angle_diff_max_min_75_50  = AngleDiffPolar(angle_pre_conversion, theta_diff_min_75_50)
    angle_diff_min_theta_50_of_min_freq = AngleDiffPolar(theta_diff_min_75_50, theta_50_of_min_freq)
    angle_diff_theta_50_of_min_freq_theta_75_of_min_freq = AngleDiffPolar(theta_50_of_min_freq, theta_75_of_min_freq)
    angle_diff_theta_75_of_min_freq_theta_50_of_max_freq = AngleDiffPolar(theta_75_of_min_freq, theta_50_of_max_freq)
    angle_diff_theta_50_of_max_freq_theta_75_of_max_freq = AngleDiffPolar(theta_50_of_max_freq, theta_75_of_max_freq)
    
    angle_diff_max_min_50_25  = AngleDiffPolar(angle_pre_conversion, theta_diff_min_50_25)
    angle_diff_min_theta_25_of_min_freq = AngleDiffPolar(theta_diff_min_50_25, theta_25_of_min_freq)
    angle_diff_theta_75_of_min_freq_theta_50_of_min_freq = AngleDiffPolar(theta_25_of_min_freq, theta_50_of_min_freq)
    angle_diff_theta_50_of_min_freq_theta_25_of_max_freq = AngleDiffPolar(theta_50_of_min_freq, theta_25_of_max_freq)
    angle_diff_theta_25_of_max_freq_theta_50_of_max_freq = AngleDiffPolar(theta_25_of_max_freq, theta_50_of_max_freq)
    
    
    ############ 1D cumsum  ###############
    cumsum_scaled = (PolarSpecHighpass.cumsum(dim = 'f') / PolarSpecHighpass.sum())
    cumsum_scaled_sum_theta = cumsum_scaled.sum(dim = 'theta')
    
    ##########  index frequencies in bandpass corresponding to cumulative sum of energy #############
    idx_25 = np.argmax(cumsum_scaled_sum_theta.values > 0.25)
    idx_50 = np.argmax(cumsum_scaled_sum_theta.values > 0.50)
    idx_75 = np.argmax(cumsum_scaled_sum_theta.values > 0.75)
    
    freq_25 = cumsum_scaled_sum_theta.f[idx_25].values*1
    freq_50 = cumsum_scaled_sum_theta.f[idx_50].values*1
    freq_75 = cumsum_scaled_sum_theta.f[idx_75].values*1
    
    

    return CartSpec, PolarSpec, PolarSpec_plot, cartesian_nrj, beam1, beam2, variance_beyond_nyquist, var_windfield, beams, var_bandpass, var_highpass, var_lowpass, var_bandpass_beam, var_polar, var_beam, \
        polar_effect, window_effect, low_pass_frac, high_pass_frac, bandpass_frac, frac_beam, density_beam, density_bandpass, density_beam_bandpass, fx, fy, fx_plot, fy_plot, freq_25, freq_50, freq_75, \
        angle_diff_max_min_75_25, angle_diff_max_min_75_50, angle_diff_max_min_50_25, angle_diff_min_theta_75_of_min_freq, angle_diff_min_theta_50_of_min_freq, angle_diff_min_theta_25_of_min_freq,\
        angle_diff_theta_25_of_min_freq_theta_75_of_min_freq, angle_diff_theta_50_of_min_freq_theta_75_of_min_freq, angle_diff_theta_75_of_min_freq_theta_50_of_min_freq, \
        angle_diff_theta_75_of_min_freq_theta_25_of_max_freq, angle_diff_theta_75_of_min_freq_theta_50_of_max_freq, angle_diff_theta_50_of_min_freq_theta_25_of_max_freq, \
        angle_diff_theta_25_of_max_freq_theta_75_of_max_freq, angle_diff_theta_50_of_max_freq_theta_75_of_max_freq, angle_diff_theta_25_of_max_freq_theta_50_of_max_freq



def ds_plot(sar_ds, CartSpec, PolarSpec, PolarSpec_plot, cartesian_nrj, beam1, beam2, angle_pre_conversion, variance_beyond_nyquist, var_windfield, beams, var_bandpass, var_highpass, var_lowpass, \
            var_bandpass_beam, var_polar, var_beam, fx, fy, sum_theta, idx_inertial_min, idx_inertial_max, w_star_normalised_deviation, freq_max = 1 / 600, freq_min = 1 / 3000, save_directory = None):
    
    """
    Function to plot previously obtained results
    
    Input:
        spectrum: cartesian spectrum
        PolarSpec: dataset containing high res. interpolated 'sigma0_detrend' on polar grid. To be used for spectral calculations
        cartesian_nrj: variance (energy) of cartesian image. To be used to determine effect of wind correction from spectral calculation
        variance_beyond_nyquist: variance in the corners of cartesian grid that falls outisde of poalr spectrum
        beam1: datarray with polar spectrum of beam 1
        beam2: datarray with polar spectrum of beam 2
        angle_pre_conversion: angle in polar spectrum with greatest energy 
        var_windfield: np.var(windfield). Variance unadultarated by polar conversion or windows
        beams: dataset with polar information for beams centred around 'angle_pre_conversion'
        var_bandpass: variance of polar spectrum contained within bandpass 
        var_highpass: variance of polar spectrum contained within frequencies greater than maximum specified (i.e. swell) including frequencies beyon nyquist but within cartesian 
        var_lowpass: variance of polar spectrum contained within frequencies smaller than minimum specified (i.e. mesoscale)
        var_bandpass_beam: variance of polar spectrum contained within bandpass section of the beam
        var_polar: variance contained in polar spectrum. Expected to be slightly less than var_windfield due to windowing and polar interpolation
        var_beam: variance contained in beams of polar spectrum
        fx: horizontal coordinates of polar spectrum on cartesian grid
        fy: vertical coordinates of polar spectrum on cartesian grid
        sum_theta: sum of energy per theta within bandpass
        idx_inertial_min: index corresponding to the inertial subrange minimum in the 1D spectrum derived from the 2D smoothed spectrum (PolarSpec_pre)
        idx_inertial_max: index corresponding to the inertial subrange maximum in the 1D spectrum derived from the 2D smoothed spectrum (PolarSpec_pre)
        w_star_normalised_deviation: std of w_star values in inertial subrange
        freq_max: frequency of bandpass shorter wavelength in 1/m, high frequencies can contain swell
        freq_min: frequency of bandpass upper wavelength in 1/m, low frequencies can contian mesoscale activity
        save_directory: directory in which to store computed plots, if 'None' no images will be stored
        
    Output:
        image
    """
    
    plt.figure(figsize = (21,10))

    # add radar image
    plt.subplot(2, 3, 1)
    plt.ylabel('Azimuth [pixels]')
    plt.xlabel('Range [pixels]')
    plt.title('Detrended SAR image')
    vmin= np.nanpercentile(sar_ds.sigma0_detrend_plotting.values,10)
    vmax= np.nanpercentile(sar_ds.sigma0_detrend_plotting.values,90)
    plt.imshow(sar_ds.sigma0_detrend_plotting.values, cmap='Greys_r', origin = 'lower', vmin = vmin, vmax = vmax) #, extent=[0,xend,0,yend])

    # add orientations
    plt.subplot(2, 3, 4)
    x_axis = ((360 - (sum_theta.theta - 90)) % 360).values
    plt.plot(x_axis[np.argsort(x_axis)], sum_theta.values[np.argsort(x_axis)])
    plt.ylabel(r'sum of (PSD $\cdot$ $f$) [$arbitrary$]')
    plt.xlabel(r'Angle w.r.t azimuth [$^{\circ}$]')
    plt.title('Sum of PSD $\cdot$ $f$ per angle in bandpass')

    # add remainder of cartesian spectrum
    plt.subplot(2, 3, 2)
    vmin = np.nanpercentile(10*np.log10(PolarSpec_plot), 50)
    vmax = np.nanpercentile(10*np.log10(PolarSpec_plot), 95)
    c = 10*np.log10(CartSpec)
    x, y = np.meshgrid(CartSpec.f_range, CartSpec.f_azimuth)
    plt.scatter(x, y, c = c, vmin = vmin, vmax = vmax, cmap = 'inferno', alpha=0.035)

    # plot complete polar spectrum
    c = 10*np.log10(PolarSpec_plot)
    plt.scatter(fx, fy, vmin = vmin, s = 2.5, c = c, vmax = vmax, cmap = 'inferno', lw = 0,) # alpha=0.05)

    plt.title('Polar PSD')
    plt.ylabel(r'$1/\lambda$ [$m^{-1}$]')
    plt.xlabel(r'$1/\lambda$ [$m^{-1}$]')

    # plot complete polar spectrum with beams and bandpass
    plt.subplot(2, 3, 5)
    c = 10*np.log10(PolarSpec_plot)
    plt.scatter(fx, fy, vmin = vmin, s = 0.8, c = c, vmax = vmax, cmap = 'inferno', lw = 0, alpha=0.3)

    # plot area within beam
    c = 10 * np.log10(beams) 
    cbar1 = plt.scatter(beams.f_range, beams.f_azimuth, s = 2.5 , c=c, vmin = vmin, vmax = vmax, cmap = 'inferno', lw = 0.1)
    # plt.colorbar(cbar1,ax=ax, fraction=0.031, pad=0.05)

    beam_size = 20
    slice_min = (angle_pre_conversion - beam_size + 360 ) % 360
    slice_max = (angle_pre_conversion + beam_size + 360 ) % 360
    
    slice_min_mirror = (slice_min + 180 ) % 360
    slice_max_mirror = (slice_max + 180 ) % 360

    # plot beams
    beam2_1 = beam2.sel(theta = slice_min_mirror , method="nearest")
    beam2_2 = beam2.sel(theta = slice_max_mirror , method="nearest")
    plt.scatter(beam2_1.freq_xtrack_m, beam2_1.freq_atrack_m, s = 1, c = 'red', lw = 0)
    plt.scatter(beam2_2.freq_xtrack_m, beam2_2.freq_atrack_m, s = 1, c = 'red', lw = 0)
    beam1_1 = beam1.sel(theta = slice_min , method="nearest")
    beam1_2 = beam1.sel(theta = slice_max , method="nearest")
    plt.scatter(beam1_1.freq_xtrack_m, beam1_1.freq_atrack_m, s = 1, c = 'red', lw = 0)
    plt.scatter(beam1_2.freq_xtrack_m, beam1_2.freq_atrack_m, s = 1, c = 'red', lw = 0)

    # plot bandpass
    high_freq_cutoff = PolarSpec_plot.sel(f = freq_max, method="nearest")
    low_freq_cutoff = PolarSpec_plot.sel(f = freq_min, method="nearest")
    plt.scatter(high_freq_cutoff.freq_xtrack_m, high_freq_cutoff.freq_atrack_m, s = 1, c = 'red', lw = 0)
    plt.scatter(low_freq_cutoff.freq_xtrack_m, low_freq_cutoff.freq_atrack_m, s = 1, c = 'red', lw = 0)

    plt.title('Beams along max gradient')
    plt.ylabel(r'$1/\lambda$ [$m^{-1}$]')
    plt.xlabel(r'$1/\lambda$ [$m^{-1}$]')

    d_theta = PolarSpec.theta.spacing * np.pi / 180
    # plot spectrum
    plt.subplot(2, 3, 3)
    PSD_plot = ((PolarSpec*PolarSpec.f).sum(dim = 'theta')) * np.prod([PolarSpec.f.spacing, d_theta]) 
    # plt.plot(beams.f[3:], PSD_plot, 'k', marker='o', markersize=1,  linewidth=0)
    plt.plot(beams.f[3:], PSD_plot[3:], c = 'gray', marker='o', markersize=1,  linewidth=0)
    # plt.plot(beams.sel(f = slice(freq_min, freq_max)).f, PSD_plot.sel(f = slice(freq_min, freq_max)), c = 'black', marker='o', markersize=1,  linewidth=0)
    plt.plot(beams.isel(f = slice(idx_inertial_max, idx_inertial_min)).f, PSD_plot.isel(f = slice(idx_inertial_max, idx_inertial_min)), c = 'black', marker='o', markersize=1,  linewidth=0)
    
    plt.title('Sum of all angles')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1e-04, 5e-03)
    plt.ylim(1e-05, 1e-01)
    plt.ylabel(r'Variance at wavelength [$m^{2}s^{-2}$]')
    plt.xlabel(r'$1/\lambda$ [$m^{-1}$]')
    # fraction of energy left after windowing with correction
    plt.text(1.1e-04, 4e-02, r'$\frac{\sigma{^2}_{cartesian}}{\sigma{^2}_{windfield}}=$' +str(np.round((cartesian_nrj/ var_windfield), 3)), fontsize = 15)
    # fraction of energy left after interpolation to polar spectrum
    plt.text(6e-04, 4e-02, r'$\frac{\sigma{^2}_{polar\ spectrum}}{\sigma{^2}_{cartesian}}=$' +str(np.round((var_polar/cartesian_nrj),3)), fontsize = 15)
    # fraction of energy in lowpass
    plt.text(1.1e-04, 1.3e-02, r'$\frac{\sigma{^2}_{polar\ lowpass}}{\sigma{^2}_{polar\ spectrum}}=$' +str(np.round((var_lowpass/var_polar),3)), fontsize = 15)
    # fraction of energy in highpass
    plt.text(6e-04, 1.3e-02, r'$\frac{\sigma{^2}_{cartesian\ highpass}}{\sigma{^2}_{polar\ spectrum}}=$' +str(np.round((var_highpass/var_polar),3)), fontsize = 15)
    plt.text(1.1e-04, 4e-03, r'$std.\ metric\ of\ slope=$' + str(np.round(w_star_normalised_deviation,3)), fontsize = 15)

    ######## plot -5/3 powerlaw #######
    # select frequency axis
    x_axis_plotting = PolarSpec.f.values
    # select point on frequency axis deemed to be within intertial subrange
    axis_kolmogorov = x_axis_plotting[idx_inertial_max:idx_inertial_min]
    # find value of highest frequency point on frequency axis
    value_kolmogorov = PSD_plot[idx_inertial_min].values
    # some things to prepare the amplitude of the powerlaw slope
    a = 1/x_axis_plotting[idx_inertial_max:idx_inertial_min]**(5/3)
    kolmogorov = value_kolmogorov * a/ (min(a))
    # plot results
    plt.plot(axis_kolmogorov, kolmogorov,'C3--',  linewidth=3)

    # plot spectrum
    plt.subplot(2, 3, 6)
    PSD_beams_plot = ((beams*beams.f).sum(dim = 'theta')) * np.prod([beams.f.spacing, d_theta])
    # plt.plot(beams.f[3:], PSD_beams_plot, 'k', marker='o', markersize=1,  linewidth=0)
    plt.plot(beams.f[3:], PSD_beams_plot[3:], c = 'gray', marker='o', markersize=1,  linewidth=0)
    # plt.plot(beams.sel(f = slice(freq_min, freq_max)).f, PSD_beams_plot.sel(f = slice(freq_min, freq_max)), c = 'black', marker='o', markersize=1,  linewidth=0)
    plt.plot(beams.isel(f = slice(idx_inertial_max, idx_inertial_min)).f, PSD_beams_plot.isel(f = slice(idx_inertial_max, idx_inertial_min)), c = 'black', marker='o', markersize=1,  linewidth=0)
    
    plt.title('Sum of beam cross-section')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1e-04, 5e-03)
    plt.ylim(1e-05, 1e-01)
    plt.ylabel(r'Variance at wavelength [$m^{2}s^{-2}$]')
    plt.xlabel(r'$1/\lambda$ [$m^{-1}$]')
    # fraction of energy in polar spectrum within beam
    plt.text(1.1e-04, 4e-02, r'$\frac{\sigma{^2}_{beam}}{\sigma{^2}_{polar\ spectrum}}=$' +str(np.round((var_beam/var_polar),3)), fontsize = 15)
    # fraction of energy in polar spectrum within beam scaled by beam area over total area
    plt.text(6e-04, 4e-02, r'$\frac{\sigma{^2}_{beam}\ /\ \sigma{^2}_{polar\ spectrum}}{A_{beam}\ /\ A_{polar\ spectrum}}=$' +str(np.round((var_beam/var_polar/ (beam_size * 4 / 360)),3)), fontsize = 15)
    # fraction of energy in bandpass scaled by total area
    plt.text(1.1e-04, 1.3e-02, r'$\frac{\sigma{^2}_{bandpass}}{\sigma{^2}_{polar\ spectrum}}=$' +str(np.round((var_bandpass/var_polar),3)), fontsize = 15)
    # fraction of energy in both band and beam scaled by total area in bandpass
    plt.text(6e-04, 1.3e-02, r'$\frac{\sigma{^2}_{bandpass\ beam}\ /\ \sigma{^2}_{bandpass}}{A_{bandpass\ beam}\ /\ A_{bandpass}}=$' +str(np.round((var_bandpass_beam/var_bandpass/ (beam_size * 4 / 360)),3)), fontsize = 15)

    ######## plot -5/3 powerlaw #######
    
    # select frequency axis
    x_axis_plotting = PolarSpec.f.values
    # select point on frequency axis deemed to be within intertial subrange
    axis_kolmogorov = x_axis_plotting[idx_inertial_max:idx_inertial_min]
    # find value of highest frequency point on frequency axis
    value_kolmogorov = PSD_beams_plot[idx_inertial_min].values
    # some things to prepare the amplitude of the powerlaw slope
    a = 1/x_axis_plotting[idx_inertial_max:idx_inertial_min]**(5/3)
    kolmogorov = value_kolmogorov * a/ (min(a))
    # plot results
    plt.plot(axis_kolmogorov, kolmogorov,'C3--',  linewidth=3)

    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.2, 
                        hspace=0.3)
    
    # path2save = '/home/owen/Documents/buoy_data/images/spectral_analysis/'
    if save_directory != None:
        name2save = sar_ds.name[121:].replace('.SAFE:', '_____')
        plt.savefig(save_directory + name2save + '_w_slopes_.png', bbox_inches='tight') 
    
    plt.show()




def loop1(sar_ds):
    """
    First loop of Young's approach. Calculates surface stress Tau , friction velocity u* and roughness length z_0
    based on neutral wind speed input'
    
    Input:
        sar_ds: sar dataset containing windfield with neutral 10 metre windspeeds in m/s
        
    Output:
        sar_ds: updated with surface stress field Tau
        u*: friction velocity in m/s
        z_0: friction length in m
        C_dn: neutral drag coefficient
        
    """
    
    # define constants
    karman = 0.40                           # Karman constant
    Charnock = 0.011                        # Charnock constant
    g = 9.8                                 # Gravitational acceleration, m/s**2
    z = 10                                  # measurements height, 10 metres for CMOD5.N 
    rho_air = 1.2                           # kg/m**3, 1.2 for 20 degrees, 1.25 for 10 degreess                           
    T = 20                                  # temperature in Celcius
    
    # kinematic viscosity of air
    nu = 1.326 * 10**(-5) *(1 + (6.542 * 10**(-3))* T + (8.301 * 10**(-6)) * T**2 - (4.840 * 10**(-9)) * T**3) # m**2/s
    
    #Calculate mean neutral 10 metre windspeed
    windspeed_median = sar_ds['windfield'].median().values * 1
    
    # prepare loop of 15 iterations
    iterations = 15
    A_friction_velocity = np.ones(iterations)    # m/s
    A_surface_stress = np.ones(iterations)       # kg/ m / s**2  [Pa]
    A_Cdn = np.ones(iterations)                  # 
    A_z_0 = np.ones(iterations)                  # m

    # Initialise loop with windspeed and iterate with refined estimates of neutral drag coefficient
    for i in range(iterations):
        if i > 0:
            A_friction_velocity[i] = np.sqrt(A_Cdn[i-1] * windspeed_median**2)
            A_surface_stress[i] = rho_air * A_friction_velocity[i]**2
            A_z_0[i] = (Charnock * A_friction_velocity[i]**2) / g + 0.11 * nu / A_friction_velocity[i]
            A_Cdn[i] = (karman / np.log( z / A_z_0[i]) )**2
    
    # calculate stress field based on retrieved constants and windspeed estimates
    # !!! use mean windfield here or windfield? --> if use windfield then u_star will change for different windspeed variances
    sar_ds['surface_stress'] = rho_air * A_Cdn[-1] *  sar_ds['windfield']**2 

    # save friction velocity and friction length based on mean stress field and neutral drag coefficient
    friction_velocity = np.sqrt(sar_ds['surface_stress'].mean().values * 1 / rho_air)
    z_0 = (Charnock * friction_velocity**2) / g + 0.11 * nu / friction_velocity
    Cdn = A_Cdn[-1]
    
    return sar_ds, friction_velocity, z_0, Cdn



def loop2B(sar_ds, friction_velocity, z_0, Zi, Cdn, PolarSpec_pre, PolarSpec, dissip_rate = 1, freq_max = 1 / 600, freq_min = 1 / 3000, label = 'Micro Convective Cell'):
    """
    Third loop of Young's approach. Requires output of loop 1. Recalculates wind field using stability correction.
    Similar to loop two but instead of using wind variance of entire field only uses inertial subrange
    Outputs recalculated parameters, obukhov Length L and kinematic heat flux B
    
    Input:
        sar_ds: sar dataset containing 2D windfield array with neutral 10 metre windspeeds in m/s
        surface_stress: Tau from loop 1
        friction_velocity: u_star from loop 1
        z_0: z_0 from loop 1
        Cdn: neutral drag coefficient from loop 1
        PolarSpec_pre: Smoothed polar spectrum to be used to find spectral peak and troughs
        PolarSpec: dataset containing high res. interpolated 'sigma0_detrend' density spectrum on polar grid. To be used for spectral calculations
        dissip_rate: approximately between 0.5 and 2.5 (kaimal et al,  1976)
        freq_max: frequency of bandpass shorter wavelength in 1/m, high frequencies can contain swell
        freq_min: frequency of bandpass upper wavelength in 1/m, low frequencies can contain mesoscale activity
        label: expected convection form, cells is standard, rolls result in slight modification

    Output:
        sigma_u: estimated wind-field variance
        L: Obukhov length in meters
        B: Kinematic heat flux in metres
        w_star: convective velocity scale in m/s
        w_star_normalised_deviation: std of w_star values in inertial subrange
        corr_fact: stability correction factor
        H: heat flux (from Ocean into atmosphere)
        Zi_estimate: wavelentgh of the spectral peak. In literature this is multipleid times a factor to get Zi
        idx_inertial_min: index corresponding to the inertial subrange minimum in the 1D spectrum derived from the 2D smoothed spectrum (PolarSpec_pre)
        idx_inertial_max: index corresponding to the inertial subrange maximum in the 1D spectrum derived from the 2D smoothed spectrum (PolarSpec_pre)
        
    
    Change such that Zi is input directly from prun rather than taken from sar_ds
    
    ###### Zi: lowest inversion height following Kaimal et al 1976, Sikora et al 1997
    
    """
    
    # find indexes in smoothed spectrum belonging to peak and trough of inertial subrange. Requires that frequency spacing of PolarSpec and PolarSpec_pre are identical
    if len(PolarSpec_pre.f) != len(PolarSpec.f):
        print('Frequencies belonging to PolarSpec_pre and PolarSpec are not equal, will result in incorrect indexing of inertial subrange')
    
    idx_start_min = np.argmin(abs(PolarSpec_pre.f.values - 1/600)) # 1000 metres = maximum value for lower limit
    idx_start_max = np.argmin(abs(PolarSpec_pre.f.values - freq_min)) # no peaks considered with wavelengths greater than the bandpass 
    
    # find lowest point in smoothed spectrum, i.e. intertial subrange trough
    idx_inertial_min = ((PolarSpec_pre*PolarSpec_pre.f).sel(f = slice(1/600, 1/300)).sum(dim = 'theta')).argmin(dim=['f'])['f'].values * 1 + idx_start_min
    
    # find highest point in smoothed spectrum, i.e. intertial subrange peak
    idx_inertial_max = ((PolarSpec_pre*PolarSpec_pre.f).sel(f = slice(freq_min, freq_max)).sum(dim = 'theta')).argmax(dim=['f'])['f'].values * 1 + idx_start_max
    
    # if idx_inertial_max is the same as idx_inertial_min the slope is continually positive over the range, meaning the multiplication with 
    # * PolarSpec_pre.f was unnecessary
    if idx_inertial_min == idx_inertial_max:
        
        idx_inertial_min = ((PolarSpec_pre).sel(f = slice(1/600, 1/300)).sum(dim = 'theta')).argmin(dim=['f'])['f'].values * 1 + idx_start_min
        
        idx_inertial_max = ((PolarSpec_pre).sel(f = slice(freq_min, freq_max)).sum(dim = 'theta')).argmax(dim=['f'])['f'].values * 1 + idx_start_max

    
    pi = 3.1415926535
    z = 10                                 # measurements height, 10 metres for CMOD5.N 
    karman = 0.40                          # Karman constant
    T_v = 293                              # virtual potential temperature in Kelvin
    g = 9.8
    rho_air = 1.2
    Cp = 1005    
    iterations = 10
    kolmogorov = 0.5
    dissip_rate = dissip_rate              # 0.6 is low and 2 about average according to fig.4 in Kaimal et al. (1976)
    windspeed_median = sar_ds['windfield'].median().values * 1
    
    # takes entire polar spectrum, averages along all theta angles, multiples density spectrum by area and divides by frequency spacing 
    # to arrive at the Power Spectral Density (PSD) needed for further calculations
    d_theta = PolarSpec.theta.spacing * np.pi / 180   # angular resolution from degrees to radians
    PSD = (PolarSpec*PolarSpec.f).sum(dim = 'theta').values * np.prod([PolarSpec.f.spacing, d_theta]) / PolarSpec.f.spacing
    
    # NOTE! PSD units are in variance over hz (m^2/s^2 / m^-1), further calculations require m^2/s^1 as the equations were made 
    # for temporal measurements rather than the spatial ones. Invoking Taylors hypothesis we assume that the windfield is unchanging though time 
    # and can thus be divided by the mean wind field to arrive at the corrected PSD
    PSD /= windspeed_median
    
    x_axis = 1 / PolarSpec.f.values    # spatial wavelengths in metre, x_axis[idx_inertial_max] should be 1.5 * Zi 
    
    Zi_estimate = x_axis[idx_inertial_max]
    
    # create arrays to store loop results
    C_w_star_normalised_deviation = np.ones(iterations)
    C_w_star = np.ones(iterations)
    C_B = np.ones(iterations)
    C_L = np.ones(iterations)
    C_x = np.ones(iterations)
    C_Psi_m = np.ones(iterations)
    C_corr_fact = np.ones(iterations)
    
    
    for i in range(iterations):
        if i > 0:
            # spatial wavelengths within selected part of inertial subrange
            Lambda = x_axis[idx_inertial_max:idx_inertial_min]

            # select PSD within inertial subrange and apply correction factor
            S = PSD[idx_inertial_max:idx_inertial_min] * C_corr_fact[i-1]**2   # !!! perhaps this should not be squared. FFT might have suared depsndence but
            # afterwards PSD is also divided by windspeed, thus dividing again by C_corr_fact[i-1]
            
            # calculate corrected wind speed
            U_corr = windspeed_median * C_corr_fact[i-1]
            
            # calculate cyclic frequency in per second
            n = 1 / Lambda * U_corr
            
            # calculate dimensionless frequency
            fi = n * Zi / U_corr
        
            # Difference between 0.20 and 0.15 due to isotropy related to cross and along wind analysis (Kaimal et al 1976)
            # if analyses is performed cross wind (i.e. NOT cells), include 4/3 isotropy factor
            if label == 'Wind Streaks':
                pre_w_star = np.sqrt((2 * pi)**(2/3) * fi**(2/3) * n * S / (4/3 * kolmogorov * dissip_rate**(2/3)))
                
            if label == 'Micro Convective Cell':
                pre_w_star = np.sqrt((2 * pi)**(2/3) * fi**(2/3) * n * S / (kolmogorov * dissip_rate**(2/3)))

            # determine weights and calculate weighted mean and std of convective velocity scale
            weights = x_axis[idx_inertial_max:idx_inertial_min] / np.min(x_axis[idx_inertial_max:idx_inertial_min])
            C_w_star[i] = weighted_avg_and_std(pre_w_star, weights)[0]
            C_w_star_normalised_deviation[i] = weighted_avg_and_std(pre_w_star, weights)[1] / np.median(pre_w_star)

            # calculate kinematic heat flux
            C_B[i] =  (C_w_star[i]**3 * T_v) / (g * Zi)
            
            # Monin Obukhov similarity theory
            C_L[i] = - (friction_velocity**3 * T_v) / (C_B[i] * karman * g)

            # structure function and emperical constant from young et al 2000
            C_x[i] = (1 + 16 * abs(z / C_L[i]))**0.25
            C_Psi_m[i] = np.log(((1 + C_x[i]**2) / 2)**2) - 2 * np.arctan(C_x[i]) + pi / 2 
        
            # stability correction factor from young et al 2000
            C_corr_fact[i] = 1 - (C_Psi_m[i] * np.sqrt(Cdn)) / karman
            
        
    # calculate final outputs to return at the end of function
    sigma_u = friction_velocity * np.sqrt(4 + 0.6 * (-Zi / C_L[-1])**(2/3)) 
    L = C_L[-1]
    B = C_B[-1]
    w_star_normalised_deviation = C_w_star_normalised_deviation[-1]
    w_star = C_w_star[-1]
    corr_fact = C_corr_fact[-1]
    H = C_B[-1] * Cp * rho_air       # heat flux
    

    return sigma_u, L, B, w_star, w_star_normalised_deviation, corr_fact, H, Zi_estimate, idx_inertial_min, idx_inertial_max


def sar_variance(sar_ds, interpolation = 'linear', freq_max = 1 / 600, freq_min = 1 / 3000):
    
    """
    Function to calculate spectral characteristics of the windfield contained in 'sar_ds'
    
    Input:

        
    Output:

    """
    
    # calculate power spectral density of windfield using (and correcting for) a Hanning window
    CartSpec = xrft.power_spectrum(sar_ds['sigma0'] , scaling  = 'density', detrend='constant', window = 'hann', window_correction = 'True')  

    # add and swap dimensions
    CartSpec = CartSpec.assign_coords({'f_range':CartSpec.freq_xtrack_m, 'f_azimuth': CartSpec.freq_atrack_m})
    CartSpec = CartSpec.swap_dims({'freq_xtrack_m':'f_range','freq_atrack_m':'f_azimuth'})
    CartSpec.f_range.attrs.update({'spacing':CartSpec.freq_xtrack_m.spacing})
    CartSpec.f_azimuth.attrs.update({'spacing':CartSpec.freq_atrack_m.spacing})

    # calculate energy that falls outside polar spectrum but within Cartesian
    x, y = np.meshgrid(CartSpec.f_range, CartSpec.f_azimuth)

    #################################################################
    ######## polar conversion at low resolution for plotting ########
    #################################################################

    Nt = 720
    theta_spacing = 360 / Nt
    theta = np.arange(0, 360, theta_spacing)
    theta = xr.DataArray(theta, dims='theta', coords={'theta':theta})

    # number of spatial frequencies
    Nf= 300
    fspacing = float(CartSpec.f_range.max() / Nf)
    f = np.arange(0, CartSpec.f_range.max(), fspacing)
    f = xr.DataArray(f, dims='f', coords={'f':f})

    # calculate equivalent coordinates of polar system in cartesian system
    fx_plot = f*np.cos(np.deg2rad(theta))
    fy_plot = f*np.sin(np.deg2rad(theta))

    # interpolate from cartesian spectrum to polar spectrum
    PolarSpec_plot = CartSpec.interp(f_range = fx_plot, f_azimuth = fy_plot, assume_sorted = True, kwargs = {'fill_value':None}, method = interpolation)
    PolarSpec_plot.f.attrs.update({'spacing':fspacing})
    PolarSpec_plot.theta.attrs.update({'spacing':theta_spacing})
    
    ###########################################################
    ######## actual polar conversion and interpolation ########
    ###########################################################
    
    # number of theta's, 3600 yields angular resultion of 0.1 degree, allows for minimal loss during interpolation
    Nt = 3600
    theta_spacing = 360 / Nt
    theta = np.arange(0, 360, theta_spacing)
    theta = xr.DataArray(theta, dims='theta', coords={'theta':theta})

    # number of spatial frequencies
    Nf= 600
    fspacing = float(CartSpec.f_range.max() / Nf)
    f = np.arange(0, CartSpec.f_range.max(), fspacing)
    f = xr.DataArray(f, dims='f', coords={'f':f})

    # calculate equivalent coordinates of polar system in cartesian system
    fx = f*np.cos(np.deg2rad(theta))
    fy = f*np.sin(np.deg2rad(theta))

    # interpolate from cartesian spectrum to polar spectrum
    PolarSpec = CartSpec.interp(f_range = fx, f_azimuth = fy, assume_sorted = True, kwargs = {'fill_value':None}, method = interpolation)
    PolarSpec.f.attrs.update({'spacing':fspacing})
    PolarSpec.theta.attrs.update({'spacing':theta_spacing})

    ########################################
    ######## find inertial subrange ########
    ########################################

    # find indexes in smoothed spectrum belonging to peak and trough of inertial subrange. Requires that frequency spacing of PolarSpec and PolarSpec_pre are identical
    if len(PolarSpec.f) != len(PolarSpec.f):
         print('Frequencies belonging to PolarSpec_pre and PolarSpec are not equal, will result in incorrect indexing of inertial subrange')
     
    idx_start_min = np.argmin(abs(PolarSpec.f.values - 1/600)) # 1000 metres = maximum value for lower limit
    idx_start_max = np.argmin(abs(PolarSpec.f.values - freq_min)) # no peaks considered with wavelengths greater than the bandpass 
     
    # find lowest point in smoothed spectrum, i.e. intertial subrange trough
    idx_inertial_min = ((PolarSpec*PolarSpec.f).sel(f = slice(1/600, 1/300)).sum(dim = 'theta')).argmin(dim=['f'])['f'].values * 1 + idx_start_min
     
    # find highest point in smoothed spectrum, i.e. intertial subrange peak
    idx_inertial_max = ((PolarSpec*PolarSpec.f).sel(f = slice(freq_min, freq_max)).sum(dim = 'theta')).argmax(dim=['f'])['f'].values * 1 + idx_start_max
     
    # if idx_inertial_max is the same as idx_inertial_min the slope is continually positive over the range, meaning the multiplication with 
    if idx_inertial_min == idx_inertial_max:
         
        idx_inertial_min = ((PolarSpec).sel(f = slice(1/600, 1/300)).sum(dim = 'theta')).argmin(dim=['f'])['f'].values * 1 + idx_start_min
         
        idx_inertial_max = ((PolarSpec).sel(f = slice(freq_min, freq_max)).sum(dim = 'theta')).argmax(dim=['f'])['f'].values * 1 + idx_start_max
 
    # used for weighting w* values in inertial subrange
    def weighted_avg_and_std(values, weights):
        """
        Return the weighted average and weighted standard deviation.
        """
        average = np.average(values, weights = weights)
        variance = np.average((values - average)**2, weights = weights)
        return (average, np.sqrt(variance))   
 

    d_theta = PolarSpec.theta.spacing * np.pi / 180   # angular resolution from degrees to radians
    PSD = (PolarSpec*PolarSpec.f).sum(dim = 'theta').values * np.prod([PolarSpec.f.spacing, d_theta]) / PolarSpec.f.spacing
    x_axis = 1 / PolarSpec.f.values
    
    # select PSD within inertial subrange and apply correction factor
    S_pre = PSD[idx_inertial_max:idx_inertial_min] * 1/ x_axis[idx_inertial_max:idx_inertial_min]

    # determine weights and calculate weighted mean and std of convective velocity scale
    weights = x_axis[idx_inertial_max:idx_inertial_min] / np.min(x_axis[idx_inertial_max:idx_inertial_min])
    S = weighted_avg_and_std(S_pre, weights)[0]
    S_normalised_deviation = weighted_avg_and_std(S_pre, weights)[1] / np.median(S_pre)

    return PSD, x_axis, S, S_normalised_deviation





def splitTrainTest(x_data, y_validation, testSize = 0.3, randomState = 42, n_splits = 1, smote = False, equalSizedClasses = False, classToIgnore = None, continuous = False):
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import ShuffleSplit
    from imblearn.over_sampling import SMOTE
    
    """
    Equation to split a stratified dataframe (multiple classes) into arrays of training and testing values.
    
    Optionally choose:
        smote: Wether to artificially oversample from under represented classes to create equal sized classes
        equalSizedClasses: Wether to pick the minimum number of points per class among all classes such that there are equal points per class
            (e.g. if two classes with 20 and 10 points respectively it will select 10 points per class)
        classToIgnore: Int or list of ints with classes to ignore 
        continuous: if False data is split into classes and input ratio of classes is identical in training and testing. If data is continuous no input ratio is considered (unstratified).
                    Does not work with 'equalSizedClasses' or 'classToIgnore'
    
    """
    y_validation = pd.DataFrame(y_validation.values, columns = ['val'])
    
    
    if equalSizedClasses == True:
        # take equal number of points from each error class
        min_number_among_classes = y_validation.groupby('val', group_keys = False).apply(lambda x: x.count()).min().values[0]

        # apply random smaple using minimum number to achieve equal sized classes
        y_validation = y_validation.groupby('val', group_keys = False).apply(lambda x: x.sample(n = min_number_among_classes))
        equalSizeIndexes = y_validation.index
        x_data = x_data[x_data.index.isin(equalSizeIndexes)]
    
    # input is dataframe so convert to arrays
    y_validation = np.ravel(y_validation.reset_index(drop=True).values)
    x_data = x_data.reset_index(drop=True).values
    
    
    # determine whether to use a stratified or unstratified classifier
    if continuous == False: # if objective consists of classes and not continuous values ...
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=testSize, random_state=randomState)  #... split the data taking into account classes ()
    elif continuous == True: # if objectvie consists of v=continuous values
        sss = ShuffleSplit(n_splits=n_splits, test_size=testSize, random_state=randomState)
    

    for train_index, test_index in sss.split(x_data , y_validation):
        
        # if specified ignore specific class
        if classToIgnore != None:
            if np.shape(classToIgnore) == ():
                train_index = train_index[y_validation[train_index]!=classToIgnore]
            else: 
                for i in classToIgnore:
                    train_index = train_index[y_validation[train_index]!=i]

        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_validation[train_index], y_validation[test_index]
        
        if smote == True:
            oversample = SMOTE(random_state=randomState)
            x_train, y_train = oversample.fit_resample(x_train, y_train)
            
        return x_train, x_test, y_train, y_test, test_index, train_index


def envelope(df, param_x, param_y, begin, end, steps =25, log = True):
    """
    function to derive the median and quantiles for a pointcloud from a df with two specified parameters
    """
    import pandas as pd 
    
    placeholder = df.copy()
    
    if log == True:
        bins = np.logspace(begin, end, steps)
    else:
        bins=np.linspace(begin, end, steps)
        
    placeholder['bins_x'], bins = pd.cut(abs(placeholder[param_x]), bins=bins, include_lowest=True, retbins=True)
    placeholder['bins_y'], bins = pd.cut(abs(placeholder[param_y]), bins=bins, include_lowest=True, retbins=True)
        
    bin_center = (bins[:-1] + bins[1:]) /2
    bin_median = abs(placeholder.groupby('bins_x')[param_y].agg(np.nanmedian))#.nanmedian())
    bin_count_x = abs(placeholder.groupby('bins_x')[param_x].count())
    bin_count_y = abs(placeholder.groupby('bins_y')[param_y].count())
    bin_std = abs(placeholder.groupby('bins_x')[param_y].agg(np.nanstd)) #.nanstd())
    bin_quantile_a = abs(placeholder.groupby('bins_x')[param_y].agg(lambda x: np.nanpercentile(x, q = 2.5)))
    bin_quantile_b = abs(placeholder.groupby('bins_x')[param_y].agg(lambda x: np.nanpercentile(x, q = 16)))
    bin_quantile_c = abs(placeholder.groupby('bins_x')[param_y].agg(lambda x: np.nanpercentile(x, q = 84)))
    bin_quantile_d = abs(placeholder.groupby('bins_x')[param_y].agg(lambda x: np.nanpercentile(x, q = 97.5)))
    return bin_center, bin_median, bin_count_x, bin_count_y, bin_std, bin_quantile_a, bin_quantile_b, bin_quantile_c, bin_quantile_d


def plot_envelope(df_plot, hist_steps, title, x_axis_title, alpha = 1):
    
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,figsize=(10,8))
    ################# first figure ##################################
    
    ax1 = axes[0,0]
    im = ax1.scatter( df_plot.y_test, df_plot.y_pred, alpha = alpha, s = 1, c = 'k')
    #                  c = df_plot.y_ML, cmap = 'jet', norm=colors.LogNorm(vmin=10, vmax=1000))
    # cbar = fig.colorbar(im, ax = ax1, location='right', pad = -0.0)
    # cbar.set_label('ratio  validation / corrected', rotation=270, labelpad = 20.3)
    ax1.plot([1, 3000], [1, 3000], '--k')
    ax1.set_ylabel('|Obukhov length| estimate')
    ax1.set_title('Obukhov length prediction (Test)')
    
    #######################  second figure  #############################
    
    ax2 = axes[0,1]
    ax2_2 = ax2.twinx()
    bin_center, bin_median, bin_count_test, bin_count_pred, bin_std, bin_quantile_a, bin_quantile_b, bin_quantile_c, bin_quantile_d = envelope(df_plot, 'y_test', 'y_pred', \
                                                                                                                             -1, 4, steps =hist_steps, log = True)
    ax2.plot(bin_center, bin_median, 'k', label = r'Median$_{estimate}$')
    ax2.fill_between(bin_center, bin_quantile_b, bin_quantile_c, color = 'gray', alpha = 0.8, label = r'$\sigma$')
    ax2.fill_between(bin_center, bin_quantile_a, bin_quantile_d, color = 'gray', alpha = 0.6, label = r'2$\sigma$')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.plot([1, 3000], [1, 3000], '--k')
    ax2.set_ylim(0.5,10000)
    ax2.set_xlim(0.5,10000)
    ax2.legend()
    ax2.set_title('Obukhov length prediction (Test)')
    
    ax2_2.bar(bin_center[1:], bin_count_pred.values[1:], width= np.diff(bin_center), color = 'b')
    ax2_2.tick_params(axis='y', colors='b')
    ax2_2.set_xscale('log')
    ylim = int(bin_count_pred.max()*3)
    ax2_2.set_ylim(0,ylim)
    ax2_2.set_ylabel('occurence', color='b')
    
    
    ###################### third figure ###################
    
    ax3 = axes[1,0]
    im = ax3.scatter( df_plot.y_test, df_plot.y_ML, alpha = alpha, c = 'k', s =1)
    ax3.plot([1, 3000], [1, 3000], '--k')
    ax3.set_ylabel('|Obukhov length| estimate')
    ax3.set_xlabel(x_axis_title)
    ax3.set_title('Obukhov length corrected (Test)')
    
    ###################### fourth figure ###################
    ax4 = axes[1,1]
    ax4_2 = ax4.twinx()
    bin_center, bin_median, bin_count_test, bin_count_pred, bin_std, bin_quantile_a, bin_quantile_b, bin_quantile_c, bin_quantile_d = envelope(df_plot, 'y_test', 'y_ML', \
                                                                                                                             -1, 4, steps =hist_steps, log = True)
    ax4.plot(bin_center, bin_median, 'k', label = r'Median$_{corrected}$')
    ax4.fill_between(bin_center, bin_quantile_b, bin_quantile_c, color = 'gray', alpha = 0.8, label = r'$\sigma$')
    ax4.fill_between(bin_center, bin_quantile_a, bin_quantile_d, color = 'gray', alpha = 0.6, label = r'2$\sigma$')
    
    
    ax4.set_xlabel(x_axis_title)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.plot([1, 3000], [1, 3000], '--k')
    ax4.set_ylim(0.5,10000)
    ax4.set_xlim(0.5,10000)
    ax4.set_title('Obukhov length corrected (Test)')
    ax4.legend()
    
    ax4_2.bar(bin_center[1:], bin_count_pred.values[1:], width= np.diff(bin_center), color = 'b')
    ax4_2.tick_params(axis='y', colors='b')
    ax4_2.set_xscale('log')
    ax4_2.set_ylim(0,ylim )
    ax4_2.set_ylabel('Occurence', color='b')
    
    fig.tight_layout()
    fig.suptitle(title, fontsize = 15)
    fig.subplots_adjust(top=0.88)
    # plt.subplots_adjust(wspace=0.3)
    plt.show()

def plot_envelope_single(df_plot, param_test, param_predict, hist_steps, title, x_axis_title, y_axis_title, alpha = 1, legend = True):

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,figsize=(7,6))
    
    fontsize = 15
    ax2 = axes
    ax2_2 = ax2.twinx()
    bin_center, bin_median, bin_count_test, bin_count_pred, bin_std, bin_quantile_a, bin_quantile_b, bin_quantile_c, bin_quantile_d = envelope(df_plot, param_test, param_predict, \
                                                                                                                             -1, 4, steps =hist_steps, log = True)
    ax2.plot(bin_center, bin_median, 'k', label = r'Median')
    ax2.fill_between(bin_center, bin_quantile_b, bin_quantile_c, color = 'gray', alpha = 0.8, label = r'$68\%%$')
    ax2.fill_between(bin_center, bin_quantile_a, bin_quantile_d, color = 'gray', alpha = 0.6, label = r'$95\%%$')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.plot([1, 3000], [1, 3000], '--k')
    ax2.set_ylim(0.5,10000)
    ax2.set_xlim(0.5,10000)
    # ax2.legend(fontsize = fontsize)
    # ax2.set_title('Obukhov length prediction (Test)', fontsize=fontsize)
    ax2.set_ylabel(y_axis_title, fontsize=fontsize)
    ax2.set_xlabel(x_axis_title, fontsize=fontsize)
    
    ax2.tick_params(which='major', width=3, length=6)
    ax2.tick_params(which='minor', width=1.5, length=3)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    
    # ax2_2.bar(bin_center[1:], bin_count_pred.values[1:], width= np.diff(bin_center), hatch="X", edgecolor = 'k', color = 'gray', alpha = 0.5)
    ax2_2.step(bin_center[1:], bin_count_pred.values[1:], where = 'mid', color = 'r', alpha = 0.5, linewidth = 3)
    ax2_2.step(bin_center[1:], bin_count_test.values[1:], where = 'mid', color = 'b', alpha = 0.5, linewidth = 3)
    # ax2_2.bar(bin_center[1:], bin_count_test.values[1:], width= np.diff(bin_center), color = 'none', edgecolor = 'k')
    ax2_2.tick_params(axis='y', colors='b')
    ax2_2.set_xscale('log')
    ylim = int(bin_count_test.max()*4)
    ax2_2.set_ylim(0,ylim)
    ax2_2.set_ylabel('Hist. count', color='b', fontsize=fontsize)
    
    # -- plot custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    if legend == True:
        legend_elements = [Line2D([0], [0], color='k', lw=4, label = 'Median'),
                           Patch(facecolor='gray', alpha = 0.8, edgecolor='none', label = '68%'),
                           Patch(facecolor='gray', alpha = 0.6, edgecolor='none', label = '95%'),
                           # Patch(hatch="X", edgecolor = 'k', facecolor = 'none', label='Val.'),
                           Line2D([0], [0], color='r', alpha = 0.5, linewidth = 3, label = 'Est.'),
                           Line2D([0], [0], color='b', alpha = 0.5, linewidth = 3, label = 'Val.')]
        plt.legend(handles = legend_elements, framealpha =0.99, edgecolor = 'black', borderpad = 0.2, 
                   loc = 'upper left', ncol = 2, fontsize = fontsize)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    # plt.subplots_adjust(wspace=0.3)
    # plt.show()
    
    return bin_center, bin_median, bin_count_test, bin_count_pred, bin_std, fig
   


def da_averaging(list_of_dataarrays, list_of_dims):

    """
    Input:
        list_of_dataarrays = list of xr.dataarays, input a list of datarrays which are to be stacked along a new dimension
        list_of_dims = list of str, list of dimensions names perpendicular to which the data ought to be satcked and averaged
    
    Output:
        average of datarraays
    """
    
    # -- create dummy dimensions
    dummy_dims = [dim +'_dummy' for dim in list_of_dims]
    # -- create dictionary from lsit of original and dummy dimensions
    dims_dict = dict(zip(list_of_dims, dummy_dims))
    # -- rename main dimensions in spectra (f and theta) such that adding a dimension to the existing main coordinates will not crash
    da_renamed = [arr.swap_dims(dims_dict) for arr in list_of_dataarrays]
    # -- stack separate polar spectra using a new arbitrary dimension
    da_stack = xr.concat(da_renamed, dim="tile", coords= list_of_dims , compat="override")
    # -- take the average over this arbitrary dimension
    da_mean = da_stack.mean(dim = 'tile')
    # -- add back previous coordinates belonging to main dimensions
    # -- NOTE!: since each tile (data array in the stack) should be clipped/interpolated to the same size, all their coordinates are identical
    coordinates_to_add = {dummy_dim: (dummy_dim, list_of_dataarrays[0][dim].values) for dim, dummy_dim in zip(list_of_dims, dummy_dims)}
    da_mean = da_mean.assign_coords(coordinates_to_add)
    # -- rename back to old names
    da_mean = da_mean.rename(dict(zip(dummy_dims, list_of_dims)))
    # -- add old attributes
    for dim in list_of_dims: # for each dimension in the list of dimensions
        try:
            for attribute in list(list_of_dataarrays[0][dim].attrs.keys()):  # add every attribute (add in try loop in case no attributes are contained)
                da_mean[dim].attrs[attribute] = list_of_dataarrays[0][dim].attrs[attribute]
            pass
        except Exception:
            pass
        
            
    return da_mean



def world_maps_single(df_input, variables, statistics, norms, cmaps, rows, columns, title = None, cbar_title = None, shrink = 0.85,  resolution = 2, pad =0.1, fontsize = 15,
                      labelsize = 10, labelpad = 15, cbar_ticks = [-1, 0, 1], cbar_labels = ['-1', '0', '1']):
    
    """
    df_input: dataframe containing all variables selected and 'lon' and 'lat' parameters
    variables: list containing variable names as strings
    statistics: list containing the statistics as string for each variable (e.g. ['mean', 'median'])
    norms: list of colorbar norms per variable (e.g. [matplotlib.colors.Normalize(vmin=0, vmax=360), matplotlib.colors.LogNorm(vmin=-15, vmax=15)])
    cmaps: list of colormap names to be used (e.g. ['Reds_r', 'jet'])
    rows: number of rows in the figure
    colums: number of columns in the figure
    shrink: shrinkage factor of the colorbars
    resolution: resolution of gridded 'lon' and 'lat'
    """

    import matplotlib.colors
    import cartopy as cart
    import cartopy.crs as ccrs
    from scipy.stats import binned_statistic_2d
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    
    # -- create figure
    fig, axes= plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True, figsize=(columns*10,rows*5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    
    # -- set original facecolour to white such that axes can be made gray without making the rest transparant
    fig.set_facecolor('white')
    
    # --------- add coastline to each plot ----------
    # -- for multiple rows and columns
    if len(np.shape(axes)) ==2:
        [(axes[x,y].add_feature(cart.feature.LAND,zorder=100, edgecolor='k'), 
          axes[x,y].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--'), 
          axes[x,y].set_facecolor('silver')
         ) for x in np.arange(0,np.shape(axes)[0],1) for y in np.arange(0,np.shape(axes)[1],1)]
        
    # -- for single row or column
    elif len(np.shape(axes)) ==1: 
        [(axes[x].add_feature(cart.feature.LAND,zorder=100, edgecolor='k'), 
          axes[x].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--'), 
          axes[x].set_facecolor('silver'),
         ) for x in np.arange(0,np.shape(axes)[0],1)]
        
    # -- for single cell
    elif ~(len(np.shape(axes)) >=1): 
          axes.add_feature(cart.feature.LAND,zorder=100, edgecolor='k') 
          gl = axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--') 
          axes.set_facecolor('silver')
          
          gl.top_labels = False
          gl.right_labels = False
          gl.xlabel_style = {'size': labelsize}
          gl.ylabel_style = {'size': labelsize}
          gl.ylocator = mticker.FixedLocator([60, 30, 0, -30, -60])
          gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
          gl.xformatter = LONGITUDE_FORMATTER
          gl.yformatter = LATITUDE_FORMATTER
          axes.set_extent([-180, 180, -75, 75], ccrs.PlateCarree())
          axes.tick_params(axis='both', which='major', labelsize=labelsize)
    # -----------------------------------------------
    
    # -- create grid for 2D histogram
    lat = np.arange(-90,90,resolution)
    lon = np.arange(-180,180,resolution)
    lons,lats = np.meshgrid(lon,lat)
    
    
    
    # -- create empty list to store plotted data
    datas = []

    #######################################
    ####### 2D binned histogram ###########
    #######################################
    for idx, ax in enumerate(np.ravel(axes)):
        if idx <= len(np.ravel(variables))-1:
            data, _, _, _ = binned_statistic_2d(df_input.lon,df_input.lat,values=df_input[variables[idx]], statistic= statistics[idx], bins=[lon, lat], expand_binnumbers=True)
            data = data.T; datas.append(data)
            im = ax.imshow(data, origin="lower", extent = [-180, 180,-90, 90], cmap = cmaps[idx], norm =norms[idx])
            if title == None:
                ax.set_title(r'$\mathbf{%s}(%s)$' %(str(statistics[idx]).replace('_', '\ '), variables[idx].replace('_', '\ ')), fontsize = fontsize)
            else: 
                ax.set_title(title, fontsize = fontsize)
            shrink = shrink
            cbar = fig.colorbar(im, ax=ax, shrink=shrink, pad=pad, ticks = cbar_ticks) #cbar3.set_ticks([-1, -0.5, 0,  0.5, 1])
            if cbar_title == None:
                'ok, no tittle for you then'
            else:
                cbar.set_label(cbar_title, rotation=270, labelpad=labelpad, fontsize = fontsize)
                cbar.ax.set_yticklabels(cbar_labels)
                cbar.ax.tick_params(labelsize=labelsize) 
                
                

#     fig.suptitle(title, fontsize = 20)
    fig.tight_layout()
    
    return fig, datas


#######################################################################################################
# ---------------------------------------------- ML ---------------------------------------------------
#######################################################################################################

def calc_PSD_spatial(KinHeatFlux, Beta, xi, Tv, Alpha = 0.5, Psi = 1, g = 9.81):
    """
    Calculates S(xi) based from kinematic heat flux and several parameters approximates by constants
        
    """
    Alpha                                   # Kolmogorov constant
    g                                       # Gravitational acceleration, m/s**2
    Psi                                     # dimensionless dissipation rate, 
    Tv                                      # Virtual potential temperature, K
    
    part1 = Alpha * Beta / xi **(5/3)
    # part2 = (Psi * KinHeatFlux * g) / (2 * np.pi * Tv)**(2/3)
    part2 = ((Psi * KinHeatFlux * g) / (2 * np.pi * Tv))**(2/3)
        
    PSD_spatial = part1 * part2
    
    PSD_spatial = float(np.where(PSD_spatial < 0, np.nan, PSD_spatial))
    
    return pd.Series([PSD_spatial])



def sigma_u_panofsky(L, Zi, u_star):
    """
    From Panofsky et al 1977
    
    """
    sigma_u = u_star * np.sqrt(4 + 0.6 * (-Zi / L)**(2/3))
    return sigma_u

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values
    
    Source:
        https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72 
    '''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values
        
    Source:
        https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72 
    '''
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' 
    creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
    
    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
    
    Returns
    ----------
    colour map
    
    Source:
        https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72 
    '''
        
    import matplotlib.colors as mcolors 
    
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def outlier_detector(df, column_key_start, column_key_end, pca_comp = 0.80, neighbours = 100, plot_PCA = False):
    """
    Function using SKlearn's 'LocalOutlierFactor' to detec outliers within a specififed range of the input df's columns
    
    Input:
        df: dataframe from which to select sub-dataframe
        column_key_start: name of column to start 
        column_key_end: nume of column end including column itself
        pca_comp: number of Principal components to include (int for number, float for fraction explained variance)
        neighbours: number of neighbours to conisder in localOutlier
        plot_PCA: if true will plot first two principal componenets and colour outliers
    """
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import LocalOutlierFactor
    
    # -- Select specified columns
    idx_measurements = list(df.keys()).index(column_key_start)
    idx_measurements_end = list(df.keys()).index(column_key_end) + 1  # plus 1 because [5:10] does not include idx 10
    data = df.iloc[:, idx_measurements : idx_measurements_end]
    
    # -- change processing depending on whether PCA should be invoked or not
    if (type(pca_comp) == float) | (type(pca_comp) == int) :

        # -- apply standard scaler (outliers will remain)
        x = StandardScaler().fit_transform(data)
        
        # -- select fraction or number of Principal componenets and create PCA
        pca = PCA(n_components = pca_comp)
        
        # -- apply PCA
        X = pca.fit_transform(x)
        
    else:
        X = data
        
    # -- create outlier detector
    outlier_detector = LocalOutlierFactor(n_neighbors=neighbours)
    
    # -- apply detector on data
    inliers = outlier_detector.fit_predict(X)
    
    # -- create df with inliers only
    df_outliers_removed = df[inliers==1] # inliers = 1
    
    if plot_PCA == True:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c = inliers)
        plt.xlabel(r"Principal Component 1"); plt.ylabel(r"Principal Component 2")
        plt.title("Outliers")
        plt.show()
        
    return df_outliers_removed


def scaler_chooser(scaler_str):
    """
    Function outputs a scaler function corresponding to input string
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    
    if scaler_str == "minmax":
        return MinMaxScaler()
    elif scaler_str == "standard":
        return StandardScaler()
    elif scaler_str == "robust":
        return RobustScaler()
   
    return None


def pca_chooser(trial = None, **kwargs):
    """
    Function outputs a pca function corresponding to input
    """
    
    if kwargs.get('pca_value') is not None:
        from sklearn.decomposition import PCA
        type_ref = type(kwargs.get('pca_value'))
        
        if type_ref is dict:
            pca = PCA(**kwargs.get('pca_value'))
        elif type_ref is int or type_ref is float or type_ref is str:
            pca = PCA(kwargs.get('pca_value'))
            # pca = PCA(n_components = kwargs['pca_value'])
        
        if trial != None:
            trial.suggest_categorical('pca_value', [pca.get_params()])
    else:
        pca = None
        if trial != None:
            trial.suggest_categorical('pca_value', [None])
        
    return pca


def poly_chooser(trial = None, **kwargs):
    """
    Function to transform input variables using polynomial features
    """
    
    if kwargs.get('poly_value') is not None:
        from sklearn.preprocessing import PolynomialFeatures
        type_ref = type(kwargs.get('poly_value'))
        
        if type_ref is dict:
            poly = PolynomialFeatures(**kwargs.get('poly_value'))
        elif type_ref is int or type_ref is float:
            poly = PolynomialFeatures(degree = kwargs['poly_value'])
            # poly = PolynomialFeatures(degree = kwargs['poly_value'])
        
        if trial != None:
            trial.suggest_categorical('poly_value', [poly.get_params()])
    else:
        poly = None
        if trial != None:
            trial.suggest_categorical('poly_value', [None])
        
    return poly



def spline_chooser(trial = None, **kwargs):
    """
    Function to transform input variables using spline features
    """
    
    if kwargs.get('spline_value') is not None:
        from sklearn.preprocessing import SplineTransformer
        type_ref = type(kwargs.get('spline_value'))
        
        if type_ref is dict:
            spline = SplineTransformer(**kwargs.get('spline_value'))
        elif type_ref is tuple or type_ref is list:
            spline = SplineTransformer(*kwargs.get('spline_value'))
        elif type_ref is int:
            spline = SplineTransformer(kwargs.get('spline_value'))
            
        if trial != None:
            trial.suggest_categorical('spline_value', [spline.get_params()])
    else:
        spline = None
        if trial != None:
            trial.suggest_categorical('spline_value', [None])
        
    return spline
    

def transformer_chooser(transformer_str, trial = None, n_quantiles = 500):
    """
    Function outputs a transformer function corresponding to input string
    """
    
    from sklearn.preprocessing import QuantileTransformer
    
    if transformer_str == "none":
        return None
    elif transformer_str == "quantile":
        
        # -- if optuna trial is provided to function determine optimal number of quantiles
        if trial != None:
            n_quantiles = trial.suggest_int('n_quantiles', 100, 4000, step = 100)
            
        return QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state = 42)

        

def model_performance(trial, X_train, y_train, cross_validation, pipeline, study_name):
    """
    function for splitting, training, assessing and pruning the regressor
    
    1. First the data is split into K-folds. 
    2. Iteratively an increasing fraction of the training folds and test fold is taken
    3. The regressor is trained and assessed iteratively
    4. If performance is for first iterations is poor, regressor is pruned thus preventing training and testing on full dataset
    
    Input:
        trial: optuna trial (machine learning run with hyperparameters selected by optuna)
        X_train: array of N samples with M measurements
        y_train: array of N validation values
        cross_validation: method of cross valdiation for splitting data into folds
        pipeline: pipeline serving as a regressor for which the optuna trial is optimizing
                  i.e. the pipeline is the regressor being tested
                  
    Output:
        MAE: Median Absolute error of regressor
        MAE_std: standard deviation of MAE
        r2: r2 score of truth and regressor estimate
        r2_std: standard deviation r2
    """
    
    from sklearn.metrics import median_absolute_error, r2_score
    import optuna
    
    # -- turn train and test arrays into temporary dataframes
    df_X_train = pd.DataFrame(X_train)
    df_y_train = pd.DataFrame(y_train)
    
    # -- Retrieve list containing with dataframes for training and testing for each fold
    indexes_train_kfold = list(cross_validation.split(df_X_train))
    
    r2_fracs = []
    r2_frac_stds = []
    MAE_fracs = []
    MAE_frac_stds = []
    
    # -- For each fraction value...
    for idx_fraction, partial_fit_frac in enumerate([0.1, 0.2, 0.3, 0.4, 0.6, 1]):
        
        # -- prepare storage lists
        r2_folds = []
        MAE_folds = []
        
        # -- select the fraction of the fold ...
        for idx_fold, fold in enumerate(indexes_train_kfold):
            
            # ... select a fold 
            fold_X_train = df_X_train.iloc[fold[0]]
            fold_X_test = df_X_train.iloc[fold[1]]
            fold_y_train = df_y_train.iloc[fold[0]]
            fold_y_test = df_y_train.iloc[fold[1]]
            
            # ... retrieve indexes belonging to fraction of the fold 
            idx_partial_fit_train = pd.DataFrame(fold_X_train).sample(frac = partial_fit_frac, random_state= 42).index
            idx_partial_fit_test = pd.DataFrame(fold_X_test).sample(frac = partial_fit_frac, random_state= 42).index

            # ... select fraction of fold 
            fold_X_train_frac = fold_X_train.loc[idx_partial_fit_train]
            fold_X_test_frac = fold_X_test.loc[idx_partial_fit_test]
            fold_y_train_frac = fold_y_train.loc[idx_partial_fit_train]
            fold_y_test_frac = fold_y_test.loc[idx_partial_fit_test]
            
            # -- determine if regressor is boosted model
            regressor_is_boosted = bool(set([study_name]) & set(['lightgbm', 'XGBRegressor'])) #catboost
            
            # ----------------------Modified------------------------ #!!!
            # ... if regressor is boosted ...
            if regressor_is_boosted:
                # -- fit training data and add early stopping function if 200 iterations did not improve data
                
                # -- fit transformers to training fold of training data
                fold_X_train_frac_transformed = pipeline[:-1].fit_transform(fold_X_train_frac)
                # -- transform testting fold of training data
                fold_X_test_frac_transformed = pipeline[:-1].transform(fold_X_test_frac)
                
                # fit pipeline using pre-fitted transformers
                pipeline.fit(fold_X_train_frac_transformed, fold_y_train_frac, 
                              regressor__eval_set=[(fold_X_test_frac_transformed, fold_y_test_frac)],
                              regressor__early_stopping_rounds=20)
                
                # ... make fold prediction
                prediction = pipeline.predict(fold_X_test_frac_transformed)
                
            # ... if regressor is NOT boosted ...
            else:
                # -- fit training data
                pipeline.fit(fold_X_train_frac, fold_y_train_frac)
                
                # ... make fold prediction
                prediction = pipeline.predict(fold_X_test_frac)
    
            # ---------------------------------------------- #!!!
            
            # # ------------------original---------------------------- #!!!
            # # ... if regressor is boosted ...
            # if regressor_is_boosted:
            #     # -- fit training data and add early stopping function if 100 iterations did not improve data
            #     pipeline.fit(fold_X_train_frac, fold_y_train_frac, 
            #                   regressor__eval_set=[(fold_X_test_frac, fold_y_test_frac)],
            #                   regressor__early_stopping_rounds=100)
                
            # # ... if regressor is NOT boosted ...
            # else:
            #     # -- fit training data
            #     pipeline.fit(fold_X_train_frac, fold_y_train_frac)
            # 
            
            # # ... make fold prediction
            # prediction = pipeline.predict(fold_X_test_frac)
            # ---------------------------------------------- #!!!
            
            # ... assess fold performance
            MAE_fold = -1* median_absolute_error(fold_y_test_frac, prediction) # multiply times -1 such that mobjectvie becomes maximisation 
            r2_fold = r2_score(fold_y_test_frac, prediction)
            
            # ... store results to assess performance per fraction
            MAE_folds.append(MAE_fold)
            r2_folds.append(r2_fold)
        
        # -- Calculate mean and std results from all folds per fraction of data
        r2_frac = np.mean(r2_folds)
        MAE_frac = np.mean(MAE_folds)
        r2_fracs_std = np.std(r2_folds)
        MAE_fracs_std = np.std(MAE_folds)
        
        # -- Save results
        r2_fracs.append(r2_frac); r2_frac_stds.append(r2_fracs_std); 
        MAE_fracs.append(MAE_frac);  MAE_frac_stds.append(MAE_fracs_std)
        
        # -- only prune if not applied on fraction containing all datapoints 
        if partial_fit_frac < 1.0:
            
            # -- Report results to decide wether to prune
            trial.report(MAE_frac, idx_fraction)
            
            # -- Prune the intermediate value if neccessary.
            if trial.should_prune():
                raise optuna.TrialPruned()

    # -- final results are those obtained for last fraction (e.g. fraction of 1/1)
    MAE = MAE_fracs[-1]
    MAE_std = r2_frac_stds[-1]
    r2 = r2_fracs[-1]
    r2_std = r2_frac_stds[-1]
    return MAE, r2, MAE_std, r2_std





def create_objective(study_name, write_path, regressor_class, create_params, X_training, y_training, study, cross_validation, **kwargs):
    """
    Nested function containing the optuna objective which has to be mini/maximised
    
    input:
        study_name: str, custom name to save study in folder
        write_path: str, path to save study
        regressor_class: regressor loaded from methods
        create_params: function loading the parameters associated with each regressor
        X_training: m x n array, independent training data
        y_training: m x 1 array, dependent training data
        study: optuna optimisation study 
        cross_validation: method of cross validation
            kwargs: 
                pca_value: int or float, pca compression to apply after scaling of X matrix
                poly_value: int, creates polynomial expension of degree n of X matrix
                spline_value_value: int, float, list, dict, creates spline expansion with n_knots and of degree n of X matrix
            
            
    Example:
        
    methods = {
        "xgboost": (
            XGBRegressor,
            lambda trial: {
                'max_depth': trial.suggest_int("max_depth", 1, 20, log = False),
                'n_estimators': trial.suggest_int("n_estimators", 20, 400, log=False),
                'eta': trial.suggest_float("eta", 1e-4, 1.0, log = True),
                'subsample': trial.suggest_float("subsample", 0.1, 1.0, log = False),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1.0, log = False),
                'random_state': 42
            },
        ),
        "catboost": (
            CatBoostRegressor,
            lambda trial: {
                # 'depth': trial.suggest_int("depth", 1, 16),
                'depth': trial.suggest_int("depth", 1, 10),
                'iterations': trial.suggest_int("iterations", 20, 600, log = True),
                'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-2, 1e1, log = True),
                'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1e0, log = True),
                'rsm': trial.suggest_float("rsm", 1e-2, 1e0, log = False),
                'logging_level': 'Silent',  
                'early_stopping_rounds': 10,
                'random_seed': 42,
            },
        ),
        "lassolars": (
            LassoLars,
            lambda trial: {
                'alpha': trial.suggest_float("alpha", 1e-8, 1e2, log = True),
                'normalize': trial.suggest_categorical("normalize", [False]),
                'random_state': 42,
            },
        ),
    }
    
    
    # -- set study parameters
    timeout = 600 # seconds
    n_trial = 200 # runs
    sampler = TPESampler(seed = 42)
    cross_validation = KFold(n_splits = 5, shuffle = True, random_state= 42)   #  KFold(n_splits=5)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3)
    
    # -- to store study results
    base_directory = '/home/owen/Documents/models/optuna/rolls/'
    folder = 'test'  
    write_path = base_directory + folder
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    
    studies = {}
    for study_name, (regressor, create_params) in methods.items():
        study = optuna.create_study(direction = 'maximize', sampler=sampler, pruner=pruner)
        study.optimize(eq.create_objective(study_name, write_path, regressor, create_params, X_training = X_train,
                                        y_training = y_train, study = study , cross_validation = cross_validation), n_trials=n_trial, timeout=timeout)
        
        # the study is saved during each trial to update with the previous trial (this stores the data even if the study does not complete)
        # here the study is saved once more to include the final iteration
        joblib.dump(study, write_path + '/' + study_name + '.pkl')
        studies[study_name] = study
        
    """
    import joblib
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    

    
    def objective(trial):
    
        # save optuna study
        joblib.dump(study, write_path + '/' + study_name + '.pkl')
        
        # -- Instantiate scaler for independents
        scalers = trial.suggest_categorical("scalers", [None, 'minmax', 'standard', 'robust'])
        scaler = scaler_chooser(scalers)
        
        # -- determine if requested feature combinations improve results
        # -- only suggest this to trial of kwargs contain at least one of the relevant parameters
        if any([i in kwargs for i in ['spline_value', 'pca_value','poly_value']]):
            # -- suggest either to include feature combination or not
            feature_combo = trial.suggest_categorical("feature_combo", [False, True])
            
            # -- if trial will try using feature combinations/compression
            if feature_combo == True:
                # -- instantiate pca compression if relevant kwargs included
                pca = pca_chooser(trial = trial, **kwargs)
    
                # -- instantiate spline transformer if relevant kwargs included
                spline = spline_chooser(trial = trial, **kwargs)
                
                # -- instantiate polynomial transformer if relevant kwargs included
                poly = poly_chooser(trial = trial, **kwargs)
            else:
                pca = spline = poly = None
        else:
            pca = spline = poly = None
            
        # -- Instantiate transformer for dependends
        transformers = trial.suggest_categorical("transformers", ['none', 'quantile'])
        transformer = transformer_chooser(transformers, trial = trial)

        # -- Tune estimator algorithm
        param = create_params(trial)
    
        # -- Create regressor
        regressor = regressor_class()
        regressor.set_params(**param)
        
        # -- Create transformed regressor
        transformed_regressor = TransformedTargetRegressor(
            regressor = regressor,
            transformer = transformer
            )
        
        # # -- Make a pipeline
        # pipeline = Pipeline([('scaler', scaler), ('regressor', transformed_regressor)])
        pipeline = Pipeline([('poly', poly), ('spline', spline), ('scaler', scaler), ('pca', pca), ('regressor', transformed_regressor)])
        
        # -- Assess model performance using specified cross validation on pipeline with pruning
        MAE, r2, MAE_std, r2_std = model_performance(trial, X_training, y_training, cross_validation, pipeline, study_name)
        return MAE 
    
    return objective


def equal_dist(X, y, param, nbins, ranges, samples = 1000, replace = False):
    """
    function to sample dependent and independent dataframes according to specific parameter
    
    Input:
        X: dataframe, contains the independend parameters. Assumed to be of shape N x M
        y: dataseries/dataframe, contains the dependend parameter. Assumed to be of shape N x 1
        param: str, parameter which to bin along specified ranges, must be contained in either X_filt or y_filt
        ranges: tuple of int/ flaot, contains the domain edges over which to bin
        samples: int, number of points to sample from each bin
        replace: boolean, if True will sample with replacement allowing for oversampling, else not
        
    Output:
        X_sampled: resampled version of X
        y_sampled: resampled version of y with identical indexes as X_resampled
        
    Example:
        resample_dict = {'param': 'L_era5', 'nbins': 8, 'ranges' : (np.log10(20),np.log10(300)), 'replace': False, 'samples': 50}
        X, y = equal_dist(X, y, **resample_dict)  
    """
    
    #X_filt = X; y_filt = y; param = 'L_era5'; nbins = 10; ranges = (0,3); replace = True; samples = 1000
    bins = np.linspace(*ranges, nbins + 1)
    
    # -- merge independent and independent dataframes
    placeholder = X.merge(y, left_index = True, right_index = True).copy()
    
    # -- Calculate bins and for each row add corresponding bin 
    placeholder['bins'], _ = pd.cut(placeholder[param], bins=bins, include_lowest=True, retbins=True)
    
    # -- Sample each bin according to desired number of samples and whether to oversample or not (i.e. sample with replacement)
    sampled = placeholder.groupby('bins').apply(lambda x: x.sample(n = samples, replace = replace, random_state = 42))
    
    # -- drop bin index
    sampled.reset_index(drop = True, inplace= True)
    
    # return to original shape
    y_sampled = sampled.iloc[:,-2]
    X_sampled = sampled.iloc[:,:-2]
    
    return X_sampled, y_sampled


def angular_diff(point1, point2):
    
    """
    point1: tuple, y and x coordinates of first point
    point2: tuple, y and x coordinates of second point
    """
    
    distance_y =  point2[0] - point1[0]
    distance_x =  point2[1] - point1[1]
    
    # -- angle of 0 means new point is north of old point in coordinate system
    # angle = 90 mean new point lies to the right, angle =-90 means point lies to the left
    angle = np.arctan2(distance_y, distance_x)*180/np.pi*-1 + 90
    
    return angle

def field_diff(da, n_points):
    import scipy
    
    """
    input:
        da: data array with coordinates atrack_m, xtrack_m
        n_points: number of points for which to compute difference and distance to all other points
    """
    

    #da = data_xr # image_sub
    dim = da.dims
    image_subsample = da[
        np.array([int(x) for x in np.linspace(0, da[dim[0]].size-1, n_points)]),
        np.array([int(x) for x in np.linspace(0, da[dim[1]].size-1, n_points)])
        ]

    image_subsample_dataframe = image_subsample.to_dataframe()
    coordinates = list(image_subsample_dataframe.index)
    # list_of_list = np.array(list(image_subsample_dataframe.index))
    # coordinates = [tuple(x) for x in list_of_list]
    
    # -- calculate angular difference between all points
    list_list_ang_diff = [[angular_diff(coordinates_clipped, coord) for i, coord in enumerate(coordinates[j+1:])] for j, coordinates_clipped in enumerate(coordinates)]
    # -- "explode" list of lists
    ang_diff = [item for sublist in list_list_ang_diff for item in sublist]
    
    # -- calculate distances between all points
    distance_grid = scipy.spatial.distance.cdist(coordinates, coordinates, metric='euclidean')
    
    # -- find diagonal indexes and lower triangle for removal
    low_tri_idx = np.tril_indices(len(distance_grid))
    
    # -- set to undesireable indexes to 0
    distance_grid[low_tri_idx] = 0
    
    # -- ravel grid and remove all points with 0 distance
    distance_ravel = np.ravel(distance_grid)
    distance = np.delete(distance_ravel, distance_ravel == 0)
    
    # -- Repeat but for wind field differencees using same indexes to delete
    param = list(image_subsample_dataframe.keys())[-1]
    field = np.array(image_subsample_dataframe[param])
    
    # Calculate absolute differences between each element 
    difference_grid = field[:,None] - field
    
    # ravel grid and remove all points with 0 distance
    difference_ravel = np.ravel(difference_grid)
    difference = np.delete(difference_ravel, distance_ravel == 0)
    
    df = pd.DataFrame()
    df['difference'] = np.ravel(difference)
    df['distance'] = np.ravel(distance)
    df['ang_diff'] = ang_diff
    df['difference_square'] = df['difference']**2
    df['difference_root'] = abs(df['difference'])**0.5
    
    return df

def semi_variogram(differences, distances, distances_bins, method = 'semi'):
    
    # -- create dataframe
    df = pd.DataFrame()
    df['difference'] = differences
    df['distance'] = distances
    
    # -- split according to bins
    df['bins'] = pd.cut(df['distance'], bins = distances_bins)
    
    # standard semi variogram
    if method == 'semi':
        
        # -- calculate semi variogram
        df['difference_square'] = df['difference']**2
        y_hat = df.groupby('bins')['difference_square'].mean()/2#
        
    # robust semi variogram
    elif method == 'robust':
        
        # -- calculate multiplier
        bins_count = df.groupby('bins')['difference'].count()
        delta = 1/ (2* (0.457 + (0.494/ bins_count)))
        
        # -- calculate semi variogram
        df['difference_root'] = abs(df['difference'])**0.5
        y_hat = delta * (df.groupby('bins')['difference_root'].mean())**4
    
    return y_hat

#%%

################################
### no longer used #############
################################














#%%

# # -- remove if wind direction is exactly range direction  --> unnecessary when using hardcoded ERA5 wind
# number = len(df_val)

# df_val = df_val.loc[abs(((df_val['mean_ground_heading'] + 360)%360)%360 - df_val['wdir_estimate']) >= 0.5]
# df_val = df_val.loc[abs(((df_val['mean_ground_heading'] + 360)%360 + 90)%360 - df_val['wdir_estimate']) >= 0.5]
# df_val = df_val.loc[abs(((df_val['mean_ground_heading'] + 360)%360 + 180)%360 - df_val['wdir_estimate']) >= 0.5]
# df_val = df_val.loc[abs(((df_val['mean_ground_heading'] + 360)%360 + 270)%360 - df_val['wdir_estimate']) >= 0.5]
# number2 =  len(df_val)
# print('# estimated wind direction is exactly along range: ' + str(number - number2))


# # -- remove outliers in all observation columns
# def outlier(df):
#     return np.where((df> np.mean(df) + np.std(df)*4) | (df< np.mean(df) - np.std(df)*4))

# T = df_val.reset_index(drop = True).iloc[:,keep_after_index:].apply(outlier, axis = 0).values[0] # find rows with outliers in any of the estimates parameters
# idx_delete = list(set([x for xs in T for x in xs])) # extract
# print('# outliers: ' + str(len(idx_delete)))
# df_val = df_val.reset_index(drop = True).drop(index = idx_delete)


#%% calcualte 1D powerspectrum 

# def psd1D(image, samplerate, plotting= False, windowed = False, scaled = False, normalised = False):
    
#     """
#     Spectra calculated following Stull (1988)
    
#     other source:
#     https://www.ap.com/blog/fft-spectrum-and-spectral-densities-same-data-different-scaling/
#     https://www.sjsu.edu/people/burford.furman/docs/me120/FFT_tutorial_NI.pdf 
    
#     input:
#         image: 2D wind speed image rotated that greatest avraince direction is either orientated horizontal or vertical
#         samplerate: image samplerate
#         plotting: boolean, whether to plot output PSD
#         windowed: boolean, whether to apply a 1D filter on per row basis prior to calculation of PSD, useful in most cases
#         scaled: boolean, whether to multiply PSD with frequency axis, only used for peak determination following sikora 1997
#         normalised: boolean, whether to normalise such that output is energy spectrum (False) or power spectrum PSD (True)
#                     sum of energy spectrum yields variance
    
#     output:
#         1D spectrum, either energy spectrum or power spectrum depending on normalisation

#     """
    
#     pi = 3.14159265358979
    
#     if windowed == True:
#         # apply window on per row basis, Hanning window for small amplitude (microscale peak) component far off a large amplitude component (near DC)
#         window_0 = np.hanning(image.shape[1])  # 
        
#         # turn into 2D window with orientation following rows
#         base = np.ones(image.shape[0])
#         window1 = np.outer(base, window_0)
        
#         # normalise window
#         window = window1 / np.mean(window1)  #/ 1.63 #np.mean(window1) 
#     else:
#         basey = np.ones(image.shape[0])
#         basex = np.ones(image.shape[1])
#         window = np.outer(basey, basex)
    
#     # N = np.shape(image)[1]                          # number of points in original FFT
#     # fs = 1 / samplerate                             # samplerate
#     # FFT1 = np.fft.fft(image * window, axis=1) / N   # Scaled FFT
#     # FFT2 = abs(FFT1[:,:])**2  #abs(FFT1[:,1:])**2                     # Square norm except DC

#     # # this basically divides by the square root of 2, don't know why it's needed 
#     # # but otherwise the windowed version gains about sqrt(2) extra power
#     # if windowed == True:
#     #     FFT2 = FFT2 * 1 #8/3 #/ 1.5 #*  1.63 #/ np.sqrt(1 / np.mean(window1))  
    
#     # if np.shape(FFT2)[1] % 2 == 0:                  # if ODD (since len(FFT2) = N - 1)
#     #     FFT3 = 2 * FFT2[:, :np.shape(FFT2)[1]//2]   # multiply times 2
#     # else:                                           # if EVEN again multiply times 2...
#     #     FFT3 = 2 * FFT2[:, :int(np.ceil(np.shape(FFT2)[1]/2))]   
#     #     FFT3[:,-1] /= 2                             # ...except Nyquist
#     fs = 1 / samplerate
#     N = np.shape(image)[1]                          # number of points in original FFT                        # samplerate
#     FFT1 = np.fft.fft(image * window, axis=1) / N   # Scaled FFT
#     FFT2 = abs(FFT1[:,:])**2  #abs(FFT1[:,1:])**2                     # Square norm except DC
#     FFT3 = 2 * FFT2[:, :1 + N // 2]
#     if N % 2 == 0:                  # if ODD (since len(FFT2) = N - 1)
#         FFT3[:,-1] /= 2                             # ...except Nyquist

        
#     NFFT = np.shape(FFT1)[1] + 0                    # number of points in FFT )
#     delta_n = fs / NFFT                             # calculated bin width
#     FFT4 = FFT3 / delta_n                           # final scaling
    
#     if normalised  == True:
#         psd1D_1 = FFT4  
#     else:
#         psd1D_1 = FFT4 * delta_n                    # if sum of output should yield sigma, undo scaling
    
#     psd1D = np.nanmean(psd1D_1, axis = 0)           # average over all rows to get 1D PSD
    
#     # to go from spatial wavelength spectra to temporal frequency spectra, divide by mean windspeed (Kaimal et al 1972)
#     if normalised == True:
#         psd1D /= np.mean(image)
    

#     ####################################
#     #### Plot with kolmogorov below ####
#     ####################################
    
#     x_axis = 2*pi*(1/((1/np.arange(1,np.shape(psd1D)[0]+1))*(2*samplerate*np.shape(psd1D)[0])))  # in radian?
    
#     # Select kolmogorov drop off 
#     begin = 0.012 #0.0042 #0.0009
#     end = 0.03  #   0.01 for 650m for 0.0063 for 1000m       0.002
    
#     axis_kolmogorov = x_axis[np.where((x_axis < end) & (x_axis > begin))]
#     idx_kolmogorov = np.argmin(x_axis < end)
#     value_kolmogorov = psd1D[idx_kolmogorov]
#     a = (1/x_axis/2/pi)[idx_kolmogorov-len(axis_kolmogorov):idx_kolmogorov]**(5/3)
#     kolmogorov = value_kolmogorov * a/ (min(a))
  
#     if scaled == True:
#         psd1D = x_axis*psd1D
#         # needs to scale -5/3 as well, not yet done!!!!
    
#     if plotting == True:
#         plt.figure(figsize=(8,5))
#         # first two and last point of PSD are skipped as these are DC (or effected by DC leakage) or very low (resultiong from nyquist/2)
#         plt.loglog(x_axis[1:-1],psd1D[1:-1], linewidth=3, label = 'Power spectrum')
#         # plt.loglog(x_axis[2:-1],np.nanstd(psd1D_1[:, 2:-1], axis = 0)  , linewidth=3, c = 'red')
#         # plt.loglog(x_axis[2:-1],psd1D[2:-1] + 2* np.nanstd(psd1D_1[:, 2:-1], axis = 0)  , linewidth=3, c = 'red')
        
#         plt.loglog(axis_kolmogorov, kolmogorov,'C3--',  linewidth=3, label = '-5/3')
#         plt.title('One-sided PSD')
#         plt.xlabel('$K\ [radians\ m^{-1}]$')
#         plt.legend(ncol=1, fontsize = 12)
        
#         if scaled == True:
#             plt.ylabel('K*S(K) [arbitrary]')
#         elif normalised == True:
#             plt.ylabel(r'S(K) [$\frac{m^2}{s^2}/Hz$]')
#         else:
#             plt.title('One-sided PSD for variance')
#             plt.ylabel(r'S(K) [$\frac{m^2}{s^2}$]')
#     return psd1D, window, psd1D_1, FFT1

# #%%

# def longWaveFilter(image, samplerate, plotting = False):
#     import cv2
#     from scipy import ndimage
#     """
#     Calculate long wave components (>10km) of NRCS image.
#     Use remove these components by dividing NRCS by output 
#     Source: Wakkerman 1996
#     """
#     # calculate how many pixels a 10kmx10km filter should be
#     kernelSize =  int(10000//samplerate)   #previous 10km instead of 5km
#     if kernelSize%2!= 1:
#         kernelSize+=1
    
#     # apply Gaussian blur kernel !!!!!! CHANGE TO MEDIAN FILTER !!!!!!!!
#     image_longwave = cv2.GaussianBlur(image, (kernelSize , kernelSize), 0)


# def rotateAndClip(image, rotation, plotting = False):
#     from scipy import ndimage
#     """
#     Clip the image such that only parts of the original image is available

#     Optimum clip retrieved from 
#     https://math.stackexchange.com/questions/828878/calculate-dimensions-of-square-inside-a-rotated-square
#     """
    
#     pi = 3.14159265358979
    
#     # rotate image
#     image_rotated = ndimage.rotate(image, rotation)
    
#     # determine remaining angle (90 degree rotations dont need clipping)
#     # so determine remaining angle after subtracting integer multiple of 90
#     mult = abs(rotation) // 90
#     if mult > 0:
#         rem_rotation = abs(rotation) % (90 * mult) * np.sign(rotation)
#     else:
#         rem_rotation = rotation
    
#     # define shape of input image
#     h  = image.shape[0]
#     w  = image.shape[1]
    
#     # formulae retrieved from attached link
#     a = int(np.ceil(h/(np.cos(abs(rem_rotation)*pi/180)+np.sin(abs(rem_rotation)*pi/180))))
#     b = int(np.ceil(a*np.cos(rem_rotation*pi/180)))
#     c = int(np.ceil(np.sin(abs(rem_rotation)*pi/180)*b))
#     d = int(np.ceil((h-b)*np.sin((90-abs(rem_rotation))*pi/180)))
    
#     # add a couple -1's and +1's to indexes in order to remove empty data on the edges
#     image_rotated_clipped = image_rotated[c+1:(c + a )-1, d+1:(d + a)-1]
    
#     if plotting == True:
#         plt.imshow(image_rotated_clipped, cmap = 'Greys_r', origin = 'lower')

#     return image_rotated_clipped


#%% Calculate angle of short wave components 

# def peakShortwave(psd2D, samplerate, plotting = False):
#     from scipy import ndimage
#     from matplotlib.patches import Arrow
#     """
#     Calculated the local peak in a 2D PSD 
#     between spectral wavelengths of 300 and 1000 meter (approx lower bound of roll vortices)
#     """
#     pi = 3.1415926535
#     h  = psd2D.shape[0]
#     w  = psd2D.shape[1]
#     wc = w//2
#     hc = h//2
#     Y, X = np.ogrid[0:h, 0:w]
#     r    =(1/np.hypot(X - wc, Y - hc).astype(np.int)*2*samplerate*(len(psd2D)//2))
    
#     # psd2D = psd2D_filtered
#     psd2D_weighted = psd2D #10**(psd2D/10)/r
#     psd2D_weighted_averaged = ndimage.gaussian_filter(psd2D_weighted,3)
#     psd2D_weighted_averaged_clipped = np.where((r<300) | (r>3000), np.min(psd2D_weighted_averaged), psd2D_weighted_averaged)
    
#     idx_max = np.unravel_index(np.argmax(np.where(Y>=hc,psd2D_weighted_averaged_clipped, -99)), psd2D_weighted_averaged_clipped.shape)

#     if idx_max[1]< wc:
#         O = abs(idx_max[0]-hc)
#         A = abs(idx_max[1]-wc)
#         angle = 90 -np.arctan(O/A)*180/pi
#     else:
#         O = abs(idx_max[0]-hc)
#         A = abs(idx_max[1]-wc)
#         angle2 = np.arctan(O/A)*180/pi
#         angle =  -(90- angle2)


#     if plotting == True:
        
#         fig, (ax1) = plt.subplots(1, 1,figsize=(8,6))

#         axis = 2*pi*(1/((1/np.arange(1,np.shape(psd2D)[0]//2+1))*(2*samplerate*np.shape(psd2D)[0]//2)))
#         vmin = np.percentile(psd2D_weighted_averaged,23.5)
#         vmax = np.percentile(psd2D_weighted_averaged,99.9)
#         plt.ylabel('Wavenumber [rad / metre]')
#         plt.xlabel('Wavenumber [rad / metre]')
#         plt.title('Spectral peak in Gaussian filtered 2D psd [arbitrary]')
#         cbar1 = ax1.imshow(psd2D_weighted_averaged, cmap = 'viridis', origin = 'lower', vmin = vmin , vmax = vmax, extent=[-axis[-1],axis[-1],-axis[-1],axis[-1]])
        
#         plt.colorbar(cbar1,ax=ax1)
        
#         idx_max_rad_0 = 2*pi/(1/(idx_max[0]-wc)*(samplerate*np.shape(psd2D)[0]))
#         idx_max_rad_1 = 2*pi/(1/(idx_max[1]-hc)*(samplerate*np.shape(psd2D)[0]))
        

#         dy, dx = (idx_max_rad_1, idx_max_rad_0) 
#         patches = Arrow(0, 0, dy, dx, width=0.002, color='red')
        
#         ax1.add_patch(patches)
#         plt.scatter(dy, dx, s=30, c='red', marker='o')
#         plt.show()
#     return angle, psd2D_weighted_averaged_clipped, r

# #%%

# def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
#     import math
#     """
#     source : https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
    
#     this function fits a gaussian function to the centre structure in the 2D spectrum in order to estimate wind direction
    
#     """
#     theta = math.radians(math.degrees(theta)%360*np.sign(theta))
#     x, y = xdata_tuple
#     xo = float(xo)
#     yo = float(yo)    
#     a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
#     b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
#     c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
#     g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
#     return g.ravel()


# def GaussFit(PSD2Fit, K_cutoff, samplerate = 50, plotting = False):
#     from scipy.optimize import curve_fit
#     from scipy import ndimage
#     """
#     Estimate angle based on 2D gaussian fit to long wavelengths in filterd 2DPS
#     """
#     pi = 3.1415926535
#     axis = 2*pi*(1/((1/np.arange(1,np.shape(PSD2Fit)[0]//2+1))*(2*samplerate*np.shape(PSD2Fit)[0]//2)))
#     idx = np.argmax(axis>= K_cutoff)

#     #create mesh grid based on data input size
#     h  = PSD2Fit.shape[0]
#     w  = PSD2Fit.shape[1]
#     hc = h//2
#     wc = w//2
#     x = np.linspace(-axis[idx], axis[idx], 2*idx)
#     y = np.linspace(-axis[idx], axis[idx], 2*idx)
#     x, y = np.meshgrid(x, y)
#     Y, X = np.ogrid[0:h, 0:w]
#     rr  = 1 #(1/np.hypot(X - wc, Y - hc).astype(np.int)*2*samplerate*(len(PSD2Fit)//2))
    
#     PSD2Fit = 10*np.log10(PSD2Fit)
#     # provide initial guess for gaussian fit and apply Gaussian fit
#     initial_guess = (np.max((PSD2Fit/rr)[hc-idx:hc+idx,hc-idx:hc+idx]), 0, 0, K_cutoff/2, K_cutoff/2, 0, np.mean((PSD2Fit/rr)[hc-idx:hc+idx,hc-idx:hc+idx])/2)
#     # initial_guess = (np.max((PSD2Fit/rr)[hc-idx:hc+idx,hc-idx:hc+idx]), 0, 0, 0.000, 0.000, 0, 0)
#     fitted_param, pcov = curve_fit(twoD_Gaussian, (x, y), ((PSD2Fit/rr)[hc-idx:hc+idx,hc-idx:hc+idx]).ravel(), p0=initial_guess, maxfev = 50000)
#     data_fitted = twoD_Gaussian((x, y), *fitted_param).reshape((2*idx),(2*idx))

#     ###################
#     # calculate angle #
#     ###################
    
#     # artificiallly increase data_fitted resolution to increase angle resolution
#     # COMMENTED 
#     data_fitted_zoomed = ndimage.interpolation.zoom(data_fitted,4) 
    
#     h2  = data_fitted_zoomed.shape[0]
#     w2  = data_fitted_zoomed.shape[1]
#     wc2 = w2//2
#     hc2 = h2//2
#     Y2, X2 = np.ogrid[0:h2, 0:w2]
#     r    = np.hypot(X2 - wc2, Y2 - hc2).astype(np.int)

#     rangeRing = np.where((r<hc2//2) | (r>hc2//2), np.min(data_fitted_zoomed), data_fitted_zoomed)
#     idx_max = np.unravel_index(np.argmax(np.where( Y2>=hc2, rangeRing, -99)), rangeRing.shape)

#     if idx_max[1]< wc2:
#         O = abs(idx_max[0]-hc2)
#         A = abs(idx_max[1]-wc2)
#         angle = 90 -np.arctan(O/A)*180/pi
        
#     else:
#         O = abs(idx_max[0]-hc2)
#         A = abs(idx_max[1]-wc2)
#         angle2 = np.arctan(O/A)*180/pi
#         angle =  -(90- angle2)
    

#     # plot results
#     if plotting == True:
#         fig = plt.figure(figsize=(8,6))
#         ax = fig.add_subplot(111)
#         # circ()
#         plt.title('Approximate orientation: %1.0i (+-180) degrees \n' %angle + ' For structures w. spatial wavelengths > %1.4f rad/meter' %K_cutoff)
#         plt.ylabel('Wavenumber [rad / metre]')
#         plt.xlabel('Wavenumber [rad / metre]')
#         vmin = np.percentile(PSD2Fit/rr,1)
#         vmax = np.percentile(PSD2Fit/rr,99.5)
#         cbar1 = ax.imshow((PSD2Fit/rr).ravel().reshape(np.shape(PSD2Fit)[0], np.shape(PSD2Fit)[1]), vmin = vmin, vmax = vmax, origin = 'lower', extent=[-axis[-1],axis[-1],-axis[-1],axis[-1]])
#         ax.contour(x, y, data_fitted, 3, colors='k')
#         plt.colorbar(cbar1,ax=ax)
        
#         idx_max_rad_0 = 2*pi/(1/(idx_max[0]-wc2)*(samplerate*np.shape(PSD2Fit)[0]))
#         idx_max_rad_1 = 2*pi/(1/(idx_max[1]-hc2)*(samplerate*np.shape(PSD2Fit)[0]))
        
#         from matplotlib.patches import Arrow
#         dy, dx = (idx_max_rad_1, idx_max_rad_0) 
#         print(dy)
#         patches = Arrow(0, 0, dy, dx, width=0.008, color='red')
        
#         ax.add_patch(patches)
#         # plt.scatter(dy, dx, s=30, c='red', marker='o')
#         plt.show()

#     return data_fitted, angle, fitted_param


# #%%
# def bandpass(image, upperwavelength, lowerwavelength, samplerate, kernelSize = 51, plotting = False):
#     from scipy import signal

#     N = kernelSize
#     if not N % 2: N += 1  # Make sure that N is odd.

#     # Compute sinc filter.
#     h  = N; w  = N
#     wc = w/2; hc = h/2
#     Y, X = np.ogrid[-hc:hc, -wc:wc]
    
#     # compute cutoff frequencies
#     fc_low = samplerate / lowerwavelength   # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
#     fc_high = samplerate / upperwavelength

#     ######################
#     # compute 2D sinc function
#     low = np.sinc(2 * fc_low * X) * np.sinc(2 * fc_low * Y)
#     high = np.sinc(2 * fc_high * X) * np.sinc(2 * fc_high * Y)
#     s_low = low    # https://en.wikipedia.org/wiki/Sinc_filter
#     s_high = high
    
#     # Compute Hamming window.
#     w_low = HammingWindow(s_low)
#     w_high = HammingWindow(s_high)
    
#     # Multiply sinc filter by window.
#     f_low = s_low * w_low
#     f_high = s_high * w_high
    
#     # Normalize to get unity gain.
#     kernel_lowpass_lowfreq = f_low / np.sum(f_low)
#     kernel_lowpass_highfreq = f_high / np.sum(f_high)
    
#     # convert lowpass filter at highest frequency to high pass filter (1 - low pass) and multiply with low pass filter to get band pass
#     kernel_bandpass = np.fft.ifft2(np.fft.ifftshift((1 - abs(np.fft.fftshift(np.fft.fft2(kernel_lowpass_highfreq)))) * abs(np.fft.fftshift(np.fft.fft2(kernel_lowpass_lowfreq)))))
    
#     #apply kernel and calculate spectrum of filtered image
#     image_filt = signal.convolve2d(image, kernel_bandpass, boundary='symm', mode='same')
#     image_filt_freq = twoDPS(image_filt, samplerate)
    
#     #plot frequency domain response of filter
#     freq_domain =  abs(np.fft.fftshift(np.fft.fft2(kernel_bandpass)))
    
#     ######################
#     if plotting == True:
#         plt.figure()
#         plt.plot(freq_domain[N//2])
#         # plt.plot(freq_domain[N//2])
#         # plt.plot(test[N//2])
#     return image_filt, image_filt_freq, freq_domain


# def twoDPS_test(image, samplerate, plotting = False):
#     """
#     Calculate 2D power spectrum of input NRCS data 
#     """    
#     # calculate sampling frequency
#     fs = 1 / samplerate
    
#     # found number of points in twoD array 
#     N = 1
#     for dim in np.shape(image): N *= dim
    
#     # create window
#     window = HammingWindow(image)
#     # F1 = np.fft.fft2(np.array(image) * window / np.mean(window)) / N
#     F1 = np.fft.fft2(np.array(image) / N) 
#     # low spatial frequencies are in the center of the 2D fourier transformed image.
#     F2 = np.fft.fftshift( F1 )
#     # Calculate a 2D power spectrum
#     psd2D = np.abs( F2 )**2 
    
#     h  = psd2D.shape[0]
#     w  = psd2D.shape[1]
    
    
#     # FFT1 = np.fft.fft(image * window, axis=1) / N   # Scaled FFT
#     # FFT2 = abs(FFT1[:,:])**2  #abs(FFT1[:,1:])**2                     # Square norm except DC
#     # FFT3 = 2 * FFT2[:, :1 + N // 2]
#     # if N % 2 == 0:                  # if ODD (since len(FFT2) = N - 1)
#     #     FFT3[:,-1] /= 2                             # ...except Nyquist
    
    

    
#     if plotting == True:
#         vmin = np.percentile(10*np.log10(psd2D),50)
#         vmax = np.percentile(10*np.log10(psd2D),99.0)
#         pi = 3.1415926535
#         axis = 2*pi*(1/((1/np.arange(1, h // 2+1))*(2*samplerate* h //2)))
        
#         fig, (ax1) = plt.subplots(1, 1,figsize=(8,6))
#         plt.title('Two-dimensional power spectra [dB]')
#         plt.ylabel(r'$k$ [$rad\ m^{-1}$]')
#         plt.xlabel(r'$k$ [$rad\ m^{-1}$]')
#         cbar1 = ax1.imshow(10*np.log10(psd2D), vmin=vmin, vmax=vmax , origin = 'lower', extent=[-axis[-1],axis[-1],-axis[-1],axis[-1]])
#         plt.colorbar(cbar1,ax=ax1)

#     return psd2D 
