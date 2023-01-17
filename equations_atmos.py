#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 10:16:55 2022

@author: owen
"""

import numpy as np
import xarray as xr
import xsar
import cv2
import os
import xrft
from scipy import ndimage
import cmod5n


def MAD(data):
    """
    Calculates the median absolute difference (MAD)
    """
    median = np.nanmedian(data)
    MAD = np.nanmedian(abs(data - median))
    return MAD

def contourStats(da_contour):
    """
    Calculates statistics from spectral contours
    
    Input:
        da_contour: dataArray, datarray contoining a single radial over which statistics are calculated
        
    Output:
        first order statistics
    """
    
    contour_val = da_contour.values
    mean = np.mean(contour_val)
    median = np.median(contour_val)
    std = np.std(contour_val)
    mad = MAD(contour_val)
    return mean, median, std, mad

def AngleDiffPolar(angle1, angle2):
    """
    Calculates the smallest angular difference taking into acount the 360 to 0 jump
    if clockwise movement from angle 1 to angle 2 is shortesdt then the angular difference is positive, else negative
    
    """
        
    AngleDiff = (angle1 - angle2 + 360 ) % 360
    AngleDiffCounterClock = AngleDiff - 360
    AngleDiffMin = min([abs(AngleDiff), abs(AngleDiffCounterClock)])
    
    AngleDiff_plus1 = ((angle1 + 0.001) - angle2 + 360 ) % 360
    AngleDiffCounterClock_plus1 = AngleDiff_plus1 - 360
    AngleDiffMin_plus1 = min([abs(AngleDiff_plus1), abs(AngleDiffCounterClock_plus1)])
    
    if AngleDiffMin_plus1 > AngleDiffMin:
        AngleDiffMin *= -1
            
    return AngleDiffMin


def HammingWindow(image):
    """
    Creates a Hamming window of equal size to the input image Apply Hamming window on input 
    
    Input:
        image: 2D array
        
    Output:
        window: 2D array, of equal size to input image
    """
    
    #create 1D Hamming windows
    windowx = np.hamming(image.shape[1]).T
    windowy = np.hamming(image.shape[0])
    
    #meshgrid to combine both 1-D filters into 2D filter
    windowX, windowY = np.meshgrid(windowx, windowy)
    window = windowX*windowY
    
    return window


# used for weighting w* values in inertial subrange
def weighted_avg_and_std(values, weights):
    """
    Returns the weighted average and weighted standard deviation.
    """
    average = np.average(values, weights = weights)
    variance = np.average((values - average)**2, weights = weights)
    return (average, np.sqrt(variance))


def da_averaging(list_of_dataarrays, list_of_dims):
    """
    Input:
        list_of_dataarrays = list of xr.dataarays, input a list of datarrays which are to be stacked along a new dimension
        list_of_dims = list of str, list of dimensions names perpendicular to which the data ought to be stacked and averaged
    
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


def da_PSD(da_polar_spec, idx_inertial_max = 0, idx_inertial_min = -1):
    
    """
    calculates 1D spectrum from 2D density spectrum and returns frequency normalised and averaged PSD
    
    Input: 
        da_polar_spec: dataArray, density spectrum with coordinates theta (anugular) and f (frequency) with 
        idx_inertial_max: int, index belonging to maximum inertial subrange spectral amplitude (inertial subrange peak)
        idx_inertial_min: int, index belonging to minimum inertial subrange spectral amplitude (inertial subrange trough)
        
    Output: 
        PSD: dataArray, 1D PSD    
        S: float, frequency normalised and averaged spectral amplitude within inertial subrange
        S_std: float, standard deviation of spectrum to compute S
    """
    
    # -- calculate density parameters
    d_theta = da_polar_spec.theta.spacing * np.pi / 180   # angular resolution from degrees to radians
    PSD_f = da_polar_spec.f
    
    # -- multiplied times frequency yields density spectrum, sum over theta to get 1D, next multiply times spacings to go to PSD in A^3/f^2
    PSD = (da_polar_spec*PSD_f).sum(dim = 'theta') * np.prod([PSD_f.spacing, d_theta]) / PSD_f.spacing
    
    # -- multiply PSD with 5/3 such that inertial subrange becomes flat (only over indexes within the inertial subrange)
    S_scaled = (PSD*PSD_f**(5/3))[idx_inertial_max:idx_inertial_min]
    
    # calculate frequency weighted average and std for frequency-multiplied PSD
    wavelengths_inertial = 1/PSD_f.values[idx_inertial_max:idx_inertial_min]
    S, S_std = weighted_avg_and_std(S_scaled, wavelengths_inertial)
    S_std_norm = S_std / np.median(S_scaled)
    
    return PSD, S, S_std_norm
    


#######################################################################
#%% main functions


def ds_prepare(sar_ds):
    
    """
    Function takes a sar dataset (with coordinates in pixels), fills the NaN's (where possible), detrends and adds coordinates in metres 
    
    Input: 
        sar_ds: xr.Dataset, sar dataset containing a 'sigma0' field with coordinates 'atrack' and 'xtrack'  
        
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



def ds_windfield(sar_ds, PolarSpecSigma0_smooth, wdir_ambiguity, label, wdir_ref = None, freq_max = 1 / 600, freq_min = 1 / 3000):
    """
    This function derives the orientation of greatest energy within a bandpass of the smoothed polar spectrum
    Next the wind field is calculated using the found orientation with respect to the radar sensor (i.e. range direction) using CMOD5.N
    
    Input:
        sar_ds: sar dataset from 'ds_prepare'
        PolarSpecSigma0_smooth: smoothed polar sigma0 dataset from 'ds_to_polar'
        label: wether to consider the structure to be rolls or cells (adds 90 deg offset)
        wdir_ambiguity: a priory wind direction with which to resolve 180 degree ambiguity
        wdir_ref: wind direction used to compute the wind field, used if you want to use a-priori knowledge not just for resolving 180 deg ambiguity
                  e.g. use wdir_ref = wdir_era5 if you want to use the era5 wind direction as the wind direction for wind-field computations
        freq_max: frequency of bandpass shorter wavelength in 1/m, high frequencies can contain swell
        freq_min: frequency of bandpass upper wavelength in 1/m, low frequencies can contian mesoscale activity
        
    Output:
        sar_ds: updated with a find field with coordinates in metre
        sum_theta: sum of energy per theta within bandpass
        angle_pre_conversion: angle in polar spectrum with greatest energy 
    """

    bandpass_subset = PolarSpecSigma0_smooth.sel(f = slice(freq_min, freq_max))

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
    if label == 'Wind Streaks':
        offset = 90 # approximate
    if label == 'Micro Convective Cell':
        offset = 0 # approximate

    wdir = (energy_dir + offset + 360) % 360

    # use ERA5 widr as reference to resolve 180 degree ambiguity        
    diff = abs(wdir_ambiguity - wdir)
    if (diff >= 90) & (diff <= 270):
        wdir = (wdir + 180) % 360
        angle = (angle + 180) % 360
        energy_dir_range = (energy_dir_range + 180) % 360        

    # add correction for convection type and convert from azimuth to range by + 90 (radar viowing direction)
    phi = (angle + offset - 90 ) % 360 
    
    # if you want to use a hardcoded value to calculate the wind field 
    if wdir_ref != None:
        phi = 360 - ((sar_ds.ground_heading.mean().values  + 360 ) % 360 - wdir_ref + 90 ) % 360

    # calculate wind field
    windfield = cmod5n.cmod5n_inverse(sar_ds.sigma0.values, phi , sar_ds.incidence.values, CMOD5 = False, iterations = 10)
        
    sar_ds['windfield'] = (('atrack_m', 'xtrack_m'), windfield)
    
    return sar_ds, sum_theta, angle_pre_conversion, energy_dir, energy_dir_range, wdir



def ds_cartesian_spectrum(sar_ds, smoothing = False, parameter = 'windfield', scaling  = 'density', detrend='constant', window = 'hann', window_correction = 'True'):
    """
    Function to calculate spectral characteristics of the windfield contained in 'sar_ds'
    
    Input:
        sar_ds: xarray dataset
        parameter: parameter in dataset for which spectrum has to be calculated
        scaling: scaling of power spectrum (e.g. density or energy)
        detrend: type of detrending in power-spectrum calculation (constant removes mean a.k.a. 0th frequency)
        window: type of window to apply in power-spectrum calculation 
        window_correction: whether to correct amplitude of spectrum for hypothetical energy loss

    Output:
        CartSpec: cartesian spectrum
        var_cartesian: variance (energy) of cartesian image. To be used to determine effect of wind correction from spectral calculation
        var_beyond_nyquist: variance in the corners of cartesian grid that falls outisde of poalr spectrum
    """
    
    # -- calculate power spectral density of windfield using (and correcting for) a Hanning window
    CartSpec = xrft.power_spectrum(sar_ds[parameter] , scaling = scaling, detrend = detrend, window = window, window_correction = window_correction)  

    # -- optionally smooth spectrum with gaussian filter
    if smoothing == True:
        sigma = [2,2] # arbitrarily selected
        spectrum_smoothed = ndimage.gaussian_filter(CartSpec, sigma, mode='constant')
        ds_CartSpec = xr.Dataset({})
        ds_CartSpec['spectrum'] = CartSpec
        ds_CartSpec['spectrum_smoothed'] = (('freq_atrack_m', 'freq_xtrack_m'), spectrum_smoothed)
        CartSpec = ds_CartSpec['spectrum_smoothed']

    # -- add and swap dimensions
    CartSpec = CartSpec.assign_coords({'f_range':CartSpec.freq_xtrack_m, 'f_azimuth': CartSpec.freq_atrack_m})
    CartSpec = CartSpec.swap_dims({'freq_xtrack_m':'f_range','freq_atrack_m':'f_azimuth'})
    CartSpec.f_range.attrs.update({'spacing':CartSpec.freq_xtrack_m.spacing})
    CartSpec.f_azimuth.attrs.update({'spacing':CartSpec.freq_atrack_m.spacing})

    # -- calculate total energy inside cartesian spectrum, dividing density spcetrum by spacing which is equal to the variance 
    var_cartesian = CartSpec.sum().values * np.prod([CartSpec.f_range.spacing, CartSpec.f_azimuth.spacing])

    # -- calculate energy that falls outside polar spectrum but within Cartesian
    x, y = np.meshgrid(CartSpec.f_range, CartSpec.f_azimuth)
    indexes_beyond_nyquist = np.where(np.sqrt(x**2 + y**2) > CartSpec.f_range.max().values, 1, 0)
    var_beyond_nyquist = (CartSpec * indexes_beyond_nyquist).sum().values * np.prod([CartSpec.f_range.spacing, CartSpec.f_azimuth.spacing])
    
    return CartSpec, var_cartesian, var_beyond_nyquist



def polar_interpolation(cartesian_spectrum, max_f, x_coord = 'f_range', y_coord = 'f_azimuth', Nt = 720, Nf = 300, interpolation = 'linear'):
        """
        Calculates cartesian spectrum from polar spectrum
        
        input:
            cartesian_spectrum: input dataraay with 
            max_freq: maximum frequency to interpolate to
            Nt: number of thetas (angles) to interpolate to
            Nf: number of frequencies to interpolate to
            
        output: 
            PolarSpec: polar spectrum with new coordinates
        """
    
        # create theta grid (from 0 to 360 degrees)
        theta_spacing = 360 / Nt
        theta = np.linspace(0, 360 - theta_spacing, Nt)
        theta = xr.DataArray(theta, dims='theta', coords={'theta':theta})
    
        # create frequency grid
        fspacing = float(max_f / Nf)
        f = np.linspace(0, max_f, Nf)   # use linspace here or np.arange()
        f = xr.DataArray(f, dims='f', coords={'f':f})
    
        # calculate equivalent coordinates of polar system in cartesian system
        fx_plot = f*np.cos(np.deg2rad(theta))
        fy_plot = f*np.sin(np.deg2rad(theta))
    
        # interpolate from cartesian spectrum to polar spectrum
        PolarSpec = cartesian_spectrum.interp(coords = {x_coord: fx_plot, y_coord: fy_plot}, assume_sorted = True, kwargs = {'fill_value':None}, method = interpolation)
        PolarSpec.f.attrs.update({'spacing':fspacing})
        PolarSpec.theta.attrs.update({'spacing':theta_spacing})
        
        return PolarSpec



def spectral_calculations(polar_spectrum, theta_spacing, frequency_spacing, var_windfield, var_cartesian, var_beyond_nyquist, angle_pre_conversion = 0, freq_max = 1 / 600, freq_min = 1 / 3000):
        """
        Calculates information from a polar interpolated spectrum
        
        input:
            polar_spectrum: spectrum with coordinates f (frequency) and theta (angle, in degrees) from which calculations are made
            theta_spacing: spacing between theta (angular resolution of polar spectrum) in radian
            frequency_spacing: spacing between frequencies 
            var_windfield: energy in raw wind field
            var_cartesian: energy in cartesian spectrum of wind field (after windowig and correction)
            var_beyond_nyquist: energy cut from polar spectrum during interpolaation from cartesian to polar
            angle_pre_conversion: angle in bandpassed polar spectrum with greatest energy (used to calculate densities in spectrum)
            freq_max: frequency of bandpass shorter wavelength in 1/m, high frequencies can contain swell
            freq_min: frequency of bandpass upper wavelength in 1/m, low frequencies can contian mesoscale activity
            
        output: 
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
            
            along with several parameters related to the contours of the spectrum
        
        """
        
        # calculate total energy within the polar spectrum, depending on the winddowing effect polar_nrj < cartesian_nrj
        polar_nrj = (polar_spectrum*polar_spectrum.f).sum().values * np.prod([frequency_spacing, theta_spacing])
   
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
            idx_beam1 = [i[0] for i in np.argwhere( (polar_spectrum.theta.values >= slice_min) | (polar_spectrum.theta.values <= slice_max))]
        # if minimum crosses 360 line
        elif (slice_max > angle_pre_conversion) & (slice_min > angle_pre_conversion):
            idx_beam1 = [i[0] for i in np.argwhere( (polar_spectrum.theta.values >= slice_min) | (polar_spectrum.theta.values <= slice_max))]
        # if neither crosses 360 line   
        elif (slice_max > angle_pre_conversion) & (slice_min < angle_pre_conversion):
            idx_beam1 = [i[0] for i in np.argwhere( (polar_spectrum.theta.values >= slice_min) & (polar_spectrum.theta.values <= slice_max))]
          
        # beam 2 (e.g. beam 1 but shifted 180 degree)
        # if maximum crosses 360 line   
        if (slice_max_mirror < angle_pre_conversion_mirror) & (slice_min_mirror < angle_pre_conversion_mirror):
            idx_beam2 = [i[0] for i in np.argwhere( (polar_spectrum.theta.values >= slice_min_mirror) | (polar_spectrum.theta.values <= slice_max_mirror))]
        # if minimum crosses 360 line
        elif (slice_max_mirror > angle_pre_conversion_mirror) & (slice_min_mirror > angle_pre_conversion_mirror):
            idx_beam2 = [i[0] for i in np.argwhere( (polar_spectrum.theta.values >= slice_min_mirror) | (polar_spectrum.theta.values <= slice_max_mirror))]
        # if neither crosses 360 line     
        elif (slice_max_mirror > angle_pre_conversion_mirror) & (slice_min_mirror < angle_pre_conversion_mirror):
            idx_beam2 = [i[0] for i in np.argwhere( (polar_spectrum.theta.values >= slice_min_mirror) & (polar_spectrum.theta.values <= slice_max_mirror))]
   
        # select beam subset (i.e. points within few degrees of angle of greatest variation)
        beam1 = polar_spectrum[:, idx_beam1]
        beam2 = polar_spectrum[:, idx_beam2]
        beams = xr.concat([beam1, beam2], "theta") # add both beams into a single dataraay
   
        polar_nrj_beams = (beams*beams.f).sum().values * np.prod([frequency_spacing, theta_spacing])
   
        # calculate energy within different parts of the polar spectrum
        spectrum_bandpass = polar_spectrum.sel(f = slice(freq_min, freq_max))
        spectrum_highpass = polar_spectrum.sel(f = slice(freq_max, 1))  # all energy in wavelengths shorter than minimum wavelength, including that which falls outside polar but still within cartesian
        spectrum_lowpass = polar_spectrum.sel(f = slice(0, freq_min)) # all energy in wavelengths longer than the maximum, should be energy in mesoscale
        var_bandpass = (spectrum_bandpass*spectrum_bandpass.f).sum().values * np.prod([frequency_spacing, theta_spacing])
        var_highpass = (spectrum_highpass*spectrum_highpass.f).sum().values * np.prod([frequency_spacing, theta_spacing]) + var_beyond_nyquist
        var_lowpass = (spectrum_lowpass*spectrum_lowpass.f).sum().values * np.prod([frequency_spacing, theta_spacing])
   
        spectrum_bandpass_beam = beams.sel(f = slice(freq_min, freq_max))
        var_bandpass_beam = (spectrum_bandpass_beam*spectrum_bandpass_beam.f).sum().values * np.prod([frequency_spacing, theta_spacing])
   
        var_polar = polar_nrj
        var_beam = polar_nrj_beams
        
        polar_effect = var_polar / var_cartesian
        window_effect = var_cartesian / var_windfield
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
        PolarSpecHighpass = polar_spectrum.sel(f = slice(freq_min, 1/polar_spectrum.f.max())) * polar_spectrum.f # 200m since at 
        PolarSpecBandpass = polar_spectrum.sel(f = slice(freq_min, freq_max)) * polar_spectrum.f
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
        
        return beam1, beam2, var_windfield, beams, var_bandpass, var_highpass, var_lowpass, var_bandpass_beam, var_polar, var_beam, \
            polar_effect, window_effect, low_pass_frac, high_pass_frac, bandpass_frac, frac_beam, density_beam, density_bandpass, density_beam_bandpass, freq_25, freq_50, freq_75, \
            angle_diff_max_min_75_25, angle_diff_max_min_75_50, angle_diff_max_min_50_25, angle_diff_min_theta_75_of_min_freq, angle_diff_min_theta_50_of_min_freq, angle_diff_min_theta_25_of_min_freq,\
            angle_diff_theta_25_of_min_freq_theta_75_of_min_freq, angle_diff_theta_50_of_min_freq_theta_75_of_min_freq, angle_diff_theta_75_of_min_freq_theta_50_of_min_freq, \
            angle_diff_theta_75_of_min_freq_theta_25_of_max_freq, angle_diff_theta_75_of_min_freq_theta_50_of_max_freq, angle_diff_theta_50_of_min_freq_theta_25_of_max_freq, \
            angle_diff_theta_25_of_max_freq_theta_75_of_max_freq, angle_diff_theta_50_of_max_freq_theta_75_of_max_freq, angle_diff_theta_25_of_max_freq_theta_50_of_max_freq


def loop1(U_n, z = 10):
    """
    First loop of Young's approach. Calculates surface stress Tau , friction velocity u* and roughness length z_0
    based on neutral wind speed input'
    
    Input:
        U_n: Neutral wind speed at z-meter elevation m/s
        z: elevation of wind speed, 10m for CMOD5.N
        
    Output:
        u*: friction velocity in m/s
        z_0: friction length in m
        C_dn: neutral drag coefficient
        
    """
    
    # define constants
    karman = 0.40                           # Karman constant
    Charnock = 0.044                        # Charnock constant
    g = 9.8                                 # Gravitational acceleration, m/s**2
    z = z                                   # measurements height, 10 metres for CMOD5.N 
    rho_air = 1.2                           # kg/m**3, 1.2 for 20 degrees, 1.25 for 10 degreess                           
    T = 20                                  # temperature in Celcius
    
    # kinematic viscosity of air
    nu = 1.326 * 10**(-5) *(1 + (6.542 * 10**(-3))* T + (8.301 * 10**(-6)) * T**2 - (4.840 * 10**(-9)) * T**3) # m**2/s
    
    # prepare loop of 15 iterations
    iterations = 15
    A_u_star = np.ones(iterations)    # m/s
    A_surface_stress = np.ones(iterations)       # kg/ m / s**2  [Pa]
    A_Cdn = np.ones(iterations)                  # 
    A_z_0 = np.ones(iterations)                  # m

    # Initialise loop with windspeed and iterate with refined estimates of neutral drag coefficient
    for i in range(iterations):
        if i > 0:
            A_u_star[i] = np.sqrt(A_Cdn[i-1] * U_n**2)
            A_surface_stress[i] = rho_air * A_u_star[i]**2
            A_z_0[i] = (Charnock * A_u_star[i]**2) / g + 0.11 * nu / A_u_star[i]
            A_Cdn[i] = (karman / np.log( z / A_z_0[i]) )**2
    
    # calculate stress field based on retrieved constants and windspeed estimates
    # !!! use mean windfield here or windfield? --> if use windfield then u_star will change for different windspeed variances
    surface_stress = rho_air * A_Cdn[-1] *  U_n**2 

    # save friction velocity and friction length based on mean stress field and neutral drag coefficient
    u_star = np.sqrt(surface_stress / rho_air)
    z_0 = (Charnock * u_star**2) / g + 0.11 * nu / u_star
    Cdn = A_Cdn[-1]
    
    return u_star, z_0, Cdn



def loop2B(U_n, u_star, z_0, Zi, Cdn, PolarSpec, label, z = 10, dissip_rate = 1, freq_max = 1 / 600, freq_min = 1 / 3000, freq_lower_lim = 1/300, smoothing = False):
    """
    Second loop of Young's approach. Requires output of loop 1. Recalculates wind field using stability correction.
    Similar to loop two but instead of using wind variance of entire field only uses inertial subrange
    Outputs recalculated parameters, obukhov Length L and kinematic heat flux B
    
    Input:
        U_n: Neutral wind speed at z-meter elevation m/s
        u_star: u_star from loop 1
        z_0: z_0 from loop 1
        Zi: a-priori value for Boundary layer height
        Cdn: neutral drag coefficient from loop 1
        PolarSpec: dataset containing high res. interpolated 'sigma0_detrend' density spectrum on polar grid. To be used for spectral calculations
        label: expected convection form, cells is standard, rolls result in slight modification
        z: wind field measurement height
        dissip_rate: approximately between 0.5 and 2.5 (kaimal et al,  1976)
        freq_max: frequency of bandpass shorter wavelength in 1/m, high frequencies can contain swell
        freq_min: frequency of bandpass upper wavelength in 1/m, low frequencies can contain mesoscale activity
        freq_lower_lim = frequency beyond which no local minima are found to match with ERA5
        smoothing: boolean, whether or not to smooth submitted spectrum for peak and trough finding

    Output:
        sigma_u: estimated wind-field variance
        L: Obukhov length in meters
        B: Kinematic heat flux in metres
        w_star: convective velocity scale in m/s
        w_star_normalised_deviation: std of w_star values in inertial subrange
        corr_fact: stability correction factor
        H: heat flux (from Ocean into atmosphere)
        Zi_estimate: wavelentgh of the spectral peak. In literature this is multipleid times a factor to get Zi
        idx_inertial_min: index corresponding to the inertial subrange minimum in the 1D spectrum derived from the 2D (smoothed) spectrum
        idx_inertial_max: index corresponding to the inertial subrange maximum in the 1D spectrum derived from the 2D (smoothed) spectrum
        

    ###### Zi: lowest inversion height following Kaimal et al 1976, Sikora et al 1997
    
    """
    # PolarSpec = ds_polar_mean
    if smoothing == True:
        sigma = [5,5] # arbitrarily selected
        spectrum_smoothed = ndimage.gaussian_filter(PolarSpec, sigma, mode='constant')
        PolarSpec_smooth = xr.Dataset({})
        PolarSpec_smooth['spectrum'] = PolarSpec#.drop('f_range')
        PolarSpec_smooth['spectrum_smoothed'] = (('f', 'theta'), spectrum_smoothed)
        PolarSpec_smooth = PolarSpec_smooth['spectrum_smoothed']
    else:
        PolarSpec_smooth=PolarSpec
    
    # find indexes in smoothed spectrum belonging to peak and trough of inertial subrange.
    # idx_start_min = np.argmin(abs(PolarSpec_smooth.f.values - freq_max)) # 1/freq_max metres = maximum value for lower limit
    idx_start_max = np.argmin(abs(PolarSpec_smooth.f.values - freq_min)) # no peaks considered with wavelengths greater than the bandpass 
    
    lower_limit = 1/freq_lower_lim 
    # --------------- NOT USING SMOOTHED SPECTRUM FOR PEAKS ----------------#
    # find highest point in smoothed spectrum, i.e. intertial subrange peak
    idx_inertial_max = ((PolarSpec_smooth*PolarSpec_smooth.f**(5/3)).sel(f = slice(freq_min, freq_max)).sum(dim = 'theta')).argmax(dim=['f'])['f'].values * 1 + idx_start_max
    
    x_axis = 1 / PolarSpec.f.values 
    idx_start_min = np.argmin(abs(PolarSpec_smooth.f.values - 1/x_axis[idx_inertial_max])) 
    # find lowest point in smoothed spectrum, i.e. intertial subrange trough
    idx_inertial_min = ((PolarSpec_smooth*PolarSpec_smooth.f**(1)).sel(f = slice(1/x_axis[idx_inertial_max], 1/lower_limit)).sum(dim = 'theta')).argmin(dim=['f'])['f'].values * 1 + idx_start_min
  

    if idx_inertial_min == idx_inertial_max:

        x_axis = 1 / PolarSpec.f.values 
        idx_start_min = np.argmin(abs(PolarSpec_smooth.f.values - 1/x_axis[idx_inertial_max])) 
        idx_inertial_min = ((PolarSpec_smooth*PolarSpec_smooth.f**(0.75)).sel(f = slice(freq_min, 1/lower_limit)).sum(dim = 'theta')).argmin(dim=['f'])['f'].values * 1 + idx_start_min
    if idx_inertial_min == idx_inertial_max:

        x_axis = 1 / PolarSpec.f.values 
        idx_start_min = np.argmin(abs(PolarSpec_smooth.f.values - 1/x_axis[idx_inertial_max])) 
        idx_inertial_min = ((PolarSpec_smooth*PolarSpec_smooth.f**(0.5)).sel(f = slice(freq_min, 1/lower_limit)).sum(dim = 'theta')).argmin(dim=['f'])['f'].values * 1 + idx_start_min
    if idx_inertial_min == idx_inertial_max:

        x_axis = 1 / PolarSpec.f.values 
        idx_start_min = np.argmin(abs(PolarSpec_smooth.f.values - 1/x_axis[idx_inertial_max])) 
        idx_inertial_min = ((PolarSpec_smooth*PolarSpec_smooth.f**(0.25)).sel(f = slice(freq_min, 1/lower_limit)).sum(dim = 'theta')).argmin(dim=['f'])['f'].values * 1 + idx_start_min
                  

    
    pi = 3.1415926535
    z = z                                  # measurements height, 10 metres for CMOD5.N 
    karman = 0.40                          # Karman constant
    T_v = 293                              # virtual potential temperature in Kelvin
    g = 9.8
    rho_air = 1.2
    Cp = 1005    
    iterations = 10
    kolmogorov = 0.5
    dissip_rate = dissip_rate              # 0.6 is low and 2 about average according to fig.4 in Kaimal et al. (1976)

    
    # takes entire polar spectrum, averages along all theta angles, multiples density spectrum by area and divides by frequency spacing 
    # to arrive at the Power Spectral Density (PSD) needed for further calculations
    d_theta = PolarSpec.theta.spacing * np.pi / 180   # angular resolution from degrees to radians
    PSD = (PolarSpec*PolarSpec.f).sum(dim = 'theta').values * np.prod([PolarSpec.f.spacing, d_theta]) / PolarSpec.f.spacing
    
    # var = np.trapz(np.trapz(((PolarSpec*PolarSpec.f)), dx = PolarSpec.f.spacing), dx = d_theta)
    # NOTE! PSD units are in variance over hz (m^2/s^2 / f), further calculations require m^2/s^1 (temporal PSD) as the equations were made 
    # for temporal measurements rather than the spatial ones. Invoking Taylors hypothesis we assume that the windfield is unchanging though time 
    # and can thus be divided by the mean wind field to arrive at the corrected PSD. According to Stull (1988) S_temporal = S_spatial / velocity
    PSD /= U_n
    
    x_axis = 1 / PolarSpec.f.values    # spatial wavelengths in metre, x_axis[idx_inertial_max] should be 1.5 * Zi 
    
    Zi_estimate = x_axis[idx_inertial_max]        # peak wavelength in spectrum
    valley_estimate = x_axis[idx_inertial_min]    # valley wavelength in spectrum
    
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
            S = PSD[idx_inertial_max:idx_inertial_min] * C_corr_fact[i-1]**1   # !!! not be squared. FFT requires squared correction but
            # afterwards PSD is also divided by windspeed, thus dividing again by C_corr_fact[i-1]
            
            # calculate corrected wind speed (corrected for non-neutrality)
            U_corr = U_n * C_corr_fact[i-1]
            
            # calculate cyclic frequency in per second
            n = 1 / Lambda * U_corr
            
            # calculate dimensionless frequency
            fi = n * Zi / U_corr
        
            # Difference between 0.20 and 0.15 due to isotropy related to cross and along wind analysis (Kaimal et al 1976)
            # if analyses is performed cross wind (i.e. NOT cells), include 4/3 isotropy factor
            if label == 'Wind Streaks':
                # pre_w_star = np.sqrt((2 * pi)**(2/3) * fi**(2/3) * n * S / (4/3 * kolmogorov * dissip_rate**(2/3)))
                pre_w_star = np.sqrt((2 * pi)**(2/3) * fi**(2/3) * n * S / (kolmogorov * dissip_rate**(2/3)))
                
            if label == 'Micro Convective Cell':
                pre_w_star = np.sqrt((2 * pi)**(2/3) * fi**(2/3) * n * S / (kolmogorov * dissip_rate**(2/3)))

            # determine weights and calculate weighted mean and std of convective velocity scale
            weights = x_axis[idx_inertial_max:idx_inertial_min] / np.min(x_axis[idx_inertial_max:idx_inertial_min])
            C_w_star[i] = weighted_avg_and_std(pre_w_star, weights)[0]
            C_w_star_normalised_deviation[i] = weighted_avg_and_std(pre_w_star, weights)[1] / np.median(pre_w_star)

            # calculate kinematic heat flux
            C_B[i] =  (C_w_star[i]**3 * T_v) / (g * Zi)
            
            # Monin Obukhov similarity theory
            C_L[i] = - (u_star**3 * T_v) / (C_B[i] * karman * g)

            # structure function and empirical constant from young et al 2000
            C_x[i] = (1 + 16 * abs(z / C_L[i]))**0.25
            C_Psi_m[i] = np.log(((1 + C_x[i]**2) / 2)**2) - 2 * np.arctan(C_x[i]) + pi / 2 
        
            # stability correction factor from young et al 2000
            C_corr_fact[i] = 1 - (C_Psi_m[i] * np.sqrt(Cdn)) / karman
            
        
    # calculate final outputs to return at the end of function
    sigma_u = u_star * np.sqrt(4 + 0.6 * (-Zi / C_L[-1])**(2/3)) 
    L = C_L[-1]
    B = C_B[-1]
    w_star_normalised_deviation = C_w_star_normalised_deviation[-1]
    w_star = C_w_star[-1]
    corr_fact = C_corr_fact[-1]
    H = C_B[-1] * Cp * rho_air       # heat flux
    

    return sigma_u, L, B, w_star, w_star_normalised_deviation, corr_fact, H, Zi_estimate, valley_estimate, idx_inertial_min, idx_inertial_max, PSD



def tiled_spectra(ds, parameter, tiles = 2, list_of_dims = ['f', 'theta']):
    """
    Function to calculate tiled spectra, e.g. for tiles = 2 the input dataray will be split into 2x2 
    
    Input: 
        ds: dataset, input datasat containing with coordinates atrack_m and xtrack_m
        parameter: str,name of parameter in dataarray for which to calculate tiled spectra
        tiles: int, number of tiles in x and y direction
        list_of_dims: list of str, list of dimensions names perpendicular to which the data ought to be stacked and averaged
        
    Output:
        ds_polar_mean: dataArray, averaged spectra of all tiled spectra
        PolarSpectras_plot: dataArray, averaged spectra of all tiled spectra. interpolated to lower resolution for plotting purposes
    
    """

    # -- find shape of dataset
    shapes = np.array(np.shape(ds[parameter]))
    # -- select fine tiles
    grid_size = min(shapes)//tiles # grid size in pixels
    x_sub, y_sub = shapes // grid_size
    
    # -- storage for in loop 
    PolarSpectras = []
    PolarSpectras_plot = []
    
    for k in range(x_sub):
    # for i in tqdm(range(1)):
        for l in range(y_sub):

            # split data into sub tiles
            ds_sub = ds[dict(atrack_m=slice((k)*grid_size, (k+1)*grid_size), xtrack_m=slice((l)*grid_size, (l+1)*grid_size))]
            ds_sub = ds_sub.assign_coords({"atrack_m": ds_sub.atrack_m - np.min(ds_sub.atrack_m)})
            ds_sub = ds_sub.assign_coords({"xtrack_m": ds_sub.xtrack_m - np.min(ds_sub.xtrack_m)})
    
            # -- compute cartesian windfield spectrum for the sub tiles
            cartesian_spectrum_sub, _, _ = ds_cartesian_spectrum(ds_sub, smoothing = False, parameter = parameter, scaling  = 'density', detrend='constant', window = 'hann', window_correction = 'True')
                
            # -- interpolate cartesian spectrum to polar spectrum in coarse and high resolution (the former for calculations, the latter for plotting)
            max_f = cartesian_spectrum_sub.f_range.max().values*1
            PolarSpec_sub = polar_interpolation(cartesian_spectrum_sub, max_f, x_coord = 'f_range', y_coord = 'f_azimuth', Nt = 3600, Nf = 600, interpolation = 'linear')
            PolarSpec_sub_plot = polar_interpolation(cartesian_spectrum_sub, max_f, x_coord = 'f_range', y_coord = 'f_azimuth', Nt = 720, Nf = 300, interpolation = 'linear')
            
            # -- store data for later
            PolarSpectras.append(PolarSpec_sub)
            PolarSpectras_plot.append(PolarSpec_sub_plot)
            
    # -- average and interpolate polar spectra of windfield tiles to a single polar spectra representative of the entire field
    ds_polar_mean = da_averaging(PolarSpectras, list_of_dims=list_of_dims)  
    ds_polar_plot_mean = da_averaging(PolarSpectras_plot, list_of_dims=list_of_dims)  
            
    return ds_polar_mean, ds_polar_plot_mean





















#%%  OLD




def ds_to_polar(sar_ds, parameter = 'sigma0', interpolation = 'linear'):
    
    """
    Input:
        sar_ds: sar dataset containing a 'sigma0_detrend' array with cooresponding coordinates in metres
        interpolation: string specifying method of interpolation between cartesian and polar grid. E.g. 'linear' 'nearest'
        parameter: name of input parameter for which to apply polar conversion m e.g. 'sigma0_detrend'
    
    Output:
        PolarSpec: dataset containing interpolated 'sigma0_detrend' on polar grid. to be used for wind direction orientation
        fx: horizontal coordinates of polar spectrum on cartesian grid
        fy: vertical coordinates of polar spectrum on cartesian grid
    """
    
    # calculate spectrum from detrended image
    # This spectrum is only used to calcualte direction so does not matter if scaling is 'spectrum' or 'density'
    # detrend such energy in DC is removed
    # xrft automatically computes frequency (1 / lambda) along axis which is why ds has to be prepared such that xtrack and atrack are in meters
    spectrum = xrft.power_spectrum(sar_ds[parameter], scaling  = 'density', detrend= 'constant')

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



def ds_spectral_calculations(sar_ds, angle_pre_conversion = 0, interpolation = 'linear', freq_max = 1 / 600, freq_min = 1 / 3000):
    
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
    CartSpec = xrft.power_spectrum(sar_ds['windfield'] , scaling  = 'density', detrend='constant', window = 'hann', window_correction = 'True')  

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
    theta = np.linspace(0, 360 - theta_spacing, Nt)
    theta = xr.DataArray(theta, dims='theta', coords={'theta':theta})

    # number of spatial frequencies
    Nf= 300
    fspacing = float(CartSpec.f_range.max() / Nf)
    f = np.linspace(0, CartSpec.f_range.max(), Nf)   # use linspace here or np.arange()
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
    theta = np.linspace(0, 360 - theta_spacing, Nt)
    theta = xr.DataArray(theta, dims='theta', coords={'theta':theta})

    # number of spatial frequencies
    Nf= 600
    fspacing = float(CartSpec.f_range.max() / Nf)
    f = np.linspace(0, CartSpec.f_range.max() - fspacing, Nf)
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






def estimate_wspd(train, wdir_era5, Zi_era5, label = 'Wind Streaks', output_dir= 'test'):
    

    path_2_save = output_dir + '/solo'
    file_2_save = path_2_save + '/'+ str(train)[121:].replace('.', '___').replace(':', '____') + '.txt'
    
    # create directory to save if doesnt exist already
    if not os.path.exists(path_2_save):
        os.makedirs(path_2_save)
        
    # if file not in directory, proceed with calculations
    if not os.path.exists(file_2_save):
#     if not 2==1:
    
#         prevent single error messing up entire process
        try:
        
            # load image
            sar_ds = xsar.open_dataset(train, resolution = '100m').sel(pol = 'VV')

            freq_max = 1 / 600
            freq_min = 1 / 3000
            interpolation = 'linear'
            save_directory = None  #'/home/owen/Documents/buoy_data/images/spectral_analysis_31_05/'
            dissip_rate = 1

            # prepare dataset by detrending, filling nans and converting units/coordinates
            sar_ds = ds_prepare(sar_ds)   # removed the "eq."


            # only continue of there are no NaNs in the detrended dataset
            if np.isnan(sar_ds.sigma0_detrend).sum() != 0:
                print("detrended nanfilled field still contains NaN's, scene skipped")

            else:
                # calculate cartesian spectrum (of detrended sigma0 field) and coarsely interpolate to polar spectrum 
                PolarSpec_pre, fx_pre, fy_pre = ds_to_polar(sar_ds, interpolation = interpolation)

                # use coarse polar spectrum to estiamte orientation of energy and compute wind field
                sar_ds, sum_theta, angle_pre_conversion, energy_dir, energy_dir_range, wdir_estimate = ds_windfield(sar_ds, PolarSpec_pre, wdir_era5, label, freq_max = freq_max, freq_min = freq_min)

                # compute high resolution spectrum of wind field, convert to polar and derive spectral information
                CartSpec, PolarSpec, PolarSpec_plot, var_cartesian, beam1, beam2, var_beyond_nyquist, var_windfield, beams, var_bandpass, var_highpass, var_lowpass, var_bandpass_beam, var_polar, var_beam, \
                polar_effect, window_effect, frac_lowpass, frac_highpass, frac_bandpass, frac_beam, density_beam, density_bandpass, density_beam_bandpass, fx, fy, fx_plot, fy_plot, freq_25, freq_50, freq_75, \
                angle_diff_max_min_75_25, angle_diff_max_min_75_50, angle_diff_max_min_50_25, angle_diff_min_theta_75_of_min_freq, angle_diff_min_theta_50_of_min_freq, angle_diff_min_theta_25_of_min_freq,\
                angle_diff_theta_25_of_min_freq_theta_75_of_min_freq, angle_diff_theta_50_of_min_freq_theta_75_of_min_freq, angle_diff_theta_75_of_min_freq_theta_50_of_min_freq, \
                angle_diff_theta_75_of_min_freq_theta_25_of_max_freq, angle_diff_theta_75_of_min_freq_theta_50_of_max_freq, angle_diff_theta_50_of_min_freq_theta_25_of_max_freq, \
                angle_diff_theta_25_of_max_freq_theta_75_of_max_freq, angle_diff_theta_50_of_max_freq_theta_75_of_max_freq, angle_diff_theta_25_of_max_freq_theta_50_of_max_freq\
                = ds_spectral_calculations(sar_ds, angle_pre_conversion, interpolation = interpolation, freq_max = freq_max, freq_min = freq_min)    
                
                # loop 1
                sar_ds, friction_velocity, z_0, Cdn = loop1(sar_ds)
    
                # loop 2B
                sigma_u, L, B, w_star, w_star_normalised_deviation, corr_fact, H, Zi_estimate, valley_estimate, idx_inertial_min, idx_inertial_max \
                = loop2B(sar_ds, friction_velocity, z_0, Zi_era5, Cdn, PolarSpec_pre, PolarSpec, label, dissip_rate = dissip_rate, freq_max = freq_max, freq_min = freq_min)  

                # additional info on sigma0 spectrum
                S, S_normalised_deviation = sar_variance(sar_ds, interpolation = interpolation, freq_max = freq_max, freq_min = freq_min)
    		
                ##########################################
                #### Energy spectrum contour analysis ####
                ##########################################
        
                speccc = PolarSpec_plot.sel(f = slice(freq_min, 200)) * PolarSpec_plot.f
                speccc_ref = PolarSpec_plot.sel(f = slice(freq_min, freq_max)) * PolarSpec_plot.f
                cumsum_scaled = (speccc.cumsum(dim = 'f') / speccc_ref.sum() * len(speccc.theta) )
                contours = cumsum_scaled.rolling(theta=1).mean().interpolate_na(dim = 'theta', method= 'linear', fill_value= 'extrapolate')

                percentile_25 = (cumsum_scaled>0.25).argmax(dim = 'f')
                percentile_50 = (cumsum_scaled>0.5).argmax(dim = 'f')
                percentile_75 = (cumsum_scaled>0.75).argmax(dim = 'f')
                percentile_diff_75_25 = xr.where((percentile_75 - percentile_25)<0, np.max(percentile_75) ,(percentile_75 - percentile_25))

                theta_diff_min = percentile_diff_75_25.theta[percentile_diff_75_25.min().values*1].values*1
                theta_25_of_min_freq = percentile_25.theta[percentile_25.min().values*1].values*1
                theta_75_of_min_freq = percentile_75.theta[percentile_75.min().values*1].values*1
                theta_25_of_max_freq = percentile_25.theta[percentile_25.max().values*1].values*1
                theta_75_of_max_freq = percentile_75.theta[percentile_75.max().values*1].values*1

                atrack_plot = speccc.freq_atrack_m[percentile_diff_75_25, :]
                xtrack_plot = speccc.freq_xtrack_m[percentile_diff_75_25, :]

                cumsum_scaled = (speccc.cumsum(dim = 'f') / speccc.sum())
                cumsum_scaled_sum_theta = cumsum_scaled.sum(dim = 'theta')

                mean_25th, median_25th, std_25th, mad_25th = contourStats(percentile_25)
                mean_50th, median_50th, std_50th, mad_50th = contourStats(percentile_50)
                mean_75th, median_75th, std_75th, mad_75th = contourStats(percentile_75)
            
                ##############################################
                #### calculate misc. info of sar ####
                ##############################################

                # retrieve time imagette observation to use later in calculating dtime (buoy observation) - (imagette observation)
                time_imagette = sar_ds.start_date

                # average values
                wspd_median = sar_ds.windfield.median().values*1  # *1 to turn array of float into float
                incidence_avg = sar_ds.incidence.mean().values*1    
                mean_ground_heading = sar_ds.ground_heading.mean().values*1
                
                row = [str(i) for i in [train, time_imagette, wspd_median, wdir_estimate, incidence_avg, mean_ground_heading, energy_dir_range, energy_dir, \
                                window_effect, polar_effect, var_cartesian, var_windfield, var_polar, var_bandpass, var_highpass, var_lowpass, var_beam, var_bandpass_beam, \
                                var_beyond_nyquist, frac_beam, frac_bandpass, frac_lowpass, frac_highpass, density_beam, density_bandpass, density_beam_bandpass, \
                                friction_velocity, z_0, Cdn, sigma_u, L, B, w_star, w_star_normalised_deviation, corr_fact, H, Zi_estimate, \
                                angle_diff_max_min_75_25, angle_diff_max_min_75_50, angle_diff_max_min_50_25, angle_diff_min_theta_75_of_min_freq, angle_diff_min_theta_50_of_min_freq, \
                                angle_diff_min_theta_25_of_min_freq,angle_diff_theta_25_of_min_freq_theta_75_of_min_freq, angle_diff_theta_50_of_min_freq_theta_75_of_min_freq, \
                                angle_diff_theta_75_of_min_freq_theta_50_of_min_freq, angle_diff_theta_75_of_min_freq_theta_25_of_max_freq, \
                                angle_diff_theta_75_of_min_freq_theta_50_of_max_freq, angle_diff_theta_50_of_min_freq_theta_25_of_max_freq, \
                                angle_diff_theta_25_of_max_freq_theta_75_of_max_freq, angle_diff_theta_50_of_max_freq_theta_75_of_max_freq, \
                                angle_diff_theta_25_of_max_freq_theta_50_of_max_freq, theta_diff_min, theta_25_of_min_freq,\
                                theta_75_of_min_freq, theta_25_of_max_freq, theta_75_of_max_freq, mean_25th, median_25th, std_25th, mad_25th,\
                                mean_50th, median_50th, std_50th, mad_50th, mean_75th, median_75th, std_75th, mad_75th, S, S_normalised_deviation]]


            
            pass
        except Exception as e:
            print(e)
            row = [e]
            pass
        

        row_string = ' '.join(row)

#             save file
        with open(file_2_save, 'a') as f:
            f.write(row_string)
        
    return row_string








