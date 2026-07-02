"""
shakeout_mcmc.py

Bayesian fault optimization routines for MCMC-based inversion of earthquake parameters
using GNSS observations and the Okada dislocation model.

This module provides log-likelihood, log-prior, and combined probability functions
for use with MCMC samplers (emcee, zeus, etc.).
"""

import numpy as np
from okada85 import displacement
import geod_transform


def fault_model_for_fitting(gps_locs, *params):
    """
    Compute predicted GPS displacements for a given fault model using Okada solution.
    
    Parameters
    ----------
    gps_locs : ndarray
        Concatenated array of GPS locations: [lon1, lon2, ..., lonN, lat1, lat2, ..., latN]
    params : tuple of floats
        Fault parameters: (latc, lonc, depthc, strike, dip, L, W, ss, ds)
        - latc, lonc: Fault center in degrees
        - depthc: Fault depth in km
        - strike: Strike angle in degrees
        - dip: Dip angle in degrees
        - L: Fault length in km
        - W: Fault width in km
        - ss: Strike-slip component in meters (positive = right-lateral)
        - ds: Dip-slip component in meters (positive = reverse/thrust)
        
    Returns
    -------
    predicted_displacements : ndarray
        Predicted displacements in layout [E1, N1, U1, E2, N2, U2, ..., EN, NN, UN]
        where length is 3 * number of GPS stations
    """
    # Expand the params tuple into individual elements
    latc, lonc, depthc, strike, dip, L, W, ss, ds = params
    
    # Extract GPS locations from concatenated array
    Ngps = int(np.size(gps_locs) / 2)
    gpslon = gps_locs[:Ngps]
    gpslat = gps_locs[Ngps:]
    
    # Convert observation points from geographic (lat/lon) to ENU coordinates
    # relative to fault center
    e_pos, n_pos, u_pos = geod_transform.geod2enu(
        gpslat, gpslon, np.zeros_like(gpslat),
        latc, lonc, 0.0
    )
    
    # Convert positions to kilometers
    e_pos_km = e_pos / 1000.0
    n_pos_km = n_pos / 1000.0
    
    # Convert slip components to rake angle and magnitude
    slip_magnitude = np.sqrt(ss**2 + ds**2)
    rake = np.arctan2(ds, ss) * 180.0 / np.pi
    
    # Call Okada solution for single rectangular dislocation
    pred_dE, pred_dN, pred_dZ = displacement(
        e_pos_km, n_pos_km, depthc,
        strike, dip,
        L, W,
        rake, slip_magnitude, 0.0
    )
    
    # Format output as concatenated [E1, N1, U1, E2, N2, U2, ...]
    predicted_displacements = np.empty((3 * Ngps,), dtype=float)
    predicted_displacements[0::3] = pred_dE
    predicted_displacements[1::3] = pred_dN
    predicted_displacements[2::3] = pred_dZ
    
    return predicted_displacements


def lnlike_fault(params, x, y, yerr):
    """
    Compute log-likelihood for fault model fitting to GNSS data.
    
    Parameters
    ----------
    params : tuple
        Fault parameters (see fault_model_for_fitting)
    x : ndarray
        GPS locations (same format as in fault_model_for_fitting)
    y : ndarray
        Observed displacements in layout [E1, N1, U1, E2, N2, U2, ...]
    yerr : ndarray
        Uncertainties for each component (same layout as y)
        
    Returns
    -------
    log_likelihood : float
        Log-likelihood value (-0.5 * chi-squared)
    """
    # Get predicted displacements
    ypred = fault_model_for_fitting(x, *params)
    
    # Compute negative chi-squared
    misfit = -0.5 * np.sum(((y - ypred) / yerr) ** 2)
    
    return misfit


def lnprior_fault(params, minvals, maxvals):
    """
    Compute log-prior for fault parameters using uniform bounds.
    
    Parameters
    ----------
    params : ndarray
        Fault parameters [latc, lonc, depthc, strike, dip, L, W, ss, ds]
    minvals : ndarray, optional
        Minimum allowed values for each parameter.
    maxvals : ndarray, optional
        Maximum allowed values for each parameter.
        
    Returns
    -------
    log_prior : float
        0.0 if all parameters are within bounds, -inf otherwise
    """
    
    # Check bounds
    if any(params - minvals < 0) or any(params - maxvals > 0):
        return -np.inf
    else:
        return 0.0


def lnprob_fault(params, x, y, yerr, minvals=None, maxvals=None):
    """
    Compute log-posterior probability for fault parameters.
    
    This combines the log-prior and log-likelihood:
    log(P) = log(prior) + log(likelihood)
    
    Parameters
    ----------
    params : ndarray
        Fault parameters
    x : ndarray
        GPS locations
    y : ndarray
        Observed displacements
    yerr : ndarray
        Uncertainties
    minvals : ndarray, optional
        Minimum parameter bounds
    maxvals : ndarray, optional
        Maximum parameter bounds
        
    Returns
    -------
    log_posterior : float
        Log-posterior probability
    """
    # Evaluate prior
    prior = lnprior_fault(params, minvals, maxvals)
    
    # If prior is -inf, return immediately
    if np.isinf(prior):
        return -np.inf
    
    # Otherwise, compute and return posterior
    return prior + lnlike_fault(params, x, y, yerr)


def format_gps_data(gpslon, gpslat, gps_dE, gps_dN, gps_dU, 
                   gps_err_E, gps_err_N, gps_err_U):
    """
    Format GPS data into the concatenated arrays used by the fitting functions.
    
    Parameters
    ----------
    gpslon, gpslat : ndarray
        GPS station coordinates in degrees
    gps_dE, gps_dN, gps_dU : ndarray
        Observed displacements in meters
    gps_err_E, gps_err_N, gps_err_U : ndarray
        Uncertainties in meters
        
    Returns
    -------
    gps_locations : ndarray
        Concatenated locations [lon1, ..., lonN, lat1, ..., latN]
    gps_displacements : ndarray
        Concatenated displacements [E1, N1, U1, ..., EN, NN, UN]
    gps_err : ndarray
        Concatenated uncertainties [same layout as displacements]
    """
    Ngps = len(gpslon)
    
    # Concatenate locations
    gps_locations = np.append(gpslon, gpslat)
    
    # Create displacement vector
    gps_displacements = np.empty((3 * Ngps,), dtype=float)
    gps_displacements[0::3] = gps_dE
    gps_displacements[1::3] = gps_dN
    gps_displacements[2::3] = gps_dU
    
    # Create error vector
    gps_err = np.empty((3 * Ngps,), dtype=float)
    gps_err[0::3] = gps_err_E
    gps_err[1::3] = gps_err_N
    gps_err[2::3] = gps_err_U
    
    return gps_locations, gps_displacements, gps_err
