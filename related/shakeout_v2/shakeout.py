"""
Shakeout: Utilities for Okada-based earthquake deformation modeling and inversion.

This module provides reusable functions for:
- Loading GNSS velocity data
- Computing fault patch geometry
- Calculating predicted displacements using the Okada solution
- Creating publication-quality visualizations
- Computing model misfit statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from okada85 import displacement
import geod_transform


def read_gnss_data(filepath, verbose=True):
    """
    Read GNSS velocity data from a file.
    
    Assumes standard format with header at row 6 (0-indexed) and columns:
    Lon, Lat, OE, ON, OU, ErE, ErN, ErU
    
    Parameters
    ----------
    filepath : str
        Path to the GNSS data file
    verbose : bool
        If True, print data summary
        
    Returns
    -------
    dict
        Dictionary with keys: 'lat', 'lon', 'oe', 'on', 'ou', 'ere', 'ern', 'eru'
        All displacements are in meters (converted from mm)
    """
    data = pd.read_csv(filepath, sep=r'\s+', skiprows=6)
    
    # Extract coordinates
    lat = data['Lat'].values
    lon = data['Lon'].values
    
    # Extract displacements (in mm, convert to m)
    oe = data['OE'].values / 1000.0  # East displacement
    on = data['ON'].values / 1000.0  # North displacement
    ou = data['OU'].values / 1000.0  # Up displacement
    
    # Extract uncertainties (in mm, convert to m)
    ere = data['ErE'].values / 1000.0  # East uncertainty
    ern = data['ErN'].values / 1000.0  # North uncertainty
    eru = data['ErU'].values / 1000.0  # Up uncertainty
    
    # # error rescaling
    ere = ere*100
    ern = ern*100
    eru = eru*100
    
    if verbose:
        print(f"Loaded GNSS data: {len(lat)} stations")
        print(f"  East displacement: {oe.min():.3f} to {oe.max():.3f} m")
        print(f"  North displacement: {on.min():.3f} to {on.max():.3f} m")
        print(f"  Up displacement: {ou.min():.3f} to {ou.max():.3f} m")
    
    return {
        'lat': lat, 'lon': lon,
        'oe': oe, 'on': on, 'ou': ou,
        'ere': ere, 'ern': ern, 'eru': eru
    }


def fault_outline_en(fault_depth_km, fault_dip_deg, fault_length_km, fault_width_km,
                     fault_strike_deg, fault_centroid_E_km, fault_centroid_N_km,
                     return_depths=False):
    """
    Compute fault patch corners in ENU coordinates.
    
    Returns fault corners ordered as:
      [top-left, top-right, bottom-right, bottom-left, top-left]
    (first two points define the TOP edge)
    
    Parameters
    ----------
    fault_depth_km : float
        Depth to center of fault patch [km]
    fault_dip_deg : float
        Dip angle [degrees]
    fault_length_km : float
        Length along strike [km]
    fault_width_km : float
        Width along dip [km]
    fault_strike_deg : float
        Strike relative to North [degrees]
    fault_centroid_E_km : float
        Plan-view centroid E coordinate [km]
    fault_centroid_N_km : float
        Plan-view centroid N coordinate [km]
    return_depths : bool
        If True, also return depths of corner points [km]
        
    Returns
    -------
    corners_EN : ndarray
        Shape (5, 2) array of corner coordinates [E, N] in km
    depths : ndarray (optional)
        Shape (5,) array of depths [km] if return_depths=True
    """
    strike = np.deg2rad(fault_strike_deg)
    dip = np.deg2rad(fault_dip_deg)

    # Unit vectors
    u_strike = np.array([np.sin(strike), np.cos(strike)])  # [E, N]
    dip_az = strike + np.pi/2.0
    u_dip_h = np.array([np.sin(dip_az), np.cos(dip_az)])   # [E, N]

    # Half-vectors
    half_L = 0.5 * fault_length_km * u_strike
    half_W_h = 0.5 * fault_width_km * np.cos(dip) * u_dip_h  # horizontal projection

    # Centroid and edge centers
    C = np.array([fault_centroid_E_km, fault_centroid_N_km])
    top_center = C - half_W_h
    bottom_center = C + half_W_h

    # Corners
    top_left = top_center - half_L
    top_right = top_center + half_L
    bottom_left = bottom_center - half_L
    bottom_right = bottom_center + half_L

    corners_EN = np.vstack([top_left, top_right, bottom_right, bottom_left, top_left])

    if not return_depths:
        return corners_EN

    # Compute depths
    top_depth = fault_depth_km - np.sin(dip) * fault_width_km / 2
    bottom_depth = fault_depth_km + np.sin(dip) * fault_width_km / 2
    depths = np.array([top_depth, top_depth, bottom_depth, bottom_depth, top_depth])

    return corners_EN, depths


def prepare_coordinates(gnss_data, lat0, lon0, alt0=0.0, verbose=True):
    """
    Transform GNSS station positions to ENU coordinates and prepare data.
    
    Parameters
    ----------
    gnss_data : dict
        Dictionary from read_gnss_data() with keys 'lat', 'lon', 'oe', etc.
    lat0, lon0, alt0 : float
        Reference origin for ENU coordinate system [degrees, meters]
        
    Returns
    -------
    dict
        Dictionary with ENU coordinates and displacements
    """
    lat, lon = gnss_data['lat'], gnss_data['lon']
    oe, on, ou = gnss_data['oe'], gnss_data['on'], gnss_data['ou']
    
    # Get positions in ENU frame relative to reference
    e_pos, n_pos, u_pos = geod_transform.geod2enu(lat, lon, np.zeros_like(lat),
                                                    lat0, lon0, alt0)
    
    if verbose:
        print(f"GNSS stations in ENU:")
        print(f"  East range: {e_pos.min():.1f} to {e_pos.max():.1f} m")
        print(f"  North range: {n_pos.min():.1f} to {n_pos.max():.1f} m")
    
    return {
        'e_pos_m': e_pos, 'n_pos_m': n_pos, 'u_pos_m': u_pos,
        'e_pos_km': e_pos / 1000.0, 'n_pos_km': n_pos / 1000.0,
        'oe': oe, 'on': on, 'ou': ou
    }


def compute_okada_displacements(e_pos_km, n_pos_km, fault_depth, fault_strike,
                                fault_dip, fault_length, fault_width,
                                fault_rake, fault_slip, verbose=True):
    """
    Compute predicted surface displacements using the Okada solution.
    
    Parameters
    ----------
    e_pos_km : ndarray
        East positions of observation points [km]
    n_pos_km : ndarray
        North positions of observation points [km]
    fault_depth : float
        Fault depth [km]
    fault_strike : float
        Strike angle [degrees]
    fault_dip : float
        Dip angle [degrees]
    fault_length : float
        Fault length [km]
    fault_width : float
        Fault width [km]
    fault_rake : float
        Rake angle [degrees]
    fault_slip : float
        Slip magnitude [m]
    verbose : bool
        If True, print displacement summary
        
    Returns
    -------
    dict
        Dictionary with keys 'dE', 'dN', 'dZ' (displacements in m)
    """
    pred_dE, pred_dN, pred_dZ = displacement(
        e_pos_km, n_pos_km, fault_depth,
        fault_strike, fault_dip,
        fault_length, fault_width,
        fault_rake, fault_slip, 0.0
    )
    
    if verbose:
        print(f"Predicted displacements:")
        print(f"  East: {pred_dE.min():.4f} to {pred_dE.max():.4f} m")
        print(f"  North: {pred_dN.min():.4f} to {pred_dN.max():.4f} m")
        print(f"  Up: {pred_dZ.min():.4f} to {pred_dZ.max():.4f} m")
    
    return {'dE': pred_dE, 'dN': pred_dN, 'dZ': pred_dZ}


def compute_misfit(obs_E, obs_N, obs_Z, pred_E, pred_N, pred_Z,
                   err_E=None, err_N=None, err_Z=None):
    """
    Compute misfit statistics between observations and predictions.
    
    Parameters
    ----------
    obs_E, obs_N, obs_Z : ndarray
        Observed displacements [m]
    pred_E, pred_N, pred_Z : ndarray
        Predicted displacements [m]
    err_E, err_N, err_Z : ndarray, optional
        Uncertainties [m]. If provided, computes chi-squared misfit.
        
    Returns
    -------
    dict
        Dictionary with RMS misfits and residuals
    """
    residual_E = obs_E - pred_E
    residual_N = obs_N - pred_N
    residual_Z = obs_Z - pred_Z
    
    rmse_E = np.sqrt(np.mean(residual_E**2))
    rmse_N = np.sqrt(np.mean(residual_N**2))
    rmse_Z = np.sqrt(np.mean(residual_Z**2))
    
    result = {
        'residual_E': residual_E, 'residual_N': residual_N, 'residual_Z': residual_Z,
        'rmse_E': rmse_E, 'rmse_N': rmse_N, 'rmse_Z': rmse_Z
    }
    
    # Compute chi-squared if errors provided
    if err_E is not None and err_N is not None and err_Z is not None:
        n_obs = 3 * len(obs_E)
        n_params = 9  # 9 model parameters for a single fault patch
        chi2 = np.sum((residual_E / err_E)**2 + (residual_N / err_N)**2 + (residual_Z / err_Z)**2)
        red_chi2 = chi2 / (n_obs - n_params)
        result['chi2'] = chi2
        result['red_chi2'] = red_chi2
        # wrong way: not squaring the errors
        wrong_redchi2 = np.sum( residual_E**2/err_E + residual_N**2/err_N + residual_Z/err_Z)/(n_obs - n_params)
        result['wrong_redchi2'] = wrong_redchi2

    return result


def fault_corners_to_latlon(fault_corners_EN, lat0, lon0, alt0=0.0):
    """
    Convert fault corners from ENU to geographic coordinates.
    
    Parameters
    ----------
    fault_corners_EN : ndarray
        Shape (5, 2) array of corner coordinates [E, N] in km
    lat0, lon0, alt0 : float
        Reference origin [degrees, meters]
        
    Returns
    -------
    ndarray
        Shape (5, 2) array of [lon, lat] coordinates
    """
    lat_corners, lon_corners, _ = geod_transform.enu2geod(
        fault_corners_EN[:, 0] * 1000,
        fault_corners_EN[:, 1] * 1000,
        np.zeros(len(fault_corners_EN)),
        lat0, lon0, alt0
    )
    return np.column_stack([lon_corners, lat_corners])


def create_geographic_map(gnss_data, fault_corners_latlon, fault_cen_lon, fault_cen_lat,
                         pred_dE, pred_dN, pred_dZ, figsize=(14, 10),
                         title='Okada Forward Model: GNSS Displacements and Fault Patch'):
    """
    Create a geographic map showing fault patch, GNSS stations, and displacement vectors.
    
    Parameters
    ----------
    gnss_data : dict
        Dictionary from read_gnss_data()
    fault_corners_latlon : ndarray
        Shape (5, 2) array of fault corner coordinates [lon, lat]
    fault_cen_lon, fault_cen_lat : float
        Fault center coordinates
    pred_dE, pred_dN, pred_dZ : ndarray
        Predicted displacements [m]
    obs_dZ : ndarray
        Observed vertical displacements [m]
    figsize : tuple
        Figure size
    title : str
        Plot title
        
    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects
    """
    lon = gnss_data['lon']
    lat = gnss_data['lat']
    oe = gnss_data['oe']
    on = gnss_data['on']
    ou = gnss_data['ou']
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Set extent
    extent = [fault_cen_lon - 3.5, fault_cen_lon + 3.5,
              fault_cen_lat - 3.5, fault_cen_lat + 3.5]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.coastlines(resolution='50m')
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

    # Fault outline
    ax.plot(fault_corners_latlon[:, 0], fault_corners_latlon[:, 1],
            'k-', linewidth=1, transform=ccrs.PlateCarree(), label='Fault patch')
    ax.plot(fault_corners_latlon[0:2, 0], fault_corners_latlon[0:2, 1],
            'r-', linewidth=2.0, transform=ccrs.PlateCarree(), label='Fault top edge')

    # Fault center
    ax.plot(fault_cen_lon, fault_cen_lat, 'r*', markersize=20,
            transform=ccrs.PlateCarree(), markerfacecolor='none',
            markeredgewidth=2, label='Fault center')

    # GNSS stations
    pts1 = ax.scatter(lon, lat, s=120, c=ou, cmap='RdBu_r',
                    vmin=-np.max(np.abs(ou)), vmax=np.max(np.abs(ou)),
                    edgecolors='k', transform=ccrs.PlateCarree(),
                    zorder=5, label='GNSS stations (observed vertical)')
    pts2 = ax.scatter(lon, lat, s=40, c=pred_dZ, cmap='RdBu_r',
                    vmin=-np.max(np.abs(ou)), vmax=np.max(np.abs(ou)),
                    edgecolors='k', transform=ccrs.PlateCarree(),
                    zorder=5, label='GNSS stations (predicted vertical)')

    cbar = plt.colorbar(pts1, ax=ax, label='Observed vertical motion (m)', pad=0.05)

    # Displacement vectors
    scale_factor = 0.015
    ax.quiver(lon, lat, oe/scale_factor, on/scale_factor,
              transform=ccrs.PlateCarree(), scale=1.5e3, width=0.003,
              color='black', alpha=0.6, zorder=4)
    ax.quiver(lon, lat, pred_dE/scale_factor, pred_dN/scale_factor,
              transform=ccrs.PlateCarree(), scale=1.5e3, width=0.003,
              color='red', alpha=0.6, zorder=4)

    # Scale reference
    scale_ref_lon = fault_cen_lon - 3.3
    scale_ref_lat = fault_cen_lat + 2.2
    scale_disp = 2.0 / scale_factor
    ax.quiver(scale_ref_lon, scale_ref_lat, scale_disp, 0,
              transform=ccrs.PlateCarree(), scale=1.5e3, width=0.003,
              color='black', alpha=0.7, zorder=10)
    ax.text(scale_ref_lon+0.05, scale_ref_lat+0.15, '2 m (obs)',
            transform=ccrs.PlateCarree(), fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    
    return fig, ax


def create_comparison_plots(obs_E, obs_N, obs_Z, pred_E, pred_N, pred_Z,
                           rmse_E, rmse_N, rmse_Z, figsize=(15, 10)):
    """
    Create comparison plots of observed vs. predicted displacements.
    
    Parameters
    ----------
    obs_E, obs_N, obs_Z : ndarray
        Observed displacements [m]
    pred_E, pred_N, pred_Z : ndarray
        Predicted displacements [m]
    rmse_E, rmse_N, rmse_Z : float
        RMS errors for each component [m]
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig, axes
        Matplotlib figure and axes array
    """
    misfit_E = obs_E - pred_E
    misfit_N = obs_N - pred_N
    misfit_Z = obs_Z - pred_Z
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Vertical displacements
    ax = axes[0, 0]
    ax.scatter(obs_Z, pred_Z, s=50, alpha=0.6, edgecolors='k')
    ax.plot([min(obs_Z.min(), pred_Z.min()), max(obs_Z.max(), pred_Z.max())],
            [min(obs_Z.min(), pred_Z.min()), max(obs_Z.max(), pred_Z.max())],
            'r--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Observed (m)', fontsize=11)
    ax.set_ylabel('Predicted (m)', fontsize=11)
    ax.set_title(f'Vertical Displacements\nRMSE={rmse_Z:.4f} m', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    # East displacements
    ax = axes[0, 1]
    ax.scatter(obs_E, pred_E, s=50, alpha=0.6, edgecolors='k')
    ax.plot([min(obs_E.min(), pred_E.min()), max(obs_E.max(), pred_E.max())],
            [min(obs_E.min(), pred_E.min()), max(obs_E.max(), pred_E.max())],
            'r--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Observed (m)', fontsize=11)
    ax.set_ylabel('Predicted (m)', fontsize=11)
    ax.set_title(f'East Displacements\nRMSE={rmse_E:.4f} m', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    # North displacements
    ax = axes[0, 2]
    ax.scatter(obs_N, pred_N, s=50, alpha=0.6, edgecolors='k')
    ax.plot([min(obs_N.min(), pred_N.min()), max(obs_N.max(), pred_N.max())],
            [min(obs_N.min(), pred_N.min()), max(obs_N.max(), pred_N.max())],
            'r--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Observed (m)', fontsize=11)
    ax.set_ylabel('Predicted (m)', fontsize=11)
    ax.set_title(f'North Displacements\nRMSE={rmse_N:.4f} m', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    # Residuals
    ax = axes[1, 0]
    ax.scatter(range(len(misfit_Z)), misfit_Z, s=50, alpha=0.6, edgecolors='k')
    ax.axhline(0, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Station Index', fontsize=11)
    ax.set_ylabel('Residual (m)', fontsize=11)
    ax.set_title('Vertical Residuals', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(range(len(misfit_E)), misfit_E, s=50, alpha=0.6, edgecolors='k')
    ax.axhline(0, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Station Index', fontsize=11)
    ax.set_ylabel('Residual (m)', fontsize=11)
    ax.set_title('East Residuals', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.scatter(range(len(misfit_N)), misfit_N, s=50, alpha=0.6, edgecolors='k')
    ax.axhline(0, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Station Index', fontsize=11)
    ax.set_ylabel('Residual (m)', fontsize=11)
    ax.set_title('North Residuals', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    
    return fig, axes
