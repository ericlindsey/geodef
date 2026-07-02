#!/usr/local/bin/python

# Eric Lindsey
#
# Version history:
#   Created 4/2014
#   Updated for Python3 4/2016
#   Added pyproj dynamic adapter for CRS support 2/2026 - Gemini 3.1 Pro

import numpy as np
import scipy.linalg
from dataclasses import dataclass
from typing import Tuple, Union, Optional

# Array-like types for hinting scalar vs numpy arrays
ArrayLike = Union[float, int, np.ndarray]

@dataclass
class Ellipsoid:
    """Defines an Earth reference ellipsoid."""
    a: float  # semi-major axis (meters)
    f: float  # flattening

    @property
    def finv(self) -> float:
        """Inverse flattening."""
        return 1.0 / self.f if self.f != 0 else 0.0

    @property
    def e2(self) -> float:
        """Squared eccentricity."""
        f = self.f
        return 2.0 * f - f * f

# Spheroid constants - defined by the World Geodetic System 1984 (WGS84)
WGS84 = Ellipsoid(a=6378137.0, f=1.0/298.257223563)

def geod2ecef(lat: ArrayLike, lon: ArrayLike, alt: ArrayLike, ellps: Ellipsoid = WGS84, crs: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Convert geodetic (lat,lon,alt) in degrees/meters to ECEF (X,Y,Z) in meters.
    Translated from ~/gg/gamit/lib/geoxyz.f
    """
    if crs is not None:
        import importlib.util
        if importlib.util.find_spec('pyproj') is None:
            raise ImportError("The 'pyproj' library is required to use the 'crs' parameter. Please install it with 'pip install pyproj'.")
        import pyproj
        # EPSG:4978 is WGS 84 / Geocentric (ECEF)
        transformer = pyproj.Transformer.from_crs(crs, "epsg:4978", always_xy=True)
        # pyproj expects (lon, lat) when always_xy=True
        xp, yp, zp = transformer.transform(lon, lat, alt)
        return xp, yp, zp

    latr = np.radians(lat)
    lonr = np.radians(lon)
    sinlat=np.sin(latr)
    coslat=np.cos(latr)
    sinlon=np.sin(lonr)
    coslon=np.cos(lonr)
    curvn = ellps.a / (np.sqrt(1. - ellps.e2 * sinlat * sinlat))
    x=(curvn + np.array(alt))*coslat*coslon
    y=(curvn + np.array(alt))*coslat*sinlon
    z=(curvn*(1. - ellps.e2) + np.array(alt))*sinlat
    return x, y, z

def ecef2geod(x: ArrayLike, y: ArrayLike, z: ArrayLike, ellps: Ellipsoid = WGS84, crs: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Convert ECEF coordinates (X,Y,Z) in meters to geodetic (lat,lon,alt) in degrees/meters.
    Translated to python from ~/gg/gamit/lib/geoxyz.f
    """
    if crs is not None:
        import importlib.util
        if importlib.util.find_spec('pyproj') is None:
            raise ImportError("The 'pyproj' library is required to use the 'crs' parameter. Please install it with 'pip install pyproj'.")
        import pyproj
        # EPSG:4978 is WGS 84 / Geocentric (ECEF)
        transformer = pyproj.Transformer.from_crs("epsg:4978", crs, always_xy=True)
        # pyproj returns (lon, lat) when always_xy=True
        lonp, latp, altp = transformer.transform(x, y, z)
        return latp, lonp, altp

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    along=np.arctan2(y, x)
    along=np.where(along < 0.,
        along + 2.*np.pi,
        along )
    #STARTING VALUE FOR LATITUDE ITERATION
    sqr=np.sqrt(x*x + y*y)
    alat0=np.arctan2(z/sqr, 1. - ellps.e2)
    alat=alat0
    while True:
        alat0=alat
        sinlat=np.sin(alat)
        curvn=ellps.a/(np.sqrt(1. - ellps.e2*sinlat*sinlat))
        alat=np.arctan2((z + ellps.e2*curvn*sinlat), sqr)
        #iterate to double precision
        if np.all(np.abs(alat - alat0) < 1.e-15):
            break
    cutoff=80.*2*np.pi/360.
    hght=np.where(alat > cutoff,
        z/np.sin(alat)-curvn+ellps.e2*curvn,
        (sqr/np.cos(alat))-curvn )
    alat=alat*180./np.pi
    along=along*180./np.pi
    along=np.where(along > 180.,
        along-360.,
        along )
    return alat, along, hght

def ecef2enu(x: ArrayLike, y: ArrayLike, z: ArrayLike, lat0: ArrayLike, lon0: ArrayLike, alt0: ArrayLike, ellps: Ellipsoid = WGS84) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Given an origin (lat0,lon0,alt0) convert ECEF (X,Y,Z) to a local ENU frame in meters.
    """
    x0, y0, z0 = geod2ecef(lat0, lon0, alt0, ellps=ellps)
    xyz=np.array([x-x0,y-y0,z-z0])
    sinlon0=np.sin(lon0*np.pi/180.)
    coslon0=np.cos(lon0*np.pi/180.)
    sinlat0=np.sin(lat0*np.pi/180.)
    coslat0=np.cos(lat0*np.pi/180.)
    Rmat=np.array([[-sinlon0,         coslon0,         0.     ],
                   [-sinlat0*coslon0,-sinlat0*sinlon0, coslat0],
                   [ coslat0*coslon0, coslat0*sinlon0, sinlat0]])
    enu=np.dot(Rmat,xyz)
    return enu[0], enu[1], enu[2]
    
def ecef2enu_vel(x: ArrayLike, y: ArrayLike, z: ArrayLike, lat0: ArrayLike, lon0: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Given an origin (lat0,lon0) convert a small offset or velocity in ECEF (X,Y,Z) to a local ENU frame.
    """
    assert np.size(x)==np.size(y) and np.size(x)==np.size(z) and np.size(x)==np.size(lat0) and np.size(x)==np.size(lon0)
    #interweave vectors into one, size 3nx1
    xyz=np.ravel(np.column_stack((x,y,z)))
    latr=np.array(np.radians(lat0),ndmin=1)
    lonr=np.array(np.radians(lon0),ndmin=1)
    sinlon0=np.sin(lonr)
    coslon0=np.cos(lonr)
    sinlat0=np.sin(latr)
    coslat0=np.cos(latr)
    R_list = []
    for i in range(np.size(lat0)):
        Ri=np.array([[-sinlon0[i],            coslon0[i],            0.         ],
                     [-sinlat0[i]*coslon0[i],-sinlat0[i]*sinlon0[i], coslat0[i] ],
                     [ coslat0[i]*coslon0[i], coslat0[i]*sinlon0[i], sinlat0[i] ]])
        R_list.append(Ri)
    # block_diag doesn't work with empty list
    Rmat = scipy.linalg.block_diag(*R_list) if R_list else np.array([])
            
    enu=np.dot(Rmat,xyz)
    return enu[0::3], enu[1::3], enu[2::3]

def enu2ecef(e: ArrayLike, n: ArrayLike, u: ArrayLike, lat0: ArrayLike, lon0: ArrayLike, alt0: ArrayLike, ellps: Ellipsoid = WGS84) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Given an origin (lat0,lon0,alt0) convert local ENU coordinates back to ECEF (X,Y,Z).
    """
    x0, y0, z0 = geod2ecef(lat0, lon0, alt0, ellps=ellps)
    enu=np.array([e,n,u])
    Rmat=__Rmat_for_enu2ecef(lat0,lon0)
    
    # Handle scalar vs array adding xyz translation correctly
    xyz_offset = np.array([x0, y0, z0])
    if enu.ndim > 1:
        xyz_offset = xyz_offset[:, np.newaxis]
    
    xyz=np.dot(Rmat,enu) + xyz_offset
    return xyz[0], xyz[1], xyz[2]
    
def enu2ecef_vel(evel: ArrayLike, nvel: ArrayLike, uvel: ArrayLike, lat0: ArrayLike, lon0: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Given an origin (lat0,lon0,alt0) convert local ENU velocity back to ECEF (X,Y,Z).
    """
    assert np.size(evel)==np.size(nvel) and np.size(evel)==np.size(uvel)
    enu=np.vstack((evel,nvel,uvel))
    Rmat=__Rmat_for_enu2ecef(lat0,lon0)
    
    if np.size(lat0) > 1:
        # If vectorized points, Rmat is 3Nx3N block diagonal, but enu is 3xN. 
        # We need to ravel ENU similar to how ecef2enu_vel does it for XYZ
        enu_flat = np.ravel(np.column_stack((evel, nvel, uvel)))
        xyz_flat = np.dot(Rmat, enu_flat)
        return xyz_flat[0::3], xyz_flat[1::3], xyz_flat[2::3]
    else:
        xyz=np.dot(Rmat,enu)
        return xyz[0], xyz[1], xyz[2]
    
def enu2ecef_sigma(esigma: ArrayLike, nsigma: ArrayLike, usigma: ArrayLike, rhoen: ArrayLike, lat0: ArrayLike, lon0: ArrayLike) -> np.ndarray:
    """ 
    Given an origin (lat0,lon0,alt0) convert local ENU covariance matrix to ECEF (X,Y,Z).
    Returns a 3n x 3n covariance matrix.
    """
    esigma=np.array(esigma,ndmin=1)
    nsigma=np.array(nsigma,ndmin=1)
    usigma=np.array(usigma,ndmin=1)
    rhoen=np.array(rhoen,ndmin=1)
    cov_list = []
    for i in range(np.size(esigma)):
        covi=np.array([[esigma[i]*esigma[i],          esigma[i]*nsigma[i]*rhoen[i], 0.                  ],
                       [esigma[i]*nsigma[i]*rhoen[i], nsigma[i]*nsigma[i],          0.                  ],
                       [0.,                           0.,                           usigma[i]*usigma[i] ]])
        cov_list.append(covi)
    # block_diag doesn't work with empty list
    covarmat = scipy.linalg.block_diag(*cov_list) if cov_list else np.array([])
    Rmat=__Rmat_for_enu2ecef(lat0,lon0)
    cov_out=np.dot(np.dot(Rmat,covarmat),Rmat.T)
    return cov_out

def __Rmat_for_enu2ecef(lat: ArrayLike, lon: ArrayLike) -> np.ndarray:
    '''Internal function to create rotation matrix for enu2ecef routines.'''
    latr=np.array(np.radians(lat),ndmin=1)
    lonr=np.array(np.radians(lon),ndmin=1)
    sinlon=np.sin(lonr)
    coslon=np.cos(lonr)
    sinlat=np.sin(latr)
    coslat=np.cos(latr)
    R_list = []
    for i in range(np.size(latr)):
        Ri=np.array([[-sinlon[i], -sinlat[i]*coslon[i], coslat[i]*coslon[i]],
                     [ coslon[i], -sinlat[i]*sinlon[i], coslat[i]*sinlon[i]],
                     [ 0.,         coslat[i],           sinlat[i]        ]])
        R_list.append(Ri)
        
    return scipy.linalg.block_diag(*R_list) if R_list else np.array([])

def geod2spher(lat: ArrayLike, ellps: Ellipsoid = WGS84) -> Union[float, np.ndarray]:
    '''Convert geodetic latitude to spherical.'''
    return np.arctan((1 - ellps.e2) * np.tan(np.radians(lat))) * 180. / np.pi

def spher2geod(lat: ArrayLike, ellps: Ellipsoid = WGS84) -> Union[float, np.ndarray]:
    '''Convert spherical latitude to geodetic.'''
    return np.arctan(np.tan(np.radians(lat)) / (1 - ellps.e2)) * 180. / np.pi

def geod2enu(lat: ArrayLike, lon: ArrayLike, alt: ArrayLike, lat0: ArrayLike, lon0: ArrayLike, alt0: ArrayLike, ellps: Ellipsoid = WGS84) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Given an origin (lat0,lon0,alt0) convert coordinates (lat,lon,alt) to a local ENU frame in meters.
    Uses geod2ecef -> ecef2enu for actual conversion.
    """
    x, y, z = geod2ecef(lat, lon, alt, ellps=ellps)
    e, n, u = ecef2enu(x, y, z, lat0, lon0, alt0, ellps=ellps)
    return e, n, u

def enu2geod(e: ArrayLike, n: ArrayLike, u: ArrayLike, lat0: ArrayLike, lon0: ArrayLike, alt0: ArrayLike, ellps: Ellipsoid = WGS84) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Given an origin (lat0,lon0,alt0) convert local ENU coordinates back to geodetic.
    Uses enu2ecef -> ecef2geod for actual conversion.
    """
    x, y, z = enu2ecef(e, n, u, lat0, lon0, alt0, ellps=ellps)
    lat, lon, alt = ecef2geod(x, y, z, ellps=ellps)
    return lat, lon, alt
    
def translate_flat(lat: ArrayLike, lon: ArrayLike, alt: ArrayLike, eoffset: ArrayLike, noffset: ArrayLike, uoffset: ArrayLike, ellps: Ellipsoid = WGS84) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Offset coordinates (lat,lon,alt) by (eoffset,noffset,uoffset) (units in meters).
    Defined for scalar inputs only.
    Uses a simple flat earth approximation - valid for small distances only.
    """
    # note, the 3rd component of this transformation, the elevation deviation from flat due to curvature of the sphere, is ignored here - hence 'translate_flat'
    la, lo, al = enu2geod(eoffset, noffset, 0, lat, lon, alt, ellps=ellps)
    return la * 1., lo * 1., alt + uoffset * 1.

def vincenty(lat0: ArrayLike, lon0: ArrayLike, lat1: ArrayLike, lon1: ArrayLike, ellps: Ellipsoid = WGS84) -> Tuple[float, float, float]:
    """ 
    Returns dist, az0, az1 (forward and back azimuths at the starting and ending points).
    Vincenty's more accurate iterative formula for distances, azimuths on an ellipsoid.
    http://en.wikipedia.org/wiki/Vincenty%27s_formulae 
    """
    lat0r=np.radians(lat0)
    lat1r=np.radians(lat1)
    
    U0=np.arctan((1-ellps.f)*np.tan(lat0r))
    U1=np.arctan((1-ellps.f)*np.tan(lat1r))
    sinU0=np.sin(U0)
    cosU0=np.cos(U0)
    sinU1=np.sin(U1)
    cosU1=np.cos(U1)
    L=np.radians(lon1-lon0)
    lam=L
    count=0
    converged=False
    while count<200:
        count+=1
        lam0 = lam
        sinsig = np.sqrt((cosU1*np.sin(lam))**2 + (cosU0*sinU1 - sinU0*cosU1*np.cos(lam))**2)
        cossig = sinU0*sinU1 + cosU0*cosU1*np.cos(lam)
        
        if sinsig == 0.0:
            if cossig > 0:
                # exactly coincident points
                return 0.0, 0.0, 0.0
            else:
                # exactly antipodal points; fails to converge
                break
                
        sig = np.arctan2(sinsig,cossig)
        sina = cosU0*cosU1*np.sin(lam)/sinsig
        cossqa = 1. - sina**2
        
        if cossqa == 0.0:
            cos2sigm = 0.0 # Equatorial line
        else:
            cos2sigm = cossig - 2.*sinU0*sinU1/cossqa
            
        C = (ellps.f/16.)*cossqa*(4. + ellps.f*(4 - 3.*cossqa))
        lam = L + (1. - C)*ellps.f*sina*(sig + C*sinsig*(cos2sigm + C*cossig*(-1.+2.*cos2sigm**2)))
        if (abs(lam-lam0) < 1.e-15):
            converged=True
            break
            
    if not converged:
        import warnings
        warnings.warn("Vincenty formula failed to converge (points are nearly antipodal).", RuntimeWarning)
        return np.nan, np.nan, np.nan
        
    b = (1 - ellps.f) * ellps.a
    usq = cossqa*(ellps.a**2-b**2)/b**2
    k1 = (np.sqrt(1 + usq) - 1)/(np.sqrt(1 + usq) + 1)
    A = (1 + 0.25*k1**2)/(1 - k1)
    B = k1*(1 - 0.375*k1**2)
    dsig = B*sinsig*(cos2sigm + 0.25*B*(cossig*(-1. + 2.*cos2sigm**2) - (1./6.)*B*cos2sigm*(-3. + 4.*sinsig**2)*(-3. + 4.*cos2sigm**2)))
    dist = b*A*(sig - dsig)
    
    # Avoid div by zero for coincident/antipodal singular azimuths
    if sinsig == 0.0:
        az0, az1 = 0.0, 0.0
    else:
        az0 = np.degrees(np.arctan2(cosU1*np.sin(lam),cosU0*sinU1 - sinU0*cosU1*np.cos(lam)))
        az1 = 180. + np.degrees(np.arctan2(cosU0*np.sin(lam),-1*sinU0*cosU1 + cosU0*sinU1*np.cos(lam)))
    return dist,az0,az1

def haversine(lat0: float, lon0: float, lat1: float, lon1: float, radius: float = 6371000.0) -> float:
    """
    Get the great-circle distance between two points in lat,lon.
    Note, uses spherical earth assumption with an optional radius.
    """
    dlat=np.radians(lat1-lat0)
    dlon=np.radians(lon1-lon0)
    lat0r=np.radians(lat0)
    lat1r=np.radians(lat1)
    x=np.sin(dlat/2.)*np.sin(dlat/2.) + np.sin(dlon/2.)*np.sin(dlon/2.)*np.cos(lat0r)*np.cos(lat1r)
    c=2.*np.arctan2(np.sqrt(x),np.sqrt(1.-x))
    dist=radius*c
    return dist

def heading(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """
    Great-circle heading/azimuth between two points in lat,lon, relative to point0 (degrees assumed).
    Note, uses spherical earth assumption.
    """
    lat0r=np.radians(lat0)
    lon0r=np.radians(lon0)
    lat1r=np.radians(lat1)
    lon1r=np.radians(lon1)
    headr=np.mod(np.arctan2(np.sin(lon1r-lon0r)*np.cos(lat1r),
           np.cos(lat0r)*np.sin(lat1r)-np.sin(lat0r)*np.cos(lat1r)*np.cos(lon1r-lon0r)),
           2.*np.pi)
    headd=np.degrees(headr)
    return headd

def midpoint(lat0: float, lon0: float, lat1: float, lon1: float) -> Tuple[float, float]:
    """
    Great-circle midpoint between two points on a sphere.
    Note, uses spherical earth assumption.
    """
    dlon=np.radians(lon1-lon0)
    lat0r=np.radians(lat0)
    lat1r=np.radians(lat1)
    bx=np.cos(lat1r)*np.cos(dlon)
    by=np.cos(lat1r)*np.sin(dlon)
    latcr=np.arctan2(np.sin(lat0r)+np.sin(lat1r), np.sqrt((np.cos(lat0r) + bx)**2 + by**2))
    loncr=np.radians(lon0) + np.arctan2(by, np.cos(lat0r)+bx)
    latc=np.degrees(latcr)
    lonc=np.degrees(loncr)
    return latc,lonc
    
