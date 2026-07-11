# `geodef.transforms` — Coordinate transforms

Geodetic coordinate transforms between geographic (lat/lon/alt), ECEF (X/Y/Z),
and local ENU (East/North/Up) frames. Uses WGS84 by default; other ellipsoids
can be passed as `ellps=`.

## Which coordinate system should I use?

- **Geodetic** latitude, longitude, and ellipsoidal altitude are convenient for
  observations and maps but are not Cartesian coordinates.
- **ECEF** is a global Cartesian frame centered on Earth and is useful for
  exact rotations and long baselines.
- **ENU** is a Cartesian tangent frame at one reference point and is the natural
  frame for local fault geometry and displacement vectors.

Latitude and longitude are angles, not distances: never subtract longitude and
latitude and treat the result as meters. GeoDef uses geodetic (ellipsoidal), not
geocentric, latitude unless a function explicitly says otherwise. See
[geodetic coordinates](https://en.wikipedia.org/wiki/Geodetic_coordinates) and
[ECEF](https://en.wikipedia.org/wiki/Earth-centered%2C_Earth-fixed_coordinate_system)
for diagrams.

---

## Geographic ↔ ECEF

### `geod2ecef(lat, lon, alt, ellps=WGS84, crs=None) → (X, Y, Z)`

Convert geodetic coordinates (degrees, meters) to ECEF (meters).

```python
from geodef.transforms import geod2ecef
X, Y, Z = geod2ecef(lat=37.0, lon=-122.0, alt=0.0)
```

### `ecef2geod(X, Y, Z, ellps=WGS84, crs=None) → (lat, lon, alt)`

Convert ECEF to geodetic.

Passing `crs=` delegates to `pyproj` and requires that optional dependency.

---

## Geographic ↔ Local ENU

### `geod2enu(lat, lon, alt, lat0, lon0, alt0, ellps=WGS84) → (e, n, u)`

Convert geodetic coordinates to local East/North/Up relative to a reference point.

```python
from geodef.transforms import geod2enu

e, n, u = geod2enu(lat, lon, alt, ref_lat, ref_lon, 0.0)
# e, n, u in meters
```

### `enu2geod(e, n, u, lat0, lon0, alt0, ellps=WGS84) → (lat, lon, alt)`

Inverse of `geod2enu`.

---

## ECEF ↔ Local ENU

### `ecef2enu(X, Y, Z, lat0, lon0, alt0, ellps=WGS84) → (e, n, u)`

### `enu2ecef(e, n, u, lat0, lon0, alt0, ellps=WGS84) → (X, Y, Z)`

### `ecef2enu_vel(X, Y, Z, lat0, lon0) → (e, n, u)`

For velocity vectors (no translation, rotation only).

### `enu2ecef_vel(e, n, u, lat0, lon0) → (X, Y, Z)`

### `enu2ecef_sigma(se, sn, su, rhoen, lat0, lon0) → cov`

Convert ENU covariance to ECEF. Returns block-diagonal `(3n, 3n)` covariance matrix.

---

## Flat-earth offset

### `translate_flat(lat, lon, alt, eoffset, noffset, uoffset) → (lat, lon, alt)`

Offset coordinates by a small East/North/Up displacement. Uses a flat-earth
approximation and ignores curvature-induced vertical change.

Use this only when offsets are small relative to Earth's radius and the desired
accuracy. For regional or global baselines, use the ECEF/ENU transformations.

```python
from geodef.transforms import translate_flat

new_lat, new_lon, _ = translate_flat(lat, lon, 0.0, east_m, north_m, 0.0)
```

---

## Spherical / geodetic latitude

### `geod2spher(lat) → lat_spher`
### `spher2geod(lat) → lat_geod`

---

## Distance and azimuth

### `vincenty(lat0, lon0, lat1, lon1, ellps=WGS84) → (dist, az0, az1)`

Accurate ellipsoidal distance and forward/back azimuths (meters, degrees).

```python
from geodef.transforms import vincenty
dist, az_fwd, az_bck = vincenty(37.0, -122.0, 34.0, -118.0)
```

### `haversine(lat0, lon0, lat1, lon1, radius=6371000.0) → dist`

Great-circle distance on a sphere (meters). Fast but less accurate than Vincenty.

Vincenty accounts for ellipsoidal flattening; haversine assumes a sphere. The
difference is often negligible for plotting or neighborhood searches but can
matter for precise long-baseline geodesy.

### `heading(lat0, lon0, lat1, lon1) → azimuth`

Initial bearing from point 0 to point 1 (degrees clockwise from north).

### `midpoint(lat0, lon0, lat1, lon1) → (lat, lon)`

Geographic midpoint on a great-circle path.

---

## Custom ellipsoids

```python
from geodef.transforms import Ellipsoid, geod2enu

GRS80 = Ellipsoid(a=6378137.0, f=1.0/298.257222101)
e, n, u = geod2enu(lat, lon, alt, lat0, lon0, alt0, ellps=GRS80)
```

`WGS84 = Ellipsoid(a=6378137.0, f=1/298.257223563)` is the default.
