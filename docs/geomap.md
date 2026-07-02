# `geodef.geomap`

Optional geographic map plotting with [Cartopy](https://scitools.org.uk/cartopy/).
Most `geodef.plot` functions work in a local Cartesian frame (East/North km);
`geomap` provides the complementary geographic view — coastlines, borders, and
topography with faults and GNSS vectors overlaid in longitude/latitude.

Cartopy is an optional dependency. Install it with:

```bash
pip install geodef[maps]
```

Every function raises a clear `ImportError` if Cartopy is missing.

## Building a basemap

### `basemap(extent=None, *, coastlines=True, borders=False, land=False, ocean=False, stock_img=False, resolution='50m', gridlines=True, ax=None) → GeoAxes`

Create a PlateCarree map axes with the requested background features.

```python
from geodef import geomap

ax = geomap.basemap(
    extent=(84, 88, 27, 29),      # lon_min, lon_max, lat_min, lat_max
    coastlines=True,
    borders=True,
    stock_img=True,               # built-in shaded relief, no download
)
```

Because the axes use PlateCarree, plot your own lon/lat data on it with
`transform=cartopy.crs.PlateCarree()`.

## Overlays

### `add_fault(ax, fault, *, edgecolor='red', facecolor='none', linewidth=0.8, **kwargs)`

Draw a fault's patch outlines (rectangular or triangular) in lon/lat.

### `add_vectors(ax, dataset, *, scale=1.0, color='black', **kwargs)`

Draw horizontal GNSS velocity arrows at their lon/lat locations.

```python
ax = geomap.basemap(extent=(84, 88, 27, 29), coastlines=True)
geomap.add_fault(ax, fault, edgecolor='k')
geomap.add_vectors(ax, gnss, scale=2e4)
```
