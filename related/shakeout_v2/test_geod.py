#!/usr/local/bin/python

#test/advertise geodetic transforms
# later, make this a 'unit test' function -- eg. see test_okada

import geod_transform
import numpy as np

lat0=1
lon0=0
alt0=0

lat=np.linspace(lat0-0.1,lat0+0.1,num=2)
lon=np.linspace(lon0-0.1,lon0+0.1,num=2)
alt=251.702+0*lat
print("lat", lat)
print("lon", lon)
print("alt", alt)

e,n,u = geod_transform.geod2enu(lat,lon,alt,lat0,lon0,alt0)
print("e", e)
print("n", n)
print("u", u)

la,lo,al = geod_transform.enu2geod(e,n,u,lat0,lon0,alt0)
print("la", la)
print("lo", lo)
print("al", al)

print("la-lat", la-lat)
print("lo-lon", lo-lon)
print("al-alt", al-alt)

print("haversine dist, az, backaz")
dist=geod_transform.haversine(lat[0],lon[0],lat[1],lon[1])
head0=geod_transform.heading(lat[0],lon[0],lat[1],lon[1])
head1=geod_transform.heading(lat[1],lon[1],lat[0],lon[0])
print(dist,head0,head1)

print("vincenty dist,az,backaz")
vd,vh0,vh1=geod_transform.vincenty(lat[0],lon[0],lat[1],lon[1])
print(vd,vh0,vh1)

print("midpoint")
latc,lonc=geod_transform.midpoint(lat[0],lon[0],lat[1],lon[1])
print(latc,lonc)
