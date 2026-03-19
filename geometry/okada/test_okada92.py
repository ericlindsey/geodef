import numpy as np

from okada92 import okada92

# Define inputs for the test
X = 10.0
Y = 0.0
Z = -10.0
strike = 0.0
depth = 50.0
dip = 70.0
length = 100.0
width = 55.0
strike_slip = 100.0
dip_slip = 0.0
opening = 0.0

G=30.0
nu=0.25

# Output heading
print("*** OUTPUT OF okada92.py ***\n")
print(f"depth, strike, dip = {depth}, {strike}, {dip}")
print(f"length, width = {length}, {width}")
print(f"strike_slip, dip_slip, opening = {strike_slip}, {dip_slip}, {opening}")
print(f"X = {X}, Y = {Y}, Z = {Z}")

displacement, strain = okada92(X,Y,Z,depth,strike,dip,length,width,strike_slip,dip_slip,opening,G,nu,allow_singular=False)

print(displacement)
print(strain)

## Extract displacement components
#UX = displacement[0, 0]
#UY = displacement[1, 0]
#UZ = displacement[2, 0]
#
## Output results
#print("\nIRET =", IRET)
#print(f"UX, UY, UZ = {UX}, {UY}, {UZ}")
##print("ANSWER = -37.8981 63.1789 14.9607")  # Expected output from the original Fortran code
