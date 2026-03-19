import numpy as np

from okada92 import DC3D

# Define inputs for the test
X = 10.0
Y = 20.0
Z = -30.0
DEPTH = 50.0
DIP = 70.0
AL1 = -80.0
AL2 = 120.0
AW1 = -30.0
AW2 = 25.0
DISL1 = 200.0
DISL2 = -150.0
DISL3 = 100.0

# Output heading
print("*** OUTPUT OF okada92.py ***\n")
print(f"DEPTH, DIP = {DEPTH}, {DIP}")
print(f"AL1, AL2, AW1, AW2 = {AL1}, {AL2}, {AW1}, {AW2}")
print(f"DISL1, DISL2, DISL3 = {DISL1}, {DISL2}, {DISL3}")
print(f"X = {X}, Y = {Y}, Z = {Z}")

# Set ALPHA and call the DC3D function
ALPHA = 2.0 / 3.0
print(f"ALPHA = {ALPHA}")
displacement, strain, IRET = DC3D(ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3)

print(displacement)

# Extract displacement components
UX = displacement[0, 0]
UY = displacement[1, 0]
UZ = displacement[2, 0]

# Output results
print("\nIRET =", IRET)
print(f"UX, UY, UZ = {UX}, {UY}, {UZ}")
print("ANSWER = -37.8981 63.1789 14.9607")  # Expected output from the original Fortran code
