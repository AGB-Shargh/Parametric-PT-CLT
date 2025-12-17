
# src/core/constants.py
"""Shared constants for the CLT design framework."""

LARGE_NUMBER = 1e6
SMALL_NUMBER = 10e-5
NEGLIGIBLE_MASS = 1e-06 
RIGID_MATERIAL_STIFFNESS = 1e10

# Units
kip = 1.0
inch = 1.0
sec = 1.0

ft = 12 * inch
ksi = kip / inch ** 2
psi = ksi / 1000

lb = kip / 1000
mm = inch / 25.4
meter = 1000 * mm
meter2 = meter ** 2
MPa = 145.038 * psi
kPa = MPa / 1000
kN = kPa * (1000 * mm) ** 2
g = 9.81 * meter / sec ** 2

