
import numpy as np
import os

# Define constants

large_number = 10e5
small_number = 10e-4

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

# Create the full dictionary
config = {
    "building": {
        "Floor Dead Load Density": 3.62372 * kN / meter2,
        "Floor Live Load Density": 0.05 * kip / ft ** 2,
        "Roof Dead Load Density": 3.70713 * kN / meter2,
        "Roof Live Load Density": 0.02 * kip / ft ** 2,
        "Story Seismic Mass": {},
        "Wall Mass": {},
        "Live Load": {},
        "Dead Load": {},
        "Stories Heights": [146, 120],
        "Half Building Area": 53.95 * meter**2,
        "Walls Tributary Area": 13.25 * meter**2
    },
    "wall": {
        'Span Number': 2,
        "PT Extender Length": 5,
        "Wall Length": 61,
        "Wall Extension": 0.61 * meter,
        "Wall Thickness": 6.875,
        "Total-to-Shear Area Ratio": 5/6,
        "Wall Elastic Modulus": 8536 * MPa,
        "Wall Shear Modulus": 552 * MPa,
        "Wall Yield Stress": 25 * MPa,
        "Weight Density": 0.037 * kip / ft ** 3,
        'Shear Correction Factor': 5 / 6,
        "Plastic Hinge Length Reference": "thickness",
        "Plastic Hinge Length Ratio": 2,
        "Wall Split Strain": 0.0056,
        "Wall Crush Strain": 0.0375,
        "Pinch Factor X": 1.0,
        "Pinch Factor Y": 1.0,
        "Initial Damage X": 0.0,
        "Initial Damage Y": 0.0,
        "Deterioration Factor": 0.0,
        "Split Strength Ratio": 0.98,
        "Crush Strength Ratio": 0.25
    },
    "UFP": {
        "Hysteresis Model": 'Baird et al.',
        "UFP Numbers": [2, 3],
        "UFP Height Ratios": [[79/146, 117/146], [16/120, 57/120, 98/120]],
        "UFP Width": 0.114 * meter,
        "UFP Thickness": 0.0095 * meter,
        "UFP Diameter": 0.0921 * meter,
        "UFP Yield Stress": 414 * MPa,
        "UFP Ultimate Stress": 1.5
    },
    "spring": {
        "Number of Rocking Springs": 40
    },
    "steel_elastic_module": 29000 * ksi,
    "bar": {
        "Material": {
            'Elastic Module': 29000,
            "Ultimate Stress": 120,
            "Yield Stress": 92,
            "Post Yield Hardening Ratio": 0.00965,
            "PT to Yield Force Ratio": 0.4,
            "Mass Density": 0.4837 * kip / ft**3 / g
        },
        "Diameter": 16.54 * mm, 
        "Number Per Panel": 4
    },
    "leaning_columns": {
    "Section Area": large_number,   
    "Section Iz": small_number,   
    "Elastic Modulus": 29000
    },
    "diaphragm": {
        "Elements": {
            "Shear Key": {
                "E": 29000 * ksi,
                "G": 11200 * ksi,
                "Story 1": {"Thickness": 22.2 * mm, "Width": 73 * mm, "Length": 416 * mm},
                "Story 2": {"Thickness": 44.5 * mm, "Width": 67.5 * mm, "Length": 416 * mm}
            }
        }
    },

    "analysis": {
        "damping": {"modes": [1, 2], "damping ratio": 0.02},
        "time_integration": {
            "beta": 0.25,
            "gamma": 0.5,
            "time_step_ratio": 1
        }
    }



}
