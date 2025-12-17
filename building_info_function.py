# building_info_function.py (modified generate_configurations)

def generate_configurations(raw_config):
    """Generate full OpenSees model configuration based on sampled parameters."""
    
    # Constants
    large_number = 10e5
    kip, inch, sec = 1.0, 1.0, 1.0
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
    
    # Extract parameters
    ufp_number = raw_config["n_ufp"]
    n_stories = raw_config["number_of_stories"]
    wall_len_m = raw_config["wall_length_m"]
    building_area = raw_config["building_area"]
    n_ply = raw_config.get("wall_n_ply", 7)  # Default 7 ply
    story_h_ft = raw_config.get("story_height_ft", 10)  # Default 10 ft
    

    ply_to_value = {
        7: 31.3  # * kip / ft
    }
    in_plane_shear_cap = ply_to_value[n_ply]

    # Calculate wall thickness and properties based on ply count
    # Typical CLT: 1.375" per ply
    wall_thickness = n_ply * 1.375 * inch
    


    # Story heights: first story 146", rest from parameter
    story_heights = [146] + [int(story_h_ft * ft)] * (n_stories - 1)
    


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
            "Stories Heights": story_heights,
            "Half Building Area": building_area * meter**2,
        },
        
        "wall": {
            "PT Extender Length": 5,
            "Wall Length": wall_len_m * meter,
            "Wall Extension": 0.61 * meter,
            "Wall Thickness": wall_thickness,
            "Span Number": 2,
            "Wall Elastic Modulus": 12400 *MPa,
            "Wall Shear Modulus": 12400/16 * MPa,
            "Wall Yield Stress": 31 * MPa,
            "Weight Density": 0.037 * kip / ft ** 3,
            'Shear Correction Factor': 5/6,
            "Plastic Hinge Length Reference": "thickness",
            "Plastic Hinge Length Ratio": 2,
            "Wall Split Strain": 0.0056,
            "Wall Crush Strain": 0.015,
            "Pinch Factor X": 1.0,
            "Pinch Factor Y": 1.0,
            "Initial Damage X": 0.0,
            "Initial Damage Y": 0.0,
            "Deterioration Factor": 0.0,
            "Split Strength Ratio": 0.98,
            "Crush Strength Ratio": 0.25,
            'In-Plane Shear Capacity': in_plane_shear_cap * kip / ft, 
        },
        
        "UFP": {
            "Hysteresis Model": 'Baird et al.',
            "UFP Numbers": [ufp_number] * n_stories,
            "UFP Height Ratios": [[i / (ufp_number+1) for i in range(1, ufp_number+1)]] * n_stories,
            "UFP Width": 0.114 * meter,
            "UFP Thickness": 0.0095 * meter,
            "UFP Diameter": 0.0921 * meter,
            "UFP Yield Stress": 414 * MPa,
            "UFP Ultimate Stress": 1.5,
            "R": 25,
            "b": 0.01,
            "cR1": 0.925,
            "cR2": 0.15,
            "a1": 0.05,
            "a2": 2.0,
            "a3": 0.05,
            "a4": 2.0,
            "Material":{
                'Mass Density':{'SI':  7850 # kg/m³
                        },
            }
        },
        
        "spring": {
            "Number of Rocking Springs": 40
        },
        
        "steel_elastic_module": 29000 * ksi,
        
        "bar": {
            "Material": {
                'Elastic Module': 29000,
                "Ultimate Stress": 125,
                "Yield Stress": 105,
                "Post Yield Hardening Ratio": 0.00965,
                "PT to Yield Force Ratio": 0.39,
                'Mass Density':
                        {'SI':  7850 # kg/m³
                        }
            },
            "Diameter": 16.54 * mm,
            "Number Per Panel": 4
        },
        
        "leaning_columns": {
            "Section Area": large_number,
            "Section Iz": large_number,
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
            "damping": {
                "modes": [1, 2],
                "damping ratio": 0.02
            },
            "time_integration": {
                "beta": 0.25,
                "gamma": 0.5,
                "time_step_ratio": 1
            }
        }
    }
    
    return config