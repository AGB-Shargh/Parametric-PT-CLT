# main_opt.py
"""
Multi-location seismic optimization for PT-CLT systems using Mixed-Integer DE.
Optimizes GWP across West Coast seismic gradient with comprehensive constraints.
"""
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from scipy.optimize import differential_evolution, Bounds
import openseespy.opensees as ops
from datetime import datetime
import time
import traceback

from building_info_function import generate_configurations
from core.model_config import ConfigLoader
from core.model_state import ModelState
from Seismic_Design.seismic_functions import SeismicFunctions
from Seismic_Design.design_objectives import DesignObjectives
from core.structural_utils import StructuralUtils
from core.analysis_runner import AnalysisRunner
from core.model_orchestrator import ModelOrchestrator
from Seismic_Design.cross_sectional_analysis import CrossSectionalAnalysis

# ============================================================================
# CONSTANTS
# ============================================================================
kip, inch, sec = 1.0, 1.0, 1.0
ft = 12 * inch
ksi = kip / inch ** 2
psi = ksi / 1000
mm = inch / 25.4
meter = 1000 * mm
MPa = 145.038 * psi
kN = MPa / 1000 * meter ** 2
g = 9.81 * meter / sec ** 2

script_dir = Path(__file__).parent

# ============================================================================
# SEISMIC DATA ACQUISITION
# ============================================================================
def get_asce7_22_params(lat: float, lon: float, risk_category: str = 'II', 
                        site_class: str = 'D', title: str = 'MySite'):
    """Query USGS ASCE7-22 web service for seismic hazard parameters."""
    url = "https://earthquake.usgs.gov/ws/designmaps/asce7-22.json"
    params = {'latitude': lat, 'longitude': lon, 'riskCategory': risk_category,
              'siteClass': site_class, 'title': title}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if data['request']['status'] != 'success':
        raise RuntimeError(f"USGS request failed: {data}")
    d = data['response']['data']
    return {k: d.get(k) for k in ['ss', 's1', 'sms', 'sm1', 'sds', 'sd1', 'tl', 'pgam']}

# ============================================================================
# MODEL BUILDER
# ============================================================================
def build_model_state(x, user_params):
    """
    Build ModelState from design variables.
    x = [wall_len_m, bar_d_mm, pt_ratio, ufp_thick_mm, ufp_width_mm, ufp_diam_mm, n_ufp]
    """
    wall_len_m, bar_d_mm, pt_ratio, ufp_thick_mm, ufp_width_mm, ufp_diam_mm, n_ufp = x
    n_ufp = int(n_ufp)  # Already integer from DE integrality
    
    raw_config = {
        "number_of_stories": user_params["n_stories"],
        "story_height_ft": user_params["story_height_ft"],
        "building_area": user_params["building_area"],
        "wall_n_ply": 7,  # Fixed 7-ply
        "wall_length_m": wall_len_m,
        "n_ufp": n_ufp,
    }
    
    config = generate_configurations(raw_config)
    
    # Override UFP properties
    config["UFP"]["UFP Thickness"] = ufp_thick_mm * mm
    config["UFP"]["UFP Width"] = ufp_width_mm * mm
    config["UFP"]["UFP Diameter"] = ufp_diam_mm * mm
    
    # Override bar properties
    config["bar"]["Diameter"] = bar_d_mm * mm
    config["bar"]["Material"]["PT to Yield Force Ratio"] = pt_ratio
    
    validated_config = ConfigLoader.load(config)
    state = ModelState(
        n_stories=validated_config['n_stories'],
        model_type=validated_config['model_type'],
        steel_elastic_module=validated_config['steel_elastic_module'],
        wall=validated_config['wall'],
        UFP=validated_config['UFP'],
        bar=validated_config['bar'],
        spring=validated_config['spring'],
        diaphragm=validated_config['diaphragm'],
        leaning_columns=validated_config['leaning_columns'],
        building=validated_config['building'],
        analysis=validated_config['analysis']
    )
    
    orchestrator = ModelOrchestrator(state)
    orchestrator.build_elastic_model()
    return state

# ============================================================================
# OBJECTIVE AND CONSTRAINTS
# ============================================================================
def objective(x, user_params):
    """Minimize GWP (kg CO2)."""
    try:
        state = build_model_state(x, user_params)
        utils = StructuralUtils(state)
        gwp_results = utils.calculate_GWP()
        return gwp_results["total"]
    except Exception as e:
        print(f"Warning: Objective failed: {e}")
        return 1e10

def aggregate_constraints(x, user_params):
    """
    Compute aggregate constraint: TOLERANCE - max(DCR).
    Returns: >0 = feasible, <=0 = infeasible
    """
    try:
        state = build_model_state(x, user_params)
        
        seismic = SeismicFunctions(
            state=state,
            SS=user_params['SS'], S1=user_params['S1'],
            SMS=user_params['SMS'], SM1=user_params['SM1'],
            site_class=user_params['site_class'],
            TL=user_params['TL'],
            R=user_params['R'], Ie=user_params['Ie'], Cd=user_params['Cd']
        )
        
        objectives = DesignObjectives(state, seismic)
        runner = AnalysisRunner(state)
        
        # Cross-sectional analysis for DBE and MCE
        csa = CrossSectionalAnalysis(state)
        state.csa_results = {
            'csa_de': csa.analysis(0.02),   # DBE: 2% drift
            'csa_mce': csa.analysis(0.03)   # MCE: 3% drift
        }
        
        # Calculate all DCRs
        dcrs = {
            'drift_de': objectives.drift_ratio_de(runner),
            'moment': objectives.moment_dcr(),
            'shear': objectives.shear_dcr(),
            'energy_dissipation': objectives.energy_dissipation_ratio(min_ratio=0.3),
            'pt_yield_de': objectives.pt_yield_dcr_de(),
            'pt_failure_mce': objectives.pt_failure_dcr_mce(),
            'ufp_failure_mce': objectives.ufp_failure_dcr_mce(),
            'wall_crush_mce': objectives.wall_crush_dcr_mce(),
        }
        
        user_params['_last_dcrs'] = dcrs.copy()
        max_dcr = max(dcrs.values())

        # Apply 1% tolerance
        TOLERANCE = 1.01
        return TOLERANCE - max_dcr  # Returns >0 if max_dcr <= 1.01

        
    except Exception as e:
        print(f"Warning: Constraint evaluation failed: {e}")
        user_params['_last_dcrs'] = {'error': str(e)}
        return -1000.0

def penalized_objective(x, user_params, penalty=5000):
    """Objective with penalty for constraint violation."""
    gwp = objective(x, user_params)
    constraint_val = aggregate_constraints(x, user_params)
    
    if constraint_val < 0:  # Infeasible
        return gwp + penalty * abs(constraint_val)
    return gwp

# ============================================================================
# LOGGING
# ============================================================================
class OptLogger:
    """Logs optimization progress."""
    def __init__(self, output_dir: Path, run_id: str):
        self.output_dir = output_dir / "Optimization-Steps" / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.start_time = time.time()
        self.best_gwp = float('inf')
        self.best_x = None
    
    def log_iteration(self, x, convergence, user_params):
        """Log every 10 iterations."""
        self.step += 1
        if self.step % 10 != 0:
            return
        
        try:
            gwp = objective(x, user_params)
            constraint_val = aggregate_constraints(x, user_params)
        except:
            return
        
        if gwp < self.best_gwp:
            self.best_gwp = gwp
            self.best_x = x.copy()
        
        log_file = self.output_dir / f"iter_{self.step:04d}.txt"
        dcrs = user_params.get('_last_dcrs', {})
        
        TOLERANCE = 1.01
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"ITERATION {self.step}\n{'='*70}\n\n")
            f.write(f"Time: {time.time() - self.start_time:.1f}s\n")
            f.write(f"Convergence: {convergence:.6e}\n\n")
            f.write(f"DESIGN: wall_len={x[0]:.3f}m, bar_d={x[1]:.1f}mm, pt_ratio={x[2]:.3f}\n")
            f.write(f"        ufp_thick={x[3]:.1f}mm, ufp_width={x[4]:.1f}mm, ufp_diam={x[5]:.1f}mm, n_ufp={int(x[6])}\n\n")
            f.write(f"GWP: {gwp:.1f} kg CO2 | Best: {self.best_gwp:.1f}\n")
            f.write(f"Constraint: {constraint_val:.4f} ({'PASS' if constraint_val > 0 else 'FAIL'})\n\n")
            if dcrs and 'error' not in dcrs:
                f.write(f"DCRs (tolerance={TOLERANCE}):\n")
                for name, dcr in dcrs.items():
                    status = "PASS" if dcr <= TOLERANCE else "FAIL"
                    f.write(f"  {name:20s}: {dcr:6.4f}  {status}\n")
    
    def log_final(self, result, user_params):
        """Log final result."""
        x = result.x
        final_gwp = objective(x, user_params)
        final_constraint = aggregate_constraints(x, user_params)
        dcrs = user_params.get('_last_dcrs', {})
        
        TOLERANCE = 1.01
        
        with open(self.output_dir / "FINAL_RESULT.txt", 'w', encoding='utf-8') as f:
            f.write(f"FINAL OPTIMIZATION RESULT\n{'='*70}\n\n")
            f.write(f"Total time: {time.time() - self.start_time:.1f}s\n")
            f.write(f"Iterations: {result.nit} | Evals: {result.nfev}\n")
            f.write(f"Success: {result.success} | Message: {result.message}\n\n")
            f.write(f"OPTIMAL DESIGN:\n")
            f.write(f"  Wall length:  {x[0]:.4f} m\n")
            f.write(f"  Bar diameter: {x[1]:.2f} mm\n")
            f.write(f"  PT ratio:     {x[2]:.4f}\n")
            f.write(f"  UFP thick:    {x[3]:.2f} mm\n")
            f.write(f"  UFP width:    {x[4]:.2f} mm\n")
            f.write(f"  UFP diameter: {x[5]:.2f} mm\n")
            f.write(f"  n_UFP:        {int(x[6])}\n\n")
            f.write(f"GWP: {final_gwp:.1f} kg CO2\n")
            f.write(f"Constraint: {final_constraint:.4f}\n\n")
            if dcrs and 'error' not in dcrs:
                f.write(f"FINAL DCRs (tolerance={TOLERANCE}):\n")
                for name, dcr in sorted(dcrs.items(), key=lambda x: x[1], reverse=True):
                    status = "PASS" if dcr <= TOLERANCE else "FAIL"
                    f.write(f"  {name:20s}: {dcr:6.4f}  {status}\n")

# ============================================================================
# OPTIMIZATION STUDY
# ============================================================================
def run_optimization_study():
    """Run multi-location optimization using Mixed-Integer DE."""
    
    CITIES = {
        'San_Diego_CA': {'lat': 32.715, 'lon': -117.162},
        'Los_Angeles_CA': {'lat': 34.052, 'lon': -118.243},
        'San_Francisco_CA': {'lat': 37.775, 'lon': -122.419},
        'Portland_OR': {'lat': 45.515, 'lon': -122.677},
        'Seattle_WA': {'lat': 47.608, 'lon': -122.335},
        'Eureka_CA': {'lat': 40.802, 'lon': -124.164},
        'San_Luis_Obispo_CA': {'lat': 35.283, 'lon': -120.660},
    }
    
    N_STORIES_OPTIONS = [2, 4, 6, 8]
    SITE_CLASS = 'D'
    
    # Design variable bounds
    bounds_array = Bounds(
        lb=[1.5, 16.0, 0.2, 6.0, 80.0, 60.0, 3],
        ub=[4.0, 50.0, 0.8, 15.0, 180.0, 140.0, 10]
    )
    
    # Mark n_ufp (index 6) as integer
    integrality = [False, False, False, False, False, False, True]
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / f"optimization_study_MIDE_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    with open(output_dir / 'metadata.txt', 'w', encoding='utf-8') as f:
        f.write(f"Multi-location PT-CLT optimization using Mixed-Integer DE\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Cities: {list(CITIES.keys())}\n")
        f.write(f"Stories: {N_STORIES_OPTIONS}\n")
        f.write(f"Total runs: {len(CITIES) * len(N_STORIES_OPTIONS)}\n")
        f.write(f"Variables: [wall_len_m, bar_d_mm, pt_ratio, ufp_thick_mm, ufp_width_mm, ufp_diam_mm, n_ufp]\n")
        f.write(f"Integrality: {integrality}\n")
    
    print("="*70)
    print("MULTI-LOCATION SEISMIC OPTIMIZATION (MIXED-INTEGER DE)")
    print("="*70)
    print(f"Cities: {len(CITIES)} | Stories: {N_STORIES_OPTIONS}")
    print(f"Total runs: {len(CITIES) * len(N_STORIES_OPTIONS)}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    all_results = []
    run_counter = 0
    
    for city_name, coords in CITIES.items():
        lat, lon = coords['lat'], coords['lon']
        
        print(f"\n{'='*70}")
        print(f"LOCATION: {city_name} ({lat:.3f}, {lon:.3f})")
        print("="*70)
        
        # Fetch seismic parameters
        try:
            seismic_params = get_asce7_22_params(lat=lat, lon=lon, site_class=SITE_CLASS, title=city_name)
            time.sleep(0.5)
            
            # Validate
            required = ['ss', 's1', 'sms', 'sm1', 'sds', 'sd1', 'tl']
            if any(seismic_params.get(k) is None for k in required):
                print(f"ERROR: Missing seismic data")
                continue
            
            print(f"SUCCESS: SS={seismic_params['ss']:.3f}g, S1={seismic_params['s1']:.3f}g")
            
        except Exception as e:
            print(f"ERROR: Seismic data fetch failed: {e}")
            continue
        
        # Loop over building heights
        for n_stories in N_STORIES_OPTIONS:
            run_counter += 1
            run_id = f"{city_name}_n{n_stories}"
            
            print(f"\n{'-'*70}")
            print(f"RUN {run_counter}: {run_id}")
            print("-"*70)
            
            # Setup parameters
            user_params = {
                "n_stories": n_stories,
                "story_height_ft": 10,
                "building_area": 53.95,
                "site_class": SITE_CLASS,
                "R": 8.0, "Ie": 1.0, "Cd": 8.0,
                "SS": seismic_params['ss'],
                "S1": seismic_params['s1'],
                "SMS": seismic_params['sms'],
                "SM1": seismic_params['sm1'],
                "TL": seismic_params['tl'],
            }
            
            logger = OptLogger(output_dir, run_id)
            
            def callback(x, convergence=0.0):
                logger.log_iteration(x, convergence, user_params)
            
            # Create diverse initial population for n_ufp exploration
            popsize = 15
            n_vars = 7
            init_pop = np.zeros((popsize, n_vars))
            
            # Force n_ufp diversity: sample across [3, 4, 5, ..., 10]
            n_ufp_values = np.linspace(3, 10, popsize).astype(int)
            
            for i in range(popsize):
                # Random continuous variables
                init_pop[i, 0] = np.random.uniform(1.5, 4.0)    # wall_len
                init_pop[i, 1] = np.random.uniform(16.0, 50.0)  # bar_d
                init_pop[i, 2] = np.random.uniform(0.2, 0.8)    # pt_ratio
                init_pop[i, 3] = np.random.uniform(6.0, 15.0)   # ufp_thick
                init_pop[i, 4] = np.random.uniform(80.0, 180.0) # ufp_width
                init_pop[i, 5] = np.random.uniform(60.0, 140.0) # ufp_diam
                init_pop[i, 6] = n_ufp_values[i]                # n_ufp: FORCED DIVERSITY
            
            # Run Mixed-Integer DE
            start_time = time.time()
            
            try:
                print("Running Mixed-Integer DE...")
                
                result = differential_evolution(
                    penalized_objective,
                    bounds_array,
                    args=(user_params,),
                    integrality=integrality,
                    init=init_pop,  # Use diverse initial population
                    strategy='best1bin',
                    maxiter=300,
                    popsize=15,
                    tol=0.01,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    callback=callback,
                    disp=True,
                    polish=True,
                    updating='deferred',
                    workers=1
                )
                
                elapsed = time.time() - start_time
                logger.log_final(result, user_params)
                
                # Extract results
                x = result.x
                final_gwp = objective(x, user_params)
                final_constraint = aggregate_constraints(x, user_params)
                dcrs = user_params.get('_last_dcrs', {})
                feasible = all(dcr <= 1.01 for dcr in dcrs.values()) if dcrs and 'error' not in dcrs else False
                
                result_dict = {
                    'run': run_counter, 'city': city_name, 'lat': lat, 'lon': lon,
                    'n_stories': n_stories, 'site_class': SITE_CLASS,
                    'SS': seismic_params['ss'], 'S1': seismic_params['s1'],
                    'gwp_kg_co2': final_gwp,
                    'wall_length_m': x[0], 'bar_diameter_mm': x[1], 'pt_ratio': x[2],
                    'ufp_thickness_mm': x[3], 'ufp_width_mm': x[4], 'ufp_diameter_mm': x[5],
                    'n_ufp': int(x[6]),
                    'feasible': feasible, 'de_success': result.success,
                    'n_iter': result.nit, 'n_eval': result.nfev,
                    'time_sec': elapsed, 'error': None,
                    **({f'dcr_{k}': v for k, v in dcrs.items()} if dcrs and 'error' not in dcrs else {})
                }
                
                all_results.append(result_dict)
                
                # Print summary
                status = "FEASIBLE" if feasible else "INFEASIBLE"
                print(f"\n{status}")
                print(f"  GWP: {final_gwp:.1f} kg CO2")
                print(f"  Design: wall={x[0]:.2f}m, bar_d={x[1]:.1f}mm, n_ufp={int(x[6])}")
                print(f"  Time: {elapsed:.1f}s | Evals: {result.nfev}")
                
            except Exception as e:
                print(f"ERROR: OPTIMIZATION FAILED: {e}")
                traceback.print_exc()
                
                result_dict = {
                    'run': run_counter, 'city': city_name, 'n_stories': n_stories,
                    'feasible': False, 'de_success': False,
                    'time_sec': time.time() - start_time, 'error': str(e)
                }
                all_results.append(result_dict)
            
            # Checkpoint
            pd.DataFrame(all_results).to_csv(output_dir / 'results_checkpoint.csv', index=False)
    
    # Final results
    print(f"\n{'='*70}")
    print("STUDY COMPLETE")
    print("="*70)
    
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / 'results_final.csv', index=False)
    
    feasible = df[df['feasible'] == True]
    print(f"\nTotal: {len(df)} | Feasible: {len(feasible)}")
    if len(feasible) > 0:
        print(f"GWP: Min={feasible['gwp_kg_co2'].min():.1f}, Mean={feasible['gwp_kg_co2'].mean():.1f} kg CO2")
    
    print(f"\nResults: {output_dir}")
    return df, output_dir

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    df, output_dir = run_optimization_study()
    print("\nDone!")