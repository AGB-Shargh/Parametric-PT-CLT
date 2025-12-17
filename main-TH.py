# fragility_main.py
"""Parametric fragility analysis runner for PT-CLT systems."""
import logging
from pathlib import Path
import time
from building_info_function import generate_configurations
from core.model_config import ConfigLoader
from core.model_state import ModelState
from core.model_orchestrator import ModelOrchestrator
from core.analysis_runner import AnalysisRunner
import openseespy.opensees as ops

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def parse_gm_dt(dt_file):
    """Parse ground motion dt values from file."""
    gm_dt = {}
    with open(dt_file, 'r') as f:
        for line in f:
            if ':' in line:
                name, dt = line.strip().split(':')
                gm_dt[name.strip()] = float(dt.strip())
    return gm_dt

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Define parameter space
    wall_lengths = [2, 2.5, 3.5]
    ufp_thicknesses = [8, 10, 12]
    pt_ratios = [0.3, 0.5, 0.7]
    n_stories_list = [2, 6]

    # Fixed parameters
    n_ufp = 6
    building_area = 53.95  # m^2
    bar_diameter = 44.5  # mm
    
    # Ground motion setup
    base_dir = Path(__file__).parent
    gm_dir = base_dir / "Records"
    dt_file = gm_dir / "dt_values.txt"
    gm_dt = parse_gm_dt(dt_file)
    
    gm_files = [f for f in gm_dir.glob("RSN*.txt")]

    # Recorders setup
    record_components = ['UFP', 'PT_Bar', 'Rocking_Spring', 'Diaphragm']
    node_responses = ['disp']
    element_responses = ['force', 'deformation']

    results_base = base_dir / "Fragility_Results"
    results_base.mkdir(exist_ok=True)
    
    total_configs = len(wall_lengths) * len(ufp_thicknesses) * len(pt_ratios) * len(n_stories_list)
    config_count = 0
    
    # Parameter loops
    for wall_len in wall_lengths:
        for ufp_thick in ufp_thicknesses:
            for pt_ratio in pt_ratios:
                for n_stories in n_stories_list:
                    config_count += 1
                    
                    # Create configuration name
                    config_name = f"L{wall_len:.1f}_UFP{ufp_thick}_PT{pt_ratio:.1f}_S{n_stories}"
                    logger.info(f"[{config_count}/{total_configs}] Running {config_name}")
                    
                    # Gnerate raw config
                    raw_config = {
                        "wall_length_m": wall_len,
                        "n_ufp": n_ufp,
                        "ufp_thickness_mm": ufp_thick,
                        "pt_ratio": pt_ratio,
                        "number_of_stories": n_stories,
                        "building_area": building_area,
                        "bar_diameter_mm": bar_diameter,
                        "wall_n_ply": 7,
                        "story_height_ft": 10
                    }
                    
                    # Generate full config
                    config = generate_configurations(raw_config)
                    validated_config = ConfigLoader.load(config)
                    
                    # Run ground motions
                    for gm_file in gm_files:
                        gm_name = gm_file.stem
                        dt = gm_dt.get(gm_name, 0.01)
                        
                        logger.info(f"  Running {gm_name} (dt={dt})")
                        
                        try:
                            # Wipe and rebuild for each analysis
                            ops.wipe()
                            
                            # Create model state
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
                            
                            # Build model
                            orchestrator = ModelOrchestrator(state)
                            orchestrator.build(pt_tolerance=0.01)
                            
                            # Run analysis with limited recorders
                            runner = AnalysisRunner(state)
                            result_dir = results_base / config_name / gm_name
                            
                            
                            runner.run_ground_motion(
                                gm_file, 
                                dt, 
                                scale_factor = 1.0, 
                                result_path = result_dir,
                                record_components = record_components,
                                node_responses = node_responses,
                                element_responses = element_responses
                            )
                            
                        except Exception as e:
                            logger.error(f"  Failed {gm_name}: {e}")
                            continue
    
    logger.info(f"Completed all {total_configs} configurations")

if __name__ == "__main__":
    main()


