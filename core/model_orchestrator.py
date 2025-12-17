# src/core/model_orchestrator.py
"""
Orchestrates model building sequence.
Replaces build() and helper methods from RockingWallBuilding.
"""
import logging
import openseespy.opensees as ops
from contextlib import contextmanager
from typing import Optional
from typing import Union
from pathlib import Path
import numpy as np

from core.model_state import ModelState
from core.tag_manager import TagGenerator
from core.material_factory import MaterialFactory
from core.structural_utils import StructuralUtils
from core.components.wall import Wall
from core.components.ufp import UFP
from core.components.pt_bar import PTBar
from core.components.diaphragm import Diaphragm
from core.components.leaning_column import LeaningColumn
from core.components.rocking_spring import RockingSpring


class ModelOrchestrator:
    """Coordinates model building sequence."""
    
    def __init__(self, state: ModelState):
        """Initialize orchestrator with model state.
        
        Args:
            state: ModelState instance containing all structural data
        """
        self.state = state
        self.logger = logging.getLogger(__name__)
        
        # Initialize managers
        self.tags = TagGenerator(state)
        self.materials = MaterialFactory(state)
        self.utils = StructuralUtils(state)
        
        # Initialize components
        self.wall = Wall(state, self.tags)
        self.ufp = UFP(state, self.tags, self.materials)
        self.pt_bar = PTBar(state, self.tags, self.materials)
        self.diaphragm = Diaphragm(state, self.tags, self.materials)
        self.leaning = LeaningColumn(state, self.tags)
        self.rocking = RockingSpring(state, self.tags, self.materials)

    def build_elastic_model(self) -> None:
        """Build fixed-based linear model.
        """
        
        self.state.model_type = 'Elastic'
        # with self._suppress_logging():

        # Reset and rebuild
        self.tags.reset_tags()
        self._initialize_opensees()
        self._compute_properties()
        self._initialize_tags()
        self._create_nodes()
        self._define_materials_pre_pt()
        self._create_elements_pre_pt()
        self.utils.assign_mass_load()
        _, _ = self.materials.define_bar_material()
        self.pt_bar.create_elements()

        # Run static analysis to get actual tension
        self.utils.run_static_analysis()

        
        # Final assembly
        self.ufp.create_nodes(location='Between Wall')
        self.materials.define_ufp_material()
        self.ufp.create_elements()

        self.tags.save_tag_documentation(Path(__file__).parent)
        
        ops.loadConst("-time", 0.0)
        ops.wipeAnalysis()

    def build(self, pt_tolerance: float = 0.01) -> None:
        """Build complete model with iterative PT tension calibration.
        
        Args:
            target_pt_tension: Target PT bar tension
            tolerance: Convergence tolerance (fraction of target)
        """
        iter_strain = None
        current_tension = 0.1
        iteration = 0
        target_pt_tension = 10
        
        while abs(target_pt_tension - current_tension) > pt_tolerance * target_pt_tension:
            iteration += 1

            with self._suppress_logging():
                # Reset and rebuild
                self.tags.reset_tags()
                self._initialize_opensees()
                self._compute_properties()
                self._initialize_tags()
                self._create_nodes()
                self._define_materials_pre_pt()
                self._create_elements_pre_pt()
                self.utils.assign_mass_load()
                
                # Define PT material with current strain
                target_pt_tension, initial_strain = self.materials.define_bar_material(
                    initial_PT_strain=iter_strain
                )
                self.pt_bar.create_elements()
                
                # Run static analysis to get actual tension
                self.utils.run_static_analysis()

                bar_ultimate_force = self.state.bar['Ultimate Force']
                pt_tag = self.state.bar["Elements"]["PT Bars"]["Left Wall"]["Left"]
                current_tension = abs(ops.eleResponse(pt_tag, "force")[1])
                
                if current_tension == 0:
                    raise ValueError("PT bar has zero tension - check model setup")
                
                print(f"  PT Target: {target_pt_tension:.2f}, Current: {current_tension:.2f}, Ultimate: {bar_ultimate_force:.2f}")
                
                # Update strain estimate
                iter_strain = (target_pt_tension / current_tension) * initial_strain
        
        # Final assembly
        self._pt_iterations = iteration
        self.ufp.create_nodes(location='Between Wall')
        self.materials.define_ufp_material()
        self.ufp.create_elements()
        self.utils.modal_damping()
        self.tags.save_tag_documentation(Path(__file__).parent.parent)

        ops.loadConst("-time", 0.0)
        ops.wipeAnalysis()
        
        self.logger.info(f"Model built successfully in {iteration} iterations")
    
    def _initialize_opensees(self) -> None:
        """Initialize OpenSees model and transformations."""
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)
        ops.geomTransf('Linear', self.state.linear_geom_transf)
        ops.geomTransf('PDelta', self.state.pdelta_geom_transf)
    
    def _compute_properties(self) -> None:
        """Compute all derived structural properties."""
        self.wall.compute_properties()
    
    def _initialize_tags(self) -> None:
        """Generate all node and element tags."""
        components = ['Wall', 'UFP', 'PT Bar', 'Diaphragms', 
                     'Leaning Columns', 'Rocking Springs']
        
        for component in components:
            self.tags.generate_node_tags(component)
            if component != 'PT Bar':
                self.tags.generate_element_tags(component)
            else:
                self.tags.generate_element_tags('PT Bars')
        
        self.tags.generate_element_tags('Shear Keys')
    
    def _create_nodes(self) -> None:
        """Create all nodes in sequence."""
        self.wall.create_nodes()
        self.ufp.create_nodes(location='On Wall')
        self.rocking.create_nodes()
        self.pt_bar.create_nodes()
        self.leaning.create_nodes()
        self.diaphragm.create_nodes()
    
    def _define_materials_pre_pt(self) -> None:
        """Define materials except PT bars."""
        self.materials.preliminaries()
        self.materials.define_rocking_springs_material()
        self.materials.define_shear_key_mat()
    
    def _create_elements_pre_pt(self) -> None:
        """Create elements except PT bars."""
        self.wall.create_elements()
        self.rocking.create_elements()
        self.leaning.create_elements()
        self.diaphragm.create_elements()
        self.diaphragm.define_shear_keys()
    
    @contextmanager
    def _suppress_logging(self):
        """Temporarily suppress logging during iteration."""
        logger = logging.getLogger()
        previous_level = logger.level
        logger.setLevel(logging.CRITICAL)
        try:
            yield
        finally:
            logger.setLevel(previous_level)

    def save_model_summary(self, output_path: Union[str, Path]) -> None:
        """Save comprehensive model characteristics to CSV after building.
        
        Args:
            output_path: Path to save CSV file
           
        Note:
            All units are imperial (kip, inch, sec base system)
        """
        from pathlib import Path
        import pandas as pd
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # =====================================================================
        # GEOMETRY
        # =====================================================================
        data = {
            'n_stories': self.state.n_stories,
            'building_height_in': self.state.building_height if hasattr(self.state, 'building_height') else sum(self.state.building['Stories Heights']),
            'wall_length_in': self.state.wall['Wall Length'],
            'wall_thickness_in': self.state.wall['Wall Thickness'],
            'wall_area_in2': self.state.wall['Wall Cross Section Area'],
            'wall_Iz_in4': self.state.wall['Wall Iz'],
            'wall_shear_area_in2': self.state.wall['Wall Shear Area'],
            'building_area_in2': self.state.building['Half Building Area'],
        }
        
        # Story heights
        for i, h in enumerate(self.state.building['Stories Heights'], 1):
            data[f'story_{i}_height_in'] = h
        
        # =====================================================================
        # MATERIAL PROPERTIES (Input)
        # =====================================================================
        data.update({
            'wall_E_psi': self.state.wall['Wall Elastic Modulus'],
            'wall_G_psi': self.state.wall['Wall Shear Modulus'],
            'wall_fy_psi': self.state.wall['Wall Yield Stress'],
            'wall_weight_density_kip_in3': self.state.wall['Weight Density'],
            'wall_split_strain': self.state.wall['Wall Split Strain'],
            'wall_crush_strain': self.state.wall['Wall Crush Strain'],
            'wall_split_strength_ratio': self.state.wall['Split Strength Ratio'],
            'wall_crush_strength_ratio': self.state.wall['Crush Strength Ratio'],
            'plastic_hinge_length_in': self.state.spring.get('Plastic Hinge Length', 0),
            'plastic_hinge_length_ratio': self.state.wall['Plastic Hinge Length Ratio'],
            'plastic_hinge_reference': self.state.wall['Plastic Hinge Length Reference'],
            
            'UFP_width_in': self.state.UFP['UFP Width'],
            'UFP_thickness_in': self.state.UFP['UFP Thickness'],
            'UFP_diameter_in': self.state.UFP['UFP Diameter'],
            'UFP_fy_psi': self.state.UFP['UFP Yield Stress'],
            'UFP_hysteresis_model': self.state.UFP.get('Hysteresis Model', 'Unknown'),
            
            'PT_bar_diameter_in': self.state.bar['Diameter'],
            'PT_bar_area_in2': self.state.bar.get('Area', 0),
            'PT_bar_E_ksi': self.state.bar['Material']['Elastic Module'],
            'PT_bar_fy_ksi': self.state.bar['Material']['Yield Stress'],
            'PT_bar_fu_ksi': self.state.bar['Material']['Ultimate Stress'],
            'PT_bar_hardening_ratio': self.state.bar['Material']['Post Yield Hardening Ratio'],
            'PT_target_ratio': self.state.bar['Material']['PT to Yield Force Ratio'],
            'n_PT_bars_per_wall': self.state.bar.get('Number Per Panel', 2),
            'n_PT_bars_total': self.state.bar.get('Number Per Panel', 2) * 2,
        })
        
        # =====================================================================
        # LOADS
        # =====================================================================
        data.update({
            'floor_DL_density_kip_in2': self.state.building['Floor Dead Load Density'],
            'roof_DL_density_kip_in2': self.state.building['Roof Dead Load Density'],
            'floor_DL_intensity_kip': self.state.building.get('Floor Dead Load Intensity', 0),
            'roof_DL_intensity_kip': self.state.building.get('Roof Dead Load Intensity', 0),
        })
        
        # =====================================================================
        # DYNAMIC PROPERTIES (Computed)
        # =====================================================================
        if self.state.periods:
            for i, T in enumerate(self.state.periods, 1):
                data[f'T{i}_sec'] = T
        
        # =====================================================================
        # SEISMIC MASSES (Computed)
        # =====================================================================
        total_mass = 0
        for story in range(1, self.state.n_stories + 1):
            mass = self.state.building.get(f'Elevation {story}', {}).get('Seismic Mass', 0)
            data[f'floor_{story}_mass_kip_s2_in'] = mass
            total_mass += mass
        data['total_seismic_mass_kip_s2_in'] = total_mass
        
        # =====================================================================
        # PT BAR PROPERTIES (Computed)
        # =====================================================================
        data.update({
            'PT_initial_force_kip': self.state.bar.get('Initial PT Force', 0),
            'PT_initial_disp_in': self.state.bar.get('Initial Disp.', 0),
            'PT_yield_force_kip': self.state.bar['Material'].get('Yield Force', 0),
            'PT_ultimate_force_kip': self.state.bar.get('Ultimate Force', 0),
            'PT_yield_strain': self.state.bar.get('Yield Strain', 0),
            'PT_ultimate_strain': self.state.bar.get('Ultimate Strain', 0),
            'PT_total_yield_capacity_kip': self.state.bar['Material'].get('Yield Force', 0) * data['n_PT_bars_total'],
            'PT_iterations': self._pt_iterations if self._pt_iterations is not None else 'N/A',
        })
        
        # =====================================================================
        # UFP PROPERTIES (Computed)
        # =====================================================================
        data.update({
            'UFP_stiffness_kip_in': self.state.UFP.get('Stiffness', 0),
            'UFP_yield_force_kip': self.state.UFP.get('Yield Force', 0),
            'UFP_ultimate_force_kip': self.state.UFP.get('Ultimate Force', 0),
            'UFP_yield_disp_in': self.state.UFP.get('Yield Disp.', 0),
        })
        
        # UFP counts per story and totals
        total_ufps = 0
        total_ufp_capacity = 0
        ufp_yield_force = self.state.UFP.get('Yield Force', 0)
        
        for i, count in enumerate(self.state.UFP['UFP Numbers'], 1):
            data[f'story_{i}_n_UFPs'] = count
            total_ufps += count
            total_ufp_capacity += count * ufp_yield_force
        
        data['total_n_UFPs'] = total_ufps
        data['UFP_total_yield_capacity_kip'] = total_ufp_capacity
        
        # =====================================================================
        # ROCKING SPRING PROPERTIES (Computed)
        # =====================================================================
        n_springs = self.state.spring['Number of Rocking Springs']
        data['n_rocking_springs'] = n_springs
        
        # Example spring properties (first spring)
        if 'Rocking' in self.state.spring and 'Material' in self.state.spring['Rocking']:
            spring_mat = self.state.spring['Rocking']['Material']
            if 'Yield Force 0' in spring_mat:
                data['rocking_spring_0_yield_force_kip'] = spring_mat['Yield Force 0']
                data['rocking_spring_0_yield_disp_in'] = spring_mat['Yield Disp. 0']
                
                # Estimate total rocking capacity (sum all springs)
                total_rocking_capacity = sum(
                    spring_mat.get(f'Yield Force {i}', 0) 
                    for i in range(n_springs)
                )
                data['rocking_total_yield_capacity_kip'] = total_rocking_capacity
        
        # =====================================================================
        # SHEAR KEY PROPERTIES (Computed)
        # =====================================================================
        for story in range(1, self.state.n_stories + 1):
            stiffness = self.state.diaphragm['Elements']['Shear Key'].get(
                f'Story {story}', {}
            ).get('Stiffness', 0)
            data[f'story_{story}_shear_key_stiffness_kip_in'] = stiffness
        
        # =====================================================================
        # DAMPING (Computed)
        # =====================================================================
        damping_config = self.state.analysis.get('damping', {})
        data.update({
            'damping_ratio': damping_config.get('damping ratio', 0),
            'damping_mode_1': damping_config.get('modes', [1, 2])[0],
            'damping_mode_2': damping_config.get('modes', [1, 2])[1] if len(damping_config.get('modes', [1, 2])) > 1 else 'N/A',
        })
        
        # Rayleigh coefficients (need to compute from periods if available)
        if self.state.periods and len(self.state.periods) >= 2:
            modes = damping_config.get('modes', [1, 2])
            damping_ratio = damping_config.get('damping ratio', 0.02)
            omega_1 = 2 * np.pi / self.state.periods[modes[0] - 1]
            omega_2 = 2 * np.pi / self.state.periods[modes[1] - 1]
            alpha_m = 2.0 * damping_ratio * omega_1 * omega_2 / (omega_1 + omega_2)
            beta_k = 2.0 * damping_ratio / (omega_1 + omega_2)
            data['rayleigh_alpha_M'] = alpha_m
            data['rayleigh_beta_K'] = beta_k
        
        # =====================================================================
        # ANALYSIS CONFIGURATION
        # =====================================================================
        data.update({
            'model_type': self.state.model_type,
            'time_integration_beta': self.state.analysis['time_integration'].get('beta', 0.25),
            'time_integration_gamma': self.state.analysis['time_integration'].get('gamma', 0.5),
        })
        
        # =====================================================================
        # SYSTEM RATIOS
        # =====================================================================
        if data['building_height_in'] > 0 and data['wall_length_in'] > 0:
            data['wall_aspect_ratio'] = data['building_height_in'] / data['wall_length_in']
        
        # =====================================================================
        # SAVE TO CSV
        # =====================================================================
        df = pd.DataFrame([data])       
        df_long = df.melt(var_name='Property', value_name='Value')
        long_output_path = Path(output_path).with_name(Path(output_path).stem + '.csv')
        df_long.to_csv(long_output_path, index=False)

        self.logger.info(f"Saved model summary to {output_path}")