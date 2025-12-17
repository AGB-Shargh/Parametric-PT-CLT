# src/core/structural_utils.py
"""
Structural utilities - mass assignment, damping, and analysis execution.
Refactored to work with ModelState and follow consistent patterns.
"""
import os
import logging
import numpy as np
import openseespy.opensees as ops
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import re

from core.model_state import ModelState
from core.constants import SMALL_NUMBER, g, meter


class StructuralUtils:
    """Handles mass assignment, modal damping, and analysis execution."""

    def __init__(self, state: ModelState):
        """Initialize with model state.
        
        Args:
            state: ModelState instance containing structural data
        """
        self.state = state
        self.logger = logging.getLogger(__name__)
        self._wall_config = state.wall
        self._building_config = state.building
        self._diaphragm_config = state.diaphragm
        self._leaning_config = state.leaning_columns
        self._analysis_config = state.analysis

    # =========================================================================
    # Mass and Load Assignment
    # =========================================================================

    def assign_mass_load(self) -> None:
        """Assign masses and loads to structure."""
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        
        self._assign_wall_weight()
        self._assign_seismic_mass()
        self._assign_leaning_column_loads()
        
        self.logger.info("Assigned all masses and loads")

    def _assign_wall_weight(self) -> None:
        """Assign wall self-weight to wall nodes."""
        required_keys = ['Weight Density', 'Wall Cross Section Area', 'Segmental Elevations']
        missing = [k for k in required_keys if k not in self._wall_config]
        if missing:
            raise KeyError(f"Wall config missing required keys: {missing}")
        
        density = self._wall_config['Weight Density']
        area = self._wall_config['Wall Cross Section Area']
        
        # Sort segments by elevation
        segments = dict(sorted(
            self._wall_config["Segmental Elevations"].items(),
            key=lambda x: int(x[0])
        ))
        
        # Initialize weights
        for seg_data in segments.values():
            seg_data["Weight"] = 0.0
        
        # Calculate segment weights (distribute to adjacent segments)
        segment_list = list(segments.items())
        total_wall_weight = 0.0
        
        for idx, (seg_key, seg_data) in enumerate(segment_list):
            current_height = seg_data["Height"]
            
            if idx > 0:
                prev_key, prev_data = segment_list[idx - 1]
                prev_height = prev_data["Height"]
                
                # Segment height and weight
                segment_height = current_height - prev_height
                segment_weight = segment_height * area * density / 2
                
                # Distribute weight to both segments
                prev_data["Weight"] += segment_weight
                seg_data["Weight"] += segment_weight
        
        # Apply weights to nodes
        for seg_key, seg_data in segments.items():
            weight = seg_data["Weight"]
            left_node = seg_data["Left Node"]
            right_node = seg_data["Right Node"]
            
            ops.load(left_node, 0.0, -weight, 0.0)
            ops.load(right_node, 0.0, -weight, 0.0)

            total_wall_weight += weight

            self.logger.debug(f"Applied wall weight {weight:.3f} to segment {seg_key}")

        self.state.wall_weight = total_wall_weight

    def _assign_seismic_mass(self) -> None:
        """Assign seismic masses to diaphragm nodes."""
        required_keys = ['Half Building Area', 'Floor Dead Load Density', 'Roof Dead Load Density']
        missing = [k for k in required_keys if k not in self._building_config]
        if missing:
            raise KeyError(f"Building config missing required keys: {missing}")
        
        area = self._building_config["Half Building Area"]
        floor_dl_density = self._building_config["Floor Dead Load Density"]
        roof_dl_density = self._building_config["Roof Dead Load Density"]
        
        # Calculate total loads
        floor_dl = floor_dl_density * area
        roof_dl = roof_dl_density * area
        
        # Store intensities
        self._building_config["Floor Dead Load Intensity"] = floor_dl
        self._building_config["Roof Dead Load Intensity"] = roof_dl
        
        # Convert to seismic mass
        floor_mass = floor_dl / g
        roof_mass = roof_dl / g
        
        for story in range(1, self.state.n_stories + 1):
            is_roof = (story == self.state.n_stories)
            mass = roof_mass if is_roof else floor_mass
            
            # Store in building config
            elevation_key = f"Elevation {story}"
            self._building_config.setdefault(elevation_key, {})
            self._building_config[elevation_key]["Seismic Mass"] = mass
            
            # Apply half mass to each wall
            diaphragm_nodes = self._diaphragm_config["Nodes"].get(elevation_key, {})
            
            for side in ["Left Wall", "Right Wall"]:
                node = diaphragm_nodes.get(side)
                if not node:
                    raise KeyError(f"Diaphragm node missing: {elevation_key}, {side}")
                
                ops.mass(node, mass / 2, 0.0, 0.0)
                
                self.logger.debug(f"Applied seismic mass {mass/2:.4f} to {elevation_key} {side}")

    def _assign_leaning_column_loads(self) -> None:
        """Assign vertical dead loads to leaning column nodes."""
        floor_dl = self._building_config.get("Floor Dead Load Intensity")
        roof_dl = self._building_config.get("Roof Dead Load Intensity")
        
        if floor_dl is None or roof_dl is None:
            raise ValueError("Dead load intensities not calculated. Run _assign_seismic_mass first.")
        
        for story in range(1, self.state.n_stories + 1):
            elevation_key = f"Elevation {story}"
            
            # Get leaning column diaphragm node
            node = self._diaphragm_config["Nodes"].get(elevation_key, {}).get("Leaning Column")
            if not node:
                raise KeyError(f"Leaning column diaphragm node missing at {elevation_key}")
            
            # Select load based on floor type
            load_value = roof_dl if story == self.state.n_stories else floor_dl
            
            # Apply vertical load (negative Y direction)
            ops.load(node, 0.0, -load_value, 0.0)
            
            self.logger.debug(f"Applied dead load {load_value:.3f} to leaning column at {elevation_key}")

    # =========================================================================
    # Modal Analysis and Damping
    # =========================================================================

    def modal_damping(self, n_modes: int = 4) -> None:
        """Define Rayleigh damping based on modal analysis.

        Args:
            n_modes: Number of modes to use for damping (default: 4)
            
        Raises:
            ValueError: If eigenvalue analysis fails
        """
        try:
            # Use default damping ratio if not specified
            damping_ratio = self._analysis_config.get('damping', {}).get('damping ratio', 0.02)

            # Use default damping modes if not specified
            modes = self._analysis_config.get('damping', {}).get('modes', [1, 2])
            if len(modes) < 2:
                first_damping_mode, second_damping_mode = 1, 2
            else:
                first_damping_mode, second_damping_mode = modes

            # Run eigenvalue analysis
            eigenvalues = ops.eigen(n_modes)
            if not eigenvalues or any(lam <= 0 for lam in eigenvalues):
                raise ValueError(f"Invalid eigenvalues from analysis: {eigenvalues}")

            # Calculate frequencies
            frequencies = [np.sqrt(lam) for lam in eigenvalues]

            # Calculate periods and store
            self.state.periods = [2 * np.pi / omega for omega in frequencies]
            self.logger.info(f"Computed periods: {[f'{T:.3f}' for T in self.state.periods]}")

            # Rayleigh damping coefficients
            omega_1 = frequencies[first_damping_mode - 1]
            omega_2 = frequencies[second_damping_mode - 1]

            alpha_m = 2.0 * damping_ratio * omega_1 * omega_2 / (omega_1 + omega_2)
            beta_k = 2.0 * damping_ratio / (omega_1 + omega_2)

            # Apply damping
            ops.rayleigh(alpha_m, beta_k, 0.0, 0.0)
            self.logger.info(f"Applied Rayleigh damping: αM={alpha_m:.6f}, βK={beta_k:.6f}")

        except Exception as e:
            self.logger.error(f"Modal damping failed: {e}")
            raise ValueError(f"Modal damping computation failed: {e}")


    # =========================================================================
    # Static Analysis
    # =========================================================================

    def run_static_analysis(self, n_steps: int = 1, tolerance: float = 1e-3, 
                           max_iter: int = 1000) -> None:
        """Run static analysis with adaptive algorithm selection.
        
        Args:
            n_steps: Number of load steps
            tolerance: Convergence tolerance
            max_iter: Maximum iterations per step
            
        Raises:
            ValueError: If analysis fails to converge
        """
        algorithms = [
            'Linear', 'Newton', 'NewtonLineSearch', 'ModifiedNewton',
            'KrylovNewton', 'SecantNewton', 'BFGS', 'Broyden'
        ]
        
        test_types = [
            'NormDispIncr', 'NormUnbalance', 'RelativeEnergyIncr',
            'RelativeNormUnbalance', 'RelativeNormDispIncr'
        ]
        
        # Setup analysis
        ops.system("BandGeneral")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0 / n_steps)
        ops.test('NormDispIncr', tolerance, max_iter)
        ops.algorithm(algorithms[0])
        ops.analysis("Static")
        
        # Run analysis with adaptive strategy
        steps_completed = 0
        
        for step in range(n_steps):
            result = ops.analyze(1)
            
            if result == 0:
                steps_completed += 1
                continue
            
            # Try different algorithms and tests
            converged = False
            attempts = 0
            max_attempts = 50
            
            for algorithm in algorithms[1:]:
                if converged:
                    break
                
                for test_type in test_types:
                    attempts += 1
                    
                    ops.algorithm(algorithm)
                    ops.test(test_type, tolerance, max_iter)
                    
                    self.logger.debug(f"Trying {algorithm} with {test_type}")
                    
                    result = ops.analyze(1)
                    
                    if result == 0:
                        steps_completed += 1
                        converged = True
                        self.logger.info(f"Converged with {algorithm} + {test_type}")
                        break
                    
                    if attempts >= max_attempts:
                        raise ValueError(
                            f"Static analysis failed after {attempts} attempts at step {step + 1}"
                        )
        
        progress = (steps_completed / n_steps) * 100
        self.logger.info(f"Static analysis completed: {progress:.1f}%")
        
        if steps_completed < n_steps:
            raise ValueError(f"Static analysis incomplete: {steps_completed}/{n_steps} steps")

    # =========================================================================
    # Dynamic Analysis
    # =========================================================================

    def EQ(self, record_path: Union[str, Path], dt: float, 
        scale_factor: float = 1, tolerance: float = 1e-6, max_iter: int = 3000) -> float:
        """Run ground motion time-history analysis."""
        
        record_path = Path(record_path)
        
        if not record_path.exists():
            raise FileNotFoundError(f"Ground motion file not found: {record_path}")
        
        # Load ground motion
        acceleration = np.loadtxt(record_path)
        n_points = len(acceleration)
        duration = n_points * dt + 20
        
        self.logger.info(f"Ground motion: {record_path.name}, "
                        f"Points: {n_points}, Duration: {duration:.2f}s, dt: {dt:.4f}s")
        
        # Define time series and load pattern
        ops.timeSeries('Path', 2, '-dt', dt, '-filePath', str(record_path), 
                    '-factor', scale_factor * g)
        ops.pattern('UniformExcitation', 2, 1, '-accel', 2)
        
        newmark_beta = self._analysis_config['time_integration'].get('beta', 0.25)
        newmark_gamma = self._analysis_config['time_integration'].get('gamma', 0.5)

        # Setup COMPLETE analysis system (all 6 components)
        ops.wipeAnalysis()
        ops.system('BandGeneral')                    # 1. System solver
        ops.constraints('Transformation')            # 2. Constraint handler
        ops.numberer('RCM')                          # 3. Numberer
        ops.test('EnergyIncr', tolerance, max_iter, 0)  # 4. Test criteria
        ops.algorithm('NewtonLineSearch')            # 5. Algorithm
        ops.integrator('Newmark', newmark_gamma, newmark_beta)  # 6. Integrator
        ops.analysis('Transient')                    # Now this has everything!
        
        # Run analysis with adaptive time stepping
        final_time = self._run_adaptive_time_history(duration, dt)
        
        # Check if analysis reached full duration
        if final_time < duration - dt:
            raise ValueError(
                f"Analysis failed at t={final_time:.2f}s (target: {duration:.2f}s)"
            )
        
        return final_time

    def _run_adaptive_time_history(self, duration: float, dt: float) -> float:
        """Run time-history analysis with enhanced adaptive strategies.
        
        CRITICAL: Never call wipeAnalysis() during the loop - it destroys time integration state.
        """
        
        time_step_ratio = self._analysis_config.get('time_integration', {}).get('time_step_ratio', 0.5)

        t_current = 0.0
        dt_current = dt * time_step_ratio  
        dt_min = dt * 0.0001
        dt_max = dt
        
        failed_steps = 0
        total_steps = 0
        progress_reported = set()

        # Enhanced strategy sequence
        strategies = [
            # 1. Fast Newton (baseline)
            {
                'test': ('EnergyIncr', 1e-6, 1000, 0),
                'algorithm': ('NewtonLineSearch',),
                'dt_factor': 1.0,
                'name': 'FastNewton'
            },
            
            # 2. KrylovNewton - Standard
            {
                'test': ('NormDispIncr', 1e-6, 2000, 0),
                'algorithm': ('KrylovNewton', '-iterate', 'initial', '-maxDim', 6),
                'dt_factor': 0.8,
                'name': 'KrylovStd'
            },
            
            # 3. Modified Newton with Initial tangent
            {
                'test': ('NormDispIncr', 1e-6, 2000, 0),
                'algorithm': ('ModifiedNewton', '-initial'),
                'dt_factor': 0.6,
                'name': 'ModNewton-Initial'
            },
            
            # 4. KrylovNewton - Higher dimension
            {
                'test': ('NormDispIncr', 1e-7, 3000, 0),
                'algorithm': ('KrylovNewton', '-iterate', 'initial', '-maxDim', 12),
                'dt_factor': 0.5,
                'name': 'KrylovHigh'
            },
            
            # 5. BFGS with line search
            {
                'test': ('EnergyIncr', 1e-6, 2500, 0),
                'algorithm': ('BFGS', '-count', 10),
                'dt_factor': 0.4,
                'name': 'BFGS'
            },
            
            # 6. Broyden fallback
            {
                'test': ('NormDispIncr', 1e-6, 3000, 0),
                'algorithm': ('Broyden', 8),
                'dt_factor': 0.3,
                'name': 'Broyden'
            },
            
            # 7. Change system solver + KrylovNewton
            {
                'test': ('NormDispIncr', 1e-6, 4000, 0),
                'algorithm': ('KrylovNewton', '-maxDim', 15),
                'system': ('UmfPack',),
                'dt_factor': 0.2,
                'name': 'KrylovUmf'
            },
            
            # 8. Relaxed tolerance + micro-steps
            {
                'test': ('EnergyIncr', 5e-6, 5000, 0),
                'algorithm': ('KrylovNewton', '-iterate', 'initial', '-maxDim', 10),
                'dt_factor': 0.1,
                'name': 'KrylovRelaxed'
            },
            
            # 9. Last resort: Penalty constraints + relaxed
            {
                'test': ('NormDispIncr', 1e-5, 8000, 0),
                'algorithm': ('ModifiedNewton', '-initial'),
                'constraints': ('Penalty', 1.0e15, 1.0e15),
                'dt_factor': 0.05,
                'name': 'PenaltyMicro'
            },
        ]
        
        # Track if we've changed system/constraints for proper reset
        system_changed = False
        constraints_changed = False
        
        while t_current < duration:
            # Progress reporting
            percent = int((100 * t_current) / duration)
            if percent % 10 == 0 and percent not in progress_reported:
                self.logger.info(
                    f"{percent}% complete (t={t_current:.2f}s, steps: {total_steps}, failures: {failed_steps})"
                )
                progress_reported.add(percent)
            
            # Try current step with baseline settings
            result = ops.analyze(1, dt_current)
            
            if result == 0:
                t_current = ops.getTime()
                total_steps += 1
                
                # Gradually increase dt if stable
                if dt_current < dt_max:
                    dt_current = min(dt_current * 1.2, dt_max)
                    
            else:
                failed_steps += 1
                converged = False
                self.logger.warning(f"Step failed at t={t_current:.4f}s, dt={dt_current:.6f}s")

                for strat in strategies:
                    if converged:
                        break

                    # Apply system changes if specified (only during recovery)
                    if 'system' in strat:
                        ops.system(*strat['system'])
                        system_changed = True
                    
                    # Apply constraint handler if specified (only during recovery)
                    if 'constraints' in strat:
                        ops.constraints(*strat['constraints'])
                        constraints_changed = True
                    
                    # Set algorithm with all parameters
                    ops.algorithm(*strat['algorithm'])
                    
                    # Set test criteria
                    ops.test(*strat['test'])

                    # Calculate substeps
                    dt_try = max(dt_current * strat['dt_factor'], dt_min)
                    n_substeps = max(1, int(np.ceil(dt_current / dt_try)))
                    dt_try = dt_current / n_substeps  # Exact division

                    self.logger.debug(
                        f"Trying {strat['name']}: {n_substeps}×{dt_try:.4e}s"
                    )

                    result = ops.analyze(n_substeps, dt_try)
                    
                    if result == 0:
                        t_current = ops.getTime()
                        total_steps += n_substeps
                        dt_current = min(dt_try * 1.1, dt_max)  # Conservative recovery
                        converged = True
                        self.logger.info(f"✓ Recovered with {strat['name']} at t={t_current:.4f}s")
                        break

                if not converged:
                    self.logger.error(
                        f"FAILED at t={t_current:.4f}s after all strategies. "
                        f"Last convergence: EnergyIncr or DispIncr exceeded limits."
                    )
                    return t_current

                # Reset system/constraints if they were changed
                if system_changed:
                    ops.system('BandGeneral')
                    system_changed = False
                
                if constraints_changed:
                    ops.constraints('Transformation')
                    constraints_changed = False
                
                # Reset to fast baseline algorithm and test
                ops.algorithm('NewtonLineSearch')
                ops.test('EnergyIncr', 1e-6, 1000, 0)


        self.logger.info(
            f"✓ Analysis COMPLETE: {total_steps} steps, {failed_steps} failures recovered"
        )
        return ops.getTime()

    # =========================================================================
    # Recorders
    # =========================================================================

    def create_recorders(self, result_path: Union[str, Path], 
                        components: Optional[List[str]] = None,
                        node_responses: Optional[List[str]] = None,
                        element_responses: Optional[List[str]] = None) -> None:
        """Create output recorders for analysis results with metadata.
        
        Args:
            result_path: Directory path for output files
            components: List of components to record. If None, records all available.
                    Options: 'Wall', 'UFP', 'PT_Bar', 'Rocking_Spring', 
                            'Leaning_Column', 'Diaphragm', 'Base_Reactions'
            node_responses: List of node responses to record. 
                        Options: 'disp', 'vel', 'accel', 'reaction'
                        Default: ['disp'] if None
            element_responses: List of element responses to record.
                            Options: 'force', 'deformation', 'stress'
                            Default: [] if None (no element recording by default)
        """
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)
        
        # Set defaults
        if components is None:
            components = ['Wall','Base_Reactions']  # Default: only base reactions
        
        if node_responses is None:
            node_responses = ['disp']  # Default: only displacements
        
        if element_responses is None:
            element_responses = []  # Default: no element recording
        
        # Validate inputs
        valid_components = {
            'Wall', 'UFP', 'PT_Bar', 'Rocking_Spring', 
            'Leaning_Column', 'Diaphragm', 'Base_Reactions'
        }
        valid_node_responses = {'disp', 'vel', 'accel', 'reaction'}
        valid_element_responses = {'force', 'deformation', 'stress'}
        
        invalid = set(components) - valid_components
        if invalid:
            raise ValueError(f"Invalid components: {invalid}. Valid: {valid_components}")
        
        invalid = set(node_responses) - valid_node_responses
        if invalid:
            raise ValueError(f"Invalid node responses: {invalid}. Valid: {valid_node_responses}")
        
        invalid = set(element_responses) - valid_element_responses
        if invalid:
            raise ValueError(f"Invalid element responses: {invalid}. Valid: {valid_element_responses}")
        
        # Collect tags based on requested components
        node_collections = self._collect_node_tags(set(components))
        element_collections = self._collect_element_tags(set(components))
        
        # Create recorders
        recorder_info = {
            'result_directory': str(result_path.absolute()),
            'components_recorded': components,
            'node_responses': node_responses,
            'element_responses': element_responses,
            'node_recorders': {},
            'element_recorders': {},
            'created_at': str(np.datetime64('now'))
        }
        
        # Handle base reactions
        if 'Base_Reactions' in components:
            base_nodes_info = self._create_base_reaction_recorders(result_path)
            recorder_info['node_recorders']['Base_Reactions'] = {
                'directory': 'Base_Reactions',
                'node_count': len(base_nodes_info['node_tags']),
                'node_tags': base_nodes_info['node_tags'],
                'node_info': base_nodes_info['node_info'],
                'responses': ['reaction'],
                'files': ['reactions.txt']
            }
        
        # Create node recorders
        if node_responses:
            for name, nodes in node_collections.items():
                if nodes:
                    self._create_node_recorders_with_metadata(
                        result_path, name, nodes, node_responses
                    )
                    recorder_info['node_recorders'][name] = {
                        'directory': name,
                        'node_count': len(nodes),
                        'node_tags': nodes,
                        'responses': node_responses,
                        'files': [f'{resp}.txt' for resp in node_responses]
                    }
        
        # Create element recorders
        if element_responses:
            for name, elements in element_collections.items():
                if elements:
                    self._create_element_recorders_with_metadata(
                        result_path, name, elements, element_responses
                    )
                    recorder_info['element_recorders'][name] = {
                        'directory': name,
                        'element_count': len(elements),
                        'element_tags': elements,
                        'responses': element_responses,
                        'files': [f'{resp}.txt' for resp in element_responses]
                    }
        
        # Save master index
        with open(result_path / 'recorder_index.json', 'w') as f:
            json.dump(recorder_info, f, indent=2)
        
        self.logger.info(
            f"Created recorders in {result_path}: "
            f"{len(recorder_info['node_recorders'])} node groups, "
            f"{len(recorder_info['element_recorders'])} element groups"
        )

    def _create_base_reaction_recorders(self, result_path: Path) -> Dict:
        """Create recorders for base reactions at fixed nodes.
        
        Args:
            result_path: Base directory for results
            
        Returns:
            Dictionary with node_tags and node_info
        """
        component_path = result_path / 'Base_Reactions'
        component_path.mkdir(exist_ok=True)
        
        # Get base nodes (fixed nodes)
        base_nodes = []
        base_node_info = {}
        
        # Wall base nodes
        if "Fixed Base" in self._wall_config.get("Nodes", {}):
            for side, node_tag in self._wall_config["Nodes"]["Fixed Base"].items():
                base_nodes.append(node_tag)
                
                # Get node coordinates
                try:
                    coords = ops.nodeCoord(node_tag)
                    coord_dict = {'x': coords[0], 'y': coords[1]}
                except:
                    coord_dict = None
                
                base_node_info[str(node_tag)] = {  # Convert to string for JSON
                    'component': 'Wall',
                    'location': side,
                    'description': f'Wall {side} base',
                    'coordinates': coord_dict
                }
        
        if not base_nodes:
            self.logger.warning("No base nodes found for reaction recording")
            return {'node_tags': [], 'node_info': {}}
        
        # Create reaction recorder
        file_path = component_path / 'reactions.txt'
        ops.recorder(
            'Node', '-file', str(file_path),
            '-time', '-node', *base_nodes, '-dof', 1, 2, 3, 'reaction'
        )
        
        # Create comprehensive metadata
        columns = ['time'] + [
            f'node_{node}_{dof}' 
            for node in base_nodes 
            for dof in ['Fx', 'Fy', 'Mz']
        ]
        
        metadata = {
            'file': 'reactions.txt',
            'type': 'node_reaction',
            'response_type': 'reaction',
            'node_tags': base_nodes,
            'node_info': base_node_info,
            'dofs': [1, 2, 3],
            'dof_names': ['Fx', 'Fy', 'Mz'],
            'dof_descriptions': [
                'Horizontal force (X)',
                'Vertical force (Y)',
                'Moment about Z'
            ],
            'columns': columns,
            'column_mapping': {col: idx for idx, col in enumerate(columns)},
            'units': {
                'time': 'seconds',
                'Fx': 'force_units',
                'Fy': 'force_units',
                'Mz': 'force_units * length_units'
            },
            'shape': {
                'rows': 'number_of_time_steps',
                'cols': len(columns)
            },
            'notes': 'Base reactions at fixed supports'
        }
        
        # Save metadata as JSON
        with open(component_path / 'reactions_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save column names as CSV (for easy pandas loading)
        pd.DataFrame({'column': columns}).to_csv(
            component_path / 'reactions_columns.csv',
            index=False, header=False
        )
        
        self.logger.debug(f"Created base reaction recorders for {len(base_nodes)} nodes")
        
        return {
            'node_tags': base_nodes,
            'node_info': base_node_info
        }

    def _collect_node_tags(self, components: Set[str]) -> Dict[str, List[int]]:
        """Collect node tags for requested components only.
        
        Args:
            components: Set of component names to collect
            
        Returns:
            Dictionary mapping component names to node tag lists
        """
        collections = {}
        
        # Wall nodes
        if 'Wall' in components:
            # Get wall nodes including elevations
            wall_nodes = []
                       
            # All elevation nodes (but NOT PT Bar nodes)
            for key, value in self._wall_config.get("Nodes", {}).items():
                if key.startswith("Elevation"):
                    wall_nodes.extend(self._flatten_dict(value))
            
            # Segmental elevations (UFP nodes on wall)
            if "Segmental Elevations" in self._wall_config:
                for seg_data in self._wall_config["Segmental Elevations"].values():
                    if "Left Node" in seg_data:
                        wall_nodes.append(seg_data["Left Node"])
                    if "Right Node" in seg_data:
                        wall_nodes.append(seg_data["Right Node"])
            
            if wall_nodes:
                # Remove duplicates and sort
                wall_nodes = sorted(list(set(wall_nodes)))
                collections["Wall_Nodes"] = wall_nodes
                
                self.logger.debug(f"Collected {len(wall_nodes)} wall nodes: {wall_nodes}")

        # UFP nodes
        if 'UFP' in components:
            ufp_nodes = self._flatten_dict(self.state.UFP.get("Nodes", {}))
            if ufp_nodes:
                collections["UFP_Nodes"] = ufp_nodes
        
        # Rocking spring nodes
        if 'Rocking_Spring' in components:
            spring_nodes = self._flatten_dict(
                self.state.spring.get("Rocking", {}).get("Nodes", {})
            )
            if spring_nodes:
                collections["Rocking_Spring_Nodes"] = spring_nodes
        
        # PT bar nodes
        if 'PT_Bar' in components:
            pt_nodes = self._flatten_dict(
                self._wall_config.get("Nodes", {}).get("PT Bar", {})
            )
            if pt_nodes:
                collections["PT_Bar_Nodes"] = pt_nodes
        
        # Leaning column nodes
        if 'Leaning_Column' in components:
            leaning_nodes = self._flatten_dict(self._leaning_config.get("Nodes", {}))
            if leaning_nodes:
                collections["Leaning_Column_Nodes"] = leaning_nodes
        
        # Diaphragm nodes
        if 'Diaphragm' in components:
            diaphragm_nodes = self._flatten_dict(self._diaphragm_config.get("Nodes", {}))
            if diaphragm_nodes:
                collections["Diaphragm_Nodes"] = diaphragm_nodes
        
        return collections

    def _collect_element_tags(self, components: Set[str]) -> Dict[str, List[int]]:
        """Collect element tags for requested components only.
        
        Args:
            components: Set of component names to collect
            
        Returns:
            Dictionary mapping component names to element tag lists
        """
        collections = {}
        
        # Wall elements
        if 'Wall' in components:
            wall_elements = self._flatten_dict(self._wall_config.get("Elements", {}))
            if wall_elements:
                collections["Wall_Elements"] = wall_elements
        
        # UFP elements (zero-length only)
        if 'UFP' in components:
            ufp_elements = self._flatten_dict(
                self.state.UFP.get("Elements", {}).get("ZeroLength Elements", {})
            )
            if ufp_elements:
                collections["UFP_Elements"] = ufp_elements
        
        # PT bar elements
        if 'PT_Bar' in components:
            pt_elements = self._flatten_dict(
                self.state.bar.get("Elements", {}).get("PT Bars", {})
            )
            if pt_elements:
                collections["PT_Bar_Elements"] = pt_elements
        
        # Rocking spring elements (zero-length only)
        if 'Rocking_Spring' in components:
            spring_elements = self._flatten_dict(
                self.state.spring.get("Elements", {}).get("ZeroLength Elements", {})
            )
            if spring_elements:
                collections["Rocking_Spring_Elements"] = spring_elements
        
        # Leaning column elements
        if 'Leaning_Column' in components:
            leaning_elements = self._flatten_dict(self._leaning_config.get("Elements", {}))
            if leaning_elements:
                collections["Leaning_Column_Elements"] = leaning_elements
        
        # Diaphragm elements
        if 'Diaphragm' in components:
            # Shear keys
            shear_keys = self._flatten_dict(
                self._diaphragm_config.get("Elements", {}).get("Shear Keys", {})
            )
            if shear_keys:
                collections["Diaphragm_Shear_Keys"] = shear_keys
            
            # Rotational hinges
            rot_hinges = self._flatten_dict(
                self._diaphragm_config.get("Elements", {}).get("Rotational Hinges", {})
            )
            if rot_hinges:
                collections["Diaphragm_Rotational_Hinges"] = rot_hinges
        
        return collections

    def _create_node_recorders_with_metadata(self, result_path: Path, 
                                            component_name: str,
                                            nodes: List[int],
                                            responses: List[str]) -> None:
        """Create node recorders with comprehensive metadata.
        
        Args:
            result_path: Base directory for results
            component_name: Name of component (e.g., 'Wall_Nodes')
            nodes: List of node tags
            responses: List of response types to record
        """
        component_path = result_path / component_name
        component_path.mkdir(exist_ok=True)
        
        # Response type mapping
        response_info = {
            'disp': {
                'opensees_arg': 'disp',
                'dof_names': ['x', 'y', 'rotation'],
                'dof_descriptions': [
                    'Horizontal displacement',
                    'Vertical displacement', 
                    'Rotation about Z-axis'
                ],
                'units': 'length_units (or radians for rotation)'
            },
            'vel': {
                'opensees_arg': 'vel',
                'dof_names': ['vx', 'vy', 'omega'],
                'dof_descriptions': [
                    'Horizontal velocity',
                    'Vertical velocity',
                    'Angular velocity'
                ],
                'units': 'length_units/time (or rad/s for rotation)'
            },
            'accel': {
                'opensees_arg': 'accel',
                'dof_names': ['ax', 'ay', 'alpha'],
                'dof_descriptions': [
                    'Horizontal acceleration',
                    'Vertical acceleration',
                    'Angular acceleration'
                ],
                'units': 'length_units/time^2 (or rad/s^2 for rotation)'
            },
            'reaction': {
                'opensees_arg': 'reaction',
                'dof_names': ['Fx', 'Fy', 'Mz'],
                'dof_descriptions': [
                    'Reaction force in X',
                    'Reaction force in Y',
                    'Reaction moment about Z'
                ],
                'units': 'force_units (or force*length for moment)'
            }
        }
        
        for response in responses:
            if response not in response_info:
                self.logger.warning(f"Unknown response type: {response}, skipping")
                continue
            
            info = response_info[response]
            file_path = component_path / f'{response}.txt'
            
            # Create OpenSees recorder
            ops.recorder(
                'Node', '-file', str(file_path),
                '-time', '-node', *nodes, '-dof', 1, 2, 3, info['opensees_arg']
            )
            
            # Build column mapping
            if component_name == 'Diaphragm_Nodes':
                # Map node tags to location info
                node_map = {}
                for elev_key, nodes_dict in self._diaphragm_config.get("Nodes", {}).items():
                    if elev_key.startswith("Elevation"):
                        story = elev_key.split()[-1]
                        for location, node_tag in nodes_dict.items():
                            if isinstance(node_tag, int):
                                loc_str = location.lower().replace(" ", "_")
                                node_map[node_tag] = f"{loc_str}_story{story}"
                
                columns = ['time'] + [
                    f'node_{node}_{dof_name}_({node_map.get(node, "unknown")})' 
                    for node in nodes 
                    for dof_name in info['dof_names']
                ]
            else:
                columns = ['time'] + [
                    f'node_{node}_{dof_name}' 
                    for node in nodes 
                    for dof_name in info['dof_names']
                ]
            
            # Get node coordinates for metadata
            node_coords = {}
            for node in nodes:
                try:
                    coords = ops.nodeCoord(node)
                    node_coords[node] = {
                        'x': coords[0],
                        'y': coords[1],
                        'z': coords[2] if len(coords) > 2 else 0.0
                    }
                except:
                    node_coords[node] = None
            
            # Create comprehensive metadata
            metadata = {
                'file': f'{response}.txt',
                'component': component_name,
                'type': f'node_{response}',
                'response_type': response,
                'opensees_command': info['opensees_arg'],
                'node_tags': nodes,
                'node_count': len(nodes),
                'node_coordinates': node_coords,
                'dofs': [1, 2, 3],
                'dof_names': info['dof_names'],
                'dof_descriptions': info['dof_descriptions'],
                'units': info['units'],
                'columns': columns,
                'column_mapping': {col: idx for idx, col in enumerate(columns)},
                'shape': {
                    'rows': 'number_of_time_steps (unknown until analysis completes)',
                    'cols': len(columns)
                },
                'data_format': {
                    'delimiter': 'whitespace',
                    'time_column': 0,
                    'data_columns': list(range(1, len(columns)))
                },
                'notes': f'{response.capitalize()} response for {component_name}'
            }
            
            # Save metadata as JSON
            with open(component_path / f'{response}_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save column names as CSV (for easy pandas loading)
            pd.DataFrame({'column': columns}).to_csv(
                component_path / f'{response}_columns.csv',
                index=False, header=False
            )
            
            self.logger.debug(
                f"Created {response} recorder for {component_name}: "
                f"{len(nodes)} nodes, {len(columns)} columns"
            )

    def _create_element_recorders_with_metadata(self, result_path: Path,
                                               component_name: str,
                                               elements: List[int],
                                               responses: List[str]) -> None:
        """Create element recorders with comprehensive metadata.
        
        Args:
            result_path: Base directory for results
            component_name: Name of component (e.g., 'Wall_Elements')
            elements: List of element tags
            responses: List of response types to record
        """
        component_path = result_path / component_name
        component_path.mkdir(exist_ok=True)
        
        # Determine if these are zero-length elements
        is_zero_length = 'UFP' in component_name or 'Spring' in component_name
        
        # Response type mapping
        response_info = {
            'force': {
                'opensees_arg': 'localForce' if is_zero_length else 'force',
                'description': 'Element forces',
                'units': 'force_units (or force*length for moments)'
            },
            'deformation': {
                'opensees_arg': 'deformation',
                'description': 'Element deformations',
                'units': 'length_units (or radians for rotations)'
            },
            'stress': {
                'opensees_arg': 'stress',
                'description': 'Element stresses',
                'units': 'force_units/area_units'
            }
        }
        
        for response in responses:
            if response not in response_info:
                self.logger.warning(f"Unknown response type: {response}, skipping")
                continue
            
            info = response_info[response]
            file_path = component_path / f'{response}.txt'
            
            # Create OpenSees recorder
            ops.recorder(
                'Element', '-file', str(file_path),
                '-time', '-ele', *elements, info['opensees_arg']
            )
            
            # Note: Element response column structure depends on element type
            # We provide general guidance in metadata
            metadata = {
                'file': f'{response}.txt',
                'component': component_name,
                'type': f'element_{response}',
                'response_type': response,
                'opensees_command': info['opensees_arg'],
                'element_tags': elements,
                'element_count': len(elements),
                'is_zero_length': is_zero_length,
                'description': info['description'],
                'units': info['units'],
                'data_format': {
                    'delimiter': 'whitespace',
                    'time_column': 0,
                    'note': 'Element response columns depend on element type. '
                           'Typically: time, then response for each element.'
                },
                'shape': {
                    'rows': 'number_of_time_steps (unknown until analysis completes)',
                    'cols': 'varies by element type'
                },
                'column_interpretation': {
                    'zero_length': 'For zero-length: forces/deformations in local DOFs',
                    'frame_element': 'For frame elements: forces at nodes i and j',
                    'truss_element': 'For truss: axial force only'
                },
                'notes': f'{response.capitalize()} response for {component_name}. '
                        f'Exact column structure depends on element formulation.'
            }
            
            # Save metadata as JSON
            with open(component_path / f'{response}_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.debug(
                f"Created {response} recorder for {component_name}: "
                f"{len(elements)} elements"
            )

    def _flatten_dict(self, data: Any) -> List[int]:
        """Recursively flatten nested dictionary/list to extract integer values.
        
        Args:
            data: Nested dictionary, list, or value
            
        Returns:
            Flat list of integer values
        """
        result = []
        
        if isinstance(data, dict):
            for value in data.values():
                result.extend(self._flatten_dict(value))
        elif isinstance(data, (list, tuple)):
            for item in data:
                result.extend(self._flatten_dict(item))
        elif isinstance(data, int):
            result.append(data)
        
        return result
    
    # =========================================================================
    # Sustainability Computations
    # =========================================================================

    def calculate_GWP(self) -> Dict[str, float]:
        """Calculate Global Warming Potential (GWP) for structural system.
        
        Returns:
            Dictionary with total GWP and component breakdown in kg CO2
            
        """

        # GWP coefficients
        GWP_COEF = {
            'clt': 137.19,   # kg CO2/m³
            'ufp': 1730,     # kg CO2/metric ton   (Plate steel - fabricated)
            'pt_bar': 854    # kg CO2/metric ton   (Rebar - fabricated)
        }

        # =========================================================================
        # 1. CLT Wall GWP
        # =========================================================================
        wall_length = self._wall_config['Wall Length']
        wall_thickness = self._wall_config['Wall Thickness']
        building_height = self.state.wall_elevations[-1]
        
        # Volume: 2 walls × length × thickness × height
        walls_volume_si = (2 * wall_length * wall_thickness * building_height) / (meter ** 3)
        wall_GWP = GWP_COEF['clt'] * walls_volume_si
        
        # =========================================================================
        # 2. PT Bar GWP
        # =========================================================================
        bar_area = self.state.bar['Area']
        bar_height = building_height
        bar_volume_si = 4 * bar_area * bar_height / (meter ** 3)
        bar_steel_density = self.state.bar['Material']['Mass Density']['SI']  # 7850 kg/m³
        
        # Mass: 4 bars (2 modelled elements per wall) × area × height × density
        bar_mass_kg = bar_volume_si * bar_steel_density
        bar_mass_ton = bar_mass_kg / 1000
        bar_GWP = bar_mass_ton * GWP_COEF['pt_bar']
        
        # =========================================================================
        # 3. UFP GWP
        # =========================================================================
        bu = self.state.UFP['UFP Width']
        tu = self.state.UFP['UFP Thickness']
        du = self.state.UFP['UFP Diameter']
        n_ufp = sum(self.state.UFP['UFP Numbers'])
        ufp_steel_density = self.state.UFP['Material']['Mass Density']['SI']  # 7850 kg/m³

        # Volume: U-shaped plate with circular cutout       
        ufp_volume = n_ufp * (np.pi * (((du + tu) / 2) ** 2 - ((du - tu) / 2) ** 2) / 2 + 2 * tu * du) * bu
        ufp_volume_si = ufp_volume / (meter ** 3)
        
        # Mass
        ufp_mass_kg = ufp_volume_si * ufp_steel_density
        ufp_mass_ton = ufp_mass_kg / 1000
        UFP_GWP = ufp_mass_ton * GWP_COEF['ufp']
        
        # =========================================================================
        # Total GWP
        # =========================================================================
        total_GWP = wall_GWP + bar_GWP + UFP_GWP
        
        results = {
            'total': total_GWP,
            'wall': wall_GWP,
            'pt_bar': bar_GWP,
            'ufp': UFP_GWP,
            'breakdown_pct': {
                'wall': (wall_GWP / total_GWP) * 100,
                'pt_bar': (bar_GWP / total_GWP) * 100,
                'ufp': (UFP_GWP / total_GWP) * 100
            },
            'metrics': {
                'walls_volume_m3': walls_volume_si,
                'bar_volume_m3': bar_volume_si,
                'ufp_volume_m3': ufp_volume_si,
                'bar_mass_ton': bar_mass_ton,
                'ufp_mass_ton': ufp_mass_ton,
                'total_steel_mass_ton': bar_mass_ton + ufp_mass_ton
            }
        }
        
        self.logger.info(f"Total GWP: {total_GWP:.1f} kg CO₂")
        self.logger.info(f"  Wall: {wall_GWP:.1f} kg CO₂ ({results['breakdown_pct']['wall']:.1f}%)")
        self.logger.info(f"  PT Bars: {bar_GWP:.1f} kg CO₂ ({results['breakdown_pct']['pt_bar']:.1f}%)")
        self.logger.info(f"  UFPs: {UFP_GWP:.1f} kg CO₂ ({results['breakdown_pct']['ufp']:.1f}%)")
        
        return results