# cross_sectional_analysis.py
"""
Cross-sectional analysis for PT-CLT rocking walls.
Uses scipy optimization for equilibrium solving.
CORRECTED VERSION - validated against pushover analysis
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
from scipy.optimize import root_scalar, minimize_scalar

from core.model_state import ModelState
from core.constants import g

# Constants
SMALL_NUMBER = 1e-12
DEFAULT_PRECISION = 1e-7


@dataclass
class AnalysisProperties:
    """Container for analysis properties."""
    building_height: float
    wall_elastic_modulus: float
    wall_shear_modulus: float
    wall_thickness: float
    wall_length: float
    wall_weight: float
    initial_pt_force: float
    pt_extender_length: float
    wall_yield_stress: float
    wall_yield_strain: float
    bar_ultimate_strain: float
    bar_yield_force: float
    initial_pt_displacement: float
    bar_relationships: Tuple[List, List]
    ufp_relationships: Tuple[List, List]


class CrossSectionalAnalysis:
    """Cross-sectional analysis for CLT rocking walls."""
    
    def __init__(self, state: ModelState):
        self.state = state
    
    def _extract_properties(self) -> AnalysisProperties:
        """Extract all necessary properties from state."""
        wall_info = self.state.wall
        bar_info = self.state.bar
        ufp_info = self.state.UFP
        
        wall_length = wall_info['Wall Length']
        building_height = self.state.wall_elevations[-1]
        
        return AnalysisProperties(
            building_height=building_height,
            wall_elastic_modulus=wall_info['Wall Elastic Modulus'],
            wall_shear_modulus=wall_info.get('Wall Shear Modulus', wall_info['Wall Elastic Modulus'] / 16),
            wall_thickness=wall_info['Wall Thickness'],
            wall_length=wall_length,
            wall_weight=self.state.wall_weight,
            initial_pt_force=bar_info['Initial PT Force'],
            pt_extender_length=wall_info.get('PT Extender Length', 0.0),
            wall_yield_stress=wall_info['Wall Yield Stress'],
            wall_yield_strain=wall_info['Wall Yield Stress'] / wall_info['Wall Elastic Modulus'],
            bar_ultimate_strain=bar_info['Ultimate Strain'],
            bar_yield_force=bar_info['Material']['Yield Force'],
            initial_pt_displacement=bar_info['Initial Disp.'],
            bar_relationships=list(zip(*bar_info['Delta-Force Relationship'])),
            ufp_relationships=list(zip(*ufp_info['Delta-Force Relationship'])),
        )
    
    def _calculate_wall_properties(self, props: AnalysisProperties) -> Dict:
        """Calculate derived wall properties."""
        area = props.wall_length * props.wall_thickness
        moment_of_inertia = props.wall_length ** 3 * props.wall_thickness / 12
    
        elastic_stiffness = 1 / (
            props.building_height ** 3 / (3 * props.wall_elastic_modulus * moment_of_inertia) +
            props.building_height / (props.wall_shear_modulus * area)
        )
        
        return {
            'area': area,
            'moment_of_inertia': moment_of_inertia,
            'elastic_stiffness': elastic_stiffness,
            'plastic_hinge_length': 2 * props.wall_thickness,
            'moment_arm': 3 * props.wall_length / 8  # At decompression
        }
    
    def _calculate_pt_forces(self, c_ratio: float, props: AnalysisProperties, 
                            theta_gap: float) -> Tuple[float, float, float]:
        """Calculate PT bar forces based on gap rotation."""
        neutral_axis_depth = c_ratio * props.wall_length
        
        # PT bar distances from neutral axis (decompression point)
        pt1_distance = props.wall_length / 2 + props.pt_extender_length - neutral_axis_depth
        pt2_distance = props.wall_length / 2 - props.pt_extender_length - neutral_axis_depth

        pt1_displacement = props.initial_pt_displacement + max(0, theta_gap * pt1_distance)
        pt2_displacement = props.initial_pt_displacement + max(0, theta_gap * pt2_distance)
    
        
        # Interpolate forces from relationships
        bar_deltas, bar_forces = props.bar_relationships
        pt1_force = np.interp(pt1_displacement, bar_deltas, bar_forces)
        pt2_force = np.interp(pt2_displacement, bar_deltas, bar_forces)
        
        return pt1_force, pt2_force, max(pt1_displacement, pt2_displacement)
    
    def _calculate_compression_force(self, c_ratio: float, props: AnalysisProperties, 
                                    wall_props: Dict, theta_gap: float) -> Tuple[float, float, float, float]:
        """Calculate compression force and centroid."""
        neutral_axis_depth = c_ratio * props.wall_length
        
        # For plastic hinge analysis
        curvature_plastic = theta_gap / wall_props['plastic_hinge_length']
        max_strain = curvature_plastic * neutral_axis_depth
        
        # Stress and plastic depth
        max_stress = min(max_strain * props.wall_elastic_modulus, props.wall_yield_stress)
        plastic_depth = (
            max(0, neutral_axis_depth * (max_strain - props.wall_yield_strain) / max_strain)
            if max_strain > props.wall_yield_strain else 0
        )
        elastic_depth = neutral_axis_depth - plastic_depth
        
        # Compression force
        compression_force_per_width = (
            plastic_depth * max_stress + elastic_depth * max_stress / 2
        )
        compression_force = props.wall_thickness * compression_force_per_width
        
        # Centroid calculation (distance from decompression point to centroid)
        if compression_force_per_width > 0:
            plastic_moment_contrib = plastic_depth * max_stress * plastic_depth / 2
            elastic_moment_contrib = (elastic_depth * max_stress / 2) * (plastic_depth + elastic_depth / 3)
            centroid = (plastic_moment_contrib + elastic_moment_contrib) / compression_force_per_width
        else:
            centroid = 0
        
        return compression_force, centroid, max_strain, plastic_depth
    
    def _calculate_ufp_force(self, props: AnalysisProperties, 
                            theta_gap: float) -> Tuple[float, float]:
        """Calculate UFP force using validated geometric relationship.
        
        UFP deformation = wall_length × sin(theta_gap)
        This captures the vertical uplift at wall edge (validated against pushover).
        """

        ufp_displacement = props.wall_length * np.sin(theta_gap)

        # Get force from material relationship
        ufp_deltas, ufp_forces = props.ufp_relationships
        ufp_force_single = np.interp(ufp_displacement, ufp_deltas, ufp_forces)
        total_ufp_force = sum(self.state.UFP['UFP Numbers']) * ufp_force_single
        
        return total_ufp_force, ufp_displacement
    
    def _equilibrium_residual(self, c_ratio: float, props: AnalysisProperties,
                             wall_props: Dict, theta_gap: float) -> float:
        """Calculate force equilibrium residual.
        
        Equilibrium: Springs + UFPs = PT + Weight
        Both springs and UFPs resist uplift.
        """
        # Calculate all forces
        pt1_force, pt2_force, _ = self._calculate_pt_forces(c_ratio, props, theta_gap)
        compression_force, centroid, _, _ = self._calculate_compression_force(
            c_ratio, props, wall_props, theta_gap
        )
        ufp_force, _ = self._calculate_ufp_force(props, theta_gap)

        # Force equilibrium: Springs + UFPs = PT + Weight
        upward = compression_force + ufp_force
        downward = props.wall_weight + pt1_force + pt2_force 
        
        return upward - downward

    def analysis(self, target_drift: float) -> Dict:
        """Perform cross-sectional analysis for given target drift using optimization."""
        props = self._extract_properties()
        wall_props = self._calculate_wall_properties(props)
        
        # Calculate decompression drift
        elastic_moment = (props.wall_weight + props.initial_pt_force) * wall_props['moment_arm']
        wall_elastic_displacement = elastic_moment / (wall_props['elastic_stiffness'] * props.building_height)
        decompression_drift = wall_elastic_displacement / props.building_height
        
        # Pre-decompression: full contact, elastic response
        if target_drift <= decompression_drift:
            base_shear = elastic_moment / props.building_height
                   
            return {
                'Wall Weight': props.wall_weight,
                'PT_Fy': props.bar_yield_force,
                'PT_Fu': max(props.bar_relationships[1]),
                
                # PT Forces - individual (equal at pre-decompression)
                'PT1 Force': props.initial_pt_force,
                'PT2 Force': props.initial_pt_force,
                'PT Total Force': 2 * props.initial_pt_force,
                
                # PT Displacements - individual (equal at pre-decompression)
                'PT1 Disp.': props.initial_pt_displacement,
                'PT2 Disp.': props.initial_pt_displacement,
                'PT Max Disp.': props.initial_pt_displacement,
                
                # PT Strains - individual (equal at pre-decompression)
                'PT1 Strain': props.initial_pt_displacement / props.building_height,
                'PT2 Strain': props.initial_pt_displacement / props.building_height,
                'Bar Max Strain': props.initial_pt_displacement / props.building_height,
                'Bar Ult. Strain': props.bar_ultimate_strain,
                
                # UFP
                'Total UFP Forces': 0.0,
                'UFP Force': 0.0,
                'UFP Disp.': 0.0,
                
                # Compression
                'Compression Force': props.wall_weight + 2 * props.initial_pt_force,
                'c_ratio': 1.0,
                'centroid': props.wall_length / 2,
                'eps_c_max': base_shear * props.building_height / (props.wall_elastic_modulus * wall_props['moment_of_inertia']) * props.wall_length / 2,
                'plastic_depth': 0.0,
                
                # Analysis info
                'converged': True,
                'target_drift': target_drift,
                'theta_gap': 0.0,
            }
        
        # Post-decompression: solve equilibrium with gap opening
        theta_gap = target_drift - decompression_drift
        
        # Use scipy to find c_ratio that satisfies equilibrium
        try:
            result = root_scalar(
                self._equilibrium_residual,
                args=(props, wall_props, theta_gap),
                bracket=[0.001, 0.9],
                method='brentq',
                xtol=DEFAULT_PRECISION
            )
            c_ratio = result.root
            converged = result.converged
        except ValueError:
            result = minimize_scalar(
                lambda c: abs(self._equilibrium_residual(c, props, wall_props, theta_gap)),
                bounds=(0.001, 0.9),
                method='bounded',
                options={'xatol': DEFAULT_PRECISION}
            )
            c_ratio = result.x
            converged = result.success
        
        # Calculate final forces with converged c_ratio
        pt1_force, pt2_force, max_pt_displacement = self._calculate_pt_forces(c_ratio, props, theta_gap)
        compression_force, centroid, max_strain, plastic_depth = self._calculate_compression_force(
            c_ratio, props, wall_props, theta_gap
        )
        total_ufp_forces, ufp_displacement = self._calculate_ufp_force(props, theta_gap)

        # Calculate individual PT displacements for output
        neutral_axis_depth = c_ratio * props.wall_length
        pt1_distance = props.wall_length / 2 + props.pt_extender_length - neutral_axis_depth
        pt2_distance = props.wall_length / 2 - props.pt_extender_length - neutral_axis_depth
        pt1_displacement = props.initial_pt_displacement + max(0, theta_gap * pt1_distance)
        pt2_displacement = props.initial_pt_displacement + max(0, theta_gap * pt2_distance)

        # Build solution dictionary
        solution = {
            'Wall Weight': props.wall_weight,
            'PT_Fy': props.bar_yield_force,
            'PT_Fu': max(props.bar_relationships[1]),
            
            # PT Forces - individual
            'PT1 Force': pt1_force,
            'PT2 Force': pt2_force,
            'PT Total Force': pt1_force + pt2_force,
            
            # PT Displacements - individual
            'PT1 Disp.': pt1_displacement,
            'PT2 Disp.': pt2_displacement,
            'PT Max Disp.': max_pt_displacement,
            
            # PT Strains - individual
            'PT1 Strain': pt1_displacement / props.building_height,
            'PT2 Strain': pt2_displacement / props.building_height,
            'Bar Max Strain': max_pt_displacement / props.building_height,
            'Bar Ult. Strain': props.bar_ultimate_strain,
            
            # UFP (all identical)
            'Total UFP Forces': total_ufp_forces,
            'UFP Force': total_ufp_forces / sum(self.state.UFP['UFP Numbers']) if sum(self.state.UFP['UFP Numbers']) > 0 else 0,
            'UFP Disp.': ufp_displacement,
            
            # Compression (max values only)
            'Compression Force': compression_force,
            'c_ratio': c_ratio,
            'centroid': centroid,
            'eps_c_max': max_strain,
            'plastic_depth': plastic_depth,
            
            # Analysis info
            'converged': converged,
            'target_drift': target_drift,
            'theta_gap': theta_gap,
        }

        return solution
