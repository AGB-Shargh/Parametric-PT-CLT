# design_objectives.py
"""
Performance-based design objectives for PT-CLT rocking walls.
Returns demand/capacity ratios (DCR < 1.0 = safe).
"""
import logging
import numpy as np
from typing import Dict, Optional

from core.model_state import ModelState
from Seismic_Design.seismic_functions import SeismicFunctions
from Seismic_Design.cross_sectional_analysis import CrossSectionalAnalysis
from core.constants import g


class DesignObjectives:
    """Performance-based design checks using demand/capacity ratios."""
    
    # Target drift limits
    DBE_DRIFT_LIMIT = 0.02   
    MCER_DRIFT_LIMIT = 0.03  
    
    def __init__(self, state: ModelState, seismic_functions: SeismicFunctions):
        """Initialize design objectives.
        
        Args:
            state: ModelState instance
            seismic_functions: SeismicFunctions instance for demand calculation
        """
        self.state = state
        self.seismic = seismic_functions
        self.logger = logging.getLogger(__name__)
        
        # Lazy-loaded CSA results
        self._csa_de = None
        self._csa_mce = None
    
    @property
    def csa_de(self) -> Dict:
        """Lazy-load DBE cross-sectional analysis results."""
        if self._csa_de is None:
            # Check if already stored in state
            if hasattr(self.state, 'csa_results') and 'csa_de' in self.state.csa_results:
                self._csa_de = self.state.csa_results['csa_de']
            else:
                # Perform analysis
                csa = CrossSectionalAnalysis(self.state)
                self._csa_de = csa.analysis(
                    self.DBE_DRIFT_LIMIT, 
                    result_key='csa_de'
                )
        return self._csa_de
    
    @property
    def csa_mce(self) -> Dict:
        """Lazy-load MCER cross-sectional analysis results."""
        if self._csa_mce is None:
            # Check if already stored in state
            if hasattr(self.state, 'csa_results') and 'csa_mce' in self.state.csa_results:
                self._csa_mce = self.state.csa_results['csa_mce']
            else:
                # Perform analysis
                csa = CrossSectionalAnalysis(self.state)
                self._csa_mce = csa.analysis(
                    self.MCER_DRIFT_LIMIT,
                    result_key='csa_mce'
                )
        return self._csa_mce
    
    @property
    def wall_geometry(self) -> Dict:
        """Extract common wall geometry parameters."""
        wall_length = self.state.wall['Wall Length']
        wall_weight = self.state.wall.get('Wall Mass', 0.0) * g
        wall_ultimate_strain = self.state.wall.get('Wall Crush Strain', 0.0375)
        
        return {
            'length': wall_length,
            'moment_arm': 3 * wall_length / 8,
            'weight': wall_weight,
            'num_panels': 2,  # Assuming dual panel configuration
            'ultimate_strain': wall_ultimate_strain
        }
    
    @property
    def ufp_properties(self) -> Dict:
        """Extract UFP properties."""
        return {
            'total_count': sum(self.state.UFP['UFP Numbers']),
            'yield_force': self.state.UFP['Yield Force'],
            'ultimate_force': self.state.UFP.get('Ultimate Force', 
                                                 1.5 * self.state.UFP['Yield Force'])
        }
    
    # =========================================================================
    # Design Objective Functions (Return Demand/Capacity Ratios)
    # =========================================================================
    
    def drift_ratio_de(self, analysis_runner, structure_seismic_features: Optional[Dict] = None) -> float:
        """Check DE drift limit state using ELF analysis.
        
        Args:
            analysis_runner: AnalysisRunner instance for running ELF analysis
            structure_seismic_features: Dict with 'Cd' (deflection amplification factor)
        
        Returns:
            Demand/Capacity Ratio (DCR < 1.0 = safe)
        """
        if structure_seismic_features is None:
            structure_seismic_features = {'Cd': self.seismic.Cd}
        
        Cd = structure_seismic_features['Cd']
        
        # Set seismic design application to 'drift'
        self.seismic.application = 'drift'

        # Get ELF forces from seismic functions
        elf_forces = self.seismic.distribute_elf_forces()
        
        # Run ELF static analysis to get elastic drifts
        story_drifts, max_elastic_drift = analysis_runner.run_elf_drift_analysis(elf_forces)
        
        # Amplify elastic drift by Cd to get inelastic drift
        max_inelastic_drift = max_elastic_drift * Cd
        
        # Check against capacity
        capacity = self.DBE_DRIFT_LIMIT
        demand = max_inelastic_drift
        
        dcr = demand / capacity


        self.logger.info(f"Drift DCR (DE): {dcr:.3f}")
        self.logger.info(f"  Elastic drift (max): {max_elastic_drift:.4f}")
        self.logger.info(f"  Amplified drift (Cd={Cd}): {max_inelastic_drift:.4f}")
        self.logger.info(f"  Capacity (DBE limit): {capacity:.4f}")
        
        # Log per-story drifts for reference
        for story, drift in story_drifts.items():
            amplified = drift * Cd
            self.logger.debug(f"  Story {story}: elastic={drift:.4f}, "
                            f"inelastic={amplified:.4f}")
        
        return dcr

    def drift_ratio_mce(self, analysis_runner, structure_seismic_features: Optional[Dict] = None) -> float:
        """Check MCE drift limit state using ELF analysis.
        
        Args:
            analysis_runner: AnalysisRunner instance for running ELF analysis
            structure_seismic_features: Dict with 'Cd' (deflection amplification factor)
        
        Returns:
            Demand/Capacity Ratio (DCR < 1.0 = safe)
        """
        if structure_seismic_features is None:
            structure_seismic_features = {'Cd': self.seismic.Cd}
        
        Cd = structure_seismic_features['Cd']
        
        # Set seismic design application to 'drift'
        self.seismic.application = 'drift'
        
        # Get ELF forces from seismic functions (uses analytical period for drift)
        elf_forces = self.seismic.distribute_elf_forces()

        # Increase forces from DE to MCE
        for node, force in elf_forces.items():
            elf_forces[node] = force * 1.5

        # Run ELF static analysis to get elastic drifts
        story_drifts, max_elastic_drift = analysis_runner.run_elf_drift_analysis(elf_forces)
        
        # Amplify elastic drift by Cd to get inelastic drift
        max_inelastic_drift = max_elastic_drift * Cd
        
        
        # Check against MCE capacity (instead of DBE)
        capacity = self.MCER_DRIFT_LIMIT 
        demand = max_inelastic_drift

        dcr = demand / capacity
        
        self.logger.info(f"Drift DCR (MCE): {dcr:.3f}")  # Changed log label
        self.logger.info(f"  Elastic drift (max): {max_elastic_drift:.4f}")
        self.logger.info(f"  Amplified drift (Cd={Cd}): {max_inelastic_drift:.4f}")
        self.logger.info(f"  Capacity (MCE limit): {capacity:.4f}")  # Changed log label
        
        # Log per-story drifts for reference
        for story, drift in story_drifts.items():
            amplified = drift * Cd
            self.logger.debug(f"  Story {story}: elastic={drift:.4f}, "
                            f"inelastic={amplified:.4f}")
        
        return dcr

    def moment_dcr(self) -> float:
        """Check moment capacity limit state.
        
        Returns:
            Demand/Capacity Ratio (DCR < 1.0 = safe)
        """
        geom = self.wall_geometry
        ufp = self.ufp_properties
        
        # Calculate nominal moment capacity
        pt_forces = 2 * self.state.bar['Initial PT Force']
        
        # Wall moment contribution
        wall_moment = geom['num_panels'] * (pt_forces + geom['weight']) * geom['moment_arm']
        
        # UFP moment contribution
        ufp_moment = ufp['total_count'] * ufp['yield_force'] * geom['length']
        
        # Nominal capacity with strength reduction factor
        phi = 0.9  # Strength reduction factor
        nominal_capacity = phi * (wall_moment + ufp_moment)
        
        # Moment demand from seismic forces
        moment_demand = self.seismic.calculate_moment_demand()
        
        dcr = moment_demand / nominal_capacity
        
        self.logger.debug(f"Moment DCR: {dcr:.3f} "
                         f"(demand={moment_demand:.2f}, capacity={nominal_capacity:.2f})")
        return dcr
    
    def shear_dcr(self) -> float:
        """Check shear capacity limit state.
        
        Returns:
            Demand/Capacity Ratio (DCR < 1.0 = safe)
        """
        geom = self.wall_geometry
        
        # Shear demand from seismic forces
        shear_demand = self.seismic.compute_base_shear()
        
        # Amplification factors
        omega_v = 1.5      # Overstrength factor
        alpha_v = 1.25     # Dynamic amplification factor
        amplified_shear = 1.5 * omega_v * alpha_v * shear_demand
        
        # Shear capacity (from CLT properties)
        shear_strength = self.state.wall['In-Plane Shear Capacity']
        
        # Nominal shear capacity parameters
        phi_v = 0.75          # Strength reduction factor
        kf = 2.88             # Configuration factor
        lambda_factor = 1.0   # Time effect factor
        
        nominal_shear = (geom['num_panels'] * kf * lambda_factor * 
                        geom['length'] * shear_strength)
        
        shear_capacity = phi_v * nominal_shear
        
        dcr = amplified_shear / shear_capacity
        
        self.logger.debug(f"Shear DCR: {dcr:.3f} "
                         f"(demand={amplified_shear:.2f}, capacity={shear_capacity:.2f})")
        return dcr
    
    def pt_yield_dcr_de(self) -> float:
        """Check PT bar yielding at DBE.
        
        Returns:
            Demand/Capacity Ratio (DCR < 1.0 = safe)
        """
        bar_yield_force = self.state.bar['Material']['Yield Force']
        max_pt_force = max(self.csa_de['PT1 Force'], self.csa_de['PT2 Force'])
        
        capacity = bar_yield_force
        demand = max_pt_force
        
        dcr = demand / capacity
        
        self.logger.debug(f"PT Yield DCR (DE): {dcr:.3f} "
                         f"(demand={demand:.2f}, capacity={capacity:.2f})")
        return dcr
    
    def pt_failure_dcr_mce(self) -> float:
        """Check PT bar failure (ultimate strain) at MCER.
        
        Returns:
            Demand/Capacity Ratio (DCR < 1.0 = safe)
        """
        ultimate_strain = self.state.bar['Ultimate Strain']
        max_strain = self.csa_mce['Bar Max Strain']
        
        capacity = ultimate_strain
        demand = max_strain
        
        dcr = demand / capacity
        
        self.logger.debug(f"PT Failure DCR (MCE): {dcr:.3f} "
                         f"(demand={demand:.6f}, capacity={capacity:.6f})")
        return dcr
    
    def energy_dissipation_ratio(self, min_ratio: float = 0.3) -> float:
        """Check energy dissipation ratio (inverted - higher ratio = better).
        
        Args:
            min_ratio: Minimum required energy dissipation ratio
            
        Returns:
            Demand/Capacity Ratio (DCR < 1.0 = safe, where demand = min_ratio)
        """
        geom = self.wall_geometry
        ufp = self.ufp_properties
        init_pt_force = self.state.bar['Initial PT Force']  # Initial PT for per each modeled bar (2 real bars)
        
        # UFP moment contribution
        ufp_moment = ufp['total_count'] * ufp['yield_force'] * geom['length']
        
        # Elastic restoring moment (2 panels, 2 bars per panel)
        elastic_moment = geom['num_panels'] * (geom['weight'] + 2 * init_pt_force) * geom['moment_arm']
        
        # Energy dissipation ratio
        beta = ufp_moment / (2 * elastic_moment)
        
        # DCR: min_ratio / actual_ratio (want actual >= min, so DCR < 1.0 is safe)
        dcr = min_ratio / beta if beta > 0 else float('inf')
        
        self.logger.debug(f"Energy Dissipation DCR: {dcr:.3f} "
                         f"(min_ratio={min_ratio:.3f}, actual={beta:.3f})")
        return dcr
    
    def ufp_failure_dcr_mce(self) -> float:
        """Check UFP failure (ultimate displacement) at MCER.
        
        Returns:
            Demand/Capacity Ratio (DCR < 1.0 = safe)
        """
        # Ultimate displacement from UFP relationship
        ufp_relationship = self.state.UFP['Delta-Force Relationship']
        ultimate_displacement = ufp_relationship[-1, 0]  # Last point in curve
        
        ufp_displacement = self.csa_mce['UFP Disp.']
        
        capacity = ultimate_displacement
        demand = ufp_displacement
        
        dcr = demand / capacity
        
        self.logger.debug(f"UFP Failure DCR (MCE): {dcr:.3f} "
                         f"(demand={demand:.6f}, capacity={capacity:.6f})")
        return dcr
    
    def wall_crush_dcr_mce(self) -> float:
        """Check wall fiber crushing (ultimate strain) at MCER.
        
        Returns:
            Demand/Capacity Ratio (DCR < 1.0 = safe)
        """
        geom = self.wall_geometry
        wall_max_strain = self.csa_mce['eps_c_max']
        
        capacity = geom['ultimate_strain']
        demand = wall_max_strain
        
        dcr = demand / capacity

        self.logger.debug(f"Wall Crush DCR (MCE): {dcr:.3f} "
                         f"(demand={demand:.6f}, capacity={capacity:.6f})")
        return dcr
    
    # =========================================================================
    # Summary Methods
    # =========================================================================
    
    def check_all_limit_states(self, analysis_runner) -> Dict[str, float]:
        """Check all design objectives and return summary.
        
        Args:
            analysis_runner: AnalysisRunner instance for drift analysis
        
        Returns:
            Dictionary of limit state names to DCR values
        """
        results = {
            'Drift (DE)': self.drift_ratio_de(analysis_runner),
            'Drift (MCE)': self.drift_ratio_mce(analysis_runner),  # Added
            'Moment Capacity': self.moment_dcr(),
            'Shear Capacity': self.shear_dcr(),
            'PT Yield (DE)': self.pt_yield_dcr_de(),
            'PT Failure (MCE)': self.pt_failure_dcr_mce(),
            'Energy Dissipation': self.energy_dissipation_ratio(),
            'UFP Failure (MCE)': self.ufp_failure_dcr_mce(),
            'Wall Crush (MCE)': self.wall_crush_dcr_mce()
        }
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("DESIGN OBJECTIVES SUMMARY (DCR < 1.0 = SAFE)")
        self.logger.info("=" * 60)
        for name, dcr in results.items():
            status = "✓ PASS" if dcr < 1.0 else "✗ FAIL"
            self.logger.info(f"{name:25s}: {dcr:6.3f}  {status}")
        self.logger.info("=" * 60)
        
        return results
    
    def get_controlling_limit_state(self) -> tuple:
        """Identify the controlling (worst) limit state.
        
        Returns:
            Tuple of (limit_state_name, dcr_value)
        """
        results = self.check_all_limit_states()
        controlling = max(results.items(), key=lambda x: x[1])
        
        self.logger.info(f"Controlling limit state: {controlling[0]} (DCR = {controlling[1]:.3f})")
        return controlling