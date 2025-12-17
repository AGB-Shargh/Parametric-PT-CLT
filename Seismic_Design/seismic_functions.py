# seismic_functions.py
"""
Seismic force calculations following ASCE 7-22.
Replaces Seismic_Functions.py with ModelState integration.
"""
import logging
import numpy as np
from typing import Dict, Optional

from core.model_state import ModelState
from core.constants import g, kip, inch, ft, sec


# ASCE 7-22 Table 12.8-1: Cu Coefficients
CU_COEFFICIENTS = {
    'S_D1': [0.1, 0.15, 0.2, 0.3, 0.4],
    'Cu': [1.7, 1.6, 1.5, 1.4, 1.4]
}


class SeismicFunctions:
    """Seismic analysis functions following ASCE 7-22."""
    
    def __init__(self, state: ModelState, 
                 SS: float = 2.25,
                 S1: float = 0.72,
                 SMS: float = 2.28,
                 SM1: float = 1.66,                 
                 site_class: str = 'D',
                 TL: float = 8.0,
                 R: float = 6.0,
                 Ie: float = 1.0,
                 Cd: float = 6.0):
        """Initialize seismic functions.
        
        Args:
            state: ModelState instance with building/wall properties
            SS: Mapped MCER spectral acceleration at short periods
            S1: Mapped MCER spectral acceleration at 1-second period
            site_class: Site class (A, B, C, D, or E)
            TL: Long-period transition period
            R: Response modification coefficient
            Ie: Importance factor
            Cd: Deflection amplification factor
        """
        self.state = state
        self.logger = logging.getLogger(__name__)
        
        # Validate site class
        valid_classes = ['A', 'B', 'C', 'D', 'E']
        if site_class not in valid_classes:
            raise ValueError(f"Site class must be one of {valid_classes}")
        
        # Force or Drift check applications
        self.application = 'strength'

        # Store seismic parameters
        self.SS = SS
        self.S1 = S1
        self.SMS = SMS
        self.SM1 = SM1
        self.site_class = site_class
        self.TL = TL
        self.R = R
        self.Ie = Ie
        self.Cd = Cd

        # Calculate design spectral accelerations (ASCE 7-22 Section 11.4.4)
        self.SDS = (2.0 / 3.0) * self.SMS
        self.SD1 = (2.0 / 3.0) * self.SM1
        
        self.logger.info(f"Seismic parameters: SS={SS:.3f}, S1={S1:.3f}, "
                        f"SDS={self.SDS:.3f}, SD1={self.SD1:.3f}, Site Class={site_class}")
    
    def _interpolate_site_coefficient(self, spectral_value: float, 
                                      coeff_table: Dict, site_class: str) -> float:
        """Interpolate site coefficient from ASCE 7-22 tables.
        
        Args:
            spectral_value: SS or S1 value
            coeff_table: Coefficient table (Fa or Fv)
            site_class: Site class letter
            
        Returns:
            Interpolated site coefficient
        """
        site_idx = ['A', 'B', 'C', 'D', 'E'].index(site_class)
        spectral_values = np.array(list(coeff_table.keys()))
        coefficients = np.array([coeff_table[sv][site_idx] for sv in spectral_values])
        
        return float(np.interp(spectral_value, spectral_values, coefficients))
    
    @property
    def seismic_weights(self) -> np.ndarray:
        """Get seismic weights from story masses.
        
        Returns:
            Array of seismic weights [n_stories x 1]
        """
        masses = []
        for story in range(1, self.state.n_stories + 1):
            elev_key = f"Elevation {story}"
            mass = self.state.building.get(elev_key, {}).get("Seismic Mass", 0.0)
            masses.append(mass)
        
        weights = np.array(masses).reshape(-1, 1) * g
        return weights
    
    @property
    def floor_heights(self) -> np.ndarray:
        """Get cumulative floor heights.
        
        Returns:
            Array of cumulative heights [n_stories x 1]
        """
        story_heights = self.state.building.get('Stories Heights', [])
        return np.cumsum(story_heights).reshape(-1, 1)
    
    def _calculate_approximate_period(self) -> float:
        """Calculate approximate period Ta per ASCE 7-22 Section 12.8.2.1.
        
        Returns:
            Approximate fundamental period (seconds)
        """
        building_height = float(np.sum(self.state.building.get('Stories Heights', [])))
        hn_ft = building_height / ft
        
        # Use Equation 12.8-7 for other structures (Ct=0.02, x=0.75)
        Ta = 0.02 * (hn_ft ** 0.75)
        
        self.logger.debug(f"Approximate period Ta = {Ta:.3f} sec (height = {hn_ft:.1f} ft)")
        return Ta
    
    def _interpolate_cu_coefficient(self) -> float:
        """Interpolate Cu coefficient from ASCE 7-22 Table 12.8-1.
        
        Returns:
            Coefficient for upper limit on calculated period
        """
        sd1_values = np.array(CU_COEFFICIENTS['S_D1'])
        cu_values = np.array(CU_COEFFICIENTS['Cu'])
        
        cu = float(np.interp(self.SD1, sd1_values, cu_values))
        self.logger.debug(f"Cu coefficient = {cu:.2f} for SD1 = {self.SD1:.3f}")
        return cu
    
    def get_design_period(self, analytical_period: Optional[float] = None,
                         application: str = 'drift') -> float:
        """Calculate design period based on application.
        
        Args:
            analytical_period: Computed period from eigenvalue analysis
            application: 'drift' or 'force' (determines Cu limit application)
            
        Returns:
            Design period (seconds)
        """
        Ta = self._calculate_approximate_period()
        
        if analytical_period is None:
            # Use first mode period from state if available
            analytical_period = self.state.periods[0] if self.state.periods else Ta

        if self.application.lower() == 'drift':
            # For drift calculation, use analytical period directly per ASCE 7-22 Section 12.8.6.2
            T_design = analytical_period
            self.logger.debug(f"Drift application: T = {T_design:.3f} sec (analytical)")
        else:
            # For force calculation, apply Cu limit per ASCE 7-22 Section 12.8.2
            Cu = self._interpolate_cu_coefficient()
            T_design = min(analytical_period, Cu * Ta)
            self.logger.debug(f"Force application: T = {T_design:.3f} sec "
                            f"(min of {analytical_period:.3f}, {Cu:.2f}*{Ta:.3f})")
        
        return T_design
    
    def _calculate_cs_coefficient(self, period: float) -> float:
        """Calculate seismic response coefficient per ASCE 7-22 Section 12.8.1.1.
        
        Args:
            period: Design period
            
        Returns:
            Seismic response coefficient Cs
        """
        R_over_Ie = self.R / self.Ie
        
        # Initial value (Equation 12.8-2)
        Cs = self.SDS / R_over_Ie
        
        # Upper limit depends on period
        if period <= self.TL:
            # Equation 12.8-4
            Cs_max = self.SD1 / (period * R_over_Ie)
        else:
            # Equation 12.8-5
            Cs_max = (self.SD1 * self.TL) / (period**2 * R_over_Ie)
        
        Cs = min(Cs, Cs_max)
        
        # Lower limits
        # Equation 12.8-6
        Cs_min = max(0.044 * self.SDS * self.Ie, 0.01)
        
        # Equation 12.8-7 (for S1 >= 0.6g)
        if self.S1 >= 0.6:
            Cs_min_s1 = 0.5 * self.S1 / R_over_Ie
            Cs_min = max(Cs_min, Cs_min_s1)
        
        Cs = max(Cs, Cs_min)
        
        self.logger.debug(f"Cs = {Cs:.4f} for T = {period:.3f} sec "
                         f"(max={Cs_max:.4f}, min={Cs_min:.4f})")
        return Cs
    
    def compute_base_shear(self, period: Optional[float] = None) -> float:
        """Calculate seismic base shear per ASCE 7-22 Section 12.8.1.
        
        Args:
            period: Design period (if None, calculates from Ta)
            
        Returns:
            Seismic base shear
        """
        if period is None:
            period = self.get_design_period(application='force')
        
        # Equation 12.8-1
        Cs = self._calculate_cs_coefficient(period)
        W = self.seismic_weights.sum()
        V = Cs * W
        
        self.logger.info(f"Base shear V = {V:.2f} (Cs={Cs:.4f}, W={W:.2f})")
        return float(V)
    
    def _get_k_coefficient(self, period: float) -> float:
        """Get vertical distribution exponent k per ASCE 7-22 Section 12.8.3.
        
        Args:
            period: Design period
            
        Returns:
            Distribution exponent k
        """
        k = np.interp(period, [0.5, 2.5], [1.0, 2.0])
        
        return k
    
    def distribute_elf_forces(self, period: Optional[float] = None) -> Dict[int, float]:
        """Distribute equivalent lateral forces per ASCE 7-22 Section 12.8.3.
        
        Args:
            period: Design period (if None, calculates from Ta)
            
        Returns:
            Dictionary mapping node tags to lateral forces
        """
        if period is None:
            period = self.get_design_period(application='force')
        
        # Get vertical distribution parameters
        k = self._get_k_coefficient(period)
        heights = self.floor_heights
        weights = self.seismic_weights
        
        # Calculate Cvx per Equation 12.8-12
        w_h_k = weights * (heights ** k)
        Cvx = w_h_k / w_h_k.sum()
        
        # Calculate base shear
        V = self.compute_base_shear(period)
        
        # Distribute forces per Equation 12.8-11
        forces = (V * Cvx).flatten()
        
        # Map to wall nodes (assuming diaphragm nodes)
        force_dict = {}
        for story in range(1, self.state.n_stories + 1):
            elev_key = f"Elevation {story}"
            # Apply to diaphragm nodes (distributed to walls through diaphragm)
            diaphragm_nodes = self.state.diaphragm.get("Nodes", {}).get(elev_key, {})
            
            # Distribute to left and right wall equally
            story_force = forces[story - 1]
            if "Left Wall" in diaphragm_nodes:
                force_dict[diaphragm_nodes["Left Wall"]] = story_force / 2
            if "Right Wall" in diaphragm_nodes:
                force_dict[diaphragm_nodes["Right Wall"]] = story_force / 2
        
        self.logger.info(f"Distributed ELF forces: k={k:.2f}, total={sum(forces):.2f}")
        return force_dict
    
    def calculate_moment_demand(self, period: Optional[float] = None) -> float:
        """Calculate overturning moment from ELF forces.
        
        Args:
            period: Design period
            
        Returns:
            Overturning moment at base
        """
        force_dist = self.distribute_elf_forces(period)
        forces_per_node = np.array(list(force_dist.values()))
        story_forces = forces_per_node.reshape(-1, 2).sum(axis=1)
        cumulative_heights = self.floor_heights.flatten()
        
        # Sum of force × height
        moment = np.sum(story_forces * cumulative_heights)
        
        self.logger.debug(f"Overturning moment = {moment:.2f}")
        return float(moment)