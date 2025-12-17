# src/core/material_factory.py
"""
Material factory - creates OpenSees materials for all structural components.
Refactored to work with ModelState and follow consistent patterns.
"""
from typing import Tuple, Optional
import numpy as np
import openseespy.opensees as ops
import logging

from core.model_state import ModelState
from core.constants import LARGE_NUMBER, SMALL_NUMBER, RIGID_MATERIAL_STIFFNESS


class MaterialFactory:
    """Handles creation of materials for model components in OpenSees."""

    def __init__(self, state: ModelState):
        """Initialize with model state.

        Args:
            state: ModelState instance with component data
        """
        self.state = state
        self.mat_tag = 100
        self.logger = logging.getLogger(__name__)
        self.material_archive = {}


        # Validate model type
        valid_types = ["Elastic", "Inelastic"]
        if state.model_type not in valid_types:
            raise ValueError(
                f"Invalid model_type: '{state.model_type}'. "
                f"Must be one of {valid_types}"
            )



    def preliminaries(self) -> None:
        """Define preliminary materials (soft and hard elastic)."""
        soft_mat_tag = self._generate_mat_tag('Soft Material')
        ops.uniaxialMaterial('Elastic', soft_mat_tag, SMALL_NUMBER)

        hard_mat_tag = self._generate_mat_tag('Hard Material')
        ops.uniaxialMaterial('Elastic', hard_mat_tag, LARGE_NUMBER)
        
        self.logger.debug("Created preliminary materials: Soft and Hard")

    def define_bar_material(self, initial_PT_strain: Optional[float] = None) -> Tuple[float, float]:
        """Define PT bar material and compute initial strain for target tension.

        Args:
            initial_PT_strain: Initial PT strain (optional, defaults to calculated value)

        Returns:
            Tuple[float, float]: Target tension and updated initial PT strain

        Raises:
            KeyError: If required bar material properties are missing
            ValueError: If bar dimensions or stresses are invalid
        """
        # Validate required keys
        required_keys = ['Ultimate Stress', 'Yield Stress', 'Post Yield Hardening Ratio', 
                        'PT to Yield Force Ratio']
        bar_material = self.state.bar.get('Material', {})
        missing = [k for k in required_keys if k not in bar_material]
        if missing:
            raise KeyError(f"Missing required PT bar material properties: {missing}")

        # Extract material properties
        bar_elastic_module = bar_material['Elastic Module']
        bar_ultimate_stress = bar_material['Ultimate Stress']
        bar_yield_stress = bar_material['Yield Stress']
        bar_hardening_ratio = bar_material['Post Yield Hardening Ratio']
        bar_init_to_yield_ratio = bar_material['PT to Yield Force Ratio']
        
        # Compute derived properties
        bar_yield_strain = bar_yield_stress / self.state.steel_elastic_module
        bar_area = self.state.bar['Area']
        bar_height = self.state.wall_elevations[-1]

        # Validate
        if bar_area <= 0:
            raise ValueError(f"Bar area must be positive, got {bar_area}")
        if bar_height <= 0:
            raise ValueError(f"Bar height must be positive, got {bar_height}")

        # Calculate forces
        bar_yield_force = bar_area * bar_yield_stress
        self.state.bar['Material']['Yield Force'] = bar_yield_force
        target_tension = bar_yield_force * bar_init_to_yield_ratio

        # Calculate initial strain
        if initial_PT_strain is None:
            initial_PT_strain = bar_init_to_yield_ratio * bar_yield_strain

        # Store computed values
        self.state.bar['Initial PT Force'] = abs(target_tension)
        self.state.bar['Initial Disp.'] = bar_height * initial_PT_strain
        self.state.bar['Ultimate Force'] = bar_ultimate_stress * bar_area
        
        bar_ultimate_strain = (bar_yield_strain + 
                              (bar_ultimate_stress - bar_yield_stress) / 
                              (bar_hardening_ratio * self.state.steel_elastic_module))
        
        self.state.bar['Yield Strain'] = bar_yield_strain
        self.state.bar['Ultimate Strain'] = bar_ultimate_strain

        # Store thresholds for pushover monitoring
        self.state.bar['Thresholds'] = {
            'yield_force': bar_yield_force,
            'ultimate_force': bar_ultimate_stress * bar_area,
            'yield_strain': bar_yield_strain,
            'ultimate_strain': bar_ultimate_strain
        }

        # Create material chain: EPP → InitStrain → MinMax
        bar_EPP_mat_tag = self._generate_mat_tag('Bar EPP Material')
        ops.uniaxialMaterial('ElasticPPGap', bar_EPP_mat_tag, bar_elastic_module, 
                           bar_yield_stress, 0.0, bar_hardening_ratio)

        bar_initStrain_mat_tag = self._generate_mat_tag('Bar Initial Strain Material')
        ops.uniaxialMaterial('InitStrainMaterial', bar_initStrain_mat_tag, 
                           bar_EPP_mat_tag, initial_PT_strain)

        bar_MinMax_mat_tag = self._generate_mat_tag('Bar MinMax Material')
        ops.uniaxialMaterial('MinMax', bar_MinMax_mat_tag, bar_initStrain_mat_tag, 
                           '-max', bar_ultimate_strain)

        # Store force-displacement relationship
        strain_list = np.linspace(0, bar_ultimate_strain, 10000)
        force_list = bar_area * np.where(
            strain_list < bar_yield_strain,
            strain_list * self.state.steel_elastic_module,
            bar_yield_strain * self.state.steel_elastic_module + 
            (strain_list - bar_yield_strain) * bar_hardening_ratio * self.state.steel_elastic_module
        )
        disp_list = bar_height * strain_list
        self.state.bar['Delta-Force Relationship'] = np.column_stack((disp_list, force_list))

        self.logger.info(f'Designed PT force ratio: {bar_init_to_yield_ratio*100:.1f}%')

        return target_tension, initial_PT_strain

    def define_rocking_springs_material(self) -> None:
        """Define rocking spring materials using distributed wall stiffness.

        Raises:
            KeyError: If required wall or spring properties are missing
            ValueError: If plastic hinge length ratio or rocking numbers are invalid
        """
        # --- Validate wall properties ---
        required_wall_keys = [
            "Wall Elastic Modulus", "Wall Length", "Wall Thickness",
            "Wall Yield Stress", "Wall Split Strain", "Wall Crush Strain",
            "Plastic Hinge Length Ratio", "Plastic Hinge Length Reference",
            "Split Strength Ratio", "Crush Strength Ratio",
            "Pinch Factor X", "Pinch Factor Y", "Initial Damage X",
            "Initial Damage Y", "Deterioration Factor"
        ]
        missing_wall = [k for k in required_wall_keys if k not in self.state.wall]
        if missing_wall:
            raise KeyError(f"Missing required wall properties: {missing_wall}")

        # --- Validate spring properties ---
        if "Number of Rocking Springs" not in self.state.spring:
            raise KeyError("Number of Rocking Springs missing in spring config")
        
        if "Spring Weights" not in self.state.spring.get("Rocking", {}):
            raise KeyError("Spring Weights not computed. Call RockingSpring.create_nodes() first")

        self.state.spring["Rocking"].setdefault("Material", {})
        behavior = self.state.model_type

        # --- Extract wall properties ---
        wall = self.state.wall
        E = wall["Wall Elastic Modulus"]
        L = wall["Wall Length"]
        t = wall["Wall Thickness"]
        fy = wall["Wall Yield Stress"]
        strain_split = wall["Wall Split Strain"]
        strain_crush = wall["Wall Crush Strain"]
        
        # Backbone degradation ratios
        split_ratio = wall["Split Strength Ratio"]
        crush_ratio = wall["Crush Strength Ratio"]
        
        # Hysteretic parameters
        pinch_x = wall["Pinch Factor X"]
        pinch_y = wall["Pinch Factor Y"]
        damage_x = wall["Initial Damage X"]
        damage_y = wall["Initial Damage Y"]
        beta = wall["Deterioration Factor"]

        # --- Determine plastic hinge length ---
        ratio = wall["Plastic Hinge Length Ratio"]
        ref = wall["Plastic Hinge Length Reference"].lower()
        if ref == "thickness":
            Lp = ratio * t
        elif ref == "length":
            Lp = ratio * L
        else:
            raise ValueError(f"Invalid Plastic Hinge Length Reference: '{ref}'. Must be 'thickness' or 'length'")
        
        self.state.spring["Plastic Hinge Length"] = Lp

        # --- Get spring weights from quadrature ---
        spring_weights = self.state.spring["Rocking"]["Spring Weights"]
        n_springs = self.state.spring["Number of Rocking Springs"]
        
        if len(spring_weights) != n_springs:
            raise ValueError(f"Spring weights length {len(spring_weights)} != n_springs {n_springs}")

        # --- Calculate stiffness (for shear springs) ---
        Ks = E * t * L / Lp

        # Initialize thresholds storage
        self.state.spring["Rocking"]["Thresholds"] = {}

        # --- Create spring materials ---
        for spring_id in range(n_springs):
            wgt = spring_weights[spring_id]

            # Yield force (distributed by quadrature weight)
            Fy_spring = fy * t * L * wgt
            
            # Yield displacement (material-level, independent of distribution)
            delta_y = fy * Lp / E
            
            # Limit state displacements
            delta_split = strain_split * Lp
            delta_crush = strain_crush * Lp

            # Store in state
            self.state.spring["Rocking"]["Material"][f"Yield Force {spring_id}"] = Fy_spring
            self.state.spring["Rocking"]["Material"][f"Yield Disp. {spring_id}"] = delta_y

            # Store thresholds for pushover monitoring
            self.state.spring["Rocking"]["Thresholds"][f"Spring {spring_id}"] = {
                'yield_force': Fy_spring,
                'yield_disp': delta_y,
                'split_disp': delta_split,
                'crush_disp': delta_crush,
                'split_force': split_ratio * Fy_spring,
                'crush_force': crush_ratio * Fy_spring,
                'yield_strain': fy / E,
                'split_strain': strain_split,
                'crush_strain': strain_crush
            }

            if behavior == "Inelastic":
                # Generate material tags
                hysteretic_tag = self._generate_mat_tag(f"Rocking Spring Hyst {spring_id}")
                ent_tag = self._generate_mat_tag(f"Rocking Spring ENT {spring_id}")
                series_tag = self._generate_mat_tag(f"Rocking Spring {spring_id}")

                # Define backbone points (compression negative, tension positive)
                n1 = [-Fy_spring, -delta_y]
                n2 = [-split_ratio * Fy_spring, -delta_split]
                n3 = [-crush_ratio * Fy_spring, -delta_crush]
                p1 = [Fy_spring, delta_y]
                p2 = [split_ratio * Fy_spring, delta_split]
                p3 = [crush_ratio * Fy_spring, delta_crush]

                # Create Hysteretic material
                ops.uniaxialMaterial(
                    "Hysteretic", hysteretic_tag,
                    *p1, *p2, *p3, *n1, *n2, *n3,
                    pinch_x, pinch_y, damage_x, damage_y, beta
                )

                # Create ENT (Elastic No Tension) material
                # ops.uniaxialMaterial("ENT", ent_tag, RIGID_MATERIAL_STIFFNESS)
                ops.uniaxialMaterial("ENT", ent_tag, 1e6)

                # Combine in series (Hysteretic in compression, zero in tension)
                ops.uniaxialMaterial("Series", series_tag, hysteretic_tag, ent_tag)
                self.state.spring["Rocking"]["Material"][f"Spring {spring_id}"] = series_tag

                self.logger.debug(
                    f"Created inelastic rocking spring {spring_id}: "
                    f"Fy={Fy_spring:.2f}, δy={delta_y:.6f}"
                )

            elif behavior == "Elastic":
                spring_tag = self._generate_mat_tag(f"Rocking Spring {spring_id}")
                ops.uniaxialMaterial("Elastic", spring_tag, RIGID_MATERIAL_STIFFNESS)
                self.state.spring["Rocking"]["Material"][f"Spring {spring_id}"] = spring_tag

        # --- Create shear material ---
        shear_tag = self._generate_mat_tag("Shear Spring Material")
        ops.uniaxialMaterial("Elastic", shear_tag, 10 * Ks)

        self.logger.info(f"Created {n_springs} rocking spring materials ({behavior} behavior)")

    def define_ufp_material(self, ufp_model: str='Baird et al.') -> None:
        """
        Define UFP material based on model type (Elastic or Inelastic).

        Raises:
            KeyError: If required UFP properties are missing
            ValueError: If UFP dimensions or stresses are invalid
        """
        # --- Required UFP properties ---
        required_keys = ['UFP Width', 'UFP Thickness', 'UFP Diameter', 'UFP Yield Stress']
        missing = [k for k in required_keys if k not in getattr(self.state, 'UFP', {})]
        if missing:
            raise KeyError(f"Missing required UFP properties: {missing}")

        # --- Extract properties safely ---
        ufp = self.state.UFP
        width = ufp['UFP Width']
        thickness = ufp['UFP Thickness']
        diameter = ufp['UFP Diameter']
        leg_length = ufp.get('UFP Length', diameter)
        yield_stress = ufp['UFP Yield Stress']
        ufp_model = ufp["Hysteresis Model"]

        # --- Validate physical values ---
        if width <= 0 or thickness <= 0 or diameter <= 0:
            raise ValueError(f"UFP dimensions must be positive, got width={width}, thickness={thickness}, diameter={diameter}")
        if yield_stress <= 0:
            raise ValueError(f"UFP yield stress must be positive, got {yield_stress}")


        # --- Steel02 parameters ---
        R   = ufp.get('R', 25)
        b   = ufp.get('b', 0.01)
        cR1 = ufp.get('cR1', 0.925)
        cR2 = ufp.get('cR2', 0.15)
        a1  = ufp.get('a1', 0.05)
        a2  = ufp.get('a2', 2.0)
        a3  = ufp.get('a3', 0.05)
        a4  = ufp.get('a4', 2.0)


        if ufp_model == 'Baird et al.':

            # --- Derived properties ---
            R = 25 
            K0_factor = ((self.state.steel_elastic_module * width / np.pi) *
                        (thickness / diameter) ** 3)
            K0 = (16 / 27) * K0_factor  # stiffness
            Fy = yield_stress * width * thickness ** 2 / (2 * diameter)  # Effective Yield force


            over_strength = 1.85
            ufp['UFP Ultimate Stress'] = over_strength * Fy


        # --- Generate material tag ---
        UFP_Mat_tag_1 = self._generate_mat_tag('UFP Dir-2 Material')
        # UFP_Mat_tag_2 = self._generate_mat_tag('UFP Dir-2 MinMax Material')

        # --- Create material based on model type ---
        behavior = getattr(self.state, 'model_type', 'Inelastic')
        
        if behavior in ['Elastic', 'Design Checks']:
            ops.uniaxialMaterial('Elastic', UFP_Mat_tag_1, K0)
            self.logger.info(f"Created elastic UFP material: K={K0:.2f}")

        elif behavior == 'Inelastic':
            ops.uniaxialMaterial(
                'Steel02', UFP_Mat_tag_1, Fy, K0, b, R, cR1, cR2, a1, a2, a3, a4, 0.0
            )

            # ops.uniaxialMaterial('MinMax', UFP_Mat_tag_2, UFP_Mat_tag_1, '-max', ...)

            self.logger.info(f"Created inelastic UFP material: Fy={Fy:.2f}, K={K0:.2f}")

        # --- Store derived UFP properties ---
        ufp['Yield Force'] = Fy
        ufp['Ultimate Force'] = over_strength * Fy
        ufp['Yield Disp.'] = Fy / K0
        ufp['Stiffness'] = K0

        # Store thresholds for pushover monitoring
        ufp['Thresholds'] = {
            'yield_force': Fy,
            'ultimate_force': over_strength * Fy,
            'yield_disp': Fy / K0,
            'yield_stress': yield_stress,
            'ultimate_stress': over_strength * yield_stress
        }

        # Get hardening ratio from earlier
        b = ufp.get('b', 0.01)
        
        # Build SIMPLIFIED monotonic backbone (for CSA usage)
        # Note: Steel02 cyclic behavior is more complex than this
        force_list = np.linspace(0, ufp['Ultimate Force'], 10000)
        delta_list = np.where(
            force_list < Fy,
            force_list / K0,  # Elastic
            (force_list - Fy) / (b * K0) + ufp['Yield Disp.']
        )
        ufp['Delta-Force Relationship'] = np.column_stack((delta_list, force_list))

    def define_shear_key_mat(self) -> None:
        """Define shear key materials for each story.

        Raises:
            KeyError: If required diaphragm or shear key properties are missing.
            ValueError: If shear key dimensions are invalid.
        """
        E_key = self.state.diaphragm["Elements"]["Shear Key"].get("E")
        G_key = self.state.diaphragm["Elements"]["Shear Key"].get("G")
        if E_key is None or G_key is None:
            raise KeyError("E or G missing in Shear Key properties.")

        for story in range(1, self.state.n_stories + 1):
            # Geometry
            t = self.state.diaphragm["Elements"]["Shear Key"][f"Story {story}"]['Thickness']
            w = self.state.diaphragm["Elements"]["Shear Key"][f"Story {story}"]['Width']
            L = self.state.diaphragm["Elements"]["Shear Key"][f"Story {story}"]['Length']

            # Flexural stiffness
            k_f = 3 * E_key * w**3 * t / (12 * L**3)

            # Shear stiffness
            k_s = G_key * w * t / L
            k_total = k_f * k_s / (k_f + k_s)

            self.state.diaphragm["Elements"]["Shear Key"][f"Story {story}"]['Stiffness'] = k_total

            shear_key_mat_tag = self._generate_mat_tag(f'Story {story} Shear Key Material')
     
            ops.uniaxialMaterial('Elastic', shear_key_mat_tag, k_total)

    def _generate_mat_tag(self, material_name: str) -> int:
        """Generate a unique material tag and archive it.

        Args:
            material_name: Name of the material for archiving

        Returns:
            Generated material tag
        """
        self.mat_tag += 1
        self.material_archive[material_name] = self.mat_tag
        self.logger.debug(f"Generated material tag {self.mat_tag} for {material_name}")
        return self.mat_tag