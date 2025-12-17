# src/components/rocking_spring.py
"""
Rocking spring component - foundation interface springs.
"""
import numpy as np
from scipy.special import roots_sh_legendre
import openseespy.opensees as ops
from typing import Dict

from core.components.base_component import BaseComponent
from core.model_state import ModelState
from core.tag_manager import TagGenerator
from core.material_factory import MaterialFactory
from core.constants import LARGE_NUMBER


class RockingSpring(BaseComponent):
    """Handles creation of rocking spring nodes and elements."""

    def __init__(self, state: ModelState, tags: TagGenerator, materials: MaterialFactory):
        """Initialize rocking spring component.
        
        Args:
            state: ModelState instance
            tags: TagGenerator instance
            materials: MaterialFactory instance
        """
        super().__init__(state, tags)
        self.materials = materials
        self._spring_config = state.spring
        self._wall_config = state.wall

    def create_nodes(self) -> None:
        """Generate rocking spring nodes using Gauss-Legendre quadrature."""
        self._validate_required_keys(
            self._wall_config,
            ["Nodes Xs", "Wall Length"],
            "Wall config"
        )
        self._validate_required_keys(
            self._spring_config,
            ["Number of Rocking Springs"],
            "Spring config"
        )
        
        n_springs = self._spring_config["Number of Rocking Springs"]
        wall_length = self._wall_config["Wall Length"]
        
        if n_springs < 1:
            raise ValueError(f"Number of rocking springs must be >= 1, got {n_springs}")
        self._validate_positive(wall_length, "Wall length")
        
        self.tags.generate_node_tags(component='Rocking Springs')
        
        # Get Gauss-Legendre quadrature points and weights
        rel_x, weights = roots_sh_legendre(n_springs)
        self._spring_config.setdefault('Rocking', {})["Spring Weights"] = weights
        
        # Create nodes for both walls at base and top
        for elevation in ["Base", "Top"]:
            for side_idx, side in enumerate(["Left Wall", "Right Wall"]):
                wall_x = self._wall_config["Nodes Xs"][side_idx]
                self._create_spring_line(side, elevation, wall_x, wall_length, 
                                       rel_x, n_springs)

    def create_elements(self) -> None:
        """Create rocking spring elements (rigid links, zero-length, shear diagonals)."""
        self._validate_required_keys(
            self._wall_config,
            ["Wall Length", "Wall Thickness", "Nodes"],
            "Wall config"
        )
        
        if 'Shear Spring Material' not in self.materials.material_archive:
            raise ValueError("Shear Spring Material must be defined before creating elements")
        
        n_springs = self._spring_config["Number of Rocking Springs"]
        
        self.tags.generate_element_tags(component="Rocking Springs")
        
        shear_mat = self.materials.material_archive['Shear Spring Material']
        shear_counter = 0
        
        for side in ["Left Wall", "Right Wall"]:
            wall_node = self._wall_config["Nodes"]["Elevation 0"][side]
            
            # Create vertical spring elements
            for spring_idx in range(n_springs):
                self._create_vertical_spring(side, spring_idx, wall_node)
            
            # Create shear diagonals
            shear_counter = self._create_shear_diagonals(
                side, n_springs, shear_mat, shear_counter
            )

    def _create_spring_line(self, side: str, elevation: str, wall_x: float, 
                           wall_length: float, rel_x: np.ndarray, 
                           n_springs: int) -> None:
        """Create line of spring nodes for one wall side and elevation.
        
        Args:
            side: 'Left Wall' or 'Right Wall'
            elevation: 'Base' or 'Top'
            wall_x: Wall x-coordinate origin
            wall_length: Wall length
            rel_x: Relative x positions from quadrature
            n_springs: Number of springs
        """
        # Scale relative positions to wall length, centered on wall
        xs = wall_x + wall_length * (rel_x - 0.5)
        
        nodes = self._spring_config['Rocking']["Nodes"][side]
        
        for spring_idx, x in enumerate(xs):
            node_data = nodes[f"{elevation} {spring_idx}"]
            node_tag = node_data['Tag']
            
            ops.node(node_tag, x, 0)
            node_data['x'] = x
            
            self.logger.debug(f"Created rocking spring node: {side}, {elevation} "
                            f"{spring_idx}, x={x:.2f}")

    def _create_vertical_spring(self, side: str, spring_idx: int, 
                               wall_node: int) -> None:
        """Create rigid link and zero-length spring for one location.
        
        Args:
            side: 'Left Wall' or 'Right Wall'
            spring_idx: Spring index
            wall_node: Wall connection node
        """
        # Get material
        mat_tag = self.materials.material_archive.get(f'Rocking Spring {spring_idx}')
        if not mat_tag:
            raise KeyError(f"Rocking Spring {spring_idx} material not found")
        
        # Get nodes
        nodes = self._spring_config["Rocking"]["Nodes"][side]
        base_node = nodes[f"Base {spring_idx}"]['Tag']
        top_node = nodes[f"Top {spring_idx}"]['Tag']
        
        # Get element tags
        elements = self._spring_config["Elements"]
        rigid_tag = elements["Rigid Elements"][f"Rocking Number {spring_idx}"][side]
        zero_tag = elements["ZeroLength Elements"][f"Rocking Number {spring_idx}"][side]
        
        if not all([base_node, top_node, rigid_tag, zero_tag]):
            raise KeyError(f"Missing node or element tag for {side}, Spring {spring_idx}")
        
        # Create rigid link from wall to spring top
        ops.element(
            'elasticBeamColumn', rigid_tag, wall_node, top_node,
            LARGE_NUMBER, LARGE_NUMBER, LARGE_NUMBER, self.state.linear_geom_transf
        )
        
        # Create zero-length spring (vertical only)
        ops.element(
            'zeroLength', zero_tag, base_node, top_node,
            '-mat', mat_tag, '-dir', 2
        )
        
        # Fix base node
        ops.fix(base_node, 1, 1, 1)
        
        self.logger.debug(f"Created rocking spring for {side}, Spring {spring_idx}: "
                        f"rigid={rigid_tag}, zero={zero_tag}")

    def _create_shear_diagonals(self, side: str, n_springs: int, 
                                shear_mat: int, counter: int) -> int:
        """Create diagonal shear elements connecting corners.
        
        Args:
            side: 'Left Wall' or 'Right Wall'
            n_springs: Number of springs
            shear_mat: Shear material tag
            counter: Current shear element counter
            
        Returns:
            Updated counter value
        """
        nodes = self._spring_config["Rocking"]["Nodes"][side]
        
        # Get corner nodes
        base_left = nodes[f"Base 0"]['Tag']
        base_right = nodes[f"Base {n_springs-1}"]['Tag']
        top_left = nodes[f"Top 0"]['Tag']
        top_right = nodes[f"Top {n_springs-1}"]['Tag']
        
        if not all([base_left, base_right, top_left, top_right]):
            raise KeyError(f"Missing shear diagonal nodes for {side}")
        
        # Create two diagonal truss elements
        elements = self._spring_config["Elements"]
        
        # Diagonal 1: base_left to top_right
        tag1 = elements[f"Shear Element {counter}"]
        ops.element("corotTruss", tag1, base_left, top_right, 1, shear_mat)
        counter += 1
        
        # Diagonal 2: base_right to top_left
        tag2 = elements[f"Shear Element {counter}"]
        ops.element("corotTruss", tag2, base_right, top_left, 1, shear_mat)
        counter += 1
        
        self.logger.debug(f"Created shear diagonals for {side}: tags={tag1},{tag2}")
        
        return counter