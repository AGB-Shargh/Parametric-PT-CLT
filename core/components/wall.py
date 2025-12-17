# src/components/wall.py
"""
Wall component - handles CLT wall nodes and elements.
"""
import numpy as np
import openseespy.opensees as ops
from typing import Tuple

from core.components.base_component import BaseComponent
from core.model_state import ModelState
from core.tag_manager import TagGenerator
from core.constants import NEGLIGIBLE_MASS

class Wall(BaseComponent):
    """Handles creation of wall nodes, elements, and properties."""
    
    def __init__(self, state: ModelState, tags: TagGenerator):
        """Initialize wall component.
        
        Args:
            state: ModelState instance
            tags: TagGenerator instance
        """
        super().__init__(state, tags)
        self._wall_config = state.wall
        self._building_config = state.building
    
    def compute_properties(self) -> None:
        """Compute wall geometric properties and elevations."""
        # Validate inputs
        self._validate_required_keys(
            self._wall_config,
            ["Wall Length", "Wall Thickness", "Shear Correction Factor", 
             "Wall Extension"],
            "Wall config"
        )
        self._validate_required_keys(
            self._building_config,
            ["Stories Heights"],
            "Building config"
        )
        
        # Extract values
        length = float(self._wall_config["Wall Length"])
        thickness = float(self._wall_config["Wall Thickness"])
        shear_factor = float(self._wall_config["Shear Correction Factor"])
        extension = float(self._wall_config["Wall Extension"])
        
        # Validate
        self._validate_positive(length, "Wall length")
        self._validate_positive(thickness, "Wall thickness")
        self._validate_positive(shear_factor, "Shear correction factor")

        
        # Compute elevations
        heights = self._building_config["Stories Heights"]
        self.state.building_height = np.sum(heights)
        if extension > 0:
            heights = np.hstack((heights, extension))
        self.state.wall_elevations = np.hstack((0.0, np.cumsum(heights)))
        
        # Compute section properties
        area, iz, shear_area = self._compute_section_properties(
            length, thickness, shear_factor
        )
        
        # Update state
        self._wall_config.update({
            "Wall Cross Section Area": area,
            "Wall Iz": iz,
            "Wall Shear Area": shear_area,
            "Nodes Xs": (0.0, length),
            "Nodes Ys": self.state.wall_elevations,
            "Segmental Elevations": {}
        })
        
        self.logger.debug(f"Computed: A={area:.2f}, Iz={iz:.2f}, As={shear_area:.2f}")
    
    def create_nodes(self) -> None:
        """Generate wall nodes at each elevation."""
        self._validate_required_keys(
            self._wall_config,
            ["Nodes Xs", "Nodes Ys"],
            "Wall nodes"
        )
        
        self.tags.generate_node_tags(component='Wall')
        
        # Fixed base nodes
        self._create_fixed_base_nodes()

        # Elevation nodes
        for elev_idx, y in enumerate(self.state.wall_elevations):
            self._validate_positive(y, f"Elevation {elev_idx}")
            self._create_elevation_nodes(elev_idx, y)
    
    def create_elements(self) -> None:
        """Create wall elements connecting nodes."""
        self._validate_required_keys(
            self.state.UFP,
            ["UFP Numbers"],
            "UFP"
        )
        
        self.tags.generate_element_tags(component='Wall')
        
        for side in ['Left Wall', 'Right Wall']:
            self._create_side_elements(side)
            
            # Extension if present
            if self._wall_config.get("Wall Extension", 0) > 0:
                self._create_extension_element(side)
    
    def _compute_section_properties(self, length: float, thickness: float, 
                                   shear_factor: float) -> Tuple[float, float, float]:
        """Compute section area, moment of inertia, and shear area.
        
        Args:
            length: Wall length
            thickness: Wall thickness
            shear_factor: Shear correction factor
            
        Returns:
            Tuple of (area, Iz, shear_area)
        """
        area = thickness * length
        iz = (thickness * length ** 3) / 12
        shear_area = area * shear_factor
        return area, iz, shear_area
    
    def _create_fixed_base_nodes(self) -> None:
        """Create and fix base nodes."""
        left_node = self._wall_config["Nodes"]["Fixed Base"]["Left Wall"]
        right_node = self._wall_config["Nodes"]["Fixed Base"]["Right Wall"]
        
        x_left, x_right = self._wall_config["Nodes Xs"]
        
        ops.node(left_node, x_left, 0)
        ops.node(right_node, x_right, 0)
        ops.fix(left_node, 1, 1, 1)
        ops.fix(right_node, 1, 1, 1)

    def _create_elevation_nodes(self, elev_idx: int, y: float) -> None:
        """Create nodes at given elevation.
        
        Args:
            elev_idx: Elevation index
            y: Y-coordinate
        """
        nodes = self._wall_config["Nodes"][f"Elevation {elev_idx}"]
        x_left, x_right = self._wall_config["Nodes Xs"]
 
        ops.node(nodes["Left Wall"], x_left, y, '-mass', NEGLIGIBLE_MASS, NEGLIGIBLE_MASS, NEGLIGIBLE_MASS)
        ops.node(nodes["Right Wall"], x_right, y, '-mass', NEGLIGIBLE_MASS, NEGLIGIBLE_MASS, NEGLIGIBLE_MASS)
        
        # Store segmental elevation
        segment = self._wall_config["Segmental Elevations"].setdefault(
            f"{elev_idx+1}0", {}
        )
        segment.update({
            "Left Node": nodes["Left Wall"],
            "Right Node": nodes["Right Wall"],
            "Height": y
        })
    
    def _create_side_elements(self, side: str) -> None:
        """Create wall elements for one side.
        
        Args:
            side: 'Left Wall' or 'Right Wall'
        """
        for story in range(1, self.state.n_stories + 1):
            prev_node = self._wall_config["Nodes"][f"Elevation {story-1}"][side]
            ufp_count = self.state.UFP['UFP Numbers'][story - 1]
            
            for seg in range(1, ufp_count + 2):
                elem_tag, curr_node = self._get_segment_info(side, story, seg, ufp_count)
                self._add_segment(elem_tag, prev_node, curr_node)
                prev_node = curr_node
    
    def _create_extension_element(self, side: str) -> None:
        """Create wall extension element.
        
        Args:
            side: 'Left Wall' or 'Right Wall'
        """
        story = self.state.n_stories
        prev_node = self._wall_config["Nodes"][f"Elevation {story}"][side]
        curr_node = self._wall_config["Nodes"][f"Elevation {story + 1}"][side]
        elem_tag = self._wall_config["Elements"]["Wall Extension"][side]
        
        self._add_segment(elem_tag, prev_node, curr_node)
    
    def _get_segment_info(self, side: str, story: int, seg: int, 
                         ufp_count: int) -> Tuple[int, int]:
        """Get element tag and node for segment.
        
        Args:
            side: Wall side
            story: Story number
            seg: Segment number
            ufp_count: Number of UFPs in story
            
        Returns:
            Tuple of (element_tag, current_node)
        """
        elem_tag = self._wall_config["Elements"][f"Story {story}"][f"{side} Segment {seg}"]
        
        if seg == ufp_count + 1:
            curr_node = self._wall_config["Nodes"][f"Elevation {story}"][side]
        else:
            curr_node = self.state.UFP["Nodes"][f"Story {story}"][f"{side} {seg}"]
        
        return elem_tag, curr_node
    
    def _add_segment(self, elem_tag: int, node_i: int, node_j: int) -> None:
        """Add ElasticTimoshenkoBeam element.
        
        Args:
            elem_tag: Element tag
            node_i: Starting node
            node_j: Ending node
        """
        self._validate_required_keys(
            self._wall_config,
            ["Wall Elastic Modulus", "Wall Shear Modulus", "Wall Cross Section Area",
             "Wall Iz", "Wall Shear Area"],
            "Wall material"
        )
        

        ops.element(
            "ElasticTimoshenkoBeam",
            elem_tag, node_i, node_j,
            self._wall_config["Wall Elastic Modulus"],
            self._wall_config["Wall Shear Modulus"],
            self._wall_config["Wall Cross Section Area"],
            self._wall_config["Wall Iz"],
            self._wall_config["Wall Shear Area"],
            self.state.linear_geom_transf
        )