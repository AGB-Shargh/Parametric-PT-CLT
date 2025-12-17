# src/components/ufp.py
"""
U-shaped flexural plate (UFP) component - energy dissipation devices.
"""
import numpy as np
import openseespy.opensees as ops
from typing import Dict, List, Tuple

from core.components.base_component import BaseComponent
from core.model_state import ModelState
from core.tag_manager import TagGenerator
from core.material_factory import MaterialFactory
from core.constants import LARGE_NUMBER, NEGLIGIBLE_MASS


class UFP(BaseComponent):
    """Handles creation of U-shaped flexural plate nodes and elements."""

    def __init__(self, state: ModelState, tags: TagGenerator, materials: MaterialFactory):
        """Initialize UFP component.
        
        Args:
            state: ModelState instance
            tags: TagGenerator instance
            materials: MaterialFactory instance
        """
        super().__init__(state, tags)
        self.materials = materials
        self._ufp_config = state.UFP
        self._wall_config = state.wall
        self._building_config = state.building
        self._wall_config = state.wall

    def create_nodes(self, location: str) -> None:
        """Generate UFP nodes for each story.
        
        Args:
            location: 'On Wall' or 'Between Wall'
            
        Raises:
            ValueError: If location invalid or UFP configuration invalid
        """
        if location not in ['On Wall', 'Between Wall']:
            raise ValueError(f"Invalid location: {location}. "
                           "Must be 'On Wall' or 'Between Wall'")
        
        self._validate_required_keys(
            self._wall_config,
            ["Nodes Xs", "Nodes Ys"],
            "Wall nodes"
        )
        self._validate_required_keys(
            self._building_config,
            ["Stories Heights"],
            "Building config"
        )
        self._validate_required_keys(
            self._ufp_config,
            ["UFP Height Ratios"],
            "UFP config"
        )
        
        if location == 'On Wall':
            self.tags.generate_node_tags(component='UFP')
        
        stories_heights = self._building_config["Stories Heights"]
        if len(stories_heights) != self.state.n_stories:
            raise ValueError("Stories Heights length does not match n_stories")
        
        x_left, x_right = self._wall_config["Nodes Xs"]
        x_center = (x_left + x_right) / 2
        
        for story_idx, story_height in enumerate(stories_heights):
            self._create_story_nodes(
                story_idx, story_height, location, x_left, x_right, x_center
            )
        
        # Sort segmental elevations if on wall
        if location == 'On Wall':
            self._sort_segmental_elevations()

    def create_elements(self) -> None:
        """Create UFP rigid connectors and zero-length elements."""
        required_materials = ['Soft Material', 'UFP Dir-2 Material']
        for mat in required_materials:
            if mat not in self.materials.material_archive:
                raise ValueError(f"{mat} must be defined before creating UFP elements")
        
        self.tags.generate_element_tags(component='UFP')
        
        mat_soft = self.materials.material_archive['Soft Material']
        mat_ufp = self.materials.material_archive['UFP Dir-2 Material']
        
        for story_idx in range(self.state.n_stories):
            story_num = story_idx + 1
            ufp_count = self._ufp_config['UFP Numbers'][story_idx]
            
            if ufp_count < 0:
                raise ValueError(f"Invalid UFP count {ufp_count} for Story {story_num}")
            
            for ufp_idx in range(ufp_count):
                self._create_ufp_assembly(story_num, ufp_idx + 1, mat_ufp)

    def _create_story_nodes(self, story_idx: int, story_height: float, 
                           location: str, x_left: float, x_right: float, 
                           x_center: float) -> None:
        """Create UFP nodes for one story.
        
        Args:
            story_idx: Story index (0-based)
            story_height: Height of story
            location: Node location type
            x_left: Left wall x-coordinate
            x_right: Right wall x-coordinate
            x_center: Center x-coordinate
        """
        story_num = story_idx + 1
        floor_elevation = self.state.wall_elevations[story_idx]
        ufp_ratios = self._ufp_config['UFP Height Ratios'][story_idx]
        
        # Validate ratios
        if not ufp_ratios or any(not 0 <= r <= 1 for r in ufp_ratios):
            raise ValueError(f"Invalid UFP height ratios for Story {story_num}")
        
        nodes = self._ufp_config["Nodes"].get(f"Story {story_num}", {})
        if not nodes:
            raise KeyError(f"UFP nodes for Story {story_num} not generated")
        
        for level_idx, ratio in enumerate(ufp_ratios):
            ufp_y = floor_elevation + ratio * story_height
            
            # Validate height within story bounds
            if not (floor_elevation <= ufp_y <= floor_elevation + story_height):
                raise ValueError(f"Invalid UFP height {ufp_y:.2f} for "
                               f"Story {story_num}, Level {level_idx + 1}")
            
            # Create nodes based on location
            if location == 'On Wall':
                self._create_wall_nodes(nodes, level_idx, ufp_y, x_left, x_right)
                self._store_segmental_elevation(story_idx, level_idx, nodes, ufp_y)
            else:  # Between Wall
                self._create_center_nodes(nodes, level_idx, ufp_y, x_center)
            
            self.logger.debug(f"Created UFP nodes for Story {story_num}, "
                            f"Level {level_idx + 1}: y={ufp_y:.2f}")

    def _create_wall_nodes(self, nodes: Dict, level_idx: int, y: float, 
                          x_left: float, x_right: float) -> None:
        """Create UFP nodes on wall faces.
        
        Args:
            nodes: Node dictionary for this story
            level_idx: Level index within story
            y: Y-coordinate
            x_left: Left x-coordinate
            x_right: Right x-coordinate
        """
        level_num = level_idx + 1
        left_tag = nodes[f"Left Wall {level_num}"]
        right_tag = nodes[f"Right Wall {level_num}"]
        
        ops.node(left_tag, x_left, y, '-mass', NEGLIGIBLE_MASS, NEGLIGIBLE_MASS, NEGLIGIBLE_MASS)
        ops.node(right_tag, x_right, y, '-mass', NEGLIGIBLE_MASS, NEGLIGIBLE_MASS, NEGLIGIBLE_MASS)

    def _create_center_nodes(self, nodes: Dict, level_idx: int, 
                           y: float, x_center: float) -> None:
        """Create UFP nodes at wall center.
        
        Args:
            nodes: Node dictionary for this story
            level_idx: Level index within story
            y: Y-coordinate
            x_center: Center x-coordinate
        """
        level_num = level_idx + 1
        mid_left_tag = nodes[f"Middle Left {level_num}"]
        mid_right_tag = nodes[f"Middle Right {level_num}"]
        
        ops.node(mid_left_tag, x_center, y, '-mass', NEGLIGIBLE_MASS, NEGLIGIBLE_MASS, NEGLIGIBLE_MASS)
        ops.node(mid_right_tag, x_center, y, '-mass', NEGLIGIBLE_MASS, NEGLIGIBLE_MASS, NEGLIGIBLE_MASS)

    def _store_segmental_elevation(self, story_idx: int, level_idx: int, 
                                   nodes: Dict, y: float) -> None:
        """Store segmental elevation data for wall segments.
        
        Args:
            story_idx: Story index (0-based)
            level_idx: Level index within story
            nodes: Node dictionary
            y: Y-coordinate
        """
        level_num = level_idx + 1
        key = f"{story_idx + 1}{level_idx + 1}"
        
        self._wall_config["Segmental Elevations"][key] = {
            "Left Node": nodes[f"Left Wall {level_num}"],
            "Right Node": nodes[f"Right Wall {level_num}"],
            "Height": y
        }

    def _sort_segmental_elevations(self) -> None:
        """Sort segmental elevations by key."""
        elevations = self._wall_config["Segmental Elevations"]
        self._wall_config["Segmental Elevations"] = {
            k: v for k, v in sorted(elevations.items(), key=lambda x: int(x[0]))
        }

    def _create_ufp_assembly(self, story_num: int, ufp_num: int, 
                            mat_ufp: int) -> None:
        """Create rigid links and zero-length element for one UFP.
        
        Args:
            story_num: Story number (1-based)
            ufp_num: UFP number within story
            mat_ufp: UFP material tag
        """
        # Get nodes
        nodes = self._ufp_config["Nodes"][f"Story {story_num}"]
        node_left_wall = nodes[f"Left Wall {ufp_num}"]
        node_mid_left = nodes[f"Middle Left {ufp_num}"]
        node_mid_right = nodes[f"Middle Right {ufp_num}"]
        node_right_wall = nodes[f"Right Wall {ufp_num}"]
        
        if not all([node_left_wall, node_mid_left, node_mid_right, node_right_wall]):
            raise KeyError(f"Missing UFP nodes for Story {story_num}, UFP {ufp_num}")
        
        # Get element tags
        elements = self._ufp_config["Elements"]
        rigid_left = elements["Rigid Elements"][f"Story {story_num}"][f"UFP Number {ufp_num}"]['Left']
        rigid_right = elements["Rigid Elements"][f"Story {story_num}"][f"UFP Number {ufp_num}"]['Right']
        zero_length = elements["ZeroLength Elements"][f"Story {story_num}"][f"UFP Number {ufp_num}"]
        

        # Get wall properties for reference
        wall_E = self._wall_config["Wall Elastic Modulus"]
        wall_A = self._wall_config["Wall Cross Section Area"]
        wall_I = self._wall_config["Wall Iz"]

        

        # Make rigid elements 1000x stiffer than wall
        rigid_multiplier = 1000
        rigid_E = wall_E * rigid_multiplier
        rigid_A = wall_A * rigid_multiplier
        rigid_I = wall_I * rigid_multiplier

        # Create rigid links
        ops.element(
            'elasticBeamColumn', rigid_left, node_left_wall, node_mid_left,
            rigid_A, rigid_E, rigid_I, self.state.linear_geom_transf
        )
        ops.element(
            'elasticBeamColumn', rigid_right, node_mid_right, node_right_wall,
            rigid_A, rigid_E, rigid_I, self.state.linear_geom_transf
        )

        
        # Create zero-length UFP element (vertical direction only)
        ops.element(
            'zeroLength', zero_length, node_mid_left, node_mid_right,
            '-mat', mat_ufp, '-dir', 2
        )
        
        self.logger.debug(f"Created UFP assembly for Story {story_num}, UFP {ufp_num}: "
                        f"rigid=({rigid_left},{rigid_right}), zero={zero_length}")