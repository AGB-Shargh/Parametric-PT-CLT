# src/components/diaphragm.py
"""
Diaphragm component - rigid floor diaphragms and shear keys.
"""
import openseespy.opensees as ops
from typing import Dict, Tuple

from core.components.base_component import BaseComponent
from core.model_state import ModelState
from core.tag_manager import TagGenerator
from core.material_factory import MaterialFactory
from core.constants import LARGE_NUMBER, NEGLIGIBLE_MASS


class Diaphragm(BaseComponent):
    """Handles creation of diaphragm nodes, elements, and shear keys."""

    def __init__(self, state: ModelState, tags: TagGenerator, materials: MaterialFactory):
        """Initialize diaphragm component.
        
        Args:
            state: ModelState instance
            tags: TagGenerator instance
            materials: MaterialFactory instance
        """
        super().__init__(state, tags)
        self.materials = materials
        self._diaphragm_config = state.diaphragm
        self._wall_config = state.wall
        self._leaning_config = state.leaning_columns

    def create_nodes(self) -> None:
        """Generate diaphragm nodes at each floor level."""
        self._validate_required_keys(
            self._wall_config,
            ["Nodes Xs", "Nodes Ys", "Wall Length"],
            "Wall config"
        )
        
        if self.state.n_stories < 1:
            raise ValueError(f"Invalid n_stories: {self.state.n_stories}")
        
        wall_length = self._wall_config["Wall Length"]
        self._validate_positive(wall_length, "Wall length")
        
        self.tags.generate_node_tags(component='Diaphragms')
        self._diaphragm_config.setdefault("Nodes", {})
        
        for floor in range(1, self.state.n_stories + 1):
            y = self._wall_config["Nodes Ys"][floor]
            x_positions = self._get_floor_x_positions(wall_length)
            
            nodes = self._diaphragm_config["Nodes"][f"Elevation {floor}"]
            
            for key, x in x_positions.items():
                node_tag = nodes[key]
                if not node_tag:
                    raise ValueError(f"Node tag for {key} at Elevation {floor} not generated")
                
                ops.node(node_tag, x, y, '-mass', NEGLIGIBLE_MASS, NEGLIGIBLE_MASS, NEGLIGIBLE_MASS)
                
                self.logger.debug(f"Created diaphragm node {node_tag} at "
                                f"Elevation {floor} ({key}): ({x:.2f}, {y:.2f})")

    def create_elements(self) -> None:
        """Create rigid diaphragm elements and rotational hinges."""
        if 'Soft Material' not in self.materials.material_archive:
            raise ValueError("Soft Material must be defined before creating elements")
        if 'Hard Material' not in self.materials.material_archive:
            raise ValueError("Hard Material must be defined before creating elements")
        
        self.tags.generate_element_tags(component='Diaphragms')
        
        for floor in range(1, self.state.n_stories + 1):
            self._create_floor_elements(floor)

    def define_shear_keys(self) -> None:
        """Define shear key elements connecting diaphragm to walls."""
        self.materials.define_shear_key_mat()
        self.tags.generate_element_tags(component='Shear Keys')
        
        soft_mat = self.materials.material_archive.get('Soft Material')
        if not soft_mat:
            raise ValueError("Soft Material not defined")
        
        # Shear keys only at floor levels (Stories 1 and 2)
        for floor in range(1, self.state.n_stories + 1):
            for position in ["Left Wall", "Right Wall"]:
                self._create_shear_key(floor, position, soft_mat)

    def _get_floor_x_positions(self, wall_length: float) -> Dict[str, float]:
        """Calculate x-positions for diaphragm nodes.
        
        Args:
            wall_length: Wall length
            
        Returns:
            Dictionary mapping position names to x-coordinates
        """
        x_left, x_right = self._wall_config["Nodes Xs"]
        
        return {
            "Left Wall": x_left,
            "Right Wall": x_right,
            "Leaning Column": x_right + wall_length
        }

    def _create_floor_elements(self, floor: int) -> None:
        """Create rigid diaphragm spans and rotational hinges for one floor.
        
        Args:
            floor: Floor number (1-based)
        """
        nodes = self._diaphragm_config["Nodes"].get(f"Elevation {floor}", {})
        if not nodes:
            raise KeyError(f"Diaphragm nodes for Elevation {floor} not found")
        
        # Create rigid spans
        self._create_rigid_spans(floor, nodes)
        
        # Create rotational hinges to leaning column
        self._create_rotational_hinges(floor, nodes)

    def _create_rigid_spans(self, floor: int, nodes: Dict[str, int]) -> None:
        """Create rigid diaphragm span elements.
        
        Args:
            floor: Floor number
            nodes: Node dictionary for this floor
        """
        span_map = {
            "Wall Span": ("Left Wall", "Right Wall"),
            "Leaning Span": ("Right Wall", "Leaning Column")
        }
        
        rigid_elements = self._diaphragm_config["Elements"]["Rigid Elements"][f"Story {floor}"]
        
        for span_name, (start_key, end_key) in span_map.items():
            if start_key not in nodes or end_key not in nodes:
                raise KeyError(f"Node {start_key} or {end_key} missing for "
                             f"{span_name} at Elevation {floor}")
            
            elem_tag = rigid_elements[span_name]
            start_node = nodes[start_key]
            end_node = nodes[end_key]
            
            ops.element(
                'elasticBeamColumn', elem_tag, start_node, end_node,
                LARGE_NUMBER, LARGE_NUMBER, LARGE_NUMBER, 
                self.state.linear_geom_transf
            )
            
            self.logger.debug(f"Created rigid diaphragm span {elem_tag} for "
                            f"{span_name} at Elevation {floor}")

    def _create_rotational_hinges(self, floor: int, nodes: Dict[str, int]) -> None:
        """Create rotational hinges connecting diaphragm to leaning column.
        
        Args:
            floor: Floor number
            nodes: Node dictionary for this floor
        """
        soft_mat = self.materials.material_archive['Soft Material']
        hard_mat = self.materials.material_archive['Hard Material']
        
        diaphragm_node = nodes.get("Leaning Column")
        if not diaphragm_node:
            raise KeyError(f"Leaning Column diaphragm node missing for Elevation {floor}")
        
        hinge_elements = self._diaphragm_config["Elements"]["Rotational Hinges"][f"Story {floor}"]
        leaning_nodes = self._leaning_config["Nodes"].get(f"Elevation {floor}", {})
        
        for position in ["Upper", "Lower"]:
            col_node = leaning_nodes.get(position)
            if not col_node:
                raise KeyError(f"Leaning column {position} node missing for Elevation {floor}")
            
            elem_tag = hinge_elements[position]
            
            # Zero-length: hard in x and y (tied), soft in rotation (hinge)
            ops.element(
                'zeroLength', elem_tag, diaphragm_node, col_node,
                '-mat', hard_mat, hard_mat, soft_mat,
                '-dir', 1, 2, 3
            )
            
            self.logger.debug(f"Created rotational hinge {elem_tag} for "
                            f"Elevation {floor} ({position})")

    def _create_shear_key(self, floor: int, position: str, soft_mat: int) -> None:
        """Create shear key element at one location.
        
        Args:
            floor: Floor number
            position: 'Left Wall' or 'Right Wall'
            soft_mat: Soft material tag
        """
        mat_tag = self.materials.material_archive.get(f'Story {floor} Shear Key Material')
        if not mat_tag:
            raise KeyError(f"Story {floor} Shear Key Material not found")
        
        # Get nodes
        wall_node = self._wall_config["Nodes"].get(f"Elevation {floor}", {}).get(position)
        diaphragm_node = self._diaphragm_config["Nodes"].get(f"Elevation {floor}", {}).get(position)
        elem_tag = self._diaphragm_config["Elements"]["Shear Keys"][f"Story {floor}"].get(position)
        
        if not all([wall_node, diaphragm_node, elem_tag]):
            raise KeyError(f"Missing node or element tag for {position} at Elevation {floor}")
        
        # Zero-length: stiff in x (shear), soft in y and rotation
        ops.element(
            'zeroLength', elem_tag, diaphragm_node, wall_node,
            '-mat', mat_tag, soft_mat, soft_mat,
            '-dir', 1, 2, 3
        )
        
        self.logger.debug(f"Created shear key {elem_tag} for Elevation {floor} ({position})")