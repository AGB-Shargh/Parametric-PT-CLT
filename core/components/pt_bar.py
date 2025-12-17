# src/components/pt_bar.py
"""
Post-tensioning bar component - handles PT bar nodes and elements.
"""
import openseespy.opensees as ops
from typing import Tuple

from core.components.base_component import BaseComponent
from core.model_state import ModelState
from core.tag_manager import TagGenerator
from core.material_factory import MaterialFactory
from core.constants import LARGE_NUMBER


class PTBar(BaseComponent):
    """Handles creation of post-tensioning bar nodes and elements."""

    def __init__(self, state: ModelState, tags: TagGenerator, materials: MaterialFactory):
        """Initialize PT bar component.
        
        Args:
            state: ModelState instance
            tags: TagGenerator instance
            materials: MaterialFactory instance
        """
        super().__init__(state, tags)
        self.materials = materials
        self._bar_config = state.bar
        self._wall_config = state.wall

    def create_nodes(self) -> None:
        """Generate PT bar nodes for each wall pier."""
        self._validate_required_keys(
            self._wall_config,
            ["Nodes Xs", "Nodes Ys", "PT Extender Length"],
            "Wall config"
        )
        
        extender_length = self._wall_config['PT Extender Length']
        self._validate_positive(extender_length, "PT Extender Length")
        
        self.tags.generate_node_tags(component='PT Bar')
        
        top_y = self._wall_config["Nodes Ys"][-1]
        base_y = self._wall_config["Nodes Ys"][0]
        
        for pier_idx, pier_side in enumerate(['Left Wall', 'Right Wall']):
            wall_x = self._wall_config["Nodes Xs"][pier_idx]
            self._create_pier_nodes(pier_side, wall_x, top_y, base_y, extender_length)

    def create_elements(self) -> None:
        """Create PT bar truss elements and rigid extenders."""
        self._validate_required_keys(
            self._bar_config,
            ["Area"],
            "Bar config"
        )
        
        if 'Bar MinMax Material' not in self.materials.material_archive:
            raise ValueError("Bar material must be defined before creating elements")
        
        self.tags.generate_element_tags(component='PT Bars')
        
        bar_area = self._bar_config['Area']
        mat_tag = self.materials.material_archive['Bar MinMax Material']
        top_elevation = f"Elevation {len(self.state.wall_elevations) - 1}"
        
        for pier_side in ['Left Wall', 'Right Wall']:
            for bar_side in ['Left', 'Right']:
                self._create_bar_and_extender(
                    pier_side, bar_side, top_elevation, bar_area, mat_tag
                )

    def _create_pier_nodes(self, pier_side: str, wall_x: float, 
                          top_y: float, base_y: float, extender_len: float) -> None:
        """Create PT bar nodes for one pier.
        
        Args:
            pier_side: 'Left Wall' or 'Right Wall'
            wall_x: Wall x-coordinate
            top_y: Top elevation
            base_y: Base elevation
            extender_len: Extender length
        """
        for bar_side in ['Left', 'Right']:
            # Calculate x-offset
            x_offset = -extender_len if bar_side == 'Left' else extender_len
            bar_x = wall_x + x_offset
            
            # Get node tags
            top_node = self._wall_config["Nodes"]["PT Bar"][pier_side][f"Top {bar_side}"]
            base_node = self._wall_config["Nodes"]["PT Bar"][pier_side][f"Bottom {bar_side}"]
            
            # Create nodes
            ops.node(top_node, bar_x, top_y)
            ops.node(base_node, bar_x, base_y)
            ops.fix(base_node, 1, 1, 1)
            
            self.logger.debug(f"Created PT nodes for {pier_side} {bar_side}: "
                            f"top={top_node}, base={base_node}")

    def _create_bar_and_extender(self, pier_side: str, bar_side: str, 
                                top_elevation: str, bar_area: float, 
                                mat_tag: int) -> None:
        """Create PT bar and extender for one location.
        
        Args:
            pier_side: 'Left Wall' or 'Right Wall'
            bar_side: 'Left' or 'Right'
            top_elevation: Top elevation key
            bar_area: Bar cross-sectional area
            mat_tag: Material tag
        """
        # Get nodes
        wall_top_node = self._wall_config["Nodes"][top_elevation][pier_side]
        pt_top_node = self._wall_config["Nodes"]["PT Bar"][pier_side][f"Top {bar_side}"]
        pt_base_node = self._wall_config["Nodes"]["PT Bar"][pier_side][f"Bottom {bar_side}"]
        
        # Get element tags
        pt_tag = self._bar_config["Elements"]["PT Bars"][pier_side][bar_side]
        extender_tag = self._bar_config["Elements"]["PT Extenders"][pier_side][bar_side]
        
        # Validate
        if not all([wall_top_node, pt_top_node, pt_base_node, pt_tag, extender_tag]):
            raise KeyError(f"Missing node or element tag for {pier_side} {bar_side}")
        
        # Create corotational truss (PT bar)
        ops.element('corotTruss', pt_tag, pt_base_node, pt_top_node, bar_area, mat_tag)
        
        # Create rigid extender
        ops.element(
            'elasticBeamColumn', extender_tag, wall_top_node, pt_top_node,
            LARGE_NUMBER, LARGE_NUMBER, LARGE_NUMBER, self.state.linear_geom_transf
        )
        
        self.logger.debug(f"Created PT bar {pt_tag} and extender {extender_tag} "
                        f"for {pier_side} {bar_side}")