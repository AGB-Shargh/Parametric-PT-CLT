# src/components/leaning_column.py
"""
Leaning column component - P-Delta gravity columns.
"""
import openseespy.opensees as ops

from core.components.base_component import BaseComponent
from core.model_state import ModelState
from core.tag_manager import TagGenerator
from core.constants import NEGLIGIBLE_MASS

class LeaningColumn(BaseComponent):
    """Handles creation of leaning column nodes and elements."""
    
    def __init__(self, state: ModelState, tags: TagGenerator):
        """Initialize leaning column component.
        
        Args:
            state: ModelState instance
            tags: TagGenerator instance
        """
        super().__init__(state, tags)
        self._leaning_config = state.leaning_columns
        self._wall_config = state.wall
    
    def create_nodes(self) -> None:
        """Generate leaning column nodes (base + upper/lower per story)."""
        self._validate_required_keys(
            self._wall_config,
            ["Nodes Xs", "Nodes Ys", "Wall Length"],
            "Wall config"
        )
        
        self.tags.generate_node_tags(component='Leaning Columns')
        
        # Calculate leaning column x-position (beyond right wall)
        x = (self._wall_config["Nodes Xs"][1] + 
             self._wall_config["Wall Length"])
        
        # Base node
        base_node = self._leaning_config["Nodes"]["Elevation 0"]
        base_y = self._wall_config["Nodes Ys"][0]
        ops.node(base_node, x, base_y)
        
        self.logger.debug(f"Created leaning column base node {base_node} at "
                        f"({x:.2f}, {base_y:.2f})")
        
        # Upper and lower nodes per story
        for story in range(1, self.state.n_stories + 1):
            y = self._wall_config["Nodes Ys"][story]
            elevation_nodes = self._leaning_config["Nodes"][f"Elevation {story}"]
            
            for position in ['Upper', 'Lower']:
                node_tag = elevation_nodes[position]
                ops.node(node_tag, x, y, '-mass', NEGLIGIBLE_MASS, NEGLIGIBLE_MASS, NEGLIGIBLE_MASS)
                
                self.logger.debug(f"Created leaning column node {node_tag} at "
                                f"Elevation {story} ({position})")
    
    def create_elements(self) -> None:
        """Create leaning column elements with P-Delta transformation."""
        self._validate_required_keys(
            self._leaning_config,
            ['Section Area', 'Section Iz', 'Elastic Modulus'],
            "Leaning column properties"
        )
        
        self.tags.generate_element_tags(component='Leaning Columns')
        
        area = self._leaning_config['Section Area']
        iz = self._leaning_config['Section Iz']
        elastic_mod = self._leaning_config['Elastic Modulus']
        
        # Fix base node
        base_node = self._leaning_config["Nodes"]["Elevation 0"]
        ops.fix(base_node, 1, 1, 0)  # Fixed in x and y, free in rotation
        
        # Track lower node (starts at base)
        lower_node = base_node
        
        # Create elements story by story
        for story in range(1, self.state.n_stories + 1):
            upper_node = self._leaning_config["Nodes"][f"Elevation {story}"]['Upper']
            elem_tag = self._leaning_config["Elements"][f"Story {story}"]
            
            # Create elastic beam-column with P-Delta
            ops.element(
                "elasticBeamColumn",
                elem_tag,
                lower_node, upper_node,
                area, elastic_mod, iz,
                self.state.pdelta_geom_transf
            )
            
            self.logger.debug(f"Created leaning column element {elem_tag} for "
                            f"Story {story}: {lower_node} -> {upper_node}")
            
            # Update lower node for next story
            lower_node = self._leaning_config["Nodes"][f"Elevation {story}"]['Lower']