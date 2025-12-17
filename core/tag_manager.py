# src/core/tag_manager.py
"""
Tag management for nodes and elements.
Refactored to work with ModelState.
"""
from typing import Dict
from enum import Enum
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

from core.model_state import ModelState


class TagRanges(Enum):
    """Tag range allocations for different components."""
    # Node tag ranges
    WALL_NODES = (1000, 2999)
    UFP_NODES = (3000, 4999)
    LEANING_COLUMN_NODES = (5000, 5999)
    DIAPHRAGM_NODES = (6000, 6999)
    ROCKING_NODES = (7000, 7999)
    FOUNDATION_NODES = (8000, 8999)

    # Element tag ranges
    WALL_ELEMENTS = (1000, 2999)
    UFP_RIGID_ELEMENTS = (3000, 3999)
    UFP_ZERO_LENGTH = (4000, 4999)
    ROCKING_ZERO_LENGTH = (5000, 5999)
    ROCKING_RIGID_ELEMENTS = (6000, 6999)
    PT_BAR_ELEMENTS = (7000, 7099)
    LEANING_COLUMN_ELEMENTS = (7100, 7499)
    DIAPHRAGM_ELEMENTS = (7500, 7999)


class TagGenerator:
    """Generates unique tags for nodes and elements."""

    def __init__(self, state: ModelState):
        """Initialize with model state.
        
        Args:
            state: ModelState instance
        """
        self.state = state
        self.logger = logging.getLogger(__name__)
        self._counters: Dict[str, int] = {}
        self._tag_registry: List[Dict[str, Any]] = []
        self.reset_tags()

    def reset_tags(self) -> None:
        """Reset all tag counters to starting values."""
        self._counters = {
            range_name: TagRanges[range_name].value[0]
            for range_name in TagRanges.__members__
        }
        self._tag_registry = []

    def generate_node_tags(self, component: str) -> None:
        """Generate node tags for specified component.
        
        Args:
            component: Component name
            
        Raises:
            ValueError: If component name invalid
        """
        valid_components = [
            "Wall", "PT Bar", "UFP", "Rocking Springs", "Foundation Springs",
            "Leaning Columns", "Diaphragms"
        ]
        
        if component not in valid_components:
            raise ValueError(f"Invalid component: {component}. Must be one of {valid_components}")
        
        generator_map = {
            "Wall": self._generate_wall_nodes,
            "PT Bar": self._generate_pt_bar_nodes,
            "UFP": self._generate_ufp_nodes,
            "Rocking Springs": self._generate_rocking_nodes,
            "Leaning Columns": self._generate_leaning_nodes,
            "Diaphragms": self._generate_diaphragm_nodes,
        }
        
        generator_map[component]()

    def generate_element_tags(self, component: str) -> None:
        """Generate element tags for specified component.
        
        Args:
            component: Component name
            
        Raises:
            ValueError: If component name invalid
        """
        valid_components = [
            "UFP", "Wall", "Rocking Springs", "Foundation Springs",
            "PT Bars", "Leaning Columns", "Diaphragms", "Shear Keys"
        ]
        
        if component not in valid_components:
            raise ValueError(f"Invalid component: {component}. Must be one of {valid_components}")
        
        generator_map = {
            "UFP": self._generate_ufp_elements,
            "Wall": self._generate_wall_elements,
            "Rocking Springs": self._generate_rocking_elements,
            "Foundation Springs": self._generate_foundation_elements,
            "PT Bars": self._generate_pt_bar_elements,
            "Leaning Columns": self._generate_leaning_elements,
            "Diaphragms": self._generate_diaphragm_elements,
            "Shear Keys": self._generate_shear_key_elements,
        }
        
        generator_map[component]()

    def _generate_tag(self, range_name: str, description: str = None) -> int:
        """Generate unique tag from specified range and track it.
        
        Args:
            range_name: Tag range name
            description: Optional human-readable description
            
        Returns:
            Generated tag
            
        Raises:
            ValueError: If tag overflow occurs
        """
        start, end = TagRanges[range_name].value
        tag = self._counters[range_name]
        
        if tag > end:
            raise ValueError(f"Tag overflow in range {range_name}: {start}-{end}")
        
        # Track the tag
        self._tag_registry.append({
            'tag': tag,
            'range_name': range_name,
            'range_start': start,
            'range_end': end,
            'description': description,
            'counter_value': self._counters[range_name] - start
        })
        
        self._counters[range_name] += 1
        return tag

    # ==================================================================
    # NODES AND ELEMENTS TAG GENERATORS
    # ==================================================================

    # Node generation methods
    def _generate_wall_nodes(self) -> None:
        """Generate wall node tags."""
        wall = self.state.wall
        wall.setdefault("Nodes", {})
        
        # Fixed base
        wall["Nodes"]["Fixed Base"] = {
            "Left Wall": self._generate_tag("WALL_NODES"),
            "Right Wall": self._generate_tag("WALL_NODES")
        }
        
        # Elevation nodes
        n_elevations = len(self.state.wall_elevations) if len(self.state.wall_elevations) > 0 else self.state.n_stories + 1
        for elev in range(n_elevations):
            wall["Nodes"][f"Elevation {elev}"] = {
                "Left Wall": self._generate_tag("WALL_NODES"),
                "Right Wall": self._generate_tag("WALL_NODES")
            }

    def _generate_pt_bar_nodes(self) -> None:
        """Generate PT bar node tags."""
        wall = self.state.wall
        wall["Nodes"].setdefault("PT Bar", {})
        
        for side in ["Left Wall", "Right Wall"]:
            wall["Nodes"]["PT Bar"][side] = {
                f"{pos} {lr}": self._generate_tag("WALL_NODES")
                for pos in ["Top", "Bottom"]
                for lr in ["Left", "Right"]
            }

    def _generate_ufp_nodes(self) -> None:
        """Generate UFP node tags."""
        ufp = self.state.UFP
        ufp.setdefault("Nodes", {})
        
        for story in range(1, self.state.n_stories + 1):
            ufp_count = ufp['UFP Numbers'][story - 1]
            ufp["Nodes"][f"Story {story}"] = {
                f"{loc} {i+1}": self._generate_tag("UFP_NODES")
                for i in range(ufp_count)
                for loc in ["Left Wall", "Right Wall", "Middle Left", "Middle Right"]
            }

    def _generate_rocking_nodes(self) -> None:
        """Generate rocking spring node tags."""
        spring = self.state.spring
        spring.setdefault("Rocking", {})
        spring["Rocking"].setdefault("Nodes", {})
        
        n_springs = spring['Number of Rocking Springs']
        
        for side in ["Left Wall", "Right Wall"]:
            spring["Rocking"]["Nodes"][side] = {
                f"{pos} {i}": {"Tag": self._generate_tag("ROCKING_NODES")}
                for pos in ["Base", "Top"]
                for i in range(n_springs)
            }

    def _generate_leaning_nodes(self) -> None:
        """Generate leaning column node tags."""
        leaning = self.state.leaning_columns
        leaning.setdefault("Nodes", {})
        
        # Base node
        leaning["Nodes"]["Elevation 0"] = self._generate_tag("LEANING_COLUMN_NODES")
        
        # Floor nodes (upper and lower)
        for story in range(1, self.state.n_stories + 1):
            leaning["Nodes"][f"Elevation {story}"] = {
                pos: self._generate_tag("LEANING_COLUMN_NODES")
                for pos in ['Upper', 'Lower']
            }

    def _generate_diaphragm_nodes(self) -> None:
        """Generate diaphragm node tags."""
        diaphragm = self.state.diaphragm
        diaphragm.setdefault("Nodes", {})
        
        for story in range(1, self.state.n_stories + 1):
            diaphragm["Nodes"][f"Elevation {story}"] = {
                side: self._generate_tag("DIAPHRAGM_NODES")
                for side in ["Left Wall", "Right Wall", "Leaning Column"]
            }

    # Element generation methods
    def _generate_wall_elements(self) -> None:
        """Generate wall element tags."""
        wall = self.state.wall
        wall.setdefault("Elements", {})
        
        for story in range(1, self.state.n_stories + 1):
            ufp_count = self.state.UFP['UFP Numbers'][story - 1]
            wall["Elements"][f"Story {story}"] = {
                f"{side} Segment {seg}": self._generate_tag("WALL_ELEMENTS")
                for side in ["Left Wall", "Right Wall"]
                for seg in range(1, ufp_count + 2)
            }
        
        # Extension if present
        if wall.get("Wall Extension", 0) > 0:
            wall["Elements"]["Wall Extension"] = {
                side: self._generate_tag("WALL_ELEMENTS")
                for side in ["Left Wall", "Right Wall"]
            }

    def _generate_ufp_elements(self) -> None:
        """Generate UFP element tags."""
        ufp = self.state.UFP
        ufp.setdefault("Elements", {})
        ufp["Elements"].setdefault("Rigid Elements", {})
        ufp["Elements"].setdefault("ZeroLength Elements", {})
        
        for story in range(1, self.state.n_stories + 1):
            ufp_count = ufp['UFP Numbers'][story - 1]
            
            ufp["Elements"]["Rigid Elements"][f"Story {story}"] = {
                f"UFP Number {i}": {
                    'Left': self._generate_tag("UFP_RIGID_ELEMENTS"),
                    'Right': self._generate_tag("UFP_RIGID_ELEMENTS")
                }
                for i in range(1, ufp_count + 1)
            }
            
            ufp["Elements"]["ZeroLength Elements"][f"Story {story}"] = {
                f"UFP Number {i}": self._generate_tag("UFP_ZERO_LENGTH")
                for i in range(1, ufp_count + 1)
            }

    def _generate_rocking_elements(self) -> None:
        """Generate rocking spring element tags."""
        spring = self.state.spring
        spring.setdefault("Elements", {})
        spring["Elements"].setdefault("Rigid Elements", {})
        spring["Elements"].setdefault("ZeroLength Elements", {})
        
        n_springs = spring['Number of Rocking Springs']
        
        for i in range(n_springs):
            spring["Elements"]["Rigid Elements"][f"Rocking Number {i}"] = {
                side: self._generate_tag("ROCKING_RIGID_ELEMENTS")
                for side in ['Left Wall', 'Right Wall']
            }
            spring["Elements"]["ZeroLength Elements"][f"Rocking Number {i}"] = {
                side: self._generate_tag("ROCKING_ZERO_LENGTH")
                for side in ['Left Wall', 'Right Wall']
            }
        
        # Shear elements
        for i in range(4):
            spring["Elements"][f"Shear Element {i}"] = self._generate_tag("ROCKING_RIGID_ELEMENTS")

    def _generate_foundation_elements(self) -> None:
        """Generate foundation spring element tags."""
        spring = self.state.spring
        spring["Elements"].setdefault("Foundation Springs", {})
        
        n_springs = spring['Number of Rocking Springs']
        
        for i in range(n_springs):
            spring["Elements"]["Foundation Springs"][f"Number {i}"] = {
                side: self._generate_tag("FOUNDATION_ELEMENTS")
                for side in ['Left Wall', 'Right Wall']
            }

    def _generate_pt_bar_elements(self) -> None:
        """Generate PT bar element tags."""
        bar = self.state.bar
        bar.setdefault("Elements", {})
        bar["Elements"].setdefault("PT Bars", {})
        bar["Elements"].setdefault("PT Extenders", {})
        
        for pier_side in ["Left Wall", "Right Wall"]:
            bar["Elements"]["PT Bars"][pier_side] = {
                bar_side: self._generate_tag("PT_BAR_ELEMENTS")
                for bar_side in ["Left", "Right"]
            }
            bar["Elements"]["PT Extenders"][pier_side] = {
                bar_side: self._generate_tag("PT_BAR_ELEMENTS")
                for bar_side in ["Left", "Right"]
            }

    def _generate_leaning_elements(self) -> None:
        """Generate leaning column element tags."""
        leaning = self.state.leaning_columns
        leaning.setdefault("Elements", {})
        
        for story in range(1, self.state.n_stories + 1):
            leaning["Elements"][f"Story {story}"] = self._generate_tag("LEANING_COLUMN_ELEMENTS")

    def _generate_diaphragm_elements(self) -> None:
        """Generate diaphragm element tags."""
        diaphragm = self.state.diaphragm
        diaphragm.setdefault("Elements", {})
        diaphragm["Elements"].setdefault("Rigid Elements", {})
        diaphragm["Elements"].setdefault("Rotational Hinges", {})
        
        for story in range(1, self.state.n_stories + 1):
            diaphragm["Elements"]["Rigid Elements"][f"Story {story}"] = {
                side: self._generate_tag("DIAPHRAGM_ELEMENTS")
                for side in ["Wall Span", "Leaning Span"]
            }
            diaphragm["Elements"]["Rotational Hinges"][f"Story {story}"] = {
                pos: self._generate_tag("DIAPHRAGM_ELEMENTS")
                for pos in ["Upper", "Lower"]
            }

    def _generate_shear_key_elements(self) -> None:
        """Generate shear key element tags."""
        diaphragm = self.state.diaphragm
        diaphragm["Elements"].setdefault("Shear Keys", {})
        
        for story in range(1, self.state.n_stories + 1):
            diaphragm["Elements"]["Shear Keys"][f"Story {story}"] = {
                side: self._generate_tag("DIAPHRAGM_ELEMENTS")
                for side in ["Left Wall", "Right Wall"]
            }


    # ==================================================================
    # TAG TRACKING AND EXPORT
    # ==================================================================

    def get_tag_registry(self) -> pd.DataFrame:
        """Get all generated tags as a DataFrame.
        
        Returns:
            DataFrame with columns: tag, type, category, path, description, range_name
        """
        if not self._tag_registry:
            self.logger.warning("No tags have been generated yet")
            return pd.DataFrame()
        
        return pd.DataFrame(self._tag_registry)

    def save_tag_documentation(self, output_dir: str = ".", 
                               formats: List[str] = ['json', 'csv', 'markdown']) -> Dict[str, Path]:
        """Save comprehensive tag documentation.
        
        Args:
            output_dir: Directory to save files
            formats: List of output formats ('json', 'csv', 'markdown', 'excel')
            
        Returns:
            Dictionary mapping format names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Build hierarchical structure
        tag_structure = self._build_tag_hierarchy()
        flat_tags = self._flatten_tag_hierarchy(tag_structure)
        
        # Save JSON (hierarchical structure)
        if 'json' in formats:
            json_file = output_path / 'tag_documentation.json'
            with open(json_file, 'w') as f:
                json.dump({
                    'hierarchical': tag_structure,
                    'flat': flat_tags,
                    'summary': self._get_tag_summary()
                }, f, indent=2)
            saved_files['json'] = json_file
            self.logger.info(f"Saved JSON documentation: {json_file}")
        
        # Save CSV (flat table)
        if 'csv' in formats:
            csv_file = output_path / 'tag_documentation.csv'
            df = pd.DataFrame(flat_tags)
            df.to_csv(csv_file, index=False)
            saved_files['csv'] = csv_file
            self.logger.info(f"Saved CSV documentation: {csv_file}")
        
        # Save Markdown (human-readable)
        if 'markdown' in formats:
            md_file = output_path / 'tag_documentation.md'
            md_content = self._generate_markdown_documentation(tag_structure, flat_tags)
            with open(md_file, 'w') as f:
                f.write(md_content)
            saved_files['markdown'] = md_file
            self.logger.info(f"Saved Markdown documentation: {md_file}")
        
        return saved_files

    def _build_tag_hierarchy(self) -> Dict[str, Any]:
        """Build hierarchical structure of all tags from ModelState.
        
        Returns:
            Nested dictionary representing tag hierarchy
        """
        hierarchy = {
            'Wall': {'Nodes': {}, 'Elements': {}},
            'UFP': {'Nodes': {}, 'Elements': {}},
            'PT_Bar': {'Nodes': {}, 'Elements': {}},
            'Rocking_Spring': {'Nodes': {}, 'Elements': {}},
            'Leaning_Column': {'Nodes': {}, 'Elements': {}},
            'Diaphragm': {'Nodes': {}, 'Elements': {}}
        }
        
        # Extract Wall tags
        self._extract_nested_tags(
            self.state.wall.get('Nodes', {}),
            hierarchy['Wall']['Nodes'],
            'Wall', 'Nodes'
        )
        self._extract_nested_tags(
            self.state.wall.get('Elements', {}),
            hierarchy['Wall']['Elements'],
            'Wall', 'Elements'
        )
        
        # Extract UFP tags
        self._extract_nested_tags(
            self.state.UFP.get('Nodes', {}),
            hierarchy['UFP']['Nodes'],
            'UFP', 'Nodes'
        )
        self._extract_nested_tags(
            self.state.UFP.get('Elements', {}),
            hierarchy['UFP']['Elements'],
            'UFP', 'Elements'
        )
        
        # Extract PT Bar tags
        if 'PT Bar' in self.state.wall.get('Nodes', {}):
            self._extract_nested_tags(
                self.state.wall['Nodes']['PT Bar'],
                hierarchy['PT_Bar']['Nodes'],
                'PT_Bar', 'Nodes'
            )
        self._extract_nested_tags(
            self.state.bar.get('Elements', {}),
            hierarchy['PT_Bar']['Elements'],
            'PT_Bar', 'Elements'
        )
        
        # Extract Rocking Spring tags
        self._extract_nested_tags(
            self.state.spring.get('Rocking', {}).get('Nodes', {}),
            hierarchy['Rocking_Spring']['Nodes'],
            'Rocking_Spring', 'Nodes'
        )
        self._extract_nested_tags(
            self.state.spring.get('Elements', {}),
            hierarchy['Rocking_Spring']['Elements'],
            'Rocking_Spring', 'Elements'
        )
        
        # Extract Leaning Column tags
        self._extract_nested_tags(
            self.state.leaning_columns.get('Nodes', {}),
            hierarchy['Leaning_Column']['Nodes'],
            'Leaning_Column', 'Nodes'
        )
        self._extract_nested_tags(
            self.state.leaning_columns.get('Elements', {}),
            hierarchy['Leaning_Column']['Elements'],
            'Leaning_Column', 'Elements'
        )
        
        # Extract Diaphragm tags
        self._extract_nested_tags(
            self.state.diaphragm.get('Nodes', {}),
            hierarchy['Diaphragm']['Nodes'],
            'Diaphragm', 'Nodes'
        )
        self._extract_nested_tags(
            self.state.diaphragm.get('Elements', {}),
            hierarchy['Diaphragm']['Elements'],
            'Diaphragm', 'Elements'
        )
        
        return hierarchy

    def _extract_nested_tags(self, source: Dict, target: Dict, 
                            component: str, category: str, path: List[str] = None):
        """Recursively extract tags from nested dictionaries.
        
        Args:
            source: Source dictionary from ModelState
            target: Target dictionary for hierarchy
            component: Component name (Wall, UFP, etc.)
            category: Category (Nodes or Elements)
            path: Current path in hierarchy
        """
        if path is None:
            path = []
        
        for key, value in source.items():
            current_path = path + [key]
            
            if isinstance(value, int):
                # Found a tag
                target[key] = {
                    'tag': value,
                    'path': ' -> '.join([component, category] + current_path),
                    'component': component,
                    'category': category
                }
            elif isinstance(value, dict):
                # Recurse into nested dict
                target[key] = {}
                self._extract_nested_tags(
                    value, target[key], component, category, current_path
                )

    def _flatten_tag_hierarchy(self, hierarchy: Dict) -> List[Dict[str, Any]]:
        """Flatten hierarchical tag structure to list of records.
        
        Args:
            hierarchy: Hierarchical tag structure
            
        Returns:
            List of flat tag records
        """
        flat_list = []
        
        def recurse(node, path=[]):
            if isinstance(node, dict):
                if 'tag' in node:
                    # This is a leaf node with a tag
                    flat_list.append({
                        'tag': node['tag'],
                        'component': node['component'],
                        'category': node['category'],
                        'path': node['path'],
                        'description': path[-1] if path else 'Unknown'                 
                    })
                else:
                    # Recurse into nested structure
                    for key, value in node.items():
                        recurse(value, path + [key])
        
        recurse(hierarchy)
        
        # Sort by tag number
        flat_list.sort(key=lambda x: x['tag'])
        
        return flat_list

    def _get_tag_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all tags.
        
        Returns:
            Dictionary with summary information
        """
        flat_tags = self._flatten_tag_hierarchy(self._build_tag_hierarchy())
        
        if not flat_tags:
            return {'total_tags': 0}
        
        df = pd.DataFrame(flat_tags)
        
        summary = {
            'total_tags': len(df),
            'total_nodes': len(df[df['category'] == 'Nodes']),
            'total_elements': len(df[df['category'] == 'Elements']),
            'tags_by_component': df['component'].value_counts().to_dict(),
            'tags_by_category': df['category'].value_counts().to_dict(),
            'tag_range': {
                'min': int(df['tag'].min()),
                'max': int(df['tag'].max())
            }
        }
        
        return summary

    def _generate_markdown_documentation(self, hierarchy: Dict, 
                                        flat_tags: List[Dict]) -> str:
        """Generate Markdown documentation.
        
        Args:
            hierarchy: Hierarchical tag structure
            flat_tags: Flat list of tags
            
        Returns:
            Markdown formatted string
        """
        md = "# Model Tag Documentation\n\n"
        
        # Summary
        summary = self._get_tag_summary()
        md += "## Summary\n\n"
        md += f"- **Total Tags**: {summary['total_tags']}\n"
        md += f"- **Total Nodes**: {summary['total_nodes']}\n"
        md += f"- **Total Elements**: {summary['total_elements']}\n"
        md += f"- **Tag Range**: {summary['tag_range']['min']} - {summary['tag_range']['max']}\n\n"
        
        # Tags by component
        md += "## Tags by Component\n\n"
        for component, count in summary['tags_by_component'].items():
            md += f"- **{component}**: {count} tags\n"
        md += "\n"
        
        # Detailed listing by component
        df = pd.DataFrame(flat_tags)
        
        for component in df['component'].unique():
            md += f"## {component}\n\n"
            
            component_df = df[df['component'] == component]
            
            for category in ['Nodes', 'Elements']:
                category_df = component_df[component_df['category'] == category]
                
                if not category_df.empty:
                    md += f"### {category}\n\n"
                    md += "| Tag | Description | Full Path |\n"
                    md += "|-----|-------------|----------|\n"
                    
                    for _, row in category_df.iterrows():
                        md += f"| {row['tag']} | {row['description']} | {row['path']} |\n"
                    
                    md += "\n"
        
        return md

    def print_tag_summary(self) -> None:
        """Print a summary of all generated tags to console."""
        summary = self._get_tag_summary()
        
        print("\n" + "="*60)
        print("MODEL TAG SUMMARY")
        print("="*60)
        print(f"Total Tags Generated: {summary['total_tags']}")
        print(f"  - Nodes: {summary['total_nodes']}")
        print(f"  - Elements: {summary['total_elements']}")
        print(f"\nTag Range: {summary['tag_range']['min']} - {summary['tag_range']['max']}")
        
        print("\nTags by Component:")
        for component, count in summary['tags_by_component'].items():
            print(f"  {component:20s}: {count:4d} tags")
        
        print("="*60 + "\n")