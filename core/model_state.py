# src/core/model_state.py
"""
Model state container - holds all structural data without behavior.
Replaces the data storage aspects of RockingWallBuilding.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np


@dataclass
class ModelState:
    """Immutable structural model state."""
    
    # Configuration
    n_stories: int
    model_type: str
    steel_elastic_module: float
    
    # Geometric transforms
    linear_geom_transf: int = 1
    pdelta_geom_transf: int = 2
    
    # Computed properties
    wall_elevations: np.ndarray = field(default_factory=lambda: np.array([]))
    periods: List[float] = field(default_factory=list)
    
    # Component data dictionaries
    wall: Dict[str, Any] = field(default_factory=dict)
    UFP: Dict[str, Any] = field(default_factory=dict)
    bar: Dict[str, Any] = field(default_factory=dict)
    spring: Dict[str, Any] = field(default_factory=dict)
    diaphragm: Dict[str, Any] = field(default_factory=dict)
    leaning_columns: Dict[str, Any] = field(default_factory=dict)
    building: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)

    
    # Tag tracking
    tags: Dict[str, Dict] = field(default_factory=lambda: {"Nodes": {}, "Elements": {}})
    
    @property
    def left_pier(self) -> str:
        """Left pier identifier."""
        return f"Pier{1}"
    
    @property
    def right_pier(self) -> str:
        """Right pier identifier."""
        return f"Pier{2}"