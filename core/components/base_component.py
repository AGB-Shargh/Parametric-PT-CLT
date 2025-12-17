# src/components/base_component.py
"""
Abstract base class for all structural components.
Ensures consistent interface and validation patterns.
"""
from abc import ABC, abstractmethod
from typing import Any
import logging

from core.model_state import ModelState
from core.tag_manager import TagGenerator


class BaseComponent(ABC):
    """Abstract base for all structural components."""
    
    def __init__(self, state: ModelState, tags: TagGenerator):
        """Initialize component with state and tag generator.
        
        Args:
            state: ModelState instance
            tags: TagGenerator instance
        """
        self.state = state
        self.tags = tags
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def create_nodes(self) -> None:
        """Create all nodes for this component."""
        pass
    
    @abstractmethod
    def create_elements(self) -> None:
        """Create all elements for this component."""
        pass
    
    def _validate_required_keys(self, data: dict, keys: list, context: str) -> None:
        """Validate required keys exist in data dictionary.
        
        Args:
            data: Dictionary to validate
            keys: Required keys
            context: Context string for error messages
            
        Raises:
            KeyError: If any required key missing
        """
        missing = [k for k in keys if k not in data]
        if missing:
            raise KeyError(f"{context} missing required keys: {missing}")
    
    def _validate_positive(self, value: float, name: str) -> None:
        """Validate value is positive.
        
        Args:
            value: Value to check
            name: Parameter name for error message
            
        Raises:
            ValueError: If value not positive
        """
        if value < 0:
            raise ValueError(f"{name} must be positive, got {value}")