# src/core/model_config.py
"""
Configuration loading and validation.
Replaces _load_config from RockingWallBuilding.
"""
import os
import yaml
import numpy as np
from typing import Dict, Any, Union
from pathlib import Path


class ConfigLoader:
    """Loads and validates model configuration."""
    
    @staticmethod
    def load(config_input: Union[Dict, str, Path]) -> Dict[str, Any]:
        """Load configuration from dict or YAML file.
        
        Args:
            config_input: Dictionary or path to YAML file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            FileNotFoundError: If YAML file missing
            ValueError: If configuration invalid
        """
        if isinstance(config_input, dict):
            config = config_input
        elif isinstance(config_input, (str, os.PathLike)):
            config = ConfigLoader._load_yaml(config_input)
        else:
            raise TypeError("config_input must be dict or file path")
        
        return ConfigLoader._validate(config)
    
    @staticmethod
    def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def _validate(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enrich configuration.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Validated and enriched configuration
            
        Raises:
            ValueError: If UFP configuration is invalid
        """
        # Derive n_stories from UFP Numbers
        ufp_numbers = config.get('UFP', {}).get('UFP Numbers', None)
        
        if ufp_numbers is None:
            raise ValueError("config['UFP']['UFP Numbers'] is required")
        
        if not isinstance(ufp_numbers, list):
            raise ValueError("config['UFP']['UFP Numbers'] must be a list")
        

        n_stories = len(config["building"]["Stories Heights"])
        config['n_stories'] = n_stories
              
        if len(ufp_numbers) != n_stories:
            raise ValueError(
                f"config['UFP']['UFP Numbers'] length ({len(ufp_numbers)}) "
                f"must equal number of stories: ({n_stories})"
            )
        
        ufp_height_ratios = config.get('UFP', {}).get('UFP Height Ratios', None)
        
        if ufp_height_ratios is None:
            raise ValueError("config['UFP']['UFP Height Ratios'] is required")
        
        if not isinstance(ufp_height_ratios, list):
            raise ValueError("config['UFP']['UFP Height Ratios'] must be a list")
        
        if len(ufp_height_ratios) != n_stories:
            raise ValueError(
                f"config['UFP']['UFP Height Ratios'] length ({len(ufp_height_ratios)}) "
                f"must equal number of stories: ({n_stories})"
            )
        
        # Each story's height ratios matches its UFP count
        for i, (n_ufp, height_ratios) in enumerate(zip(ufp_numbers, ufp_height_ratios), start=1):
            if not isinstance(height_ratios, list):
                raise ValueError(
                    f"Story {i}: UFP Height Ratios must be a list, got {type(height_ratios)}"
                )
            
            if len(height_ratios) != n_ufp:
                raise ValueError(
                    f"Story {i}: UFP Height Ratios length ({len(height_ratios)}) "
                    f"must equal UFP Numbers ({n_ufp})"
                )
            
            # Optional: Check that ratios are between 0 and 1
            if not all(0 < ratio < 1 for ratio in height_ratios):
                raise ValueError(
                    f"Story {i}: All UFP Height Ratios must be between 0 and 1, "
                    f"got {height_ratios}"
                )

            # Set defaults
            config.setdefault('model_type', 'Inelastic')
            config.setdefault('steel_elastic_module', 29000)
            
            # Compute bar area if diameter provided
            bar = config.setdefault('bar', {})
            if 'Diameter' in bar:
                bar['Area'] = 2 * (np.pi * (bar['Diameter'] / 2) ** 2)
            
            # Ensure diaphragm structure
            diaphragm = config.setdefault('diaphragm', {})
            elements = diaphragm.setdefault('Elements', {})
            shear_key = elements.setdefault('Shear Key', {})
            
            # Default shear key properties
            shear_key.setdefault('E', 200000 * 1e6)
            shear_key.setdefault('G', shear_key['E'] / 2.6)
            
            # Initialize story shear keys
            n_stories = config['n_stories']
            for story in range(1, n_stories + 1):
                shear_key.setdefault(f"Story {story}", {
                    "Thickness": 1.75,
                    "Width": 2.66,
                    "Length": 16.38
                })
            
            return config