# src/core/analysis_runner.py
"""
Handles static and dynamic analysis execution.
Replaces analysis methods from RockingWallBuilding.
"""
import logging
import numpy as np
import openseespy.opensees as ops
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from core.model_state import ModelState
from core.structural_utils import StructuralUtils


class AnalysisRunner:
    """Executes structural analyses on built model."""
    
    def __init__(self, state: ModelState):
        """Initialize with model state.
        
        Args:
            state: Built ModelState instance
        """
        self.state = state
        self.utils = StructuralUtils(state)
        self.logger = logging.getLogger(__name__)
    
    def run_elf_drift_analysis(self, elf_forces: Dict[int, float]) -> Tuple[Dict[int, float], float]:
        """Run ELF static analysis and compute interstory drifts.
        
        Args:
            elf_forces: Dictionary mapping node tags to lateral forces from ELF
            
        Returns:
            Tuple of (drift_ratios_per_story, max_drift_ratio)
            - drift_ratios_per_story: Dict[story_number, drift_ratio]
            - max_drift_ratio: Maximum interstory drift ratio
        
        """
        self.logger.info(f"Starting ELF drift analysis (model_type={self.state.model_type})")
        
        # Clear any existing loads
        ops.remove('loadPattern', -1)
        
        # Create load pattern for ELF forces
        ops.timeSeries('Linear', 100)
        ops.pattern('Plain', 100, 100)
        
        # Apply ELF forces to nodes
        for node_tag, force in elf_forces.items():
            ops.load(node_tag, force, 0.0, 0.0)
            self.logger.debug(f"Applied force {force:.2f} to node {node_tag}")
        
        # Run static analysis
        self.logger.info("Running static analysis with ELF forces...")
        self.utils.run_static_analysis(n_steps=1)
        
        # Extract displacements and compute interstory drifts
        story_drifts = {}
        displacements = []
        

        # Get displacement at each floor level
        for story in range(1, self.state.n_stories + 1):
            elev_key = f"Elevation {story}"
            
            # Get a representative wall node at this elevation
            diaphragm_nodes = self.state.diaphragm.get("Nodes", {}).get(elev_key, {})
            if "Left Wall" in diaphragm_nodes:
                node_tag = diaphragm_nodes["Left Wall"]
            elif "Right Wall" in diaphragm_nodes:
                node_tag = diaphragm_nodes["Right Wall"]
            else:
                raise KeyError(f"No wall node found at {elev_key}")
            
            # Get horizontal displacement (DOF 1)
            disp = ops.nodeDisp(node_tag, 1)
            displacements.append(disp)
            self.logger.debug(f"Story {story} displacement: {disp:.6f}")
        
        # Compute interstory drift ratios
        story_heights = self.state.building.get('Stories Heights', [])
        
        for story in range(1, self.state.n_stories + 1):
            story_idx = story - 1
            story_height = story_heights[story_idx]
            
            # Interstory drift = (disp_i - disp_i-1) / height_i
            if story == 1:
                # First story: drift relative to base (displacement = 0)
                interstory_disp = displacements[0]
            else:
                interstory_disp = displacements[story_idx] - displacements[story_idx - 1]
            
            drift_ratio = abs(interstory_disp) / story_height
            story_drifts[story] = drift_ratio
            
            self.logger.debug(f"Story {story}: drift ratio = {drift_ratio:.6f} "
                            f"(Δ={interstory_disp:.6f}, h={story_height:.2f})")
        
        # Find maximum drift
        max_drift = max(story_drifts.values())
        max_story = max(story_drifts, key=story_drifts.get)
        
        self.logger.info(f"ELF drift analysis complete: "
                        f"Max drift = {max_drift:.4f} at story {max_story}")
        
        # Wipe analysis
        ops.wipeAnalysis()
        
        return story_drifts, max_drift
    
    def run_ground_motion(self, record_path: Union[str, Path], 
                         dt: float, 
                         scale_factor: float = 1,
                         result_path: Union[str, Path] = Path(__file__).parent / "TH_Result",
                         record_components: Optional[List[str]] = None,
                         node_responses: Optional[List[str]] = None,
                         element_responses: Optional[List[str]] = None) -> float:
        
        """Run time-history analysis for ground motion.
        
        Args:
            record_path: Path to ground motion file
            dt: Time step
            result_path: Directory for output files
            record_components: Components to record (default: Base_Reactions only)
            node_responses: Node responses to record (default: ['disp'])
            element_responses: Element responses to record (default: [])
            
        Returns:
            Final analysis time reached
            
        Example:
            # Default: only base reactions and displacements
            runner.run_ground_motion(record_file, dt, result_dir)
            
            # Record everything for Wall
            runner.run_ground_motion(
                record_file, dt, result_dir,
                record_components=['Wall', 'Base_Reactions'],
                node_responses=['disp', 'accel'],
                element_responses=['force']
            )
        """

        record_name = Path(record_path).stem
        self.logger.info(f"Starting ground motion analysis: {record_name}, dt={dt:.4f}s")
        
        record_components = ['Wall', 'UFP', 'PT_Bar', 'Rocking_Spring', 'Leaning_Column', 'Diaphragm', 'Base_Reactions']
        node_responses = ['disp', 'vel', 'accel', 'reaction']
        element_responses = ['force', 'deformation', 'stress']

        # Create recorders before analysis
        self.utils.create_recorders(
            result_path, 
            components=record_components,
            node_responses=node_responses,
            element_responses=element_responses
        )
        
        # Run analysis
        final_time = self.utils.EQ(record_path, dt, scale_factor)
        
        self.logger.info(f"Completed ground motion analysis at t={final_time:.2f}s")
        return final_time