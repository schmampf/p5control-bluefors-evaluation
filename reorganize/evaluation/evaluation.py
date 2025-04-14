"""
Evaluation Module for p5control-bluefors Data Processing

This module provides an evaluation class (`Evaluation`) to analyze measurement
data recorded using the p5control-bluefors system. It is designed to work with HDF5
files and facilitates accessing, processing, and visualizing measurement data.

Main Features:
---------------
- **File Handling**:
  - Opens and reads HDF5 files.
  - Extracts available measurement data.
  - Lists available keys for specific measurements.

- **Measurement Processing**:
  - Allows selecting a specific measurement for analysis.
  - Extracts and processes relevant measurement keys.
  - Stores and updates measurement-related parameters.

- **Amplification Analysis**:
  - Retrieves voltage amplification values over time.
  - Plots amplification changes during measurements.
  - Sets user-defined amplification factors for further evaluation.

- **Logging and Debugging**:
  - Logs key steps and potential errors during file access and data processing.
  - Provides detailed debug information for troubleshooting.

Usage:
------
This class serves as a base for more specific evaluation routines.
A typical workflow might include:

1. **Initialize the evaluation instance**:
   ```python
   evaluator = BaseEvaluation(name="Test Evaluation")
"""

# region imports
# std lib
import os
import sys
from dataclasses import dataclass, field
from enum import Enum

# 3rd party
import logging
import numpy as np
# import matplotlib.pyplot as plt
# from h5py import File

# local
from integration.files import Files, FileAPI
#endregion

logger = logging.getLogger(__name__)

@dataclass
class Parameters:
    volt_amp: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=float))
    ref_resistor: float = field(default_factory=lambda: 51.689E3)

@dataclass 
class MeasurementHeader:
    type: str = field(default_factory=str)
  
# class Measurements(Enum):
    
    

class Evaluation(FileAPI):
    files: Files
    params: Parameters
    headers: MeasurementHeader
    
    # def __setattr__(self, name, value):
    #     if not name in ["files", "params", "headers"]:
    #         for group in ["files", "params", "headers"]:
    #             if name in getattr(self, group).__dict__:
    #                 group.__dict__[name] = value
    #                 break
    #     else:
    #         super().__setattr__(name, value)
            
    # def __getattr__(self, name):
        
				
   
   
    def __init__(self, name: str = "Evaluation"):
        """
        Initialize the Evaluation class.
        Args:
        name (str): Name of the evaluation instance.
        """
        self.files = Files(name=name+"_fileData")