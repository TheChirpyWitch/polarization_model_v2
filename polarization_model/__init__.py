"""
Polarization Model package.

Public surface area intentionally minimal — import from submodules directly.
"""

from polarization_model.model import EnhancedPolarizationModel
from polarization_model.parameters import ModelParameters

__all__ = ["EnhancedPolarizationModel", "ModelParameters"]
