"""
Polarization Model package.

Public surface area intentionally minimal — import from submodules directly.
"""

from polarization_model.model import EnhancedPolarizationModel
from polarization_model.parameters import ModelParameters
from polarization_model.event_a import draw_event_a, DEFAULT_EVENT_A_PARAMS

__all__ = ["EnhancedPolarizationModel", "ModelParameters", "draw_event_a", "DEFAULT_EVENT_A_PARAMS"]
