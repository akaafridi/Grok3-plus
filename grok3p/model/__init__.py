"""
Model components for Grok-3+ architecture
DOI: 10.5281/zenodo.15341810
"""

from .core import Grok3pModel
from .fp8_layer import FP8Linear
from .moe_layer import MixtureOfExperts, MoERouter

__all__ = [
    "Grok3pModel", 
    "FP8Linear", 
    "MixtureOfExperts", 
    "MoERouter"
]
