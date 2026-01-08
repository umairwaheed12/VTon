"""
Virtual Try-On Module for Fooocus
Integrates dress warping and masking pipeline for automatic virtual try-on with inpainting.
"""

from .dresss import ShoulderHeightDressWarper
from .masking import ClothMasker

__all__ = ['ShoulderHeightDressWarper', 'ClothMasker']
