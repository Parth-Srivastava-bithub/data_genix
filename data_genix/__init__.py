# ==============================================================================
# File: datagenix/__init__.py
# Description: This makes the 'datagenix' directory a Python package and
#              makes the DataGenerator class directly accessible to the user.
# ==============================================================================

from .generator import DataGenerator

__all__ = ['DataGenerator']