"""
Football Leagues ML Pipeline
Modulo reutilizable para entrenar y predecir modelos XGBoost
para multiples ligas de futbol.
"""

__version__ = "0.1.0"

from .config import LIGAS_CONFIG
from .model import Entrenador, Reentrenador
from .features import FeatureBuilder
from .data import DataLoader

__all__ = [
    'LIGAS_CONFIG',
    'Entrenador',
    'Reentrenador',
    'FeatureBuilder',
    'DataLoader',
]
