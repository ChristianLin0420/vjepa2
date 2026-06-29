"""JEPA-4D WorldModel: RGB-first structured world-model components."""

from jepa4d.data.schemas import JEPATokenBundle, RGBInputBatch, RobotState
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor

__all__ = ["JEPATokenBundle", "RGBInputBatch", "RobotState", "VJEPA21FeatureExtractor"]
__version__ = "0.1.0"
