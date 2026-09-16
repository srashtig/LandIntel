"""Land Intelligence pipeline — importable, parameterized modules refactored
from ``land_intelligence_tier1_tier2_folium_hydrology.ipynb``.

Public entry points:
    - :func:`data_analysis_pipeline.aoi.build_aoi`
    - :func:`data_analysis_pipeline.orchestrator.run_analysis`
"""

from .config import ENGINE_VERSION

__all__ = ["ENGINE_VERSION"]
__version__ = ENGINE_VERSION
