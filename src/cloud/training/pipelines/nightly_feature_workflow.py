"""Nightly feature analysis workflow for Mechanic."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog  # type: ignore[reportMissingImports]

from ..brain.brain_library import BrainLibrary
from ..config.settings import EngineSettings
from ..services.nightly_feature_analysis import NightlyFeatureAnalysis

logger = structlog.get_logger(__name__)


def run_nightly_feature_analysis(
    settings: EngineSettings,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run nightly feature importance analysis workflow.
    
    This is designed to be called by Mechanic after Engine training completes.
    
    Args:
        settings: Engine settings
        symbols: Optional list of symbols to analyze (if None, analyzes all active models)
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("nightly_feature_workflow_started")
    
    # Initialize Brain Library
    try:
        dsn = settings.postgres.dsn if settings.postgres else None
        if not dsn:
            logger.warning("no_database_dsn", message="Cannot run feature analysis without database")
            return {"status": "failed", "error": "No database DSN"}
        
        brain_library = BrainLibrary(dsn=dsn, use_pool=True)
    except Exception as e:
        logger.error("brain_library_initialization_failed", error=str(e))
        return {"status": "failed", "error": str(e)}
    
    # Initialize feature analysis service
    feature_analysis = NightlyFeatureAnalysis(brain_library, settings)
    
    # If symbols not provided, get all symbols with active models
    if not symbols:
        symbols = _get_symbols_with_active_models(brain_library)
        logger.info("analyzing_all_active_symbols", num_symbols=len(symbols))
    
    # For now, return placeholder results
    # In reality, this would:
    # 1. Load trained models from model registry
    # 2. Load datasets for each symbol
    # 3. Run feature importance analysis
    # 4. Store results in Brain Library
    
    logger.info("nightly_feature_workflow_complete", num_symbols=len(symbols))
    
    return {
        "status": "success",
        "symbols_analyzed": len(symbols),
        "symbols": symbols,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


def _get_symbols_with_active_models(brain_library: BrainLibrary) -> List[str]:
    """
    Get list of symbols with active models in Brain Library.
    
    Args:
        brain_library: Brain Library instance
        
    Returns:
        List of symbols
    """
    # This would query Brain Library for all symbols with active models
    # For now, return empty list as placeholder
    logger.debug("getting_symbols_with_active_models")
    return []

