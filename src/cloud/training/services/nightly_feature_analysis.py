"""Nightly feature importance analysis service for Mechanic."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog  # type: ignore[reportMissingImports]

from ..brain.brain_library import BrainLibrary
from ..brain.feature_importance_analyzer import FeatureImportanceAnalyzer
from ..config.settings import EngineSettings

logger = structlog.get_logger(__name__)


class NightlyFeatureAnalysis:
    """
    Nightly feature importance analysis service.
    
    Designed for Mechanic to run after Engine training completes.
    Analyzes feature importance and updates rankings in Brain Library.
    """

    def __init__(
        self,
        brain_library: BrainLibrary,
        settings: Optional[EngineSettings] = None,
    ) -> None:
        """
        Initialize nightly feature analysis service.
        
        Args:
            brain_library: Brain Library instance
            settings: Engine settings
        """
        self.brain = brain_library
        self.settings = settings
        self.feature_analyzer = FeatureImportanceAnalyzer(brain_library)
        
        logger.info("nightly_feature_analysis_initialized")

    def run_nightly_analysis(
        self,
        symbols: List[str],
        models: Dict[str, Any],
        datasets: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run nightly feature importance analysis for all symbols.
        
        Args:
            symbols: List of symbols to analyze
            models: Dictionary mapping symbol to trained model
            datasets: Dictionary mapping symbol to dataset (X, y, feature_names)
            
        Returns:
            Dictionary mapping symbol to analysis results
        """
        logger.info("nightly_feature_analysis_started", num_symbols=len(symbols))
        
        results = {}
        
        for symbol in symbols:
            try:
                if symbol not in models or symbol not in datasets:
                    logger.warning("missing_data_for_symbol", symbol=symbol)
                    continue
                
                model = models[symbol]
                dataset = datasets[symbol]
                
                X = dataset.get("X")
                y = dataset.get("y")
                feature_names = dataset.get("feature_names", [])
                
                if X is None or y is None:
                    logger.warning("invalid_dataset", symbol=symbol)
                    continue
                
                # Analyze feature importance
                importance_results = self.feature_analyzer.analyze_feature_importance(
                    symbol=symbol,
                    model=model,
                    X=X,
                    y=y,
                    feature_names=feature_names,
                    methods=['shap', 'permutation'] if len(feature_names) <= 50 else ['correlation'],
                )
                
                # Get top features
                top_features = self.feature_analyzer.get_top_features_for_symbol(
                    symbol=symbol,
                    top_n=20,
                    method='shap',
                )
                
                results[symbol] = {
                    "status": "success",
                    "importance_results": importance_results,
                    "top_features": top_features,
                    "analysis_date": datetime.now(tz=timezone.utc).date(),
                }
                
                logger.info(
                    "feature_analysis_complete",
                    symbol=symbol,
                    top_features_count=len(top_features),
                )
            except Exception as e:
                logger.error(
                    "feature_analysis_failed",
                    symbol=symbol,
                    error=str(e),
                )
                results[symbol] = {
                    "status": "failed",
                    "error": str(e),
                }
        
        logger.info(
            "nightly_feature_analysis_complete",
            total_symbols=len(symbols),
            successful=sum(1 for r in results.values() if r.get("status") == "success"),
            failed=sum(1 for r in results.values() if r.get("status") == "failed"),
        )
        
        return results

    def get_feature_importance_trends(
        self,
        symbol: str,
        days: int = 30,
    ) -> Dict[str, List[float]]:
        """
        Get feature importance trends over time.
        
        Args:
            symbol: Trading symbol
            days: Number of days to look back
            
        Returns:
            Dictionary mapping feature names to importance scores over time
        """
        # This would query Brain Library for historical feature importance
        # For now, return empty dict as placeholder
        logger.debug("getting_feature_importance_trends", symbol=symbol, days=days)
        return {}

    def identify_feature_shifts(
        self,
        symbol: str,
        threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Identify significant feature importance shifts.
        
        Args:
            symbol: Trading symbol
            threshold: Threshold for significant shift (default: 10%)
            
        Returns:
            List of features with significant shifts
        """
        # This would compare current vs previous feature importance
        # For now, return empty list as placeholder
        logger.debug("identifying_feature_shifts", symbol=symbol, threshold=threshold)
        return []

