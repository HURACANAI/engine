"""
Correlation / Cluster Engine

Uses PCA or network graphs to trade pair spreads.
Exploits correlation clusters and mean-reverting spreads.

Key Features:
1. Correlation clustering (PCA, network graphs)
2. Pair spread trading (mean-reverting spreads)
3. Cluster detection (groups of correlated assets)
4. Spread prediction (when spreads will revert)
5. Statistical arbitrage (pairs trading)

Best in: RANGE regime (mean-reverting spreads)
Strategy: Trade spreads between correlated assets
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Optional: Import PCA and network analysis
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = structlog.get_logger(__name__)


@dataclass
class ClusterSignal:
    """Signal from correlation/cluster engine."""
    direction: str  # "buy", "sell", "hold"
    confidence: float  # 0-1
    spread_zscore: float  # Spread z-score (how far from mean)
    expected_reversion: float  # Expected reversion in bps
    cluster_id: Optional[str] = None  # Cluster identifier
    reasoning: str = ""
    key_features: Optional[Dict[str, float]] = None


class CorrelationClusterEngine:
    """
    Correlation / Cluster Engine.
    
    Uses PCA or network graphs to trade pair spreads.
    Exploits correlation clusters and mean-reverting spreads.
    
    Key Features:
    - Correlation clustering (PCA, network graphs)
    - Pair spread trading
    - Cluster detection
    - Spread prediction
    - Statistical arbitrage
    """
    
    def __init__(
        self,
        min_correlation: float = 0.70,  # Minimum correlation for pairs
        max_spread_zscore: float = 2.0,  # Maximum spread z-score to trade
        min_spread_zscore: float = 1.0,  # Minimum spread z-score to trade
        use_pca: bool = True,  # Use PCA for clustering
        n_clusters: int = 5,  # Number of clusters
    ):
        """
        Initialize correlation/cluster engine.
        
        Args:
            min_correlation: Minimum correlation for pairs
            max_spread_zscore: Maximum spread z-score to trade
            min_spread_zscore: Minimum spread z-score to trade
            use_pca: Whether to use PCA for clustering
            n_clusters: Number of clusters
        """
        self.min_correlation = min_correlation
        self.max_spread_zscore = max_spread_zscore
        self.min_spread_zscore = min_spread_zscore
        self.use_pca = use_pca and HAS_SKLEARN
        self.n_clusters = n_clusters
        
        # Correlation matrix (symbol -> symbol -> correlation)
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        
        # Spread history (pair -> spread history)
        self.spread_history: Dict[Tuple[str, str], List[float]] = {}
        
        # Clusters (cluster_id -> [symbols])
        self.clusters: Dict[str, List[str]] = {}
        
        # PCA model (if using PCA)
        self.pca_model: Optional[PCA] = None
        self.kmeans_model: Optional[KMeans] = None
        
        logger.info(
            "correlation_cluster_engine_initialized",
            min_correlation=min_correlation,
            use_pca=use_pca,
            n_clusters=n_clusters,
        )
    
    def update_correlation(
        self,
        symbol1: str,
        symbol2: str,
        correlation: float,
    ) -> None:
        """Update correlation between two symbols."""
        if symbol1 not in self.correlation_matrix:
            self.correlation_matrix[symbol1] = {}
        if symbol2 not in self.correlation_matrix:
            self.correlation_matrix[symbol2] = {}
        
        self.correlation_matrix[symbol1][symbol2] = correlation
        self.correlation_matrix[symbol2][symbol1] = correlation
    
    def update_spread(
        self,
        symbol1: str,
        symbol2: str,
        spread_bps: float,
    ) -> None:
        """Update spread between two symbols."""
        pair = tuple(sorted([symbol1, symbol2]))
        
        if pair not in self.spread_history:
            self.spread_history[pair] = []
        
        self.spread_history[pair].append(spread_bps)
        
        # Keep only last 1000 spreads
        if len(self.spread_history[pair]) > 1000:
            self.spread_history[pair] = self.spread_history[pair][-1000:]
    
    def detect_clusters(
        self,
        symbols: List[str],
        returns_matrix: np.ndarray,
    ) -> Dict[str, List[str]]:
        """
        Detect correlation clusters using PCA or network graphs.
        
        Args:
            symbols: List of symbols
            returns_matrix: Returns matrix (n_symbols x n_periods)
        
        Returns:
            Dict of {cluster_id: [symbols]}
        """
        if not self.use_pca or not HAS_SKLEARN:
            # Use simple correlation-based clustering
            return self._cluster_by_correlation(symbols)
        
        # Use PCA for dimensionality reduction
        try:
            # Fit PCA
            n_components = min(10, len(symbols) - 1)
            pca = PCA(n_components=n_components)
            pca_features = pca.fit_transform(returns_matrix.T)  # Transpose for PCA
            
            # Cluster in PCA space
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(pca_features)
            
            # Group symbols by cluster
            clusters = {}
            for i, symbol in enumerate(symbols):
                cluster_id = f"cluster_{cluster_labels[i]}"
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(symbol)
            
            self.pca_model = pca
            self.kmeans_model = kmeans
            self.clusters = clusters
            
            logger.info("clusters_detected", n_clusters=len(clusters), symbols=len(symbols))
            
            return clusters
        except Exception as e:
            logger.warning("pca_clustering_failed_falling_back", error=str(e))
            return self._cluster_by_correlation(symbols)
    
    def _cluster_by_correlation(
        self,
        symbols: List[str],
    ) -> Dict[str, List[str]]:
        """Cluster symbols by correlation (simple method)."""
        clusters = {}
        assigned = set()
        
        for symbol in symbols:
            if symbol in assigned:
                continue
            
            # Find correlated symbols
            cluster = [symbol]
            assigned.add(symbol)
            
            for other_symbol in symbols:
                if other_symbol in assigned:
                    continue
                
                # Check correlation
                corr = self.get_correlation(symbol, other_symbol)
                if corr and corr >= self.min_correlation:
                    cluster.append(other_symbol)
                    assigned.add(other_symbol)
            
            if len(cluster) > 1:
                cluster_id = f"cluster_{len(clusters)}"
                clusters[cluster_id] = cluster
        
        self.clusters = clusters
        return clusters
    
    def get_correlation(
        self,
        symbol1: str,
        symbol2: str,
    ) -> Optional[float]:
        """Get correlation between two symbols."""
        if symbol1 in self.correlation_matrix and symbol2 in self.correlation_matrix[symbol1]:
            return self.correlation_matrix[symbol1][symbol2]
        return None
    
    def generate_signal(
        self,
        symbol1: str,
        symbol2: str,
        price1: float,
        price2: float,
        features: Dict[str, float],
        current_regime: str,
    ) -> ClusterSignal:
        """
        Generate correlation/cluster signal.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            price1: Price of first symbol
            price2: Price of second symbol
            features: Feature dictionary
            current_regime: Current market regime
        
        Returns:
            ClusterSignal
        """
        # Check correlation
        correlation = self.get_correlation(symbol1, symbol2)
        
        if not correlation or abs(correlation) < self.min_correlation:
            return ClusterSignal(
                direction="hold",
                confidence=0.0,
                spread_zscore=0.0,
                expected_reversion=0.0,
                reasoning=f"Correlation too low ({correlation:.2f} < {self.min_correlation:.2f})",
                key_features={
                    "correlation": correlation or 0.0,
                },
            )
        
        # Calculate spread
        spread_bps = ((price1 / price2) - 1.0) * 10000.0  # Spread in bps
        
        # Get spread history
        pair = tuple(sorted([symbol1, symbol2]))
        spread_history = self.spread_history.get(pair, [])
        
        if len(spread_history) < 100:
            # Not enough history
            return ClusterSignal(
                direction="hold",
                confidence=0.0,
                spread_zscore=0.0,
                expected_reversion=0.0,
                reasoning="Insufficient spread history",
                key_features={
                    "spread_bps": spread_bps,
                    "n_history": len(spread_history),
                },
            )
        
        # Calculate spread z-score
        spread_mean = np.mean(spread_history)
        spread_std = np.std(spread_history)
        
        if spread_std == 0:
            return ClusterSignal(
                direction="hold",
                confidence=0.0,
                spread_zscore=0.0,
                expected_reversion=0.0,
                reasoning="Spread has no variance",
                key_features={
                    "spread_bps": spread_bps,
                },
            )
        
        spread_zscore = (spread_bps - spread_mean) / spread_std
        
        # Check if spread is extreme enough
        if abs(spread_zscore) < self.min_spread_zscore:
            return ClusterSignal(
                direction="hold",
                confidence=0.0,
                spread_zscore=spread_zscore,
                expected_reversion=0.0,
                reasoning=f"Spread not extreme enough (z={spread_zscore:.2f} < {self.min_spread_zscore:.2f})",
                key_features={
                    "spread_bps": spread_bps,
                    "spread_zscore": spread_zscore,
                },
            )
        
        if abs(spread_zscore) > self.max_spread_zscore:
            # Spread too extreme (might be regime change, not mean reversion)
            return ClusterSignal(
                direction="hold",
                confidence=0.0,
                spread_zscore=spread_zscore,
                expected_reversion=0.0,
                reasoning=f"Spread too extreme (z={spread_zscore:.2f} > {self.max_spread_zscore:.2f})",
                key_features={
                    "spread_bps": spread_bps,
                    "spread_zscore": spread_zscore,
                },
            )
        
        # Determine direction (mean reversion)
        if spread_zscore > 0:
            # Spread is high → SELL spread (sell symbol1, buy symbol2)
            direction = "sell"
            expected_reversion = -abs(spread_zscore) * spread_std  # Expected reversion in bps
            reasoning = f"Spread high (z={spread_zscore:.2f}): SELL spread (sell {symbol1}, buy {symbol2})"
        else:
            # Spread is low → BUY spread (buy symbol1, sell symbol2)
            direction = "buy"
            expected_reversion = abs(spread_zscore) * spread_std  # Expected reversion in bps
            reasoning = f"Spread low (z={spread_zscore:.2f}): BUY spread (buy {symbol1}, sell {symbol2})"
        
        # Calculate confidence
        # Higher confidence when:
        # - Spread is more extreme (higher z-score)
        # - Correlation is higher
        # - Regime is RANGE (mean-reverting)
        zscore_confidence = min(1.0, abs(spread_zscore) / self.max_spread_zscore)
        correlation_confidence = abs(correlation)
        regime_confidence = 1.0 if current_regime == "RANGE" else 0.5
        
        confidence = (zscore_confidence + correlation_confidence + regime_confidence) / 3.0
        
        # Find cluster
        cluster_id = None
        for cid, cluster_symbols in self.clusters.items():
            if symbol1 in cluster_symbols and symbol2 in cluster_symbols:
                cluster_id = cid
                break
        
        return ClusterSignal(
            direction=direction,
            confidence=confidence,
            spread_zscore=spread_zscore,
            expected_reversion=expected_reversion,
            cluster_id=cluster_id,
            reasoning=reasoning,
            key_features={
                "spread_bps": spread_bps,
                "spread_zscore": spread_zscore,
                "correlation": correlation,
                "spread_mean": spread_mean,
                "spread_std": spread_std,
                "expected_reversion": expected_reversion,
            },
        )

