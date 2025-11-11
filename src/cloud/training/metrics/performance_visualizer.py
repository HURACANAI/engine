"""
Performance Visualization Layer

Visualizes wealth index and drawdowns to improve model explainability.
Auto-generates graphs after every HTF backtest cycle.

Key Features:
- Plot wealth index with annotations (marking max drawdown)
- Drawdown visualization
- Performance metrics charts
- Auto-save to files for dashboard integration

Author: Huracan Engine Team
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import polars as pl
import structlog

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib_not_available")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = structlog.get_logger(__name__)


class PerformanceVisualizer:
    """
    Performance visualization for wealth index and drawdowns.
    
    Usage:
        visualizer = PerformanceVisualizer(output_dir="plots/")
        
        # Plot wealth index and drawdowns
        visualizer.plot_wealth_index(
            data=df,
            symbol="BTC/USDT",
            save_path="wealth_index.png"
        )
        
        # Plot drawdowns
        visualizer.plot_drawdowns(
            data=df,
            symbol="BTC/USDT",
            save_path="drawdowns.png"
        )
    """
    
    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        use_plotly: bool = False,
        figsize: tuple = (12, 8)
    ):
        """
        Initialize performance visualizer.
        
        Args:
            output_dir: Directory to save plots (optional)
            use_plotly: Whether to use Plotly instead of Matplotlib (default: False)
            figsize: Figure size for Matplotlib (width, height)
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_plotly = use_plotly and HAS_PLOTLY
        self.figsize = figsize
        
        if not HAS_MATPLOTLIB and not HAS_PLOTLY:
            logger.warning("no_plotting_library_available")
        
        logger.info(
            "performance_visualizer_initialized",
            output_dir=str(self.output_dir) if self.output_dir else None,
            use_plotly=self.use_plotly
        )
    
    def plot_wealth_index(
        self,
        data: pl.DataFrame | pd.DataFrame,
        symbol: Optional[str] = None,
        wealth_column: str = 'wealth_index',
        timestamp_column: str = 'timestamp',
        save_path: Optional[str | Path] = None,
        show_max_drawdown: bool = True,
        title: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot wealth index with annotations (marking max drawdown).
        
        Args:
            data: DataFrame with wealth_index and timestamp columns
            symbol: Trading symbol (for title)
            wealth_column: Name of wealth index column
            timestamp_column: Name of timestamp column
            save_path: Path to save plot (optional)
            show_max_drawdown: Whether to annotate max drawdown
            title: Custom title (optional)
        
        Returns:
            Path to saved file if saved, None otherwise
        """
        if not HAS_MATPLOTLIB and not self.use_plotly:
            logger.error("no_plotting_library_available")
            return None
        
        # Convert to pandas if needed
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        else:
            df = data.copy()
        
        # Ensure timestamp is datetime
        if timestamp_column in df.columns:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            x_data = df[timestamp_column]
        else:
            x_data = range(len(df))
        
        y_data = df[wealth_column].values
        
        if self.use_plotly and HAS_PLOTLY:
            return self._plot_wealth_index_plotly(
                x_data, y_data, symbol, save_path, show_max_drawdown, title
            )
        else:
            return self._plot_wealth_index_matplotlib(
                x_data, y_data, symbol, save_path, show_max_drawdown, title, df, wealth_column
            )
    
    def _plot_wealth_index_matplotlib(
        self,
        x_data,
        y_data,
        symbol: Optional[str],
        save_path: Optional[str | Path],
        show_max_drawdown: bool,
        title: Optional[str],
        df: pd.DataFrame,
        wealth_column: str
    ) -> Optional[Path]:
        """Plot wealth index using Matplotlib."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot wealth index
        ax.plot(x_data, y_data, linewidth=2, label='Wealth Index', color='#2E86AB')
        ax.fill_between(x_data, y_data, alpha=0.3, color='#2E86AB')
        
        # Annotate max drawdown if requested
        if show_max_drawdown and 'drawdown' in df.columns:
            # Find max drawdown point
            max_dd_idx = df['drawdown'].idxmin()
            max_dd_value = df.loc[max_dd_idx, wealth_column]
            max_dd_time = x_data.iloc[max_dd_idx] if hasattr(x_data, 'iloc') else x_data[max_dd_idx]
            
            # Mark max drawdown
            ax.scatter(
                max_dd_time, max_dd_value,
                color='red', s=100, zorder=5,
                label=f'Max Drawdown: {df.loc[max_dd_idx, "drawdown"]:.2%}'
            )
            ax.annotate(
                f'Max DD: {df.loc[max_dd_idx, "drawdown"]:.2%}',
                xy=(max_dd_time, max_dd_value),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )
        
        # Formatting
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Wealth Index', fontsize=12)
        ax.set_title(
            title or f'Wealth Index{" - " + symbol if symbol else ""}',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Format x-axis if datetime
        if hasattr(x_data, 'dtype') and pd.api.types.is_datetime64_any_dtype(x_data):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("wealth_index_plot_saved", path=str(save_path))
            plt.close()
            return save_path
        elif self.output_dir:
            filename = f"wealth_index_{symbol or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("wealth_index_plot_saved", path=str(save_path))
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def _plot_wealth_index_plotly(
        self,
        x_data,
        y_data,
        symbol: Optional[str],
        save_path: Optional[str | Path],
        show_max_drawdown: bool,
        title: Optional[str]
    ) -> Optional[Path]:
        """Plot wealth index using Plotly."""
        fig = go.Figure()
        
        # Plot wealth index
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            name='Wealth Index',
            line=dict(color='#2E86AB', width=2),
            fill='tonexty'
        ))
        
        # Formatting
        fig.update_layout(
            title=title or f'Wealth Index{" - " + symbol if symbol else ""}',
            xaxis_title='Time',
            yaxis_title='Wealth Index',
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Save or show
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path))
            logger.info("wealth_index_plot_saved", path=str(save_path))
            return save_path
        elif self.output_dir:
            filename = f"wealth_index_{symbol or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            save_path = self.output_dir / filename
            fig.write_html(str(save_path))
            logger.info("wealth_index_plot_saved", path=str(save_path))
            return save_path
        else:
            fig.show()
            return None
    
    def plot_drawdowns(
        self,
        data: pl.DataFrame | pd.DataFrame,
        symbol: Optional[str] = None,
        drawdown_column: str = 'drawdown',
        timestamp_column: str = 'timestamp',
        save_path: Optional[str | Path] = None,
        title: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot drawdowns over time.
        
        Args:
            data: DataFrame with drawdown and timestamp columns
            symbol: Trading symbol (for title)
            drawdown_column: Name of drawdown column
            timestamp_column: Name of timestamp column
            save_path: Path to save plot (optional)
            title: Custom title (optional)
        
        Returns:
            Path to saved file if saved, None otherwise
        """
        if not HAS_MATPLOTLIB and not self.use_plotly:
            logger.error("no_plotting_library_available")
            return None
        
        # Convert to pandas if needed
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        else:
            df = data.copy()
        
        # Ensure timestamp is datetime
        if timestamp_column in df.columns:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            x_data = df[timestamp_column]
        else:
            x_data = range(len(df))
        
        y_data = df[drawdown_column].values * 100  # Convert to percentage
        
        if self.use_plotly and HAS_PLOTLY:
            return self._plot_drawdowns_plotly(x_data, y_data, symbol, save_path, title)
        else:
            return self._plot_drawdowns_matplotlib(x_data, y_data, symbol, save_path, title)
    
    def _plot_drawdowns_matplotlib(
        self,
        x_data,
        y_data,
        symbol: Optional[str],
        save_path: Optional[str | Path],
        title: Optional[str]
    ) -> Optional[Path]:
        """Plot drawdowns using Matplotlib."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot drawdowns (fill below zero)
        ax.fill_between(x_data, y_data, 0, alpha=0.5, color='red', label='Drawdown')
        ax.plot(x_data, y_data, linewidth=1.5, color='darkred')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Formatting
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title(
            title or f'Drawdown Analysis{" - " + symbol if symbol else ""}',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Format x-axis if datetime
        if hasattr(x_data, 'dtype') and pd.api.types.is_datetime64_any_dtype(x_data):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("drawdown_plot_saved", path=str(save_path))
            plt.close()
            return save_path
        elif self.output_dir:
            filename = f"drawdowns_{symbol or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("drawdown_plot_saved", path=str(save_path))
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def _plot_drawdowns_plotly(
        self,
        x_data,
        y_data,
        symbol: Optional[str],
        save_path: Optional[str | Path],
        title: Optional[str]
    ) -> Optional[Path]:
        """Plot drawdowns using Plotly."""
        fig = go.Figure()
        
        # Plot drawdowns
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=1.5),
            fillcolor='rgba(255,0,0,0.5)'
        ))
        
        # Formatting
        fig.update_layout(
            title=title or f'Drawdown Analysis{" - " + symbol if symbol else ""}',
            xaxis_title='Time',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Save or show
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path))
            logger.info("drawdown_plot_saved", path=str(save_path))
            return save_path
        elif self.output_dir:
            filename = f"drawdowns_{symbol or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            save_path = self.output_dir / filename
            fig.write_html(str(save_path))
            logger.info("drawdown_plot_saved", path=str(save_path))
            return save_path
        else:
            fig.show()
            return None
    
    def plot_combined_performance(
        self,
        data: pl.DataFrame | pd.DataFrame,
        symbol: Optional[str] = None,
        wealth_column: str = 'wealth_index',
        drawdown_column: str = 'drawdown',
        timestamp_column: str = 'timestamp',
        save_path: Optional[str | Path] = None
    ) -> Optional[Path]:
        """
        Plot combined wealth index and drawdowns in subplots.
        
        Args:
            data: DataFrame with wealth_index, drawdown, and timestamp columns
            symbol: Trading symbol (for title)
            wealth_column: Name of wealth index column
            drawdown_column: Name of drawdown column
            timestamp_column: Name of timestamp column
            save_path: Path to save plot (optional)
        
        Returns:
            Path to saved file if saved, None otherwise
        """
        if not HAS_MATPLOTLIB and not self.use_plotly:
            logger.error("no_plotting_library_available")
            return None
        
        # Convert to pandas if needed
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        else:
            df = data.copy()
        
        # Ensure timestamp is datetime
        if timestamp_column in df.columns:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            x_data = df[timestamp_column]
        else:
            x_data = range(len(df))
        
        if self.use_plotly and HAS_PLOTLY:
            return self._plot_combined_plotly(
                x_data, df, symbol, wealth_column, drawdown_column, save_path
            )
        else:
            return self._plot_combined_matplotlib(
                x_data, df, symbol, wealth_column, drawdown_column, save_path
            )
    
    def _plot_combined_matplotlib(
        self,
        x_data,
        df: pd.DataFrame,
        symbol: Optional[str],
        wealth_column: str,
        drawdown_column: str,
        save_path: Optional[str | Path]
    ) -> Optional[Path]:
        """Plot combined performance using Matplotlib."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5), sharex=True)
        
        # Plot wealth index
        ax1.plot(x_data, df[wealth_column], linewidth=2, label='Wealth Index', color='#2E86AB')
        ax1.fill_between(x_data, df[wealth_column], alpha=0.3, color='#2E86AB')
        ax1.set_ylabel('Wealth Index', fontsize=12)
        ax1.set_title(f'Performance Analysis{" - " + symbol if symbol else ""}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Plot drawdowns
        drawdowns_pct = df[drawdown_column].values * 100
        ax2.fill_between(x_data, drawdowns_pct, 0, alpha=0.5, color='red', label='Drawdown')
        ax2.plot(x_data, drawdowns_pct, linewidth=1.5, color='darkred')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        # Format x-axis if datetime
        if hasattr(x_data, 'dtype') and pd.api.types.is_datetime64_any_dtype(x_data):
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("combined_performance_plot_saved", path=str(save_path))
            plt.close()
            return save_path
        elif self.output_dir:
            filename = f"combined_performance_{symbol or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("combined_performance_plot_saved", path=str(save_path))
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def _plot_combined_plotly(
        self,
        x_data,
        df: pd.DataFrame,
        symbol: Optional[str],
        wealth_column: str,
        drawdown_column: str,
        save_path: Optional[str | Path]
    ) -> Optional[Path]:
        """Plot combined performance using Plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Wealth Index', 'Drawdown'),
            vertical_spacing=0.1
        )
        
        # Plot wealth index
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=df[wealth_column],
                mode='lines',
                name='Wealth Index',
                line=dict(color='#2E86AB', width=2),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Plot drawdowns
        drawdowns_pct = df[drawdown_column].values * 100
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=drawdowns_pct,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=1.5),
                fillcolor='rgba(255,0,0,0.5)'
            ),
            row=2, col=1
        )
        
        # Formatting
        fig.update_layout(
            title=f'Performance Analysis{" - " + symbol if symbol else ""}',
            hovermode='x unified',
            template='plotly_white',
            height=800
        )
        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_yaxes(title_text='Wealth Index', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
        
        # Save or show
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path))
            logger.info("combined_performance_plot_saved", path=str(save_path))
            return save_path
        elif self.output_dir:
            filename = f"combined_performance_{symbol or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            save_path = self.output_dir / filename
            fig.write_html(str(save_path))
            logger.info("combined_performance_plot_saved", path=str(save_path))
            return save_path
        else:
            fig.show()
            return None

