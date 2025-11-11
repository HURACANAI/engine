#!/usr/bin/env python3
"""
Run Engine Training on Top 3 Coins (BTC, ETH, SOL)

This script:
1. Uses cached candle data (already downloaded)
2. Builds features
3. Trains models
4. Reports how many days of data were used
5. Uploads models to Dropbox

Usage:
    python scripts/run_engine_top3.py --dropbox-token YOUR_TOKEN
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import polars as pl
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.integrations.dropbox_sync import DropboxSync
from src.shared.contracts.writer import ContractWriter
from src.shared.contracts.per_coin import PerCoinMetrics
from src.shared.features.feature_builder import FeatureBuilder
from src.shared.features.feature_builder import FeatureRecipe as FBRecipe
from src.shared.config_loader import load_config

# Try to import model dependencies
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not available, will use LinearRegression")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn not available")


def load_cached_data(symbol: str, lookback_days: int = 180) -> Optional[pd.DataFrame]:
    """Load candle data from cache."""
    cache_dir = Path("data/candles")
    symbol_clean = symbol.replace("/", "-")
    base_coin = symbol.split("/")[0]
    
    # Look for cached files
    coin_dir = cache_dir / base_coin
    if not coin_dir.exists():
        print(f"   ⚠️  No cache directory found: {coin_dir}")
        return None
    
    # Find parquet files for this symbol
    parquet_files = list(coin_dir.glob(f"{symbol_clean}*.parquet"))
    if not parquet_files:
        print(f"   ⚠️  No cached data found for {symbol}")
        return None
    
    # Use the most recent file
    latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
    print(f"   📦 Loading from cache: {latest_file.name}")
    
    try:
        frame = pl.read_parquet(latest_file)
        if frame.is_empty():
            return None
        
        # Convert to pandas
        df = frame.to_pandas()
        
        # Fix timestamp if it's in milliseconds (timestamp column)
        if "timestamp" in df.columns and "ts" not in df.columns:
            # Convert timestamp from milliseconds to datetime
            df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        elif "timestamp" in df.columns and df["timestamp"].dtype in ["int64", "Int64"]:
            # Timestamp is in milliseconds, convert to datetime
            df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        
        # Ensure ts column exists and is datetime
        if "ts" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
                # Try to convert if it's not already datetime
                try:
                    df["ts"] = pd.to_datetime(df["ts"], utc=True)
                except Exception:
                    pass
        
        # Note: Don't filter by date range - use all available cached data
        # The cached data is already the full 1095 days, which is more than the 180 days lookback
        # The model will use the most recent lookback_days worth of data during training
        
        return df
    except Exception as e:
        print(f"   ⚠️  Failed to load cache: {e}")
        return None


def build_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Build features from candle data."""
    if df.empty:
        return pd.DataFrame()
    
    # Create feature recipe
    recipe = FBRecipe(
        timeframes=["1d"],
        indicators={
            "rsi": {"window": 14},
            "ema": {"window": 20},
            "volatility": {"window": 20},
            "momentum": {"window": 10},
        },
        fill_rules={"strategy": "forward_fill"},
        normalization={"type": "standard", "scaler": "StandardScaler"},
        window_sizes={"short": 20, "medium": 50, "long": 200},
    )
    recipe.hash = recipe.compute_hash()
    
    builder = FeatureBuilder(recipe=recipe)
    
    # Build features using rolling window
    features_list = []
    min_window = max(recipe.window_sizes.get("long", 200), 50)
    
    for idx in range(min_window, len(df)):
        window_df = df.iloc[max(0, idx-min_window):idx+1].copy()
        features = builder.build_features(window_df, symbol)
        features_list.append(features)
    
    if features_list:
        features_df = pd.DataFrame(features_list)
        features_df.index = df.index[min_window:]
    else:
        # Fallback: simple features
        features_df = pd.DataFrame()
        if "close" in df.columns:
            features_df['ret_1'] = df['close'].pct_change()
            features_df['ret_5'] = df['close'].pct_change(5)
            features_df['ret_20'] = df['close'].pct_change(20)
        if "volume" in df.columns:
            features_df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features_df = features_df.dropna()
    
    return features_df


def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = "xgboost"):
    """Train a model."""
    if X.empty or len(y) == 0:
        return None, {}
    
    # Remove non-numeric columns
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        return None, {}
    
    # Split into train/test (temporal split)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    # Train model
    if model_type == "xgboost" and HAS_XGBOOST:
        model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif HAS_SKLEARN:
        model = LinearRegression()
    else:
        return None, {}
    
    model.fit(X_train.values, y_train.values)
    
    # Make predictions
    y_pred_train = pd.Series(model.predict(X_train.values), index=X_train.index)
    y_pred_test = pd.Series(model.predict(X_test.values), index=X_test.index)
    
    # Calculate metrics
    test_mse = mean_squared_error(y_test, y_pred_test) if HAS_SKLEARN else 0.0
    test_r2 = r2_score(y_test, y_pred_test) if HAS_SKLEARN else 0.0
    
    # Hit rate
    correct_direction = ((y_pred_test > 0) == (y_test > 0)).sum()
    hit_rate = correct_direction / len(y_test) if len(y_test) > 0 else 0.0
    
    # Sharpe
    mean_return = y_test.mean()
    std_return = y_test.std()
    sharpe = mean_return / std_return if std_return > 0 else 0.0
    
    metrics = {
        "sample_size": len(X_train),
        "sharpe": float(sharpe),
        "hit_rate": float(hit_rate),
        "test_r2": float(test_r2),
        "test_mse": float(test_mse),
        "net_pnl_pct": float(mean_return * 100),
        "gross_pnl_pct": float(mean_return * 100),
        "max_drawdown_pct": 0.0,  # Simplified
    }
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Run engine training on top 3 coins")
    parser.add_argument("--dropbox-token", type=str, help="Dropbox access token")
    parser.add_argument("--days", type=int, default=180, help="Lookback days (default: 180)")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe (default: 1d)")
    
    args = parser.parse_args()
    
    # Top 3 coins
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    print("=" * 70)
    print("🚀 ENGINE TRAINING ON TOP 3 COINS")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Lookback days: {args.days} days")
    print(f"Timeframe: {args.timeframe}")
    print("=" * 70)
    print()
    
    # Load config
    config = load_config()
    lookback_days = config.get("engine", {}).get("lookback_days", args.days)
    print(f"📋 Config lookback_days: {lookback_days} days")
    print()
    
    # Initialize Dropbox (optional)
    dropbox_sync = None
    contract_writer = None
    if args.dropbox_token:
        try:
            settings = EngineSettings.load(environment=os.getenv("HURACAN_ENV", "local"))
            dropbox_sync = DropboxSync(
                access_token=args.dropbox_token,
                app_folder=settings.dropbox.app_folder,
                enabled=True,
            )
            contract_writer = ContractWriter(
                dropbox_sync=dropbox_sync,
                base_folder="huracan",
            )
            print("✅ Dropbox sync enabled")
        except Exception as e:
            print(f"⚠️  Dropbox initialization failed: {e}")
    print()
    
    # Train each symbol
    results = {}
    
    for idx, symbol in enumerate(symbols, 1):
        print(f"[{idx}/{len(symbols)}] 🎯 Training {symbol}...")
        print("-" * 70)
        
        try:
            # Load data
            print(f"   📥 Loading data...")
            df = load_cached_data(symbol, lookback_days=lookback_days)
            
            if df is None or df.empty:
                print(f"   ❌ No data available for {symbol}")
                results[symbol] = {
                    "success": False,
                    "error": "No data available",
                    "days": 0,
                }
                continue
            
            # Calculate actual days used
            if "ts" in df.columns and pd.api.types.is_datetime64_any_dtype(df["ts"]):
                start_date = df["ts"].min()
                end_date = df["ts"].max()
                actual_days = (end_date - start_date).days
                rows = len(df)
                
                # Limit to lookback_days for training (use most recent data)
                if actual_days > lookback_days:
                    # Use only the most recent lookback_days
                    cutoff_date = end_date - pd.Timedelta(days=lookback_days)
                    df = df[df["ts"] >= cutoff_date]
                    training_days = lookback_days
                    training_rows = len(df)
                    print(f"   ✅ Loaded {rows} rows (full cache)")
                    print(f"   📅 Full date range: {start_date.date()} to {end_date.date()} ({actual_days} days)")
                    print(f"   📅 Training on: {cutoff_date.date()} to {end_date.date()} ({training_days} days)")
                    print(f"   📊 Training samples: {training_rows} candles")
                else:
                    training_days = actual_days
                    training_rows = rows
                    print(f"   ✅ Loaded {rows} rows")
                    print(f"   📅 Date range: {start_date.date()} to {end_date.date()}")
                    print(f"   📅 Training days: {training_days} days (all available data)")
            else:
                training_days = lookback_days
                training_rows = len(df)
                print(f"   ✅ Loaded {training_rows} rows")
                print(f"   ⚠️  No timestamp column, using all data (assumed {lookback_days} days)")
            
            # Build features
            print(f"   🔧 Building features...")
            features_df = build_features(df, symbol)
            
            if features_df.empty:
                print(f"   ❌ Feature building failed")
                results[symbol] = {
                    "success": False,
                    "error": "Feature building failed",
                    "days": actual_days,
                }
                continue
            
            print(f"   ✅ Built {len(features_df.columns)} features, {len(features_df)} samples")
            
            # Prepare target
            if "close" in df.columns:
                target = df['close'].pct_change().shift(-1)
                aligned_target = target.loc[features_df.index]
                aligned_target = aligned_target.dropna()
                aligned_features = features_df.loc[aligned_target.index]
            else:
                print(f"   ❌ No close price column")
                results[symbol] = {
                    "success": False,
                    "error": "No close price",
                    "days": actual_days,
                }
                continue
            
            # Train model
            print(f"   🎯 Training model...")
            model, metrics = train_model(aligned_features, aligned_target, model_type="xgboost")
            
            if model is None:
                print(f"   ❌ Model training failed")
                results[symbol] = {
                    "success": False,
                    "error": "Model training failed",
                    "days": actual_days,
                }
                continue
            
            print(f"   ✅ Model trained")
            print(f"      Sample size: {metrics.get('sample_size', 0)}")
            print(f"      Sharpe: {metrics.get('sharpe', 0):.4f}")
            print(f"      Hit rate: {metrics.get('hit_rate', 0):.4f}")
            print(f"      R²: {metrics.get('test_r2', 0):.4f}")
            
            # Save model (optional)
            if dropbox_sync and contract_writer:
                try:
                    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
                    
                    # Save model locally first
                    model_dir = Path("models") / symbol.replace("/", "") / timestamp
                    model_dir.mkdir(parents=True, exist_ok=True)
                    
                    import pickle
                    model_path = model_dir / "model.bin"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    
                    # Upload to Dropbox
                    remote_path = contract_writer.write_model_file(
                        model_path=str(model_path),
                        symbol=symbol.replace("/", ""),
                        date_str=date_str,
                    )
                    
                    if remote_path:
                        print(f"   📤 Uploaded model to Dropbox: {remote_path}")
                    
                    # Upload metrics
                    metrics_obj = PerCoinMetrics(
                        symbol=symbol,
                        sample_size=metrics.get('sample_size', 0),
                        gross_pnl_pct=metrics.get('gross_pnl_pct', 0.0),
                        net_pnl_pct=metrics.get('net_pnl_pct', 0.0),
                        sharpe=metrics.get('sharpe', 0.0),
                        hit_rate=metrics.get('hit_rate', 0.0),
                        max_drawdown_pct=metrics.get('max_drawdown_pct', 0.0),
                        avg_trade_bps=metrics.get('net_pnl_pct', 0.0) * 100,
                        validation_windows={},
                        costs_bps_used={},
                    )
                    
                    metrics_remote_path = contract_writer.write_metrics(metrics_obj, date_str)
                    if metrics_remote_path:
                        print(f"   📤 Uploaded metrics to Dropbox: {metrics_remote_path}")
                
                except Exception as e:
                    print(f"   ⚠️  Failed to upload to Dropbox: {e}")
            
            # Get training days (after filtering)
            if "ts" in df.columns and pd.api.types.is_datetime64_any_dtype(df["ts"]):
                training_start = df["ts"].min()
                training_end = df["ts"].max()
                training_days_final = (training_end - training_start).days
            else:
                training_start = None
                training_end = None
                training_days_final = training_days
            
            results[symbol] = {
                "success": True,
                "days": training_days_final,  # Days actually used for training
                "rows": training_rows,  # Rows actually used for training
                "start_date": training_start.isoformat() if training_start is not None else None,
                "end_date": training_end.isoformat() if training_end is not None else None,
                "metrics": metrics,
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results[symbol] = {
                "success": False,
                "error": str(e),
                "days": 0,
            }
        
        print()
    
    # Final summary
    print("=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    print()
    
    successful = sum(1 for r in results.values() if r.get("success"))
    print(f"✅ Successful: {successful}/{len(symbols)}")
    print(f"❌ Failed: {len(symbols) - successful}/{len(symbols)}")
    print()
    
    print("📅 DATA USAGE SUMMARY")
    print("-" * 70)
    print(f"Configured lookback: {lookback_days} days")
    print()
    
    for symbol, result in results.items():
        if result.get("success"):
            days = result.get("days", 0)
            rows = result.get("rows", 0)
            start_date = result.get("start_date", "N/A")
            end_date = result.get("end_date", "N/A")
            metrics = result.get("metrics", {})
            
            print(f"   {symbol}:")
            print(f"      ✅ Trained successfully")
            print(f"      📅 Days used: {days} days")
            print(f"      📊 Rows: {rows} candles")
            print(f"      📅 Date range: {start_date} to {end_date}")
            print(f"      📈 Metrics: Sharpe={metrics.get('sharpe', 0):.4f}, Hit Rate={metrics.get('hit_rate', 0):.4f}")
        else:
            print(f"   {symbol}:")
            print(f"      ❌ Failed: {result.get('error', 'Unknown error')}")
            print(f"      📅 Days: {result.get('days', 0)}")
    
    print()
    print("=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print()
    
    # Report total days
    total_days = sum(r.get("days", 0) for r in results.values() if r.get("success"))
    avg_days = total_days / successful if successful > 0 else 0
    
    print(f"📊 Average days per coin: {avg_days:.1f} days")
    print(f"📊 Total days across all coins: {total_days} days")
    print()


if __name__ == "__main__":
    main()

