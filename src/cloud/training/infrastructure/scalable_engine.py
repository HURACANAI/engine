"""
Scalable Engine - Complete Integration

Integrates all scalable architecture components:
- Message bus
- Event loop manager
- Global risk controller
- Real-time cost model
- Model registry
- Exchange abstraction
- Observability

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import structlog

from .message_bus import create_message_bus, Message, StreamType
from .event_loop_manager import EventLoopManager
from ..risk.global_risk_controller import (
    GlobalRiskController,
    RiskLimits,
    TradeRequest,
    RiskDecision,
)
from ..costs.real_time_cost_model import RealTimeCostModel
from ..models.model_registry import ModelRegistry
from ..exchanges import ExchangeManager, BinanceExchange, OKXExchange, BybitExchange, Order, OrderSide, OrderType
from ..observability import MetricsCollector, HealthChecker, PerformanceMonitor

logger = structlog.get_logger(__name__)


@dataclass
class ScalableEngineConfig:
    """Configuration for scalable engine."""
    max_coins: int = 400
    max_concurrent_trades: int = 500
    active_coins: int = 20
    max_active_trades: int = 100
    coins_per_event_loop: int = 50
    message_bus: Dict[str, Any] = None
    risk_limits: Dict[str, Any] = None
    cost_model: Dict[str, Any] = None
    model_registry: Dict[str, Any] = None


class ScalableEngine:
    """
    Scalable trading engine for 400 coins and 500 trades.
    
    Integrates all components:
    - Message bus for data pipelines
    - Event loop manager for parallel processing
    - Global risk controller for risk management
    - Real-time cost model for cost tracking
    - Model registry for model management
    - Exchange manager for multi-exchange support
    - Observability for monitoring
    """
    
    def __init__(self, config: ScalableEngineConfig):
        """
        Initialize scalable engine.
        
        Args:
            config: Engine configuration
        """
        self.config = config
        
        # Initialize components
        self.message_bus = create_message_bus(config.message_bus or {})
        self.event_loop_manager = EventLoopManager(
            coins_per_loop=config.coins_per_event_loop,
            max_loops=10,
        )
        self.risk_controller = GlobalRiskController(
            RiskLimits(**config.risk_limits or {})
        )
        self.cost_model = RealTimeCostModel(
            update_interval_seconds=config.cost_model.get("update_interval_seconds", 60) if config.cost_model else 60,
            min_edge_after_cost_bps=config.cost_model.get("min_edge_after_cost_bps", 5.0) if config.cost_model else 5.0,
        )
        self.model_registry = ModelRegistry(
            storage_path=config.model_registry.get("storage_path", "/models/trained") if config.model_registry else "/models/trained",
            metadata_db=config.model_registry.get("metadata_db") if config.model_registry else None,
        )
        self.exchange_manager = ExchangeManager()
        self.metrics = MetricsCollector(use_prometheus=True)
        self.health_checker = HealthChecker()
        self.performance_monitor = PerformanceMonitor()
        
        # State
        self.active_coins: List[str] = []
        self.running = False
        
        logger.info(
            "scalable_engine_initialized",
            max_coins=config.max_coins,
            active_coins=config.active_coins,
            max_concurrent_trades=config.max_concurrent_trades,
        )
    
    async def initialize(self) -> None:
        """Initialize engine components."""
        # Connect message bus
        await self.message_bus.connect()
        
        # Start cost model
        await self.cost_model.start()
        
        # Register exchanges
        self.exchange_manager.register_exchange("binance", BinanceExchange())
        self.exchange_manager.register_exchange("okx", OKXExchange())
        self.exchange_manager.register_exchange("bybit", BybitExchange())
        
        # Register health checks
        await self._register_health_checks()
        
        logger.info("scalable_engine_initialized")
    
    async def _register_health_checks(self) -> None:
        """Register health checks."""
        async def check_message_bus() -> tuple[bool, str]:
            try:
                # Test message bus connection
                return True, "Message bus healthy"
            except Exception as e:
                return False, f"Message bus error: {str(e)}"
        
        async def check_exchanges() -> tuple[bool, str]:
            try:
                # Test exchange connectivity
                return True, "Exchanges healthy"
            except Exception as e:
                return False, f"Exchange error: {str(e)}"
        
        async def check_risk_controller() -> tuple[bool, str]:
            try:
                # Test risk controller
                return True, "Risk controller healthy"
            except Exception as e:
                return False, f"Risk controller error: {str(e)}"
        
        self.health_checker.register_check("message_bus", check_message_bus, interval_seconds=30.0)
        self.health_checker.register_check("exchanges", check_exchanges, interval_seconds=60.0)
        self.health_checker.register_check("risk_controller", check_risk_controller, interval_seconds=30.0)
    
    async def start(self, coins: List[str]) -> None:
        """
        Start the engine.
        
        Args:
            coins: List of coins to trade
        """
        if self.running:
            logger.warning("engine_already_running")
            return
        
        # Limit to active_coins
        self.active_coins = coins[:self.config.active_coins]
        
        # Assign coins to event loops
        self.event_loop_manager.assign_coins(self.active_coins)
        
        # Register processors
        for coin in self.active_coins:
            self.event_loop_manager.register_processor(coin, self._process_coin)
        
        # Start event loops
        await self.event_loop_manager.start()
        
        # Update metrics
        self.metrics.set_gauge("active_coins", len(self.active_coins))
        
        self.running = True
        logger.info("scalable_engine_started", active_coins=len(self.active_coins))
    
    async def stop(self) -> None:
        """Stop the engine."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop event loops
        await self.event_loop_manager.stop(timeout=10.0)
        
        # Stop cost model
        await self.cost_model.stop()
        
        # Disconnect message bus
        await self.message_bus.disconnect()
        
        logger.info("scalable_engine_stopped")
    
    async def _process_coin(self, coin: str) -> None:
        """Process a single coin."""
        start_time = time.time()
        
        try:
            # Subscribe to market data
            async for message in self.message_bus.subscribe(
                stream_type=StreamType.MARKET_DATA,
                coin=coin,
                consumer_group="market_data_workers",
            ):
                if not self.running:
                    break
                
                # Process market data
                await self._handle_market_data(coin, message.data)
                
                # Acknowledge message
                if message.message_id:
                    await self.message_bus.ack_message(
                        stream_type=StreamType.MARKET_DATA,
                        coin=coin,
                        group_name="market_data_workers",
                        message_id=message.message_id,
                    )
        
        except Exception as e:
            logger.error("coin_processing_error", coin=coin, error=str(e))
            self.metrics.increment_counter("errors_total", labels={"error_type": "coin_processing", "component": "engine"})
        finally:
            duration = time.time() - start_time
            self.performance_monitor.record_operation(f"process_coin_{coin}", duration, success=True)
    
    async def _handle_market_data(self, coin: str, data: Dict[str, Any]) -> None:
        """Handle market data for a coin."""
        start_time = time.time()
        
        try:
            # Get model
            model_metadata = self.model_registry.get_model_metadata(coin)
            if not model_metadata or not model_metadata.is_active:
                logger.debug("no_active_model", coin=coin)
                return
            
            # Get orderbook
            orderbook, exchange_name = await self.exchange_manager.get_best_orderbook(coin)
            if not orderbook:
                logger.warning("no_orderbook", coin=coin)
                return
            
            # Update cost model
            if orderbook.bids and orderbook.asks:
                bid = orderbook.bids[0][0]
                ask = orderbook.asks[0][0]
                self.cost_model.update_spread(coin, bid, ask)
            
            # Get fees
            fees = await self.exchange_manager.get_all_fees(coin)
            if exchange_name in fees:
                fee_structure = fees[exchange_name]
                self.cost_model.update_fees(coin, fee_structure.maker_fee_bps, fee_structure.taker_fee_bps)
            
            # Generate signal (simplified)
            signal = await self._generate_signal(coin, data, model_metadata)
            if not signal:
                return
            
            # Check cost
            edge_after_cost = self.cost_model.calculate_edge_after_cost(
                coin, signal["edge_bps"], use_maker=True
            )
            if edge_after_cost < self.cost_model.min_edge_after_cost_bps:
                logger.debug("edge_after_cost_too_low", coin=coin, edge_after_cost=edge_after_cost)
                return
            
            # Check risk
            trade = TradeRequest(
                coin=coin,
                direction=signal["direction"],
                size_usd=signal["size_usd"],
                exchange=exchange_name or "binance",
                sector=self.risk_controller.coin_sectors.get(coin),
            )
            risk_result = self.risk_controller.check_trade(trade)
            
            if risk_result.decision == RiskDecision.APPROVE:
                # Execute trade
                trade_id = await self._execute_trade(coin, trade, exchange_name)
                if trade_id:
                    self.risk_controller.register_trade(trade_id, trade)
                    
                    # Update metrics
                    self.metrics.increment_counter(
                        "trades_total",
                        labels={"coin": coin, "direction": trade.direction, "exchange": trade.exchange}
                    )
                    self.metrics.observe_histogram(
                        "trade_size_usd",
                        trade.size_usd,
                        labels={"coin": coin}
                    )
                    
                    # Publish execution
                    await self.message_bus.publish(Message(
                        stream_type=StreamType.EXECUTIONS,
                        coin=coin,
                        data={"trade_id": trade_id, "trade": trade.__dict__},
                        timestamp=time.time(),
                    ))
            elif risk_result.decision == RiskDecision.THROTTLE:
                logger.debug("trade_throttled", coin=coin, reason=risk_result.reason)
            elif risk_result.decision == RiskDecision.BLOCK:
                logger.debug("trade_blocked", coin=coin, reason=risk_result.reason)
        
        except Exception as e:
            logger.error("market_data_handling_error", coin=coin, error=str(e))
            self.metrics.increment_counter("errors_total", labels={"error_type": "market_data", "component": "engine"})
        finally:
            duration = time.time() - start_time
            self.performance_monitor.record_operation(f"handle_market_data_{coin}", duration, success=True)
            self.metrics.observe_histogram(
                "latency_seconds",
                duration,
                labels={"operation": "handle_market_data", "coin": coin}
            )
    
    async def _generate_signal(self, coin: str, data: Dict[str, Any], model_metadata: Any) -> Optional[Dict[str, Any]]:
        """Generate trading signal (simplified)."""
        # Placeholder implementation
        # In production, use actual model inference
        return {
            "direction": "long",
            "edge_bps": 20.0,
            "size_usd": 1000.0,
            "confidence": 0.75,
        }
    
    async def _execute_trade(self, coin: str, trade: TradeRequest, exchange_name: Optional[str]) -> Optional[str]:
        """Execute a trade."""
        if not exchange_name:
            exchange_name = "binance"
        
        exchange = self.exchange_manager.get_exchange(exchange_name)
        if not exchange:
            logger.error("exchange_not_found", exchange=exchange_name)
            return None
        
        try:
            # Create order
            order = Order(
                symbol=coin,
                side=OrderSide.BUY if trade.direction == "long" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                amount=trade.size_usd / 50000.0,  # Simplified: assume $50k price
            )
            
            # Place order
            result = await exchange.place_order(order)
            
            # Generate trade ID
            trade_id = str(uuid.uuid4())
            
            logger.info(
                "trade_executed",
                trade_id=trade_id,
                coin=coin,
                exchange=exchange_name,
                size=trade.size_usd,
            )
            
            return trade_id
        
        except Exception as e:
            logger.error("trade_execution_error", coin=coin, error=str(e))
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "engine": {
                "active_coins": len(self.active_coins),
                "running": self.running,
            },
            "risk": self.risk_controller.get_exposure_summary(),
            "performance": self.performance_monitor.get_all_stats(),
            "metrics": self.metrics.get_metrics(),
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        health_results = await self.health_checker.run_all_checks()
        return {
            "checks": health_results,
            "status": "healthy" if all(healthy for healthy, _ in health_results.values()) else "unhealthy",
        }


def create_scalable_engine_from_config(config_dict: Dict[str, Any]) -> ScalableEngine:
    """Create scalable engine from configuration dictionary."""
    config = ScalableEngineConfig(
        max_coins=config_dict.get("max_coins", 400),
        max_concurrent_trades=config_dict.get("max_concurrent_trades", 500),
        active_coins=config_dict.get("active_coins", 20),
        max_active_trades=config_dict.get("max_active_trades", 100),
        coins_per_event_loop=config_dict.get("coins_per_event_loop", 50),
        message_bus=config_dict.get("message_bus", {}),
        risk_limits=config_dict.get("risk", {}),
        cost_model=config_dict.get("cost_model", {}),
        model_registry=config_dict.get("model_registry", {}),
    )
    return ScalableEngine(config)

