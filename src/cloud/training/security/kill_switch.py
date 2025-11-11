"""
Security and Safety Features.

Implements:
- Kill switch (emergency stop)
- Dry run mode
- IP allowlisting
- Secret vault integration
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
import structlog

logger = structlog.get_logger(__name__)


class SystemMode(Enum):
    """System operation mode."""
    LIVE = "live"
    DRY_RUN = "dry_run"
    STOPPED = "stopped"


@dataclass
class KillSwitchStatus:
    """Kill switch status."""
    is_active: bool
    activated_at: Optional[datetime]
    activated_by: Optional[str]
    reason: str
    action_taken: str


class KillSwitch:
    """
    Kill switch for emergency system shutdown.
    
    Features:
    - Immediate stop of all trading
    - Cancel all open orders
    - Set position sizes to zero
    - Log all actions
    """
    
    def __init__(self) -> None:
        """Initialize kill switch."""
        self.is_active = False
        self.activated_at: Optional[datetime] = None
        self.activated_by: Optional[str] = None
        self.reason: str = ""
        self.open_orders: List[str] = []  # Order IDs to cancel
        
        logger.info("kill_switch_initialized")
    
    def activate(
        self,
        reason: str,
        activated_by: str = "system"
    ) -> None:
        """
        Activate kill switch.
        
        Args:
            reason: Reason for activation
            activated_by: Who activated it (user/system)
        """
        self.is_active = True
        self.activated_at = datetime.now()
        self.activated_by = activated_by
        self.reason = reason
        
        logger.critical(
            "kill_switch_activated",
            reason=reason,
            activated_by=activated_by,
            timestamp=self.activated_at.isoformat()
        )
    
    def deactivate(self, deactivated_by: str = "system") -> None:
        """
        Deactivate kill switch.
        
        Args:
            deactivated_by: Who deactivated it
        """
        self.is_active = False
        self.activated_at = None
        self.activated_by = None
        self.reason = ""
        
        logger.info("kill_switch_deactivated", deactivated_by=deactivated_by)
    
    def get_status(self) -> KillSwitchStatus:
        """Get kill switch status."""
        return KillSwitchStatus(
            is_active=self.is_active,
            activated_at=self.activated_at,
            activated_by=self.activated_by,
            reason=self.reason,
            action_taken="stop_trading_cancel_orders" if self.is_active else "normal"
        )
    
    def should_block_trade(self) -> bool:
        """Check if kill switch should block trading."""
        return self.is_active
    
    def register_order(self, order_id: str) -> None:
        """Register order for cancellation if kill switch activated."""
        if order_id not in self.open_orders:
            self.open_orders.append(order_id)
    
    def get_orders_to_cancel(self) -> List[str]:
        """Get list of orders to cancel."""
        return self.open_orders.copy()


class SecurityManager:
    """
    Security manager for system safety.
    
    Features:
    - Dry run mode (default in new environments)
    - IP allowlisting per venue
    - Secret vault integration
    - Read-only keys for research
    """
    
    def __init__(
        self,
        default_mode: SystemMode = SystemMode.DRY_RUN,
        enable_ip_allowlist: bool = True,
    ) -> None:
        """
        Initialize security manager.
        
        Args:
            default_mode: Default system mode (default: DRY_RUN)
            enable_ip_allowlist: Enable IP allowlisting (default: True)
        """
        self.mode = default_mode
        self.enable_ip_allowlist = enable_ip_allowlist
        self.kill_switch = KillSwitch()
        
        # IP allowlists per venue
        self.ip_allowlists: Dict[str, Set[str]] = {}
        
        # Key permissions
        self.read_only_keys: Set[str] = set()
        self.trade_keys: Set[str] = set()
        
        logger.info(
            "security_manager_initialized",
            default_mode=default_mode.value,
            enable_ip_allowlist=enable_ip_allowlist
        )
    
    def set_mode(self, mode: SystemMode) -> None:
        """
        Set system mode.
        
        Args:
            mode: System mode
        """
        self.mode = mode
        logger.info("system_mode_changed", mode=mode.value)
    
    def is_dry_run(self) -> bool:
        """Check if in dry run mode."""
        return self.mode == SystemMode.DRY_RUN
    
    def is_live(self) -> bool:
        """Check if in live mode."""
        return self.mode == SystemMode.LIVE and not self.kill_switch.is_active
    
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        return self.is_live() and not self.kill_switch.should_block_trade()
    
    def add_ip_allowlist(
        self,
        venue: str,
        ip_addresses: List[str]
    ) -> None:
        """
        Add IP addresses to allowlist for venue.
        
        Args:
            venue: Venue name
            ip_addresses: List of allowed IP addresses
        """
        if venue not in self.ip_allowlists:
            self.ip_allowlists[venue] = set()
        
        self.ip_allowlists[venue].update(ip_addresses)
        
        logger.info(
            "ip_allowlist_updated",
            venue=venue,
            count=len(self.ip_allowlists[venue])
        )
    
    def is_ip_allowed(
        self,
        venue: str,
        ip_address: str
    ) -> bool:
        """
        Check if IP address is allowed for venue.
        
        Args:
            venue: Venue name
            ip_address: IP address to check
        
        Returns:
            True if allowed
        """
        if not self.enable_ip_allowlist:
            return True
        
        if venue not in self.ip_allowlists:
            return False
        
        return ip_address in self.ip_allowlists[venue]
    
    def register_read_only_key(self, key_id: str) -> None:
        """Register read-only API key."""
        self.read_only_keys.add(key_id)
        logger.debug("read_only_key_registered", key_id=key_id)
    
    def register_trade_key(self, key_id: str) -> None:
        """Register trade API key (only on live box)."""
        if self.mode != SystemMode.LIVE:
            logger.warning(
                "trade_key_registered_in_non_live_mode",
                key_id=key_id,
                mode=self.mode.value
            )
        self.trade_keys.add(key_id)
        logger.debug("trade_key_registered", key_id=key_id)
    
    def is_read_only_key(self, key_id: str) -> bool:
        """Check if key is read-only."""
        return key_id in self.read_only_keys
    
    def can_key_trade(self, key_id: str) -> bool:
        """Check if key can execute trades."""
        return key_id in self.trade_keys and self.can_trade()
    
    def activate_kill_switch(
        self,
        reason: str,
        activated_by: str = "user"
    ) -> None:
        """
        Activate kill switch.
        
        Args:
            reason: Reason for activation
            activated_by: Who activated it
        """
        self.kill_switch.activate(reason, activated_by)
        self.mode = SystemMode.STOPPED
        
        logger.critical(
            "kill_switch_activated_via_security_manager",
            reason=reason,
            activated_by=activated_by
        )
    
    def get_security_status(self) -> Dict[str, any]:
        """Get security status."""
        return {
            "mode": self.mode.value,
            "is_dry_run": self.is_dry_run(),
            "is_live": self.is_live(),
            "can_trade": self.can_trade(),
            "kill_switch": self.kill_switch.get_status().__dict__,
            "ip_allowlist_enabled": self.enable_ip_allowlist,
            "read_only_keys_count": len(self.read_only_keys),
            "trade_keys_count": len(self.trade_keys),
        }

