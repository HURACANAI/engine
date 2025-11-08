"""
Model Factory Adapter for AI Council

This adapter allows the AI Council to use the unified Model Factory
from the strategy-research pipeline instead of individual LLM clients.

Benefits:
- Single interface for all LLM providers
- Easier to add new models
- Consistent error handling
- Shared configuration

Usage:
    from observability.ai_council.model_factory_adapter import ModelFactoryAdapter

    adapter = ModelFactoryAdapter()

    # Get any analyst using Model Factory
    analyst = adapter.get_analyst("gpt4")
    report = analyst.analyze(metrics)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Any, cast
import importlib
import logging

try:
    import structlog
except ImportError:  # pragma: no cover - structlog is optional in some environments
    structlog = None  # type: ignore[assignment]


class _StructlogShim:
    """Lightweight shim to emulate structlog style logging calls."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(self, level: str, event: str, **kwargs: Any) -> None:
        if kwargs:
            extra = ", ".join(f"{key}={value}" for key, value in kwargs.items())
            message = f"{event} | {extra}"
        else:
            message = event
        getattr(self._logger, level)(message)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log("info", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log("warning", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log("error", event, **kwargs)


logging.basicConfig(level=logging.INFO)

if structlog is not None:
    logger = cast(Any, structlog).get_logger(__name__)
else:
    logger = _StructlogShim(__name__)

# Add strategy-research to path
# From observability/ai_council/model_factory_adapter.py
# Go up to project root: observability/ai_council -> observability -> (project root)
# Then look for strategy-research in project root
_adapter_file = Path(__file__).resolve()
_project_root = _adapter_file.parent.parent.parent  # observability -> project root
STRATEGY_RESEARCH_PATH = _project_root / "strategy-research"
if str(STRATEGY_RESEARCH_PATH) not in sys.path:
    sys.path.insert(0, str(STRATEGY_RESEARCH_PATH))

MODEL_FACTORY_AVAILABLE = False
ModelFactoryType: Optional[type[Any]] = None

try:
    # Try to import ModelFactory from strategy-research
    # First ensure the path is correct
    if STRATEGY_RESEARCH_PATH.exists():
        model_factory_module = importlib.import_module("models.model_factory")
        ModelFactoryType = getattr(model_factory_module, "ModelFactory", None)
    else:
        logger.warning(
            "strategy_research_path_not_found",
            path=str(STRATEGY_RESEARCH_PATH),
            message="Strategy-research directory not found"
        )
except ImportError as import_err:
    logger.warning(
        "model_factory_import_failed",
        message="Install strategy-research component to enable Model Factory",
        error=str(import_err),
    )
else:
    if ModelFactoryType is None:
        logger.warning(
            "model_factory_missing_class",
            message="ModelFactory class not found in models.model_factory",
        )
    else:
        MODEL_FACTORY_AVAILABLE = True


class ModelFactoryAdapter:
    """
    Adapter to use Model Factory with AI Council analysts.

    This bridges the strategy-research Model Factory with the Engine's AI Council.
    """

    # Mapping: AI Council analyst name → Model Factory provider
    ANALYST_TO_PROVIDER = {
        "gpt4": ("openai", "gpt-4"),
        "claude_opus": ("anthropic", "claude-opus-4"),
        "claude_sonnet": ("anthropic", "claude-3-5-sonnet-latest"),
        "gemini": ("google", "gemini-2.5-flash"),
        "grok": ("xai", "grok-4-fast-reasoning"),
        "llama": ("ollama", "llama3.2"),
        "deepseek": ("deepseek", "deepseek-reasoner"),
        "perplexity": ("openrouter", "perplexity/llama-3.1-sonar-large-128k-online"),  # via OpenRouter
    }

    def __init__(self, use_factory: bool = True):
        """
        Initialize the adapter.

        Args:
            use_factory: If True, use Model Factory. If False, fall back to direct clients.
        """
        factory_cls = ModelFactoryType
        self.use_factory = bool(factory_cls) and use_factory
        self.factory: Optional[Any] = None
        self.models: Dict[str, Any] = {}

        if self.use_factory and factory_cls is not None:
            logger.info("model_factory_adapter_init", status="using_model_factory")
            self.factory = factory_cls()
            self._initialize_models()
        else:
            if use_factory and not factory_cls:
                logger.warning(
                    "model_factory_unavailable",
                    message="Model Factory requested but not available; falling back",
                )
            logger.info("model_factory_adapter_init", status="model_factory_disabled")

    def _initialize_models(self):
        """Initialize all analyst models using Model Factory"""
        if not self.factory:
            return

        logger.info("initializing_analyst_models", count=len(self.ANALYST_TO_PROVIDER))

        for analyst_name, (provider, model_name) in self.ANALYST_TO_PROVIDER.items():
            try:
                # ModelFactory uses get_model() method, not create_model()
                model = self.factory.get_model(provider, model_name)
                if model:
                    self.models[analyst_name] = {
                        "provider": provider,
                        "model_name": model_name,
                        "instance": model
                    }
                    logger.info(
                        "analyst_model_initialized",
                        analyst=analyst_name,
                        provider=provider,
                        model=model_name
                    )
                else:
                    logger.warning(
                        "analyst_model_not_available",
                        analyst=analyst_name,
                        provider=provider,
                        model=model_name,
                        message="Model factory returned None (API key may be missing)"
                    )
            except Exception as e:
                logger.warning(
                    "analyst_model_failed",
                    analyst=analyst_name,
                    provider=provider,
                    error=str(e)
                )

    def get_model(self, analyst_name: str) -> Optional[Any]:
        """
        Get Model Factory model instance for an analyst.

        Args:
            analyst_name: Name of analyst (gpt4, claude_opus, etc.)

        Returns:
            Model instance or None if not available
        """
        if not self.use_factory:
            return None

        model_data = self.models.get(analyst_name)
        if model_data:
            return model_data["instance"]

        return None

    def generate_response(
        self,
        analyst_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Generate response using Model Factory.

        Args:
            analyst_name: Which analyst to use
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text or None if failed
        """
        model = self.get_model(analyst_name)
        if not model:
            logger.warning(
                "model_not_available",
                analyst=analyst_name,
                available=list(self.models.keys())
            )
            return None

        try:
            response = model.generate_response(
                system_prompt=system_prompt,
                user_content=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            logger.error(
                "model_generation_failed",
                analyst=analyst_name,
                error=str(e)
            )
            return None

    def get_available_analysts(self) -> list[str]:
        """Get list of analysts that are currently available"""
        if not self.use_factory:
            return []
        return list(self.models.keys())

    def get_model_info(self, analyst_name: str) -> Optional[Dict[str, str]]:
        """Get provider and model name for an analyst"""
        if not self.use_factory:
            return None

        model_data = self.models.get(analyst_name)
        if model_data:
            return {
                "provider": model_data["provider"],
                "model_name": model_data["model_name"]
            }
        return None


# Singleton instance
_adapter_instance: Optional[ModelFactoryAdapter] = None


def get_adapter(use_factory: bool = True) -> ModelFactoryAdapter:
    """
    Get singleton adapter instance.

    Args:
        use_factory: Whether to use Model Factory

    Returns:
        ModelFactoryAdapter instance
    """
    global _adapter_instance

    if _adapter_instance is None:
        _adapter_instance = ModelFactoryAdapter(use_factory=use_factory)

    return _adapter_instance


# Example: Updating an existing analyst to use Model Factory
class ModelFactoryAnalyst:
    """
    Base class for analysts using Model Factory.

    Subclass this instead of creating direct LLM clients.
    """

    def __init__(self, analyst_name: str, use_factory: bool = True):
        """
        Initialize analyst.

        Args:
            analyst_name: Name of analyst (must be in ANALYST_TO_PROVIDER)
            use_factory: Whether to use Model Factory
        """
        self.analyst_name = analyst_name
        self.adapter = get_adapter(use_factory=use_factory)

        # Get model info
        info = self.adapter.get_model_info(analyst_name)
        if info:
            logger.info(
                "analyst_initialized",
                analyst=analyst_name,
                provider=info["provider"],
                model=info["model_name"]
            )
        else:
            logger.warning(
                "analyst_no_model_factory",
                analyst=analyst_name,
                message="Model Factory not available, using fallback"
            )

    def analyze(
        self,
        metrics: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """
        Analyze metrics using Model Factory.

        Args:
            metrics: Metrics to analyze
            system_prompt: Custom system prompt (optional)
            temperature: Sampling temperature

        Returns:
            Analysis text
        """
        if system_prompt is None:
            system_prompt = self.get_default_system_prompt()

        user_prompt = self.format_metrics(metrics)

        response = self.adapter.generate_response(
            analyst_name=self.analyst_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )

        return response or "Error: Model not available"

    def get_default_system_prompt(self) -> str:
        """Override this in subclasses"""
        return f"You are {self.analyst_name}, an AI analyst for the Huracan Engine."

    def format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Override this in subclasses"""
        import json
        return f"Analyze these metrics:\n\n{json.dumps(metrics, indent=2)}"


def main():
    """Example usage"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          Model Factory Adapter for AI Council               ║
║          Unified LLM Interface                               ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Initialize adapter
    adapter = get_adapter(use_factory=True)

    # Show available analysts
    print(f"\n✅ Available Analysts:")
    for analyst in adapter.get_available_analysts():
        info = adapter.get_model_info(analyst)
        if not info:
            print(f"   - {analyst}: model details unavailable")
            continue
        print(f"   - {analyst}: {info['provider']}/{info['model_name']}")

    # Example: Generate response
    print(f"\n🧪 Testing GPT-4 analyst...")
    response = adapter.generate_response(
        analyst_name="gpt4",
        system_prompt="You are a trading analyst.",
        user_prompt="What is the key to successful trading?",
        temperature=0.5
    )

    if response:
        print(f"\n📝 Response preview:")
        print(f"   {response[:200]}...")
    else:
        print(f"\n❌ Response generation failed")


if __name__ == "__main__":
    main()
