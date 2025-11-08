"""
Strategy Translator: Converts moon-dev backtests into Huracan AlphaEngines

This adapter bridges the strategy-research pipeline with the Huracan Engine.

Workflow:
1. Parse backtest code from RBI agent
2. Extract signal logic (entry/exit conditions, indicators)
3. Generate AlphaEngine subclass
4. Validate on Engine's data pipeline
5. Register as new alpha engine

Author: Huracan Engine Team
Based on moon-dev-ai-agents integration
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add strategy-research to path for Model Factory import
STRATEGY_RESEARCH_PATH = Path(__file__).parent.parent.parent.parent.parent / "strategy-research"
if str(STRATEGY_RESEARCH_PATH) not in sys.path:
    sys.path.insert(0, str(STRATEGY_RESEARCH_PATH))

try:
    from models.model_factory import ModelFactory
except ImportError:
    print("⚠️  Warning: Model Factory not available. Install strategy-research component.")
    ModelFactory = None


@dataclass
class ExtractedStrategy:
    """Represents extracted strategy logic from backtest code"""
    name: str
    strategy_type: str  # trend, reversal, breakout, etc.
    indicators: List[str]  # Required indicators (RSI, MACD, etc.)
    entry_conditions: List[str]  # Entry logic
    exit_conditions: List[str]  # Exit logic
    timeframe: str  # Recommended timeframe
    stop_loss: Optional[str]  # SL logic if present
    take_profit: Optional[str]  # TP logic if present
    confidence_factors: List[str]  # What increases confidence
    raw_code: str  # Original backtest code


@dataclass
class GeneratedEngine:
    """Represents a generated AlphaEngine"""
    engine_name: str
    engine_code: str
    file_path: Path
    strategy: ExtractedStrategy
    validation_status: str  # pending, passed, failed
    performance_metrics: Optional[Dict] = None


class StrategyTranslator:
    """
    Translates backtest strategies into Huracan AlphaEngines.

    Uses AI to extract signal logic and generate production-ready engine code.
    """

    def __init__(self, llm_provider: str = "anthropic", llm_model: str = None):
        """
        Initialize the Strategy Translator.

        Args:
            llm_provider: LLM provider (anthropic, openai, deepseek, etc.)
            llm_model: Specific model name (optional)
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        # Initialize Model Factory
        if ModelFactory:
            self.factory = ModelFactory()
            try:
                # ModelFactory uses get_model() method
                if llm_model:
                    self.model = self.factory.get_model(llm_provider, llm_model)
                else:
                    self.model = self.factory.get_model(llm_provider)
                if self.model:
                print(f"✅ Strategy Translator initialized with {llm_provider}/{llm_model or 'default'}")
                else:
                    print(f"⚠️  Model not available - check API keys for {llm_provider}")
                    self.model = None
            except Exception as e:
                print(f"❌ Failed to initialize LLM model: {e}")
                self.model = None
        else:
            self.model = None
            print("⚠️  Running without AI model (manual translation only)")

        # Output directories
        self.engine_dir = Path(__file__).parent.parent / "models" / "ai_generated_engines"
        self.engine_dir.mkdir(parents=True, exist_ok=True)

    def extract_strategy_logic(self, backtest_code: str) -> ExtractedStrategy:
        """
        Extract strategy logic from backtest code using AI.

        Args:
            backtest_code: Python code from RBI agent

        Returns:
            ExtractedStrategy with parsed logic
        """
        if not self.model:
            raise RuntimeError("AI model not initialized. Cannot extract strategy logic.")

        print("\n🔍 Extracting strategy logic from backtest...")

        system_prompt = """You are an expert at analyzing trading backtest code and extracting the core strategy logic.

Your task: Analyze the backtest code and extract:
1. Strategy name (short, descriptive)
2. Strategy type (trend/reversal/breakout/arbitrage/scalping/etc.)
3. Technical indicators used (RSI, MACD, EMA, volume, etc.)
4. Entry conditions (when to open a position)
5. Exit conditions (when to close a position)
6. Stop loss logic (if any)
7. Take profit logic (if any)
8. Confidence factors (what makes this a high-confidence trade)

Output as JSON:
{
    "name": "strategy_name",
    "strategy_type": "type",
    "indicators": ["RSI", "EMA_20", ...],
    "entry_conditions": ["RSI < 30", "price > EMA_20", ...],
    "exit_conditions": ["RSI > 70", "price < EMA_20", ...],
    "timeframe": "15m",
    "stop_loss": "2% below entry" or null,
    "take_profit": "5% above entry" or null,
    "confidence_factors": ["strong volume", "regime = TREND", ...]
}"""

        user_prompt = f"""Analyze this backtest code and extract the strategy logic:

```python
{backtest_code}
```

Extract the core logic that can be converted into an AlphaEngine."""

        response = self.model.generate_response(
            system_prompt=system_prompt,
            user_content=user_prompt,
            temperature=0.2,
            max_tokens=2000
        )

        # Parse JSON response
        import json
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: manual extraction
            print("⚠️  Failed to parse JSON, using fallback extraction")
            data = self._fallback_extract(backtest_code)

        strategy = ExtractedStrategy(
            name=data.get("name", "unknown_strategy"),
            strategy_type=data.get("strategy_type", "unknown"),
            indicators=data.get("indicators", []),
            entry_conditions=data.get("entry_conditions", []),
            exit_conditions=data.get("exit_conditions", []),
            timeframe=data.get("timeframe", "15m"),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            confidence_factors=data.get("confidence_factors", []),
            raw_code=backtest_code
        )

        print(f"✅ Extracted strategy: {strategy.name} ({strategy.strategy_type})")
        print(f"   Indicators: {', '.join(strategy.indicators)}")
        print(f"   Entry conditions: {len(strategy.entry_conditions)}")
        print(f"   Exit conditions: {len(strategy.exit_conditions)}")

        return strategy

    def _fallback_extract(self, code: str) -> dict:
        """Fallback extraction using regex and AST parsing"""
        # Simple heuristic-based extraction
        indicators = []
        if "rsi" in code.lower():
            indicators.append("RSI")
        if "macd" in code.lower():
            indicators.append("MACD")
        if "ema" in code.lower() or "sma" in code.lower():
            indicators.append("MA")

        return {
            "name": "extracted_strategy",
            "strategy_type": "unknown",
            "indicators": indicators,
            "entry_conditions": ["extracted from code"],
            "exit_conditions": ["extracted from code"],
            "timeframe": "15m",
            "confidence_factors": []
        }

    def generate_alpha_engine(self, strategy: ExtractedStrategy) -> GeneratedEngine:
        """
        Generate AlphaEngine code from extracted strategy.

        Args:
            strategy: Extracted strategy logic

        Returns:
            GeneratedEngine with production-ready code
        """
        if not self.model:
            raise RuntimeError("AI model not initialized. Cannot generate engine code.")

        print(f"\n💻 Generating AlphaEngine for: {strategy.name}")

        system_prompt = """You are an expert at writing Huracan AlphaEngine subclasses.

Generate production-ready AlphaEngine code that:
1. Is a standalone class (no base class inheritance)
2. Implements calculate_features() method (optional preprocessing)
3. Implements generate_signal() method (core logic)
4. Returns AlphaSignal with ALL required fields:
   - technique: TradingTechnique enum
   - direction: "buy", "sell", or "hold"
   - confidence: float (0.0-1.0)
   - reasoning: str explaining the signal
   - key_features: Dict[str, float] with relevant feature values
   - regime_affinity: float (0.0-1.0) - how well this regime suits the strategy
5. Uses regime awareness (TREND, RANGE, PANIC)
6. Is clean, well-documented, and error-free

Template:
```python
from cloud.training.models.alpha_engines import AlphaSignal, TradingTechnique

class AIGeneratedEngine_{NAME}:
    \"\"\"
    {DESCRIPTION}

    Strategy Type: {TYPE}
    Entry: {ENTRY}
    Exit: {EXIT}
    \"\"\"

    METADATA = {{
        "source": "rbi_agent",
        "generation_date": "{DATE}",
        "strategy_type": "{TYPE}",
        "status": "testing",
        "description": "{DESCRIPTION}"
    }}

    def __init__(self):
        self.name = "{name}"

    def calculate_features(self, df):
        \"\"\"Calculate required indicators\"\"\"
        # Add indicator calculations here if needed
        # Most features come from FeatureRecipe
        return df

    def generate_signal(self, symbol, df, regime, meta):
        \"\"\"
        Generate trading signal based on strategy logic.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            df: DataFrame with OHLCV + features
            regime: Current market regime (TREND/RANGE/PANIC)
            meta: Metadata dict

        Returns:
            AlphaSignal with all required fields
        \"\"\"
        # Check for sufficient data
        if len(df) < 20:
            return AlphaSignal(
                technique=TradingTechnique.RANGE,  # Choose appropriate technique
                direction="hold",
                confidence=0.0,
                reasoning="Insufficient data",
                key_features={{}},
                regime_affinity=0.0
            )

        # Implement entry/exit logic here

        # Example BUY signal:
        # if entry_condition:
        #     return AlphaSignal(
        #         technique=TradingTechnique.TREND,  # Or RANGE, BREAKOUT, etc.
        #         direction="buy",
        #         confidence=0.7,
        #         reasoning="Entry condition met: [explain why]",
        #         key_features={{"feature_name": feature_value}},
        #         regime_affinity=0.8  # Higher if regime is favorable
        #     )

        # Example SELL signal:
        # elif exit_condition:
        #     return AlphaSignal(
        #         technique=TradingTechnique.TREND,
        #         direction="sell",
        #         confidence=0.6,
        #         reasoning="Exit condition met: [explain why]",
        #         key_features={{"feature_name": feature_value}},
        #         regime_affinity=0.7
        #     )

        # No signal (hold)
        return AlphaSignal(
            technique=TradingTechnique.RANGE,
            direction="hold",
            confidence=0.0,
            reasoning="No clear signal",
            key_features={{}},
            regime_affinity=0.5
        )
```

IMPORTANT:
- Use only features available in df (from FeatureRecipe)
- Handle missing data gracefully
- Adjust confidence and regime_affinity based on current regime
- ALWAYS return AlphaSignal with ALL 6 required fields
- Choose appropriate TradingTechnique (TREND, RANGE, BREAKOUT, etc.)
- Provide clear reasoning for each signal"""

        user_prompt = f"""Generate an AlphaEngine for this strategy:

**Strategy:** {strategy.name}
**Type:** {strategy.strategy_type}
**Indicators:** {', '.join(strategy.indicators)}
**Entry Conditions:**
{chr(10).join('  - ' + c for c in strategy.entry_conditions)}
**Exit Conditions:**
{chr(10).join('  - ' + c for c in strategy.exit_conditions)}
**Timeframe:** {strategy.timeframe}
**Stop Loss:** {strategy.stop_loss or 'None'}
**Take Profit:** {strategy.take_profit or 'None'}

Generate complete AlphaEngine code."""

        code = self.model.generate_response(
            system_prompt=system_prompt,
            user_content=user_prompt,
            temperature=0.1,
            max_tokens=3000
        )

        # Extract code from markdown
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        # Generate engine name
        engine_name = f"AIGeneratedEngine_{strategy.name.replace(' ', '_')}"
        class_match = re.search(r"class\s+(\w+)", code)
        if class_match:
            engine_name = class_match.group(1)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.engine_dir / f"{engine_name.lower()}_{timestamp}.py"
        file_path.write_text(code)

        print(f"✅ Generated engine: {engine_name}")
        print(f"   File: {file_path.name}")

        return GeneratedEngine(
            engine_name=engine_name,
            engine_code=code,
            file_path=file_path,
            strategy=strategy,
            validation_status="pending"
        )

    def translate_backtest(self, backtest_path: Path) -> GeneratedEngine:
        """
        Full pipeline: Extract logic + Generate engine.

        Args:
            backtest_path: Path to backtest Python file

        Returns:
            GeneratedEngine ready for validation
        """
        print(f"\n{'='*80}")
        print(f"🔄 Translating Backtest → AlphaEngine")
        print(f"{'='*80}")
        print(f"📄 Input: {backtest_path.name}")

        # Load backtest code
        backtest_code = backtest_path.read_text()

        # Extract strategy logic
        strategy = self.extract_strategy_logic(backtest_code)

        # Generate AlphaEngine
        engine = self.generate_alpha_engine(strategy)

        print(f"\n✅ Translation complete!")
        print(f"   Engine: {engine.engine_name}")
        print(f"   File: {engine.file_path}")
        print(f"   Status: {engine.validation_status}")

        return engine

    def batch_translate(self, backtest_dir: Path) -> List[GeneratedEngine]:
        """
        Translate all backtests in a directory.

        Args:
            backtest_dir: Directory containing backtest files

        Returns:
            List of generated engines
        """
        backtest_files = list(backtest_dir.glob("*.py"))
        print(f"\n📂 Found {len(backtest_files)} backtest files")

        engines = []
        for i, backtest_file in enumerate(backtest_files, 1):
            print(f"\n[{i}/{len(backtest_files)}] Processing {backtest_file.name}...")
            try:
                engine = self.translate_backtest(backtest_file)
                engines.append(engine)
            except Exception as e:
                print(f"❌ Failed to translate {backtest_file.name}: {e}")

        # Summary
        print(f"\n{'='*80}")
        print(f"📊 Translation Summary")
        print(f"{'='*80}")
        print(f"Total backtests: {len(backtest_files)}")
        print(f"Successfully translated: {len(engines)}")
        print(f"Failed: {len(backtest_files) - len(engines)}")
        print(f"Output directory: {self.engine_dir}")

        return engines


def main():
    """Example usage"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              Strategy Translator                             ║
║          Backtest → AlphaEngine                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    translator = StrategyTranslator(llm_provider="anthropic")

    # Example: Translate a single backtest
    strategy_research_dir = Path(__file__).parent.parent.parent.parent.parent / "strategy-research"
    example_backtest = strategy_research_dir / "data" / "rbi" / "backtests" / "example_backtest.py"

    if example_backtest.exists():
        engine = translator.translate_backtest(example_backtest)
        print(f"\n✅ Generated engine: {engine.file_path}")
    else:
        print(f"\n⚠️  No example backtest found at {example_backtest}")
        print("   Run the RBI agent first to generate backtests")


if __name__ == "__main__":
    main()
