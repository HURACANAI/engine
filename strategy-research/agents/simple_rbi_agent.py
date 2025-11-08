"""
Simplified RBI Agent for Huracan Engine Integration
Researches trading strategies and generates backtests.

Based on moon-dev-ai-agents by Moon Dev
Adapted for Huracan Engine ecosystem
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_factory import ModelFactory

# Load environment
load_dotenv()

# Model configuration
RESEARCH_CONFIG = {
    "type": "deepseek",      # Cheap and good for research
    "name": "deepseek-chat"
}

BACKTEST_CONFIG = {
    "type": "deepseek",
    "name": "deepseek-reasoner"  # Better for code generation
}

DEBUG_CONFIG = {
    "type": "openai",
    "name": "gpt-4"  # Best for debugging (fallback to deepseek if no key)
}

# Performance thresholds
MIN_RETURN_PCT = float(os.getenv("MIN_RETURN_PCT", "5.0"))
TARGET_RETURN_PCT = float(os.getenv("TARGET_RETURN_PCT", "50.0"))

# Data sources for backtesting
DATA_SOURCES = os.getenv("BACKTEST_DATA_SOURCES", "BTC-USD,ETH-USD,SOL-USD").split(",")


class SimpleRBIAgent:
    """
    Research-Backtest-Implement Agent

    1. Research: Analyzes trading strategy idea
    2. Backtest: Generates backtest code
    3. Validates: Tests on multiple data sources
    """

    def __init__(self):
        print("\n🤖 Initializing Simple RBI Agent...")

        # Initialize models
        self.factory = ModelFactory()

        try:
            self.research_model = self.factory.create_model(
                RESEARCH_CONFIG["type"],
                RESEARCH_CONFIG["name"]
            )
            print(f"✅ Research model: {RESEARCH_CONFIG['type']}/{RESEARCH_CONFIG['name']}")
        except Exception as e:
            print(f"❌ Failed to initialize research model: {e}")
            self.research_model = None

        try:
            self.backtest_model = self.factory.create_model(
                BACKTEST_CONFIG["type"],
                BACKTEST_CONFIG["name"]
            )
            print(f"✅ Backtest model: {BACKTEST_CONFIG['type']}/{BACKTEST_CONFIG['name']}")
        except Exception as e:
            print(f"❌ Failed to initialize backtest model: {e}")
            self.backtest_model = None

        try:
            self.debug_model = self.factory.create_model(
                DEBUG_CONFIG["type"],
                DEBUG_CONFIG["name"]
            )
            print(f"✅ Debug model: {DEBUG_CONFIG['type']}/{DEBUG_CONFIG['name']}")
        except Exception as e:
            print(f"⚠️  Debug model not available, using backtest model for debugging")
            self.debug_model = self.backtest_model

        # Set up output directories
        self.output_dir = Path(__file__).parent.parent / "data" / "rbi"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create dated folder
        self.run_folder = self.output_dir / datetime.now().strftime("%m_%d_%Y")
        self.run_folder.mkdir(exist_ok=True)

        (self.run_folder / "research").mkdir(exist_ok=True)
        (self.run_folder / "backtests").mkdir(exist_ok=True)
        (self.run_folder / "strategies").mkdir(exist_ok=True)

        print(f"📁 Output directory: {self.run_folder}")

    def research_strategy(self, idea: str) -> dict:
        """
        Phase 1: Research the trading strategy concept.

        Args:
            idea: Trading strategy description (text, URL, etc.)

        Returns:
            dict with strategy analysis
        """
        if not self.research_model:
            raise RuntimeError("Research model not initialized")

        print(f"\n🔍 Researching strategy: {idea[:100]}...")

        system_prompt = """You are a quantitative trading strategy analyst.

Your task is to analyze trading strategy ideas and extract the core logic.

Output format (JSON):
{
    "strategy_name": "short descriptive name",
    "strategy_type": "trend/reversal/breakout/arbitrage/etc",
    "entry_conditions": ["list of entry conditions"],
    "exit_conditions": ["list of exit conditions"],
    "indicators_needed": ["list of technical indicators"],
    "timeframe": "recommended timeframe",
    "risk_reward": "estimated risk/reward ratio",
    "market_conditions": "when this strategy works best",
    "feasibility": "high/medium/low - can this be backtested?"
}"""

        user_prompt = f"Analyze this trading strategy idea:\n\n{idea}"

        response = self.research_model.generate_response(
            system_prompt=system_prompt,
            user_content=user_prompt,
            temperature=0.3,
            max_tokens=2000
        )

        # Save research output
        research_file = self.run_folder / "research" / f"{datetime.now().strftime('%H%M%S')}_research.txt"
        research_file.write_text(response)

        print(f"✅ Research complete. Saved to: {research_file.name}")

        return {
            "idea": idea,
            "analysis": response,
            "timestamp": datetime.now().isoformat()
        }

    def generate_backtest(self, research: dict) -> str:
        """
        Phase 2: Generate backtest code from research.

        Args:
            research: Output from research_strategy()

        Returns:
            Generated Python backtest code
        """
        if not self.backtest_model:
            raise RuntimeError("Backtest model not initialized")

        print(f"\n💻 Generating backtest code...")

        system_prompt = """You are an expert at writing backtests using the backtesting.py library.

Generate clean, working backtest code that:
1. Uses pandas_ta or talib for indicators (NOT backtesting.lib)
2. Implements the strategy logic in the next() method
3. Has proper entry/exit conditions
4. Includes position sizing
5. Is production-ready and runs without errors

Output ONLY the Python code, no explanations."""

        user_prompt = f"""Generate a backtest for this strategy:

{research['analysis']}

Requirements:
- Use backtesting.py library
- Use pandas_ta for indicators
- Include stop loss and take profit
- Simple and clean code
- No complex dependencies"""

        code = self.backtest_model.generate_response(
            system_prompt=system_prompt,
            user_content=user_prompt,
            temperature=0.2,
            max_tokens=3000
        )

        # Extract code from markdown if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        # Save backtest code
        backtest_file = self.run_folder / "backtests" / f"{datetime.now().strftime('%H%M%S')}_backtest.py"
        backtest_file.write_text(code)

        print(f"✅ Backtest code generated. Saved to: {backtest_file.name}")

        return code

    def process_idea(self, idea: str) -> dict:
        """
        Full pipeline: Research → Generate → Validate

        Args:
            idea: Trading strategy idea

        Returns:
            dict with complete results
        """
        print(f"\n{'='*80}")
        print(f"📋 Processing Strategy Idea")
        print(f"{'='*80}")

        try:
            # Phase 1: Research
            research = self.research_strategy(idea)

            # Phase 2: Generate backtest
            backtest_code = self.generate_backtest(research)

            # Phase 3: Validate (placeholder - actual backtesting requires data)
            print("\n⏭️  Validation skipped (requires historical data)")
            print("   To validate: Run backtest on actual market data")

            result = {
                "success": True,
                "idea": idea,
                "research": research,
                "backtest_code": backtest_code,
                "output_dir": str(self.run_folder)
            }

            print(f"\n✅ Strategy processing complete!")
            print(f"📁 Files saved to: {self.run_folder}")

            return result

        except Exception as e:
            print(f"\n❌ Error processing strategy: {e}")
            return {
                "success": False,
                "idea": idea,
                "error": str(e)
            }

    def process_ideas_file(self, ideas_file: Path = None) -> list:
        """
        Process all ideas from ideas.txt file.

        Args:
            ideas_file: Path to ideas.txt (default: data/rbi/ideas.txt)

        Returns:
            list of results
        """
        if ideas_file is None:
            ideas_file = self.output_dir / "ideas.txt"

        if not ideas_file.exists():
            print(f"❌ Ideas file not found: {ideas_file}")
            print(f"   Creating example file...")
            ideas_file.write_text("""# Trading Strategy Ideas
# One idea per line. Lines starting with # are ignored.

Buy when RSI < 30 and sell when RSI > 70
Moving average crossover with volume confirmation
Breakout strategy with ATR-based stops
""")
            print(f"✅ Example file created at: {ideas_file}")
            return []

        print(f"\n📖 Reading ideas from: {ideas_file}")

        ideas = []
        with open(ideas_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    ideas.append(line)

        print(f"   Found {len(ideas)} strategy ideas")

        results = []
        for i, idea in enumerate(ideas, 1):
            print(f"\n{'='*80}")
            print(f"Strategy {i}/{len(ideas)}")
            print(f"{'='*80}")

            result = self.process_idea(idea)
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*80}")
        print(f"📊 Summary")
        print(f"{'='*80}")
        print(f"Total strategies: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"Output directory: {self.run_folder}")

        return results


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          Simple RBI Agent for Huracan Engine                 ║
║          Research → Backtest → Implement                     ║
╚══════════════════════════════════════════════════════════════╝
""")

    agent = SimpleRBIAgent()

    # Process all ideas from ideas.txt
    results = agent.process_ideas_file()

    print(f"\n✅ All done! Check {agent.run_folder} for outputs.")


if __name__ == "__main__":
    main()
