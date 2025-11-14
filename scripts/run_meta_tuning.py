from __future__ import annotations

import argparse
from typing import List, Tuple

from cloud.training.brain.brain_library import BrainLibrary
from cloud.training.config.settings import EngineSettings
from cloud.training.meta.meta_tuner import run_meta_tuning


def _build_symbol_mode_pairs(settings: EngineSettings, symbols: List[str], modes: List[str]) -> List[Tuple[str, str]]:
    if not symbols:
        configured = settings.training.per_coin.symbols_allowed
        symbols = configured if configured else ["SOL/USDT"]
    if not modes:
        modes = ["scalp"]
    return [(symbol, mode) for symbol in symbols for mode in modes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run meta tuning across symbols/modes.")
    parser.add_argument("--env", dest="environment", default=None, help="Config environment (default: HURACAN_ENV or local)")
    parser.add_argument("--symbols", nargs="*", default=[], help="Override symbols (e.g. SOL/USDT BTC/USDT)")
    parser.add_argument("--modes", nargs="*", default=[], help="Trading modes (e.g. scalp swing)")
    args = parser.parse_args()

    settings = EngineSettings.load(environment=args.environment)
    if not settings.postgres:
        raise RuntimeError("Postgres DSN required to run meta tuner")

    pairs = _build_symbol_mode_pairs(settings, args.symbols, args.modes)
    brain = BrainLibrary(dsn=settings.postgres.dsn, use_pool=True)
    run_meta_tuning(brain, pairs)


if __name__ == "__main__":
    main()


