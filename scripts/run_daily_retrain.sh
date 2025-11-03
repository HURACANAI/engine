#!/usr/bin/env bash
set -euo pipefail

# Placeholder script for invoking the daily retraining job once the Engine is implemented.
poetry run python -m cloud.training.pipelines.daily_retrain "$@"
