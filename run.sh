#!/bin/bash
# Super simple one-liner to run the engine
# Usage: ./run.sh

cd "$(dirname "$0")" && export PYTHONPATH="$(pwd):$PYTHONPATH" && python3 src/cloud/training/pipelines/daily_retrain.py

