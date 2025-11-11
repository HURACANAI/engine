#!/bin/bash
# Run tests with coverage analysis

set -e

echo "==========================================="
echo "Running Tests with Coverage Analysis"
echo "==========================================="
echo ""

# Install test dependencies if needed
echo "Installing test dependencies..."
pip install -q pytest pytest-cov pytest-mock coverage || true

echo ""
echo "Running unit tests..."
echo ""

# Run tests with coverage
pytest tests/unit/ \
    --cov=src/shared/database \
    --cov=src/shared/exceptions \
    --cov=src/shared/config \
    --cov=src/shared/contracts \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=json \
    -v \
    "$@"

echo ""
echo "==========================================="
echo "Coverage Report Generated"
echo "==========================================="
echo "HTML report: htmlcov/index.html"
echo "JSON report: coverage.json"
echo ""

# Display summary
if command -v coverage &> /dev/null; then
    echo "Coverage Summary:"
    coverage report --include="src/shared/*"
fi
