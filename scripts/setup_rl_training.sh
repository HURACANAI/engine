#!/bin/bash
# Setup script for RL training system

set -e

echo "🚀 Setting up RL-based self-learning trading engine..."

# Check if PostgreSQL is running
echo "📊 Checking PostgreSQL connection..."
if ! psql "$DATABASE_URL" -c '\q' 2>/dev/null; then
    echo "❌ Error: Cannot connect to PostgreSQL"
    echo "Please set DATABASE_URL environment variable"
    echo "Example: export DATABASE_URL='postgresql://user:pass@localhost/huracan'"
    exit 1
fi

echo "✅ PostgreSQL connection successful"

# Install pgvector extension
echo "📦 Installing pgvector extension..."
psql "$DATABASE_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;" || {
    echo "⚠️  Warning: Could not install pgvector extension"
    echo "You may need superuser privileges"
    echo "Run manually: psql -c 'CREATE EXTENSION vector;'"
}

# Run schema migrations
echo "🗄️  Creating database schema..."
psql "$DATABASE_URL" -f src/cloud/training/memory/schema.sql

echo "✅ Database schema created successfully"

# Install Python dependencies
echo "📚 Installing Python dependencies..."
poetry add torch polars psycopg2-binary scipy

echo "✅ Dependencies installed"

# Create models directory
echo "📁 Creating models directory..."
mkdir -p models
mkdir -p docs

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Configure config/base.yaml with your settings"
echo "2. Run training: python -m src.cloud.training.pipelines.rl_training_pipeline"
echo "3. Monitor results in PostgreSQL database"
echo ""
echo "📖 Read docs/RL_TRAINING_GUIDE.md for detailed instructions"
