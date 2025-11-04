#!/bin/bash
set -e

echo "=========================================="
echo "  Huracan Engine Database Setup"
echo "=========================================="
echo ""

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ ERROR: DATABASE_URL environment variable not set"
    echo ""
    echo "Please set it like:"
    echo "  export DATABASE_URL='postgresql://user:password@localhost:5432/huracan'"
    echo ""
    exit 1
fi

echo "✅ DATABASE_URL found"
echo ""

# Extract connection details
DB_URL=$DATABASE_URL
echo "📊 Database URL: ${DB_URL%%\?*}"  # Hide any query params
echo ""

# Check if psql is available
if ! command -v psql &> /dev/null; then
    echo "❌ ERROR: psql command not found"
    echo "Please install PostgreSQL client tools"
    exit 1
fi

echo "1️⃣  Checking database connection..."
if psql "$DB_URL" -c "SELECT 1" > /dev/null 2>&1; then
    echo "   ✅ Database connection successful"
else
    echo "   ❌ Cannot connect to database"
    exit 1
fi
echo ""

echo "2️⃣  Installing pgvector extension..."
if psql "$DB_URL" -c "CREATE EXTENSION IF NOT EXISTS vector" > /dev/null 2>&1; then
    echo "   ✅ pgvector extension installed"
else
    echo "   ⚠️  Could not install pgvector extension"
    echo "   You may need superuser privileges. Run manually:"
    echo "   CREATE EXTENSION IF NOT EXISTS vector;"
    echo ""
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

echo "3️⃣  Creating RL training schema..."
SCHEMA_FILE="$(dirname "$0")/../src/cloud/training/memory/schema.sql"

if [ ! -f "$SCHEMA_FILE" ]; then
    echo "   ❌ Schema file not found at: $SCHEMA_FILE"
    exit 1
fi

if psql "$DB_URL" -f "$SCHEMA_FILE" > /dev/null 2>&1; then
    echo "   ✅ Schema created successfully"
else
    echo "   ❌ Failed to create schema"
    echo "   Check the error messages above"
    exit 1
fi
echo ""

echo "4️⃣  Verifying tables..."
TABLES=(
    "trade_memory"
    "post_exit_tracking"
    "win_analysis"
    "loss_analysis"
    "pattern_library"
    "model_performance"
)

for table in "${TABLES[@]}"; do
    if psql "$DB_URL" -c "SELECT 1 FROM $table LIMIT 1" > /dev/null 2>&1; then
        echo "   ✅ $table"
    else
        echo "   ❌ $table (not found or not accessible)"
    fi
done
echo ""

echo "5️⃣  Checking pgvector functionality..."
if psql "$DB_URL" -c "SELECT '[1,2,3]'::vector" > /dev/null 2>&1; then
    echo "   ✅ pgvector is working"
else
    echo "   ❌ pgvector test failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ Database setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: poetry install"
echo "  2. Configure Telegram (optional): config/base.yaml"
echo "  3. Test the system with a single symbol"
echo ""
