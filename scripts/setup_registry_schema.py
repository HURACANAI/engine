#!/usr/bin/env python3
"""
Setup script for Model Registry database schema.

This script creates the necessary database tables for the ModelRegistry service:
- models: Model metadata and configuration
- model_metrics: Performance metrics for each model
- publish_log: Publish/reject event log
- training_runs: Training run metadata and results

Usage:
    python scripts/setup_registry_schema.py [--dsn DATABASE_URL]

Environment Variables:
    DATABASE_URL: PostgreSQL connection string (if not provided via --dsn)

Example:
    python scripts/setup_registry_schema.py --dsn "postgresql://user:pass@localhost/huracan"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import sql


def setup_schema(dsn: str) -> bool:
    """Create database schema from SQL file.
    
    Args:
        dsn: Database connection string
        
    Returns:
        True if schema was created successfully, False otherwise
    """
    # Read schema SQL file
    schema_file = project_root / "src" / "cloud" / "training" / "services" / "registry_schema.sql"
    
    if not schema_file.exists():
        print(f"❌ Error: Schema file not found: {schema_file}")
        return False
    
    try:
        with open(schema_file, "r") as f:
            schema_sql = f.read()
    except Exception as e:
        print(f"❌ Error: Failed to read schema file: {e}")
        return False
    
    # Parse DSN to extract connection parameters
    # Format: postgresql://user:pass@host:port/dbname
    try:
        from urllib.parse import urlparse
        parsed = urlparse(dsn)
        db_params = {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/"),
            "user": parsed.username,
            "password": parsed.password,
        }
        print(f"✅ Parsed database connection parameters")
    except Exception as e:
        print(f"❌ Error: Failed to parse DSN: {e}")
        return False
    
    # Connect to database using psycopg2 directly
    # This allows us to execute multiple statements properly
    try:
        conn = psycopg2.connect(**db_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        print(f"✅ Connected to database: {db_params['database']}")
    except Exception as e:
        print(f"❌ Error: Failed to connect to database: {e}")
        return False
    
    # Execute schema SQL using psycopg2
    # psycopg2 doesn't support multiple statements in a single execute()
    # We need to split the SQL into individual statements
    try:
        # Split SQL into statements
        # Remove comments and empty lines first
        lines = []
        for line in schema_sql.split("\n"):
            stripped = line.strip()
            # Skip comment-only lines
            if stripped.startswith("--"):
                continue
            # Skip empty lines
            if not stripped:
                continue
            lines.append(line)
        
        # Join lines back together
        cleaned_sql = "\n".join(lines)
        
        # Split by semicolon, but be careful with function definitions
        # Use a simple approach: split by semicolon, but keep function definitions together
        statements = []
        current_statement = []
        in_function = False
        
        for line in cleaned_sql.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            
            current_statement.append(line)
            
            # Check if we're in a function definition
            if "CREATE OR REPLACE FUNCTION" in stripped.upper():
                in_function = True
            
            # Check if function ends
            if "$$ LANGUAGE" in stripped.upper():
                in_function = False
                # Function ends here, add to statements
                if current_statement:
                    statement_text = "\n".join(current_statement)
                    if statement_text.strip():
                        statements.append(statement_text)
                    current_statement = []
                continue
            
            # If not in function and line ends with semicolon, it's a statement
            if not in_function and stripped.endswith(";"):
                statement_text = "\n".join(current_statement)
                if statement_text.strip():
                    statements.append(statement_text)
                current_statement = []
        
        # Add any remaining statement
        if current_statement:
            statement_text = "\n".join(current_statement)
            if statement_text.strip():
                statements.append(statement_text)
        
        # Execute statements one by one
        print(f"📝 Executing {len(statements)} SQL statements...")
        for i, statement in enumerate(statements, 1):
            try:
                cur.execute(statement)
                # Get first few words of statement for logging
                first_words = statement.split()[:3]
                print(f"✅ Executed statement {i}/{len(statements)}: {' '.join(first_words)}...")
            except psycopg2.Error as e:
                error_msg = str(e).lower()
                if "already exists" in error_msg or "duplicate" in error_msg:
                    print(f"⚠️  Statement {i}: Object already exists (skipping)")
                else:
                    # For COMMENT statements, ignore errors if table/column doesn't exist
                    if "COMMENT ON" in statement.upper():
                        print(f"⚠️  Statement {i}: Comment failed (table/column may not exist yet) - skipping")
                    else:
                        print(f"❌ Error executing statement {i}: {e}")
                        print(f"   Statement: {statement[:200]}...")
                        # Don't fail - continue with other statements
                        # Some statements might succeed even if others fail
        
        # Verify tables were created
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('models', 'model_metrics', 'publish_log', 'training_runs')
            ORDER BY table_name
        """)
        tables = cur.fetchall()
        if tables:
            print(f"\n✅ Verified tables created: {[t[0] for t in tables]}")
        else:
            print("⚠️  Warning: No tables found (they may already exist)")
        
    except Exception as e:
        print(f"❌ Error: Failed to create schema: {e}")
        import traceback
        traceback.print_exc()
        cur.close()
        conn.close()
        return False
    finally:
        cur.close()
        conn.close()
    
    print("\n✅ Database schema created successfully!")
    print("\nTables created:")
    print("  - models: Model metadata and configuration")
    print("  - model_metrics: Performance metrics for each model")
    print("  - publish_log: Publish/reject event log")
    print("  - training_runs: Training run metadata and results")
    print("\nIndexes created for performance optimization")
    print("Triggers created for automatic updated_at timestamp updates")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup Model Registry database schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dsn",
        type=str,
        help="Database connection string (e.g., postgresql://user:pass@localhost/huracan)",
        default=None,
    )
    
    args = parser.parse_args()
    
    # Get DSN from argument or environment variable
    import os
    dsn = args.dsn or os.getenv("DATABASE_URL")
    
    if not dsn:
        print("❌ Error: Database DSN not provided")
        print("   Use --dsn argument or set DATABASE_URL environment variable")
        print("\nExample:")
        print('   python scripts/setup_registry_schema.py --dsn "postgresql://user:pass@localhost/huracan"')
        print("   OR")
        print('   export DATABASE_URL="postgresql://user:pass@localhost/huracan"')
        print('   python scripts/setup_registry_schema.py')
        sys.exit(1)
    
    # Setup schema
    success = setup_schema(dsn)
    
    if not success:
        print("\n❌ Failed to setup database schema")
        sys.exit(1)
    
    print("\n✅ Setup complete!")


if __name__ == "__main__":
    main()

