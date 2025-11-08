"""
Test Integration - Verify all components are properly set up

This script tests the integration without requiring API keys.
Run this first to verify everything is installed correctly.
"""

import sys
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════════╗
║          Strategy Research Integration Test                 ║
╚══════════════════════════════════════════════════════════════╝
""")

# Test 1: Check directory structure
print("\n1️⃣  Testing directory structure...")
required_dirs = [
    "agents",
    "models",
    "data/rbi",
    "config",
]

all_exist = True
for dir_path in required_dirs:
    path = Path(dir_path)
    if path.exists():
        print(f"   ✅ {dir_path}/")
    else:
        print(f"   ❌ {dir_path}/ - MISSING")
        all_exist = False

if all_exist:
    print("   ✅ All directories present")
else:
    print("   ❌ Some directories missing")
    sys.exit(1)

# Test 2: Check required files
print("\n2️⃣  Testing required files...")
required_files = [
    "agents/simple_rbi_agent.py",
    "models/model_factory.py",
    "data/rbi/ideas.txt",
    ".env",
]

all_exist = True
for file_path in required_files:
    path = Path(file_path)
    if path.exists():
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ {file_path} - MISSING")
        all_exist = False

if all_exist:
    print("   ✅ All required files present")
else:
    print("   ❌ Some files missing")
    sys.exit(1)

# Test 3: Import Model Factory
print("\n3️⃣  Testing Model Factory import...")
try:
    from models.model_factory import ModelFactory
    print("   ✅ Model Factory imported successfully")

    # Check available models
    print("   📋 Available model implementations:")
    for model_type in ModelFactory.MODEL_IMPLEMENTATIONS.keys():
        print(f"      - {model_type}")

except ImportError as e:
    print(f"   ❌ Failed to import Model Factory: {e}")
    sys.exit(1)

# Test 4: Check .env configuration
print("\n4️⃣  Testing .env configuration...")
from dotenv import load_dotenv
import os

load_dotenv()

api_keys = [
    "ANTHROPIC_KEY",
    "OPENAI_KEY",
    "DEEPSEEK_KEY",
    "GEMINI_KEY",
    "GROQ_API_KEY",
]

found_keys = []
for key in api_keys:
    value = os.getenv(key)
    if value and value.strip():
        found_keys.append(key)
        print(f"   ✅ {key} is set")

if not found_keys:
    print(f"   ⚠️  No API keys configured yet")
    print(f"   💡 Add at least one API key to .env to run the RBI agent")
else:
    print(f"   ✅ Found {len(found_keys)} API key(s)")

# Test 5: Check ideas.txt
print("\n5️⃣  Testing ideas.txt...")
ideas_file = Path("data/rbi/ideas.txt")
if ideas_file.exists():
    with open(ideas_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if lines:
        print(f"   ✅ ideas.txt has {len(lines)} active strategy idea(s)")
        print(f"   📝 Preview:")
        for i, line in enumerate(lines[:3], 1):
            preview = line[:60] + "..." if len(line) > 60 else line
            print(f"      {i}. {preview}")
    else:
        print(f"   ⚠️  ideas.txt has no active ideas (all lines are comments)")
else:
    print(f"   ❌ ideas.txt not found")

# Test 6: Check Engine integration points
print("\n6️⃣  Testing Engine integration points...")
engine_files = [
    "../engine/src/cloud/training/adapters/strategy_translator.py",
    "../engine/src/cloud/training/models/ai_generated_engines/__init__.py",
    "../engine/observability/ai_council/model_factory_adapter.py",
]

all_exist = True
for file_path in engine_files:
    path = Path(file_path)
    if path.exists():
        print(f"   ✅ {path.name}")
    else:
        print(f"   ❌ {path.name} - MISSING")
        all_exist = False

if all_exist:
    print("   ✅ All Engine integration points present")

# Summary
print("\n" + "="*70)
print("📊 Test Summary")
print("="*70)

if not found_keys:
    print("""
Status: ⚠️  Setup incomplete
Action: Add API keys to .env file

To continue:
1. Edit .env file and add at least ONE API key:
   nano .env

2. Run a test with mock data:
   python agents/simple_rbi_agent.py --test

3. Or skip to testing the Strategy Translator:
   cd ../engine
   python -m cloud.training.adapters.strategy_translator
""")
else:
    print("""
Status: ✅ Ready to run!
Action: Test the full pipeline

Next steps:
1. Run RBI Agent to generate strategies:
   python agents/simple_rbi_agent.py

2. Check the output:
   ls -lh data/rbi/*/backtests/

3. Translate to AlphaEngines:
   cd ../engine
   python -m cloud.training.adapters.strategy_translator
""")

print("\n✅ Integration test complete!")
