#!/bin/bash

# Run Me First - Strategy Research Setup Script
# ==============================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Strategy Research Pipeline - First Run Setup            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3.11 --version 2>&1 || echo "not found")
if [[ $python_version == *"not found"* ]]; then
    echo "❌ Python 3.11 not found. Please install it first."
    exit 1
else
    echo "✅ $python_version"
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3.11 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API keys!"
    echo "   Required: At least ONE of:"
    echo "   - ANTHROPIC_KEY (Claude)"
    echo "   - OPENAI_KEY (GPT-4/5)"
    echo "   - DEEPSEEK_KEY (DeepSeek - cheapest!)"
    echo ""
    echo "   Open .env file:"
    echo "   nano .env"
    echo ""
else
    echo "✅ .env file already exists"
fi

# Check if ideas.txt has content
ideas_file="data/rbi/ideas.txt"
if [ -f "$ideas_file" ]; then
    non_comment_lines=$(grep -v "^#" "$ideas_file" | grep -v "^$" | wc -l | tr -d ' ')
    if [ "$non_comment_lines" -eq 0 ]; then
        echo ""
        echo "⚠️  ideas.txt has no active strategy ideas (all lines are comments)"
        echo "   Edit data/rbi/ideas.txt and add some strategies"
        echo "   Example: Buy when RSI < 30, sell when RSI > 70"
    else
        echo "✅ ideas.txt has $non_comment_lines strategy ideas"
    fi
else
    echo "⚠️  ideas.txt not found at $ideas_file"
fi

# Done
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete! ✅                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "1. Edit .env and add API keys (if not done)"
echo "2. Edit data/rbi/ideas.txt and add strategy ideas"
echo "3. Run the RBI agent:"
echo "   python agents/simple_rbi_agent.py"
echo ""
echo "For detailed instructions, see:"
echo "   ../QUICKSTART.md"
echo ""

