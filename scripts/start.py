#!/usr/bin/env python3
"""
Simple unified startup script for Huracan Engine
Works consistently across different laptops and operating systems
Usage: python start.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_header():
    """Print startup header."""
    print(f"{Colors.BLUE}╔════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.BLUE}║   Huracan Engine - Unified Startup    ║{Colors.NC}")
    print(f"{Colors.BLUE}╚════════════════════════════════════════╝{Colors.NC}")
    print()

def check_python_version():
    """Check if Python version is 3.8+."""
    print(f"{Colors.YELLOW}[1/6]{Colors.NC} Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"{Colors.RED}❌ Python 3.8+ required. Found: Python {version.major}.{version.minor}{Colors.NC}")
        sys.exit(1)
    print(f"{Colors.GREEN}✅ Python {version.major}.{version.minor}.{version.micro} found{Colors.NC}")
    print()

def setup_venv(project_root: Path):
    """Setup and activate virtual environment."""
    print(f"{Colors.YELLOW}[2/6]{Colors.NC} Setting up virtual environment...")
    venv_dir = project_root / "venv"
    
    if not venv_dir.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        print(f"{Colors.GREEN}✅ Virtual environment created{Colors.NC}")
    else:
        print(f"{Colors.GREEN}✅ Virtual environment already exists{Colors.NC}")
    
    # Determine activation script based on OS
    if sys.platform == "win32":
        activate_script = venv_dir / "Scripts" / "activate.bat"
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        activate_script = venv_dir / "bin" / "activate"
        python_exe = venv_dir / "bin" / "python"
    
    if not python_exe.exists():
        print(f"{Colors.RED}❌ Virtual environment Python not found at {python_exe}{Colors.NC}")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✅ Virtual environment ready{Colors.NC}")
    print()
    return python_exe

def upgrade_pip(python_exe: Path):
    """Upgrade pip."""
    print(f"{Colors.YELLOW}[3/6]{Colors.NC} Upgrading pip...")
    subprocess.run(
        [str(python_exe), "-m", "pip", "install", "--quiet", "--upgrade", "pip", "setuptools", "wheel"],
        check=False
    )
    print(f"{Colors.GREEN}✅ pip upgraded{Colors.NC}")
    print()

def install_dependencies(python_exe: Path, project_root: Path):
    """Install dependencies from requirements.txt."""
    print(f"{Colors.YELLOW}[4/6]{Colors.NC} Installing dependencies...")
    requirements_file = project_root / "requirements.txt"
    
    if requirements_file.exists():
        print("Installing from requirements.txt...")
        result = subprocess.run(
            [str(python_exe), "-m", "pip", "install", "--quiet", "-r", str(requirements_file)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"{Colors.YELLOW}⚠️  Some dependencies may have failed, continuing anyway...{Colors.NC}")
        else:
            print(f"{Colors.GREEN}✅ Dependencies installed{Colors.NC}")
    else:
        print(f"{Colors.YELLOW}⚠️  requirements.txt not found, skipping dependency installation{Colors.NC}")
    print()

def setup_environment(project_root: Path):
    """Setup environment variables."""
    print(f"{Colors.YELLOW}[5/6]{Colors.NC} Setting up environment...")
    
    # Set PYTHONPATH
    pythonpath = str(project_root)
    if "PYTHONPATH" in os.environ:
        pythonpath = f"{pythonpath}{os.pathsep}{os.environ['PYTHONPATH']}"
    os.environ["PYTHONPATH"] = pythonpath
    
    # Check for optional environment variables
    optional_vars = {
        "DROPBOX_ACCESS_TOKEN": "optional for some operations",
        "DATABASE_URL": "using default if configured",
        "TELEGRAM_TOKEN": "optional for notifications",
    }
    
    for var, description in optional_vars.items():
        if var not in os.environ:
            print(f"{Colors.YELLOW}⚠️  {var} not set ({description}){Colors.NC}")
    
    print(f"{Colors.GREEN}✅ Environment configured{Colors.NC}")
    print()

def run_application(python_exe: Path, project_root: Path):
    """Run the main application."""
    print(f"{Colors.YELLOW}[6/6]{Colors.NC} Starting Huracan Engine...")
    print()
    print(f"{Colors.BLUE}════════════════════════════════════════{Colors.NC}")
    print(f"{Colors.BLUE}Starting Engine...{Colors.NC}")
    print(f"{Colors.BLUE}════════════════════════════════════════{Colors.NC}")
    print()
    
    # Determine which entry point to use
    entry_points = [
        project_root / "scripts" / "run_daily.py",
        project_root / "engine" / "run.py",
    ]
    
    for entry_point in entry_points:
        if entry_point.exists():
            print(f"Running: {entry_point}")
            result = subprocess.run([str(python_exe), str(entry_point)])
            return result.returncode
    
    # Fallback to module execution
    print("Running: python -m src.cloud.training.pipelines.daily_retrain")
    result = subprocess.run([
        str(python_exe), "-m", "src.cloud.training.pipelines.daily_retrain"
    ])
    return result.returncode

def main():
    """Main entry point."""
    print_header()
    
    # Get project root (scripts/ is one level down)
    project_root = Path(__file__).parent.parent.absolute()
    os.chdir(project_root)
    
    # Step 1: Check Python version
    check_python_version()
    
    # Step 2: Setup virtual environment
    python_exe = setup_venv(project_root)
    
    # Step 3: Upgrade pip
    upgrade_pip(python_exe)
    
    # Step 4: Install dependencies
    install_dependencies(python_exe, project_root)
    
    # Step 5: Setup environment
    setup_environment(project_root)
    
    # Step 6: Run application
    exit_code = run_application(python_exe, project_root)
    
    print()
    if exit_code == 0:
        print(f"{Colors.GREEN}╔════════════════════════════════════════╗{Colors.NC}")
        print(f"{Colors.GREEN}║   ✅ Engine finished successfully      ║{Colors.NC}")
        print(f"{Colors.GREEN}╚════════════════════════════════════════╝{Colors.NC}")
    else:
        print(f"{Colors.RED}╔════════════════════════════════════════╗{Colors.NC}")
        print(f"{Colors.RED}║   ❌ Engine exited with error code {exit_code}   ║{Colors.NC}")
        print(f"{Colors.RED}╚════════════════════════════════════════╝{Colors.NC}")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()

