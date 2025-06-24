#!/usr/bin/env python3
"""
GAIA Unit 4 - Virtual Environment Setup Script
By Arte(r)m Sedov

This script automates the setup of a Python virtual environment for the GAIA Unit 4 agent.

Usage:
    python setup_venv.py

This script will:
1. Check Python version
2. Create a virtual environment
3. Install all required dependencies
4. Verify the installation
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, check=True, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check, 
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {command}")
        print(f"Error: {e}")
        return None

def get_python_command():
    """Get the appropriate Python command for the current platform."""
    if platform.system() == "Windows":
        # Try different Python commands on Windows
        commands = ["py", "python", "python3"]
        for cmd in commands:
            try:
                result = subprocess.run(f"{cmd} --version", shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    return cmd
            except:
                continue
        return "python"  # fallback
    else:
        return "python3"

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ is required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected.")
    return True

def create_virtual_environment():
    """Create a virtual environment."""
    print("\n📦 Creating virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("⚠️  Virtual environment 'venv' already exists.")
        response = input("Do you want to recreate it? (y/N): ").lower().strip()
        if response == 'y':
            print("🗑️  Removing existing virtual environment...")
            if platform.system() == "Windows":
                run_command("rmdir /s /q venv", check=False)
            else:
                run_command("rm -rf venv", check=False)
        else:
            print("✅ Using existing virtual environment.")
            return True
    
    # Get the appropriate Python command
    python_cmd = get_python_command()
    print(f"Using Python command: {python_cmd}")
    
    # Create virtual environment
    result = run_command(f"{python_cmd} -m venv venv")
    if result and result.returncode == 0:
        print("✅ Virtual environment created successfully.")
        return True
    else:
        print("❌ Failed to create virtual environment.")
        print("Try running manually:")
        print(f"  {python_cmd} -m venv venv")
        return False

def get_activation_command():
    """Get the appropriate activation command based on the platform."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def get_python_path():
    """Get the path to the virtual environment's Python executable."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\python.exe"
    else:
        return "venv/bin/python"

def get_pip_path():
    """Get the path to the virtual environment's pip executable."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\pip.exe"
    else:
        return "venv/bin/pip"

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("\n📚 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found in current directory.")
        return False
    
    python_path = get_python_path()
    pip_path = get_pip_path()
    
    # Upgrade pip first
    print("⬆️  Upgrading pip...")
    result = run_command(f"{python_path} -m pip install --upgrade pip")
    if not result or result.returncode != 0:
        print("⚠️  Failed to upgrade pip, continuing anyway...")
    
    # Install requirements
    print("📦 Installing packages from requirements.txt...")
    result = run_command(f"{pip_path} install -r requirements.txt")
    
    if result and result.returncode == 0:
        print("✅ Dependencies installed successfully.")
        return True
    else:
        print("❌ Failed to install dependencies.")
        print("Try running manually:")
        print(f"  {pip_path} install -r requirements.txt")
        return False

def verify_installation():
    """Verify that key packages are installed correctly."""
    print("\n🔍 Verifying installation...")
    
    test_script = """
import sys
try:
    import langchain
    import supabase
    import gradio
    import pandas
    import numpy
    import requests
    print("✅ All core packages imported successfully!")
    print(f"Python path: {sys.executable}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
    
    python_path = get_python_path()
    result = run_command(f'{python_path} -c "{test_script}"')
    
    if result and result.returncode == 0:
        print("✅ Installation verification passed.")
        return True
    else:
        print("❌ Installation verification failed.")
        return False

def main():
    """Main setup function."""
    print("🚀 GAIA Unit 4 - Virtual Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        sys.exit(1)
    
    # Success message
    print("\n🎉 Virtual environment setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate the virtual environment:")
    activation_cmd = get_activation_command()
    print(f"   {activation_cmd}")
    print("\n2. Set up your .env file with API keys")
    print("3. Run the vector store setup:")
    print("   python setup_vector_store.py")
    print("\n4. Start the application:")
    print("   python app.py")
    
    print(f"\n💡 To activate the environment later, run: {activation_cmd}")

if __name__ == "__main__":
    main() 