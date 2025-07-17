#!/usr/bin/env python
"""
Installation script for CeyeHao package.
"""

import subprocess
import sys
import os

def install_package():
    """Install the CeyeHao package in development mode."""
    try:
        # Install in development mode
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("✅ CeyeHao package installed successfully!")
        print("\nYou can now use the following commands:")
        print("  ceyehao gui                    # Launch the GUI")
        print("  ceyehao train                  # Run training")
        print("  ceyehao search                 # Run search")
        print("  ceyehao --help                 # Show help")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_package() 