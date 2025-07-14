#!/usr/bin/env python3
"""
Script to help run the Jupyter notebook with the correct environment
"""

import subprocess
import sys
import os

def main():
    """Start Jupyter notebook with the correct environment"""
    print("Starting Jupyter Notebook...")
    print("Make sure your virtual environment is activated!")
    print("If not, run: source venv/bin/activate")
    print()
    
    try:
        # Start Jupyter notebook
        subprocess.run([sys.executable, "-m", "jupyter", "notebook"], check=True)
    except KeyboardInterrupt:
        print("\nJupyter notebook stopped.")
    except Exception as e:
        print(f"Error starting Jupyter: {e}")
        print("Make sure you have activated the virtual environment:")
        print("source venv/bin/activate")

if __name__ == "__main__":
    main() 