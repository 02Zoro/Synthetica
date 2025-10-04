#!/usr/bin/env python3
"""
Demo startup script for SABDE without API keys.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def main():
    """Start Synthetica in demo mode."""
    print("Starting Synthetica Demo Mode")
    print("=" * 50)
    
    # Set demo mode environment variable
    os.environ["DEMO_MODE"] = "true"
    os.environ["DEBUG"] = "true"
    
    # Create necessary directories
    os.makedirs("chroma_db", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("Demo mode enabled")
    print("Directories created")
    
    # Start the demo backend
    print("\nStarting Demo Backend...")
    print("Backend will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the demo")
    print("-" * 50)
    
    try:
        # Start the demo backend
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main_demo:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
    except Exception as e:
        print(f"\nError starting demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
