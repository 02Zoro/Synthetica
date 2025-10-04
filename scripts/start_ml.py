"""
Start SABDE with ML Pipeline integration
"""
import os
import subprocess
import sys
import time
from pathlib import Path

def main():
    """Start Synthetica with ML Pipeline integration."""
    print("Starting Synthetica ML-Enhanced Mode")
    print("=" * 50)
    
    # Set ML pipeline environment variables
    os.environ["ML_PIPELINE_ENABLED"] = "true"
    os.environ["DEBUG"] = "true"
    os.environ["DEMO_MODE"] = "false"  # Use real ML pipeline
    
    # Create necessary directories
    os.makedirs("chroma_db", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("app/ml_pipeline", exist_ok=True)
    
    print("ML Pipeline enabled")
    print("Directories created")
    
    # Check if ML dependencies are installed
    print("\nChecking ML dependencies...")
    try:
        import torch
        import torch_geometric
        import chromadb
        import neo4j
        print("✅ ML dependencies found")
    except ImportError as e:
        print(f"❌ Missing ML dependencies: {e}")
        print("Please install ML requirements:")
        print("pip install -r requirements-ml.txt")
        return
    
    # Start the ML-enhanced backend
    print("\nStarting ML-Enhanced Backend...")
    print("Backend will be available at: http://localhost:8001")
    print("API Documentation: http://localhost:8001/docs")
    print("ML Status: http://localhost:8001/api/v1/research/ml-status")
    print("\nPress Ctrl+C to stop the ML-enhanced demo")
    print("-" * 50)
    
    try:
        # Start the ML-enhanced backend
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "app.main_ml:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8001"
        ])
    except KeyboardInterrupt:
        print("\n\nML-Enhanced demo stopped by user")
    except Exception as e:
        print(f"\nError starting ML-enhanced demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

