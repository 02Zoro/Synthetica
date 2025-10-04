#!/usr/bin/env python3
"""
Test script for ML pipeline with real gene data
"""
import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def test_ml_pipeline():
    """Test the ML pipeline with real gene data"""
    
    print("Testing ML pipeline with real gene data...")
    print("=" * 60)
    
    try:
        from app.services.ml_pipeline_service_real import ml_pipeline_service
        
        # Test initialization
        print("1. Testing initialization...")
        result = await ml_pipeline_service.initialize()
        print(f"   Initialization result: {result}")
        
        if result:
            print("\n2. Testing hypothesis generation...")
            hypotheses = await ml_pipeline_service.generate_advanced_hypotheses(
                query="What are the latest developments in cancer immunotherapy?",
                domain="biomedical",
                max_hypotheses=2
            )
            
            print(f"   Generated {len(hypotheses)} hypotheses")
            for i, hyp in enumerate(hypotheses):
                print(f"\n   Hypothesis {i+1}:")
                print(f"   Title: {hyp['title']}")
                print(f"   Description: {hyp['description'][:200]}...")
                print(f"   Confidence: {hyp['confidence_score']}")
        else:
            print("   Initialization failed - falling back to demo mode")
            
    except Exception as e:
        print(f"Error testing ML pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ml_pipeline())

