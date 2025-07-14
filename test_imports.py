#!/usr/bin/env python3
"""
Test script to verify all imports are working correctly
"""

def test_imports():
    """Test all necessary imports"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn imported successfully")
    except ImportError as e:
        print(f"✗ seaborn import failed: {e}")
        return False
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, f1_score
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    print("\nAll imports successful! 🎉")
    return True

def test_data_loading():
    """Test if the dataset can be loaded"""
    print("\nTesting data loading...")
    
    try:
        import pandas as pd
        data = pd.read_csv("Breast_cancer_data.csv")
        print(f"✓ Dataset loaded successfully")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        return True
    except FileNotFoundError:
        print("✗ Dataset file not found: Breast_cancer_data.csv")
        return False
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    print("Breast Cancer Prediction - Import Test")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test data loading
    data_ok = test_data_loading()
    
    if imports_ok and data_ok:
        print("\n✅ All tests passed! You're ready to run the main script.")
        print("Run: python breast_cancer_naive_bayes.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.") 