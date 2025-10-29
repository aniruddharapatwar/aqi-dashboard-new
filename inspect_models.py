"""
Inspect Model Pickle Files - See what keys are actually in your model files
"""

import pickle
from pathlib import Path
import sys

print("=" * 80)
print("🔍 MODEL FILE INSPECTOR")
print("=" * 80)
print()

model_dir = Path("Classification_trained_models")

if not model_dir.exists():
    print(f"❌ Model directory not found: {model_dir}")
    sys.exit(1)

# Get first model file as example
model_files = list(model_dir.glob("*.pkl"))

if len(model_files) == 0:
    print("❌ No .pkl files found in model directory")
    sys.exit(1)

print(f"Found {len(model_files)} model files")
print()

# Inspect first few models
for model_file in model_files[:3]:  # Check first 3
    print(f"📦 Inspecting: {model_file.name}")
    print("-" * 80)
    
    try:
        with open(model_file, 'rb') as f:
            model_artifact = pickle.load(f)
        
        print(f"Type: {type(model_artifact)}")
        
        if isinstance(model_artifact, dict):
            print(f"Keys in model artifact:")
            for key in model_artifact.keys():
                value = model_artifact[key]
                print(f"  • '{key}': {type(value).__name__}", end="")
                
                # Show additional details for certain types
                if isinstance(value, list):
                    print(f" (length: {len(value)})")
                    if len(value) > 0 and len(value) <= 10:
                        print(f"      Values: {value}")
                    elif len(value) > 0:
                        print(f"      First few: {value[:3]}...")
                elif isinstance(value, dict):
                    print(f" (keys: {list(value.keys())[:5]}...)")
                elif hasattr(value, 'shape'):
                    print(f" (shape: {value.shape})")
                else:
                    print()
            
            print()
            
            # Check for expected keys
            expected_keys = ['calibrated_model', 'feature_names', 'classes', 'thresholds']
            missing_keys = [key for key in expected_keys if key not in model_artifact]
            
            if missing_keys:
                print(f"⚠️  MISSING EXPECTED KEYS: {missing_keys}")
            else:
                print(f"✅ All expected keys present")
            
            # Check for alternative key names
            alternative_names = {
                'feature_names': ['features', 'feature_list', 'columns', 'X_columns', 'feature_columns'],
                'classes': ['class_names', 'categories', 'labels', 'target_classes'],
                'calibrated_model': ['model', 'classifier', 'estimator', 'trained_model'],
                'thresholds': ['threshold', 'decision_thresholds', 'class_thresholds']
            }
            
            print(f"\n🔍 Checking for alternative key names:")
            for expected, alternatives in alternative_names.items():
                if expected not in model_artifact:
                    found_alternatives = [alt for alt in alternatives if alt in model_artifact]
                    if found_alternatives:
                        print(f"  • '{expected}' not found, but found: {found_alternatives}")
                    else:
                        print(f"  • '{expected}' not found (no alternatives found)")
            
        else:
            print(f"⚠️  Model artifact is not a dictionary! It's a {type(model_artifact)}")
            print(f"     This is unexpected. Models should be saved as dictionaries.")
            
            # Try to show attributes if it's an object
            if hasattr(model_artifact, '__dict__'):
                print(f"     Object attributes:")
                for attr in dir(model_artifact):
                    if not attr.startswith('_'):
                        print(f"       • {attr}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()

print("=" * 80)
print("💡 DIAGNOSIS")
print("=" * 80)
print()

# Load one model and give detailed diagnosis
try:
    with open(model_files[0], 'rb') as f:
        model_artifact = pickle.load(f)
    
    if isinstance(model_artifact, dict):
        if 'feature_names' not in model_artifact:
            print("❌ PROBLEM: Your model files are missing 'feature_names' key")
            print()
            print("This is why you're getting the error:")
            print("  ERROR:app:Failed PM25 1h: 'feature_names'")
            print()
            
            # Check if there's an alternative
            possible_keys = ['features', 'feature_list', 'columns', 'X_columns', 'feature_columns']
            found = [k for k in possible_keys if k in model_artifact]
            
            if found:
                print(f"✅ SOLUTION: Your models have '{found[0]}' instead of 'feature_names'")
                print()
                print("You need to either:")
                print(f"  1. Update app.py to use model['{found[0]}'] instead of model['feature_names']")
                print("  2. OR run a migration script to rename the key in all model files")
                print()
                print("I'll create a migration script for you!")
            else:
                print("❌ No feature names found in model file at all")
                print()
                print("This means your model files were saved incorrectly.")
                print("You may need to re-save them with the correct format.")
        else:
            print("✅ Your model files have 'feature_names' key")
            print("The error might be something else. Let me know if you need more help.")
    else:
        print("❌ Your model files are not in the expected dictionary format")
        print()
        print("Models should be saved as dictionaries with these keys:")
        print("  • 'calibrated_model': The trained model")
        print("  • 'feature_names': List of feature names")
        print("  • 'classes': List of category names")
        print("  • 'thresholds': Dictionary of thresholds per class")
        
except Exception as e:
    print(f"❌ Could not diagnose: {e}")

print()
print("=" * 80)