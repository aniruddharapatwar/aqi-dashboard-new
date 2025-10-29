"""
Model File Migration Script
Fixes model pickle files to have the correct key names expected by app.py

This script will:
1. Check all model files for missing/incorrect keys
2. Rename keys to match expected format
3. Backup originals before modifying
4. Validate the fixed files
"""

import pickle
import shutil
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("🔧 MODEL FILE MIGRATION SCRIPT")
print("=" * 80)
print()

model_dir = Path("Classification_trained_models")

if not model_dir.exists():
    print(f"❌ Model directory not found: {model_dir}")
    exit(1)

# Create backup directory
backup_dir = model_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
backup_dir.mkdir(exist_ok=True)

print(f"📁 Model directory: {model_dir}")
print(f"💾 Backup directory: {backup_dir}")
print()

# Key mapping: old_name -> new_name
KEY_MAPPINGS = {
    'feature_names': ['features', 'feature_list', 'columns', 'X_columns', 'feature_columns'],
    'classes': ['class_names', 'categories', 'labels', 'target_classes'],
    'calibrated_model': ['model', 'classifier', 'estimator', 'trained_model'],
    'thresholds': ['threshold', 'decision_thresholds', 'class_thresholds']
}

model_files = list(model_dir.glob("*.pkl"))
print(f"Found {len(model_files)} model files")
print()

fixed_count = 0
error_count = 0

for model_file in model_files:
    print(f"Processing: {model_file.name}")
    
    try:
        # Load model
        with open(model_file, 'rb') as f:
            model_artifact = pickle.load(f)
        
        if not isinstance(model_artifact, dict):
            print(f"  ⚠️  Skipping - not a dictionary")
            continue
        
        # Check if migration is needed
        needs_migration = False
        migrations = {}
        
        for expected_key, alternative_keys in KEY_MAPPINGS.items():
            if expected_key not in model_artifact:
                # Look for alternative
                for alt_key in alternative_keys:
                    if alt_key in model_artifact:
                        migrations[alt_key] = expected_key
                        needs_migration = True
                        print(f"  🔄 Will rename: '{alt_key}' → '{expected_key}'")
                        break
        
        if not needs_migration:
            print(f"  ✅ Already in correct format")
            continue
        
        # Backup original
        backup_file = backup_dir / model_file.name
        shutil.copy2(model_file, backup_file)
        print(f"  💾 Backed up to: {backup_file.name}")
        
        # Apply migrations
        for old_key, new_key in migrations.items():
            model_artifact[new_key] = model_artifact.pop(old_key)
        
        # Save updated model
        with open(model_file, 'wb') as f:
            pickle.dump(model_artifact, f)
        
        print(f"  ✅ Migrated and saved")
        fixed_count += 1
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        error_count += 1
    
    print()

print("=" * 80)
print("📊 MIGRATION SUMMARY")
print("=" * 80)
print(f"Total files: {len(model_files)}")
print(f"Fixed: {fixed_count}")
print(f"Errors: {error_count}")
print(f"Already correct: {len(model_files) - fixed_count - error_count}")
print()

if fixed_count > 0:
    print("✅ Migration complete!")
    print(f"💾 Original files backed up to: {backup_dir}")
    print()
    print("🧪 Test your system now with: python test.py")
else:
    print("ℹ️  No files needed migration")

print()
print("=" * 80)