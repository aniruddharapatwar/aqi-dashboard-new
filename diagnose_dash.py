#!/usr/bin/env python3
"""
AQI Dashboard Diagnostic Tool
Checks for missing components and provides actionable fixes
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

class Colors:
    """Terminal colors for better readability"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def check_data_file():
    """Check if inference data file exists and validate format"""
    print_header("DATA FILE VALIDATION")
    
    data_path = Path("data/inference_data.csv")
    
    if not data_path.exists():
        print_error(f"Data file not found: {data_path}")
        print_info("Expected location: data/inference_data.csv")
        print_info("\nTo fix:")
        print_info("  1. Create 'data' directory: mkdir -p data")
        print_info("  2. Place your historical data CSV as: data/inference_data.csv")
        print_info("  3. Or create symlink: ln -s /path/to/your/data.csv data/inference_data.csv")
        return False
    
    print_success(f"Data file found: {data_path}")
    
    # Try to read and validate
    try:
        df = pd.read_csv(data_path)
        print_success(f"Data file readable: {len(df)} rows, {len(df.columns)} columns")
        
        # Check required columns
        required_cols = ['timestamp', 'lat', 'lon', 'location', 'PM25', 'PM10', 'NO2', 'OZONE']
        missing_cols = [col for col in required_cols if col not in df.columns and col.lower() not in [c.lower() for c in df.columns]]
        
        if missing_cols:
            print_error(f"Missing required columns: {', '.join(missing_cols)}")
            print_info(f"Available columns: {', '.join(df.columns.tolist()[:10])}...")
            return False
        
        print_success("All required columns present")
        
        # Check data completeness
        if 'timestamp' in df.columns or 'date' in df.columns:
            date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            date_range = (df[date_col].max() - df[date_col].min()).days
            print_info(f"Date range: {date_range} days ({df[date_col].min()} to {df[date_col].max()})")
            
            if date_range < 60:
                print_warning(f"Only {date_range} days of data. Recommended: 60+ days")
            else:
                print_success(f"Sufficient data: {date_range} days")
        
        # Check locations
        if 'location' in df.columns:
            locations = df['location'].unique()
            print_info(f"Locations in data: {len(locations)}")
            print_info(f"  Sample locations: {', '.join([str(l) for l in locations[:5]])}")
        
        # Check for nulls in critical columns
        pollutants = ['PM25', 'PM10', 'NO2', 'OZONE']
        for pol in pollutants:
            if pol in df.columns:
                null_pct = (df[pol].isna().sum() / len(df)) * 100
                if null_pct > 10:
                    print_warning(f"{pol}: {null_pct:.1f}% missing values")
                else:
                    print_success(f"{pol}: {null_pct:.1f}% missing values")
        
        return True
        
    except Exception as e:
        print_error(f"Error reading data file: {e}")
        return False

def check_models():
    """Check if LSTM models exist"""
    print_header("LSTM MODELS VALIDATION")
    
    model_path = Path("Classification_trained_models")
    
    if not model_path.exists():
        print_error(f"Models directory not found: {model_path}")
        print_info("\nTo fix:")
        print_info("  Option 1 - Use existing models:")
        print_info("    mkdir -p Classification_trained_models")
        print_info("    cp -r /path/to/your/trained_models/* Classification_trained_models/")
        print_info("\n  Option 2 - Train new models:")
        print_info("    python lstm_model_training_vf.py")
        print_info("\n  Option 3 - Create mock models for testing:")
        print_info("    python create_mock_models.py")
        return False
    
    print_success(f"Models directory found: {model_path}")
    
    # Check for all required models
    pollutants = ['PM25', 'PM10', 'NO2', 'OZONE']
    horizons = ['1h', '6h', '12h', '24h']
    
    total_models = len(pollutants) * len(horizons)
    found_models = 0
    missing_models = []
    incomplete_models = []
    
    for pollutant in pollutants:
        for horizon in horizons:
            model_dir = model_path / f"{pollutant}_{horizon}"
            keras_model = model_dir / "lstm_pure_regression.h5"
            artifacts = model_dir / "model_artifacts.pkl"
            
            if model_dir.exists():
                if keras_model.exists() and artifacts.exists():
                    found_models += 1
                    print_success(f"{pollutant}_{horizon}: Complete")
                else:
                    incomplete_models.append(f"{pollutant}_{horizon}")
                    missing_files = []
                    if not keras_model.exists():
                        missing_files.append("lstm_pure_regression.h5")
                    if not artifacts.exists():
                        missing_files.append("model_artifacts.pkl")
                    print_warning(f"{pollutant}_{horizon}: Missing {', '.join(missing_files)}")
            else:
                missing_models.append(f"{pollutant}_{horizon}")
                print_error(f"{pollutant}_{horizon}: Directory not found")
    
    print_info(f"\nModel Summary: {found_models}/{total_models} complete")
    
    if missing_models:
        print_warning(f"Missing model directories: {', '.join(missing_models)}")
    
    if incomplete_models:
        print_warning(f"Incomplete models (missing files): {', '.join(incomplete_models)}")
    
    return found_models == total_models

def check_whitelist():
    """Check if location whitelist exists"""
    print_header("LOCATION WHITELIST VALIDATION")
    
    whitelist_path = Path("region_wise_popular_places_from_inference.csv")
    
    if not whitelist_path.exists():
        print_error(f"Whitelist not found: {whitelist_path}")
        return False
    
    print_success(f"Whitelist found: {whitelist_path}")
    
    try:
        df = pd.read_csv(whitelist_path)
        print_success(f"Whitelist readable: {len(df)} locations")
        
        # Check required columns
        required_cols = ['Region', 'Place', 'Latitude', 'Longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print_error(f"Missing columns: {', '.join(missing_cols)}")
            return False
        
        print_success("All required columns present")
        
        # Show regions
        regions = df['Region'].unique()
        print_info(f"Regions: {', '.join(regions)}")
        print_info(f"Sample locations: {', '.join(df['Place'].head(3).tolist())}")
        
        return True
        
    except Exception as e:
        print_error(f"Error reading whitelist: {e}")
        return False

def check_dependencies():
    """Check if required Python packages are installed"""
    print_header("DEPENDENCIES VALIDATION")
    
    required_packages = {
        'fastapi': 'FastAPI web framework',
        'uvicorn': 'ASGI server',
        'pandas': 'Data processing',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning (scikit-learn)',
        'tensorflow': 'Deep learning (LSTM models)',
        'pickle': 'Model serialization'
    }
    
    all_installed = True
    
    for package, description in required_packages.items():
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print_success(f"{package}: {description}")
        except ImportError:
            print_error(f"{package}: NOT INSTALLED ({description})")
            all_installed = False
    
    if not all_installed:
        print_info("\nTo install missing packages:")
        print_info("  pip install -r requirements.txt")
    
    return all_installed

def check_code_files():
    """Check if required code files exist"""
    print_header("CODE FILES VALIDATION")
    
    required_files = {
        'app.py': 'Main FastAPI application',
        'predictor.py': 'Prediction logic',
        'feature_engineer.py': 'Feature engineering',
        'requirements.txt': 'Python dependencies'
    }
    
    all_exist = True
    
    for file, description in required_files.items():
        file_path = Path(file)
        if file_path.exists():
            print_success(f"{file}: {description}")
        else:
            print_error(f"{file}: NOT FOUND ({description})")
            all_exist = False
    
    return all_exist

def provide_summary(results):
    """Provide summary and next steps"""
    print_header("DIAGNOSTIC SUMMARY")
    
    all_good = all(results.values())
    
    if all_good:
        print_success("All checks passed! Your dashboard should work correctly.")
        print_info("\nNext steps:")
        print_info("  1. Start the server: uvicorn app:app --reload")
        print_info("  2. Test health: curl http://localhost:8000/api/health")
        print_info("  3. Test prediction: curl -X POST http://localhost:8000/api/predict \\")
        print_info('       -H "Content-Type: application/json" \\')
        print_info('       -d \'{"location": "Connaught Place (Rajiv Chowk)", "standard": "IN"}\'')
    else:
        print_error("Some checks failed. Please fix the issues above.")
        print_info("\nPriority fixes:")
        
        if not results['data']:
            print_info("  1. ⚠️  CRITICAL: Add data file (data/inference_data.csv)")
        
        if not results['models']:
            print_info("  2. ⚠️  CRITICAL: Add LSTM models (Classification_trained_models/)")
        
        if not results['dependencies']:
            print_info("  3. Install missing dependencies: pip install -r requirements.txt")
        
        print_info("\nSee AQI_Dashboard_Debug_Report.md for detailed solutions")

def main():
    """Run all diagnostic checks"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║          AQI DASHBOARD DIAGNOSTIC TOOL                            ║")
    print("║          Checking for missing components...                       ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    results = {
        'code': check_code_files(),
        'dependencies': check_dependencies(),
        'whitelist': check_whitelist(),
        'data': check_data_file(),
        'models': check_models()
    }
    
    provide_summary(results)
    
    print()
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())