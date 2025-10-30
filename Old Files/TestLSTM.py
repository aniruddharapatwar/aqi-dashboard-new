"""
Test Script for LSTM Prediction Pipeline
Run this to verify your LSTM models are working correctly
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test 1: Verify all required imports"""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Checking imports...")
    logger.info("="*80)
    
    try:
        import tensorflow as tf
        logger.info(f"✓ TensorFlow version: {tf.__version__}")
    except ImportError as e:
        logger.error("✗ TensorFlow not installed!")
        logger.error("  Run: pip install tensorflow>=2.15.0")
        return False
    
    try:
        import keras
        logger.info(f"✓ Keras version: {keras.__version__}")
    except ImportError as e:
        logger.error("✗ Keras not installed!")
        logger.error("  Run: pip install keras>=3.0.0")
        return False
    
    try:
        import sklearn
        logger.info(f"✓ Scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        logger.error("✗ Scikit-learn not installed!")
        return False
    
    logger.info("\n✓ All imports successful!")
    return True


def test_model_files(model_path: Path):
    """Test 2: Check model files exist"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Checking model files...")
    logger.info("="*80)
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.error(f"✗ Model directory not found: {model_path}")
        return False
    
    logger.info(f"✓ Model directory exists: {model_path}")
    
    # Check for expected structure
    pollutants = ['PM25', 'PM10', 'NO2', 'OZONE']
    horizons = ['1h', '6h', '12h', '24h']
    
    missing_models = []
    found_models = []
    
    for pollutant in pollutants:
        for horizon in horizons:
            model_name = f"{pollutant}_{horizon}"
            model_dir = model_path / model_name
            keras_file = model_dir / "lstm_pure_regression.h5"
            artifacts_file = model_dir / "model_artifacts.pkl"
            
            if model_dir.exists() and keras_file.exists() and artifacts_file.exists():
                found_models.append(model_name)
                logger.info(f"  ✓ {model_name}")
            else:
                missing_models.append(model_name)
                logger.warning(f"  ✗ {model_name} - missing files")
    
    logger.info(f"\nFound {len(found_models)}/{len(pollutants) * len(horizons)} models")
    
    if missing_models:
        logger.warning(f"\nMissing models: {', '.join(missing_models)}")
    
    return len(found_models) > 0


def test_model_loading(model_path: Path):
    """Test 3: Try loading a model"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Testing model loading...")
    logger.info("="*80)
    
    try:
        from predictor_fixed import ModelManager
        
        model_manager = ModelManager(model_path)
        
        # Try loading PM25 1h model
        logger.info("Attempting to load PM25_1h model...")
        artifact = model_manager.load_model("PM25", "1h")
        
        logger.info(f"✓ Model loaded successfully!")
        logger.info(f"  - Model type: {type(artifact['model'])}")
        logger.info(f"  - Sequence length: {artifact['sequence_length']}")
        logger.info(f"  - Number of features: {len(artifact['feature_names'])}")
        logger.info(f"  - Feature scaler: {type(artifact['feature_scaler'])}")
        logger.info(f"  - Target scaler: {type(artifact['target_scaler'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineering(data_path: Path):
    """Test 4: Feature engineering"""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Testing feature engineering...")
    logger.info("="*80)
    
    try:
        from feature_engineer import FeatureEngineer
        
        # Load data
        if not data_path.exists():
            logger.error(f"✗ Data file not found: {data_path}")
            return False
        
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Handle date column
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        
        logger.info(f"  Loaded {len(df)} rows")
        logger.info(f"  Columns: {list(df.columns)[:10]}...")
        
        # Get data for a location
        if 'location' in df.columns:
            locations = df['location'].unique()
            test_location = locations[0]
            logger.info(f"  Testing with location: {test_location}")
            
            location_data = df[df['location'] == test_location].sort_values('timestamp')
            
            if len(location_data) < 2:
                logger.error(f"✗ Insufficient data for location: {test_location}")
                return False
            
            current = location_data.iloc[[-1]]
            historical = location_data.iloc[:-1]
            
            logger.info(f"  Current: {len(current)} rows")
            logger.info(f"  Historical: {len(historical)} rows")
            
            # Test feature engineering
            fe = FeatureEngineer()
            features = fe.engineer_features(
                current_data=current,
                historical_data=historical,
                pollutant='PM25',
                horizon='1h'
            )
            
            logger.info(f"✓ Engineered {len(features.columns)} features")
            logger.info(f"  Feature names: {list(features.columns)[:10]}...")
            
            return True
        else:
            logger.error("✗ No 'location' column in data")
            return False
            
    except Exception as e:
        logger.error(f"✗ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_prediction(model_path: Path, data_path: Path):
    """Test 5: End-to-end prediction"""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Testing end-to-end prediction...")
    logger.info("="*80)
    
    try:
        from predictor_fixed import PollutantPredictor
        
        # Load data
        df = pd.read_csv(data_path)
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        
        # Get test location
        locations = df['location'].unique()
        test_location = locations[0]
        location_data = df[df['location'] == test_location].sort_values('timestamp')
        
        current = location_data.iloc[[-1]]
        historical = location_data.iloc[:-1]
        
        # Create predictor
        predictor = PollutantPredictor(model_path)
        
        # Test single prediction
        logger.info(f"Testing prediction for {test_location}...")
        category, value = predictor.predict_single_pollutant(
            current_data=current,
            historical_data=historical,
            pollutant='PM25',
            horizon='1h'
        )
        
        logger.info(f"✓ Prediction successful!")
        logger.info(f"  - Predicted concentration: {value:.2f} µg/m³")
        logger.info(f"  - AQI category: {category}")
        
        # Test all horizons
        logger.info("\nTesting all horizons for PM25...")
        all_horizons = predictor.predict_all_horizons(
            current_data=current,
            historical_data=historical,
            pollutant='PM25',
            standard='IN'
        )
        
        for horizon, result in all_horizons.items():
            logger.info(f"  {horizon}: {result['predicted_value']:.2f} µg/m³ → {result['category']}")
        
        logger.info("\n✓ End-to-end prediction successful!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("LSTM PREDICTION PIPELINE TEST SUITE")
    logger.info("="*80)
    
    # Configuration
    MODEL_PATH = Path("./Classification_trained_models")  # or ./lstm_models
    DATA_PATH = Path("./data/inference_data.csv")
    
    # Allow command line args
    if len(sys.argv) > 1:
        MODEL_PATH = Path(sys.argv[1])
    if len(sys.argv) > 2:
        DATA_PATH = Path(sys.argv[2])
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Model path: {MODEL_PATH}")
    logger.info(f"  Data path: {DATA_PATH}")
    
    # Run tests
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Model Files", test_model_files(MODEL_PATH)))
    results.append(("Model Loading", test_model_loading(MODEL_PATH)))
    results.append(("Feature Engineering", test_feature_engineering(DATA_PATH)))
    results.append(("End-to-End Prediction", test_end_to_end_prediction(MODEL_PATH, DATA_PATH)))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")
    
    logger.info(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Your LSTM pipeline is ready.")
    else:
        logger.info("\n⚠️  Some tests failed. Please fix the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)