"""
Testing Script - Verify Feature Engineering Corrections
========================================================

This script helps you verify that the corrected feature_engineer.py
produces the correct features that match your trained models.

Usage:
1. Replace your feature_engineer.py with feature_engineer_CORRECTED.py
2. Run this script: python test_feature_corrections.py
3. Review the results

"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Import the corrected feature engineer
try:
    from feature_engineer import FeatureEngineer, FeatureAligner
    print("✅ Successfully imported FeatureEngineer\n")
except ImportError as e:
    print(f"❌ Failed to import FeatureEngineer: {e}")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = Path("Classification_trained_models")  # Update if different
DATA_PATH = Path("data/inference_data.csv")  # Update if different

POLLUTANTS = ['PM25', 'PM10', 'NO2', 'OZONE']
HORIZONS = ['1h', '6h', '12h', '24h']

# Expected feature counts from training
EXPECTED_COUNTS = {
    '1h': 55,
    '6h': 83,
    '12h': 83,
    '24h': 83
}


# =============================================================================
# TEST 1: FEATURE COUNT VERIFICATION
# =============================================================================

def test_feature_counts():
    """Test if generated features match expected counts"""
    
    print("="*80)
    print("TEST 1: FEATURE COUNT VERIFICATION")
    print("="*80)
    
    engineer = FeatureEngineer()
    results = []
    
    # Create dummy data for testing
    current_data = pd.DataFrame({
        'lat': [28.6315],
        'lng': [77.2167],
        'temperature': [25.0],
        'humidity': [60.0],
        'dewPoint': [18.0],
        'apparentTemperature': [24.0],
        'precipIntensity': [0.0],
        'pressure': [1013.0],
        'surfacePressure': [1010.0],
        'cloudCover': [0.3],
        'windSpeed': [5.0],
        'windBearing': [180.0],
        'windGust': [7.0],
        'year': [2025],
        'month': [10],
        'day': [31],
        'hour': [12],
        'weekday': [4],
        'traffic_density_score': [3.5],
        'industrial_proximity': [2.0],
        'industrial_density_score': [1.5],
        'near_industrial_1km': [0],
        'near_industrial_3km': [1],
        'near_industrial_5km': [1],
        'exposure_index': [2.5]
    })
    
    for pollutant in POLLUTANTS:
        # Add pollutant to current data
        current_data[pollutant] = [50.0]
        
        # Create historical data (300 rows for sequence building)
        historical_data = pd.concat([current_data] * 300, ignore_index=True)
        historical_data[pollutant] = np.random.uniform(30, 70, 300)
        
        print(f"\n📊 Testing {pollutant}:")
        print("-" * 80)
        
        for horizon in HORIZONS:
            try:
                # Engineer features
                features = engineer.engineer_features(
                    current_data=current_data,
                    historical_data=historical_data,
                    pollutant=pollutant,
                    horizon=horizon
                )
                
                actual_count = len(features.columns)
                expected = EXPECTED_COUNTS[horizon]
                diff = abs(actual_count - expected)
                
                # Check if within tolerance (±15 features)
                status = "✅ PASS" if diff <= 15 else "⚠️  CHECK"
                
                print(f"  {horizon:>3s}: {actual_count:3d} features (expected ~{expected:3d}) {status}")
                
                results.append({
                    'pollutant': pollutant,
                    'horizon': horizon,
                    'actual': actual_count,
                    'expected': expected,
                    'diff': diff,
                    'pass': diff <= 15
                })
                
            except Exception as e:
                print(f"  {horizon:>3s}: ❌ ERROR - {e}")
                results.append({
                    'pollutant': pollutant,
                    'horizon': horizon,
                    'actual': 0,
                    'expected': expected,
                    'diff': expected,
                    'pass': False,
                    'error': str(e)
                })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY:")
    passed = sum(1 for r in results if r['pass'])
    total = len(results)
    print(f"  Passed: {passed}/{total} tests ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("  🎉 All tests passed!")
    else:
        print(f"  ⚠️  {total - passed} tests need attention")
    
    print("="*80 + "\n")
    
    return results


# =============================================================================
# TEST 2: LAG FEATURE VERIFICATION
# =============================================================================

def test_lag_features():
    """Verify lag features are created correctly"""
    
    print("="*80)
    print("TEST 2: LAG FEATURE VERIFICATION")
    print("="*80)
    
    engineer = FeatureEngineer()
    
    # Expected lag windows
    expected_lags = {
        '1h': [1, 2, 3, 6, 12, 24],
        '6h': [6, 12, 24, 48],
        '12h': [12, 24, 48, 72],
        '24h': [24, 48, 72, 96, 168]  # ← Should include 168h
    }
    
    print(f"\nExpected lag windows:")
    for horizon, lags in expected_lags.items():
        print(f"  {horizon}: {lags}")
    
    # Test with dummy data
    current_data = pd.DataFrame({'PM25': [50.0]})
    historical_data = pd.DataFrame({'PM25': np.random.uniform(30, 70, 200)})
    
    print(f"\nGenerating features for PM25...")
    all_pass = True
    
    for horizon in HORIZONS:
        try:
            features = engineer.engineer_features(
                current_data=current_data,
                historical_data=historical_data,
                pollutant='PM25',
                horizon=horizon
            )
            
            # Check which lag features exist
            lag_cols = [col for col in features.columns if 'PM25_lag_' in col or 'pm25_lag_' in col]
            
            # Extract lag values from column names
            actual_lags = []
            for col in lag_cols:
                # Extract number from "PM25_lag_24h" or "pm25_lag_24h"
                lag_str = col.split('_lag_')[1].replace('h', '')
                actual_lags.append(int(lag_str))
            
            actual_lags.sort()
            expected = expected_lags[horizon]
            
            if actual_lags == expected:
                print(f"  {horizon}: ✅ PASS - Lags: {actual_lags}")
            else:
                print(f"  {horizon}: ⚠️  MISMATCH")
                print(f"    Expected: {expected}")
                print(f"    Actual:   {actual_lags}")
                missing = set(expected) - set(actual_lags)
                extra = set(actual_lags) - set(expected)
                if missing:
                    print(f"    Missing:  {sorted(missing)}")
                if extra:
                    print(f"    Extra:    {sorted(extra)}")
                all_pass = False
                
        except Exception as e:
            print(f"  {horizon}: ❌ ERROR - {e}")
            all_pass = False
    
    print("\n" + "="*80)
    if all_pass:
        print("✅ All lag features are correct!")
    else:
        print("⚠️  Some lag features need fixing")
    print("="*80 + "\n")
    
    return all_pass


# =============================================================================
# TEST 3: ROLLING FEATURE VERIFICATION
# =============================================================================

def test_rolling_features():
    """Verify rolling features are created correctly"""
    
    print("="*80)
    print("TEST 3: ROLLING FEATURE VERIFICATION")
    print("="*80)
    
    engineer = FeatureEngineer()
    
    # Expected rolling windows
    expected_windows = {
        '1h': [6, 12, 24, 48],  # ← Should have rolling features
        '6h': [6, 12],
        '12h': [12, 24],
        '24h': [24, 48]
    }
    
    # Expected stats per horizon
    expected_stats = {
        '1h': ['mean', 'std'],  # Only 2 stats for 1h
        '6h': ['mean', 'std', 'min', 'max'],  # All 4 stats for others
        '12h': ['mean', 'std', 'min', 'max'],
        '24h': ['mean', 'std', 'min', 'max']
    }
    
    print(f"\nExpected rolling windows:")
    for horizon, windows in expected_windows.items():
        stats = expected_stats[horizon]
        print(f"  {horizon}: windows {windows} with stats {stats}")
    
    # Test with dummy data
    current_data = pd.DataFrame({'PM25': [50.0]})
    historical_data = pd.DataFrame({'PM25': np.random.uniform(30, 70, 200)})
    
    print(f"\nGenerating features for PM25...")
    all_pass = True
    
    for horizon in HORIZONS:
        try:
            features = engineer.engineer_features(
                current_data=current_data,
                historical_data=historical_data,
                pollutant='PM25',
                horizon=horizon
            )
            
            # Check rolling features
            rolling_cols = [col for col in features.columns 
                          if 'PM25_rolling_' in col or 'pm25_rolling_' in col]
            
            expected_wins = expected_windows[horizon]
            expected_sts = expected_stats[horizon]
            expected_count = len(expected_wins) * len(expected_sts)
            actual_count = len(rolling_cols)
            
            if actual_count == expected_count:
                print(f"  {horizon}: ✅ PASS - {actual_count} rolling features")
            else:
                print(f"  {horizon}: ⚠️  MISMATCH")
                print(f"    Expected: {expected_count} features ({len(expected_wins)} windows × {len(expected_sts)} stats)")
                print(f"    Actual:   {actual_count} features")
                print(f"    Features: {rolling_cols[:5]}{'...' if len(rolling_cols) > 5 else ''}")
                all_pass = False
                
        except Exception as e:
            print(f"  {horizon}: ❌ ERROR - {e}")
            all_pass = False
    
    print("\n" + "="*80)
    if all_pass:
        print("✅ All rolling features are correct!")
    else:
        print("⚠️  Some rolling features need fixing")
    print("="*80 + "\n")
    
    return all_pass


# =============================================================================
# TEST 4: MODEL ARTIFACT COMPARISON (if models available)
# =============================================================================

def test_model_artifact_alignment():
    """Compare generated features with model artifacts"""
    
    print("="*80)
    print("TEST 4: MODEL ARTIFACT ALIGNMENT")
    print("="*80)
    
    if not MODEL_PATH.exists():
        print(f"\n⚠️  Model path not found: {MODEL_PATH}")
        print("   Skipping model artifact test.")
        print("="*80 + "\n")
        return None
    
    engineer = FeatureEngineer()
    aligner = FeatureAligner()
    results = []
    
    for pollutant in POLLUTANTS:
        print(f"\n📊 Testing {pollutant}:")
        print("-" * 80)
        
        for horizon in HORIZONS:
            # Check if model exists
            model_dir = MODEL_PATH / f"{pollutant}_{horizon}"
            artifacts_path = model_dir / "model_artifacts.pkl"
            
            if not artifacts_path.exists():
                print(f"  {horizon}: ⚠️  Model artifacts not found")
                continue
            
            try:
                # Load model artifacts
                with open(artifacts_path, 'rb') as f:
                    artifacts = pickle.load(f)
                
                model_features = artifacts.get('feature_names', [])
                
                if not model_features:
                    print(f"  {horizon}: ⚠️  No feature_names in artifacts")
                    continue
                
                # Generate features
                current_data = pd.DataFrame({pollutant: [50.0]})
                historical_data = pd.DataFrame({pollutant: np.random.uniform(30, 70, 250)})
                
                features = engineer.engineer_features(
                    current_data=current_data,
                    historical_data=historical_data,
                    pollutant=pollutant,
                    horizon=horizon
                )
                
                # Align features
                aligned = aligner.align_features(features, model_features)
                
                # Check alignment
                model_count = len(model_features)
                generated_count = len(features.columns)
                aligned_count = len(aligned.columns)
                
                # Check for mismatches
                model_set = set(model_features)
                generated_set = set(features.columns)
                
                missing = model_set - generated_set
                extra = generated_set - model_set
                
                if not missing and not extra:
                    status = "✅ PERFECT"
                elif len(missing) <= 5 and len(extra) <= 5:
                    status = "✅ GOOD"
                else:
                    status = "⚠️  MISMATCH"
                
                print(f"  {horizon}: {status}")
                print(f"    Model expects: {model_count} features")
                print(f"    Generated:     {generated_count} features")
                
                if missing:
                    print(f"    Missing:       {len(missing)} features")
                    if len(missing) <= 10:
                        for feat in list(missing)[:10]:
                            print(f"      - {feat}")
                
                if extra:
                    print(f"    Extra:         {len(extra)} features")
                    if len(extra) <= 10:
                        for feat in list(extra)[:10]:
                            print(f"      - {feat}")
                
                results.append({
                    'pollutant': pollutant,
                    'horizon': horizon,
                    'model_count': model_count,
                    'generated_count': generated_count,
                    'missing': len(missing),
                    'extra': len(extra),
                    'perfect': not missing and not extra
                })
                
            except Exception as e:
                print(f"  {horizon}: ❌ ERROR - {e}")
                results.append({
                    'pollutant': pollutant,
                    'horizon': horizon,
                    'error': str(e)
                })
    
    # Summary
    if results:
        print("\n" + "="*80)
        print("SUMMARY:")
        perfect = sum(1 for r in results if r.get('perfect', False))
        total = len([r for r in results if 'error' not in r])
        if total > 0:
            print(f"  Perfect alignment: {perfect}/{total} models ({perfect/total*100:.1f}%)")
            
            if perfect == total:
                print("  🎉 All models have perfect feature alignment!")
            else:
                print(f"  ⚠️  {total - perfect} models have feature mismatches")
        else:
            print("  ⚠️  No model artifacts could be tested")
        print("="*80 + "\n")
    
    return results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all tests"""
    
    print("\n" + "🔬"*40)
    print("FEATURE ENGINEERING CORRECTION TESTS")
    print("🔬"*40 + "\n")
    
    # Run tests
    results = {}
    
    print("Running tests...\n")
    
    results['feature_counts'] = test_feature_counts()
    results['lag_features'] = test_lag_features()
    results['rolling_features'] = test_rolling_features()
    results['model_alignment'] = test_model_artifact_alignment()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    passed = sum([
        all(r['pass'] for r in results['feature_counts']),
        results['lag_features'],
        results['rolling_features']
    ])
    total = 3
    
    if results['model_alignment']:
        model_passed = sum(1 for r in results['model_alignment'] if r.get('perfect', False))
        model_total = len([r for r in results['model_alignment'] if 'error' not in r])
        if model_total > 0:
            print(f"\nCore Tests:  {passed}/{total} passed")
            print(f"Model Tests: {model_passed}/{model_total} perfect alignment")
            total_passed = passed + model_passed
            total_tests = total + model_total
            print(f"\nOverall:     {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")
        else:
            print(f"\nCore Tests: {passed}/{total} passed")
    else:
        print(f"\nCore Tests: {passed}/{total} passed")
    
    if passed == total:
        print("\n✅ All core tests passed! Feature engineering is correct.")
        print("\nNext steps:")
        print("1. Replace feature_engineer.py with feature_engineer_CORRECTED.py")
        print("2. Test the dashboard with real data")
        print("3. Verify predictions are reasonable")
    else:
        print("\n⚠️  Some tests failed. Review the output above for details.")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    run_all_tests()