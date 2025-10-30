"""Test the enhanced feature engineer with your real data"""

import pandas as pd
from feature_engineer import FeatureEngineer
from pathlib import Path

# Configuration
CSV_PATH = "data/inference_data.csv"
POLLUTANTS = ['PM25', 'PM10', 'NO2', 'OZONE']
HORIZONS = ['1h', '6h', '12h', '24h']

def test_feature_engineering():
    """Test feature engineering with real data"""
    
    print("="*80)
    print("TESTING ENHANCED FEATURE ENGINEER")
    print("="*80)
    
    # Load data
    if not Path(CSV_PATH).exists():
        print(f"❌ CSV not found: {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    print(f"\n✅ Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
    
    # Get current and historical data
    current = df.iloc[[-1]].copy()
    historical = df.tail(250).copy()
    
    print(f"   Current row: {current['date'].iloc[0] if 'date' in current.columns else 'N/A'}")
    print(f"   Historical rows: {len(historical)}")
    
    # Create feature engineer
    engineer = FeatureEngineer()
    
    # Test each combination
    print("\n" + "="*80)
    print("FEATURE GENERATION TEST")
    print("="*80)
    
    expected = {
        '1h': 50,
        '6h': 70,
        '12h': 70,
        '24h': 71
    }
    
    for pollutant in POLLUTANTS:
        print(f"\n📊 Testing {pollutant}:")
        print("-"*80)
        
        for horizon in HORIZONS:
            try:
                features = engineer.engineer_features(
                    current_data=current,
                    historical_data=historical,
                    pollutant=pollutant,
                    horizon=horizon
                )
                
                actual = len(features.columns)
                exp = expected[horizon]
                diff = abs(actual - exp)
                
                if diff <= 10:
                    status = "✅ PASS"
                elif diff <= 20:
                    status = "⚠️  CLOSE"
                else:
                    status = "❌ FAIL"
                
                print(f"  {horizon:>4s}: {actual:3d} features (expected {exp:3d}) {status}")
                
            except Exception as e:
                print(f"  {horizon:>4s}: ❌ ERROR - {str(e)[:50]}")
    
    # Show sample features
    print("\n" + "="*80)
    print("SAMPLE FEATURES (PM25, 1h)")
    print("="*80)
    
    features = engineer.engineer_features(
        current_data=current,
        historical_data=historical,
        pollutant='PM25',
        horizon='1h'
    )
    
    print(f"\nGenerated {len(features.columns)} features:")
    print("\nBase features:")
    base_cols = [c for c in features.columns if any(x in c for x in ['lat', 'lng', 'year', 'month', 'day', 'hour', 'weekday', 'temperature', 'sin', 'cos', 'weekend', 'traffic', 'industrial', 'exposure'])]
    for i, col in enumerate(base_cols[:15], 1):
        print(f"  {i:2d}. {col}")
    if len(base_cols) > 15:
        print(f"  ... and {len(base_cols) - 15} more base features")
    
    print("\nEngineered features:")
    eng_cols = [c for c in features.columns if 'lag' in c or 'rolling' in c]
    for i, col in enumerate(eng_cols[:15], 1):
        print(f"  {i:2d}. {col}")
    if len(eng_cols) > 15:
        print(f"  ... and {len(eng_cols) - 15} more engineered features")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_feature_engineering()