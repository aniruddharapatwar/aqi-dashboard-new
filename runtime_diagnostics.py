"""
Runtime Prediction Diagnostics
Tests actual predictions with real data to identify where zeros come from
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add proper imports
sys.path.append('.')
from feature_engineer import FeatureEngineer
from predictor import ModelManager, SequenceGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def test_single_prediction(model_path: Path, data_path: Path, 
                          pollutant: str, horizon: str, location: str = "India Gate"):
    """Test a single model with actual data"""
    
    print(f"\n{'='*80}")
    print(f"TESTING: {pollutant}_{horizon}")
    print(f"{'='*80}")
    
    try:
        # 1. Load model artifacts
        model_dir = model_path / f"{pollutant}_{horizon}"
        artifacts_path = model_dir / "model_artifacts.pkl"
        
        print(f"Loading model from: {artifacts_path}")
        with open(artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        model = artifacts['model']
        feature_scaler = artifacts['feature_scaler']
        target_scaler = artifacts['target_scaler']
        sequence_length = artifacts['sequence_length']
        feature_names = artifacts.get('feature_names', None)
        
        print(f"✓ Model loaded successfully")
        print(f"  - Sequence length: {sequence_length}")
        print(f"  - Feature count: {len(feature_names) if feature_names else 'Unknown'}")
        print(f"  - Target scaler mean: {target_scaler.mean_[0]:.2f}")
        print(f"  - Target scaler scale: {target_scaler.scale_[0]:.2f}")
        
        # 2. Load data
        print(f"\nLoading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Filter for location
        loc_data = df[df['place'] == location].copy()
        if len(loc_data) == 0:
            print(f"❌ No data found for location: {location}")
            return None
        
        print(f"✓ Found {len(loc_data)} rows for {location}")
        
        # Sort by date
        loc_data = loc_data.sort_values('date').reset_index(drop=True)
        
        # Get enough historical data
        if len(loc_data) < sequence_length:
            print(f"⚠️  Not enough data: have {len(loc_data)}, need {sequence_length}")
            print(f"   Proceeding anyway...")
        
        # Take last 240 rows for sequences
        historical_data = loc_data.tail(min(240, len(loc_data)))
        
        print(f"✓ Using {len(historical_data)} historical rows")
        
        # 3. Engineer features
        print(f"\nEngineering features...")
        feature_engineer = FeatureEngineer()
        
        feature_list = []
        for idx in range(len(historical_data)):
            row_data = historical_data.iloc[[idx]]
            hist_before = historical_data.iloc[:idx] if idx > 0 else historical_data.iloc[[0]]
            
            try:
                features = feature_engineer.engineer_features(
                    current_data=row_data,
                    historical_data=hist_before,
                    pollutant=pollutant,
                    horizon=horizon
                )
                
                # Check for NaN or inf
                if features.isnull().any().any():
                    nan_cols = features.columns[features.isnull().any()].tolist()
                    print(f"⚠️  Row {idx}: Found NaN in features: {nan_cols[:5]}")
                
                if np.isinf(features.values).any():
                    print(f"⚠️  Row {idx}: Found inf values in features")
                
                feature_list.append(features)
                
            except Exception as e:
                print(f"❌ Feature engineering failed at row {idx}: {e}")
                raise
        
        historical_features = pd.concat(feature_list, axis=0, ignore_index=True)
        print(f"✓ Created features: {len(historical_features.columns)} columns, {len(historical_features)} rows")
        
        # Show feature statistics
        print(f"\nFeature statistics:")
        print(f"  - NaN count: {historical_features.isnull().sum().sum()}")
        print(f"  - Inf count: {np.isinf(historical_features.values).sum()}")
        print(f"  - Mean: {historical_features.mean().mean():.2f}")
        print(f"  - Std: {historical_features.std().mean():.2f}")
        
        # 4. Align features if needed
        if feature_names is not None:
            expected = set(feature_names)
            actual = set(historical_features.columns)
            
            missing = expected - actual
            extra = actual - expected
            
            if missing:
                print(f"⚠️  Missing features ({len(missing)}): {list(missing)[:5]}")
            if extra:
                print(f"⚠️  Extra features ({len(extra)}): {list(extra)[:5]}")
            
            # Align to expected features
            historical_features = historical_features.reindex(
                columns=feature_names,
                fill_value=0.0
            )
            print(f"✓ Aligned features to {len(feature_names)} columns")
        
        # 5. Scale features
        print(f"\nScaling features...")
        try:
            X_scaled = feature_scaler.transform(historical_features.values)
            print(f"✓ Scaled features: shape {X_scaled.shape}")
            print(f"  - Scaled mean: {X_scaled.mean():.4f}")
            print(f"  - Scaled std: {X_scaled.std():.4f}")
            print(f"  - Scaled min: {X_scaled.min():.4f}")
            print(f"  - Scaled max: {X_scaled.max():.4f}")
        except Exception as e:
            print(f"❌ Feature scaling failed: {e}")
            print(f"  - Scaler expects: {feature_scaler.n_features_in_} features")
            print(f"  - We have: {historical_features.shape[1]} features")
            raise
        
        # 6. Create sequence
        print(f"\nCreating sequence...")
        sequence_gen = SequenceGenerator(sequence_length)
        X_seq = sequence_gen.create_sequence(X_scaled)
        print(f"✓ Sequence shape: {X_seq.shape}")
        
        # Check for NaN or inf in sequence
        if np.isnan(X_seq).any():
            print(f"⚠️  NaN detected in sequence!")
        if np.isinf(X_seq).any():
            print(f"⚠️  Inf detected in sequence!")
        
        # 7. Make prediction
        print(f"\nMaking prediction...")
        y_pred_scaled = model.predict(X_seq, verbose=0)
        
        print(f"  📊 Scaled prediction: {y_pred_scaled[0][0]:.6f}")
        
        # Check if prediction is reasonable in scaled space
        if y_pred_scaled[0][0] < -10:
            print(f"  ⚠️  Extremely negative scaled prediction!")
        elif y_pred_scaled[0][0] > 10:
            print(f"  ⚠️  Extremely positive scaled prediction!")
        
        # 8. Inverse transform
        print(f"\nInverse transforming...")
        y_pred_raw = target_scaler.inverse_transform(y_pred_scaled).flatten()[0]
        
        print(f"  📊 Raw prediction (before clip): {y_pred_raw:.2f}")
        
        # Analysis
        if y_pred_raw < 0:
            print(f"  ❌ NEGATIVE PREDICTION!")
            print(f"     Formula: ({y_pred_scaled[0][0]:.4f} × {target_scaler.scale_[0]:.2f}) + {target_scaler.mean_[0]:.2f} = {y_pred_raw:.2f}")
        elif y_pred_raw == 0:
            print(f"  ⚠️  ZERO PREDICTION!")
        elif y_pred_raw < 1:
            print(f"  ⚠️  VERY LOW PREDICTION!")
        else:
            print(f"  ✓ Prediction looks reasonable")
        
        # 9. Apply clipping
        y_pred_final = np.clip(y_pred_raw, 0, 1000)
        print(f"  📊 Final prediction (after clip): {y_pred_final:.2f}")
        
        if y_pred_final != y_pred_raw:
            print(f"  ⚠️  Clipping changed the value!")
        
        # 10. Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY: {pollutant}_{horizon}")
        print(f"{'='*80}")
        print(f"Scaled prediction:  {y_pred_scaled[0][0]:.6f}")
        print(f"After inverse:      {y_pred_raw:.2f}")
        print(f"Final (clipped):    {y_pred_final:.2f}")
        
        if y_pred_final == 0.0:
            print(f"\n❌ ZERO PREDICTION DETECTED!")
            print(f"Root cause:")
            if y_pred_raw < 0:
                print(f"  - Model predicted negative value: {y_pred_raw:.2f}")
                print(f"  - Clipped to 0.0")
                print(f"  - Issue: Model is predicting too-negative scaled values")
                print(f"  - Solution: Retrain this specific model")
            else:
                print(f"  - Model predicted near-zero: {y_pred_raw:.2f}")
                print(f"  - Issue: Model behavior problem")
        else:
            print(f"✓ Prediction is non-zero")
        
        return {
            'pollutant': pollutant,
            'horizon': horizon,
            'scaled_prediction': float(y_pred_scaled[0][0]),
            'raw_prediction': float(y_pred_raw),
            'final_prediction': float(y_pred_final),
            'is_zero': y_pred_final == 0.0,
            'is_negative': y_pred_raw < 0
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run runtime diagnostics"""
    
    MODEL_PATH = Path("Classification_trained_models")
    DATA_PATH = Path("data/inference_data.csv")
    
    print("\n" + "="*80)
    print("RUNTIME PREDICTION DIAGNOSTICS")
    print("="*80)
    
    # Check paths
    if not MODEL_PATH.exists():
        print(f"❌ Model path not found: {MODEL_PATH}")
        return
    
    if not DATA_PATH.exists():
        print(f"❌ Data path not found: {DATA_PATH}")
        print("Please update DATA_PATH in this script")
        return
    
    print(f"\n✓ Model path: {MODEL_PATH}")
    print(f"✓ Data path: {DATA_PATH}")
    
    # Test the problematic models
    problem_models = [
        ('PM25', '6h'),
        ('PM10', '24h'),
    ]
    
    # Also test some working models for comparison
    working_models = [
        ('PM25', '1h'),
        ('PM10', '1h'),
    ]
    
    results = []
    
    print(f"\n" + "="*80)
    print("TESTING PROBLEMATIC MODELS")
    print("="*80)
    
    for pollutant, horizon in problem_models:
        result = test_single_prediction(MODEL_PATH, DATA_PATH, pollutant, horizon)
        if result:
            results.append(result)
        input("\nPress Enter to continue...")
    
    print(f"\n" + "="*80)
    print("TESTING WORKING MODELS (FOR COMPARISON)")
    print("="*80)
    
    for pollutant, horizon in working_models:
        result = test_single_prediction(MODEL_PATH, DATA_PATH, pollutant, horizon)
        if result:
            results.append(result)
        input("\nPress Enter to continue...")
    
    # Final summary
    print(f"\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for result in results:
        status = "❌ ZERO" if result['is_zero'] else "✓ OK"
        print(f"{status} {result['pollutant']:6s} {result['horizon']:4s}: "
              f"scaled={result['scaled_prediction']:8.4f} → "
              f"raw={result['raw_prediction']:8.2f} → "
              f"final={result['final_prediction']:8.2f}")
    
    # Identify patterns
    zero_models = [r for r in results if r['is_zero']]
    negative_models = [r for r in results if r['is_negative']]
    
    if zero_models:
        print(f"\n❌ Models producing zero predictions: {len(zero_models)}")
    
    if negative_models:
        print(f"❌ Models producing negative predictions: {len(negative_models)}")
        print(f"\nThese models need to be RETRAINED:")
        for r in negative_models:
            print(f"   - {r['pollutant']}_{r['horizon']}")

if __name__ == "__main__":
    main()
