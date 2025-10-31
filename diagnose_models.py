"""
Model Diagnostics Script
Tests all LSTM models to identify which ones have issues
"""

import pickle
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_model(model_path: Path, pollutant: str, horizon: str):
    """Diagnose a single model"""
    model_dir = model_path / f"{pollutant}_{horizon}"
    artifacts_path = model_dir / "model_artifacts.pkl"
    
    if not artifacts_path.exists():
        return {
            'status': 'missing',
            'error': f'File not found: {artifacts_path}'
        }
    
    try:
        # Load artifacts
        with open(artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        # Check target scaler
        target_scaler = artifacts.get('target_scaler')
        if target_scaler is None:
            return {
                'status': 'error',
                'error': 'No target scaler found'
            }
        
        # Test with dummy prediction
        dummy_scaled = np.array([[0.0]])  # Neutral scaled value
        try:
            dummy_pred = target_scaler.inverse_transform(dummy_scaled)[0][0]
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Scaler transform failed: {e}'
            }
        
        # Check scaler parameters
        scaler_mean = target_scaler.mean_[0] if hasattr(target_scaler, 'mean_') else None
        scaler_scale = target_scaler.scale_[0] if hasattr(target_scaler, 'scale_') else None
        
        # Check for suspicious scaler parameters
        issues = []
        if scaler_mean is not None and scaler_mean < 0:
            issues.append(f"Negative mean: {scaler_mean:.2f}")
        if scaler_scale is not None and scaler_scale == 0:
            issues.append(f"Zero scale (division by zero risk)")
        if dummy_pred < 0:
            issues.append(f"Zero-scaled input produces negative output: {dummy_pred:.2f}")
        
        return {
            'status': 'ok' if not issues else 'warning',
            'scaler_mean': float(scaler_mean) if scaler_mean is not None else None,
            'scaler_scale': float(scaler_scale) if scaler_scale is not None else None,
            'test_prediction_at_zero': float(dummy_pred),
            'issues': issues
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def main():
    """Run diagnostics on all models"""
    MODEL_PATH = Path("Classification_trained_models")
    
    if not MODEL_PATH.exists():
        print(f"❌ Model path not found: {MODEL_PATH}")
        print("Please run this script from the directory containing 'Classification_trained_models'")
        return
    
    pollutants = ['PM25', 'PM10', 'NO2', 'OZONE']
    horizons = ['1h', '6h', '12h', '24h']
    
    print("\n" + "="*80)
    print("MODEL DIAGNOSTICS REPORT")
    print("="*80 + "\n")
    
    failed_models = []
    warning_models = []
    
    for pollutant in pollutants:
        print(f"\n{pollutant}:")
        print("-" * 60)
        
        for horizon in horizons:
            result = diagnose_model(MODEL_PATH, pollutant, horizon)
            status_icon = {
                'ok': '✓',
                'warning': '⚠️',
                'error': '❌',
                'missing': '❓'
            }.get(result['status'], '?')
            
            print(f"  {status_icon} {horizon:4s}: ", end='')
            
            if result['status'] == 'ok':
                print(f"OK (mean={result['scaler_mean']:.2f}, scale={result['scaler_scale']:.4f})")
            elif result['status'] == 'warning':
                print(f"WARNING - {', '.join(result['issues'])}")
                warning_models.append(f"{pollutant}_{horizon}")
                print(f"       Mean: {result['scaler_mean']:.2f}, Scale: {result['scaler_scale']:.4f}")
            elif result['status'] == 'error':
                print(f"ERROR - {result['error']}")
                failed_models.append(f"{pollutant}_{horizon}")
            else:
                print(f"MISSING - {result['error']}")
                failed_models.append(f"{pollutant}_{horizon}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if failed_models:
        print(f"\n❌ FAILED MODELS ({len(failed_models)}):")
        for model in failed_models:
            print(f"   - {model}")
        print("\n   These models need to be RETRAINED from scratch.")
    
    if warning_models:
        print(f"\n⚠️  MODELS WITH WARNINGS ({len(warning_models)}):")
        for model in warning_models:
            print(f"   - {model}")
        print("\n   These models may be producing zero or negative predictions.")
        print("   They should be RETRAINED with proper target scaling.")
    
    if not failed_models and not warning_models:
        print("\n✓ All models passed diagnostics!")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
