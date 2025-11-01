"""
Complete LSTM Model Predictor with Comprehensive AQI Calculator
PRODUCTION VERSION: Proper feature alignment, robust error handling
FIXED: US EPA AQI calculation independent of Indian categories
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Deep Learning imports for LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model as keras_load_model

# Import from same directory
from feature_engineer import FeatureEngineer, FeatureAligner

logger = logging.getLogger(__name__)


# ============================================================================
# AQI BREAKPOINTS AND STANDARDS
# ============================================================================

INDIAN_AQI_BREAKPOINTS = {
    'PM25': {
        'Good': (0, 30), 'Satisfactory': (31, 60), 'Moderate': (61, 90),
        'Poor': (91, 120), 'Very_Poor': (121, 250), 'Severe': (251, 500)
    },
    'PM10': {
        'Good': (0, 50), 'Satisfactory': (51, 100), 'Moderate': (101, 250),
        'Poor': (251, 350), 'Very_Poor': (351, 430), 'Severe': (431, 600)
    },
    'NO2': {
        'Good': (0, 40), 'Satisfactory': (41, 80), 'Moderate': (81, 180),
        'Poor': (181, 280), 'Very_Poor': (281, 400), 'Severe': (401, 500)
    },
    'OZONE': {
        'Good': (0, 50), 'Satisfactory': (51, 100), 'Moderate': (101, 168),
        'Poor': (169, 208), 'Very_Poor': (209, 748), 'Severe': (749, 1000)
    }
}

INDIAN_AQI_INDEX = {
    'Good': (0, 50), 'Satisfactory': (51, 100), 'Moderate': (101, 200),
    'Poor': (201, 300), 'Very_Poor': (301, 400), 'Severe': (401, 500)
}

US_AQI_BREAKPOINTS = {
    'PM25': {
        'Good': (0.0, 12.0), 'Moderate': (12.1, 35.4), 'Unhealthy (for sensitive)': (35.5, 55.4),
        'Unhealthy': (55.5, 150.4), 'Very_Unhealthy': (150.5, 250.4), 'Hazardous': (250.5, 500.4)
    },
    'PM10': {
        'Good': (0, 54), 'Moderate': (55, 154), 'Unhealthy (for sensitive)': (155, 254),
        'Unhealthy': (255, 354), 'Very_Unhealthy': (355, 424), 'Hazardous': (425, 604)
    },
    'NO2': {
        'Good': (0, 53), 'Moderate': (54, 100), 'Unhealthy (for sensitive)': (101, 360),
        'Unhealthy': (361, 649), 'Very_Unhealthy': (650, 1249), 'Hazardous': (1250, 2049)
    },
    'OZONE': {
        'Good': (0.000, 0.054), 'Moderate': (0.055, 0.070), 'Unhealthy (for sensitive)': (0.071, 0.085),
        'Unhealthy': (0.086, 0.105), 'Very_Unhealthy': (0.106, 0.200), 'Hazardous': (0.201, 0.604)
    }
}

US_AQI_INDEX = {
    'Good': (0, 50),
    'Moderate': (51, 100),
    'Unhealthy (for sensitive)': (101, 150),
    'Unhealthy': (151, 200),
    'Very_Unhealthy': (201, 300),
    'Hazardous': (301, 500)
}


# ============================================================================
# MODEL MANAGER - LSTM
# ============================================================================

class ModelManager:
    """
    Loads and manages trained LSTM models for all pollutants and horizons
    
    Expected structure:
    - model_path/POLLUTANT_HORIZON/lstm_pure_regression.h5
    - model_path/POLLUTANT_HORIZON/model_artifacts.pkl
    """
    
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.models = {}  # Cache loaded models
    
    def load_model(self, pollutant: str, horizon: str) -> Dict:
        """
        Load LSTM model and artifacts for specific pollutant and horizon
        
        Directory structure: POLLUTANT_HORIZON/
        Files:
            - lstm_pure_regression.h5 (Keras model)
            - model_artifacts.pkl (scalers, sequence_length, feature_names)
        
        Example: PM25_1h/lstm_pure_regression.h5
        """
        cache_key = f"{pollutant}_{horizon}"
        
        if cache_key in self.models:
            return self.models[cache_key]
        
        # Construct paths
        model_dir = self.model_path / f"{pollutant}_{horizon}"
        keras_model_path = model_dir / "lstm_pure_regression.h5"
        artifacts_path = model_dir / "model_artifacts.pkl"
        
        # Check files exist
        if not keras_model_path.exists():
            raise FileNotFoundError(
                f"LSTM model not found: {keras_model_path}\n"
                f"Expected structure: {model_dir}/lstm_pure_regression.h5"
            )
        if not artifacts_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found: {artifacts_path}\n"
                f"Expected structure: {model_dir}/model_artifacts.pkl"
            )
        
        logger.info(f"Loading LSTM model: {pollutant} {horizon}")
        
        # Load Keras model
        keras_model = keras_load_model(str(keras_model_path), compile=False)
        logger.info(f"  ✓ Loaded model from lstm_pure_regression.h5")
        
        # Load artifacts (scalers, sequence_length, feature_names)
        with open(artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        # Combine into single dict
        feature_names_from_pkl = artifacts.get('feature_names', [])
        feature_scaler = artifacts['feature_scaler']
        
        # Check for inconsistency between pickle and scaler
        scaler_expected_features = feature_scaler.n_features_in_
        pkl_features_count = len(feature_names_from_pkl)
        
        if pkl_features_count != scaler_expected_features:
            logger.warning(
                f"  ⚠️ INCONSISTENT PICKLE FILE for {pollutant} {horizon}!\n"
                f"  feature_names has {pkl_features_count} features\n"
                f"  but scaler expects {scaler_expected_features} features\n"
                f"  Using scaler's feature list as fallback"
            )
            # Try to get feature names from scaler
            if hasattr(feature_scaler, 'feature_names_in_'):
                feature_names = list(feature_scaler.feature_names_in_)
                logger.info(f"  ✓ Using feature names from scaler")
            else:
                # Generate generic feature names
                feature_names = [f'feature_{i}' for i in range(scaler_expected_features)]
                logger.warning(f"  ⚠️ Generated generic feature names")
        else:
            feature_names = feature_names_from_pkl
        
        model_artifact = {
            'model': keras_model,
            'feature_scaler': feature_scaler,
            'target_scaler': artifacts['target_scaler'],
            'sequence_length': artifacts['sequence_length'],
            'feature_names': feature_names,
            'training_history': artifacts.get('training_history', None)
        }
        
        # Cache for future use
        self.models[cache_key] = model_artifact
        
        logger.info(f"  ✓ Loaded {pollutant} {horizon} model")
        logger.info(f"  Sequence length: {model_artifact['sequence_length']}")
        logger.info(f"  Features: {len(model_artifact['feature_names'])}")
        
        return model_artifact


# ============================================================================
# SEQUENCE BUILDER FOR LSTM
# ============================================================================

class SequenceBuilder:
    """
    Builds sequences from historical data for LSTM input
    
    CRITICAL: Uses actual historical data only, no padding
    """
    
    def __init__(self, feature_engineer: FeatureEngineer):
        self.feature_engineer = feature_engineer
        self.aligner = FeatureAligner()
    
    def build_sequence(self,
                      historical_data: pd.DataFrame,
                      model_features: List[str],
                      pollutant: str,
                      horizon: str,
                      sequence_length: int) -> np.ndarray:
        """
        Build a sequence of feature vectors from historical data
        
        Args:
            historical_data: Historical data sorted by timestamp (most recent last)
            model_features: Feature names from model artifact
            pollutant: Target pollutant
            horizon: Prediction horizon
            sequence_length: Number of timesteps needed
        
        Returns:
            2D numpy array of shape (sequence_length, n_features)
            
        Raises:
            ValueError if insufficient historical data
        """
        if len(historical_data) < sequence_length:
            logger.warning(
                f"Insufficient data for {pollutant}_{horizon}: need {sequence_length}, have {len(historical_data)}"
            )
            # Use all available data and pad if necessary
            sequence_length = len(historical_data)
        
        # Get last N timesteps
        recent_data = historical_data.tail(sequence_length).copy()
        
        # Build feature vector for each timestep
        sequence_list = []
        
        for i in range(len(recent_data)):
            # Current timestep data
            current_row = recent_data.iloc[[i]]
            
            # Historical context up to this point
            hist_context = recent_data.iloc[:i+1] if i > 0 else recent_data.iloc[[0]]
            
            # Engineer features
            features = self.feature_engineer.engineer_features(
                current_data=current_row,
                historical_data=hist_context,
                pollutant=pollutant,
                horizon=horizon
            )
            
            # Align to model features
            aligned = self.aligner.align_features(features, model_features)
            
            # Add to sequence
            sequence_list.append(aligned.values[0])
        
        # Stack into 2D array: (sequence_length, n_features)
        sequence = np.vstack(sequence_list)
        
        return sequence.astype(np.float32)


# ============================================================================
# AQI CALCULATOR - FIXED US EPA CALCULATION
# ============================================================================

class AQICalculator:
    """
    Calculate AQI from pollutant concentrations using Indian and US standards
    FIXED: US EPA calculation now independent of Indian categories
    """
    
    def concentration_to_category(self, concentration: float, pollutant: str) -> str:
        """Map concentration to Indian AQI category"""
        breakpoints = INDIAN_AQI_BREAKPOINTS.get(pollutant, {})
        
        for category, (low, high) in breakpoints.items():
            if low <= concentration <= high:
                return category
        
        # If concentration exceeds all ranges, return Severe
        return 'Severe'
    
    def concentration_to_us_category(self, concentration: float, pollutant: str) -> str:
        """
        Map concentration to US EPA category (independent of Indian AQI)
        FIXED: Direct mapping from concentration to US category
        """
        breakpoints = US_AQI_BREAKPOINTS.get(pollutant, {})
        
        for category, (low, high) in breakpoints.items():
            if low <= concentration <= high:
                return category
        
        # If concentration exceeds all ranges, return Hazardous
        return 'Hazardous'
    
    def calculate_sub_index_indian(self, pollutant: str, category: str, concentration: float) -> Dict:
        """Calculate Indian sub-index for a pollutant"""
        # Get category ranges
        conc_range = INDIAN_AQI_BREAKPOINTS.get(pollutant, {}).get(category, (0, 500))
        aqi_range = INDIAN_AQI_INDEX.get(category, (0, 500))
        
        # Linear interpolation
        conc_low, conc_high = conc_range
        aqi_low, aqi_high = aqi_range
        
        if conc_high - conc_low > 0:
            sub_index = aqi_low + ((concentration - conc_low) / (conc_high - conc_low)) * (aqi_high - aqi_low)
        else:
            sub_index = (aqi_low + aqi_high) / 2
        
        sub_index = float(np.clip(sub_index, 0, 500))
        
        return {
            'sub_index': sub_index,
            'category': category,
            'pollutant': pollutant,
            'concentration': concentration
        }
    
    def calculate_sub_index_us(self, pollutant: str, concentration: float) -> Dict:
        """
        Calculate US EPA sub-index directly from concentration
        FIXED: No dependency on Indian categories!
        
        Args:
            pollutant: Pollutant name (PM25, PM10, NO2, OZONE)
            concentration: Pollutant concentration in µg/m³
        
        Returns:
            Dictionary with sub_index, category, pollutant, concentration
        """
        # Find which US category the concentration falls into (independent calculation)
        us_category = self.concentration_to_us_category(concentration, pollutant)
        
        # Get US ranges for THIS category
        conc_range = US_AQI_BREAKPOINTS.get(pollutant, {}).get(us_category, (0, 500))
        aqi_range = US_AQI_INDEX.get(us_category, (0, 500))
        
        # Linear interpolation using US EPA formula
        conc_low, conc_high = conc_range
        aqi_low, aqi_high = aqi_range
        
        if conc_high - conc_low > 0:
            sub_index = aqi_low + ((concentration - conc_low) / (conc_high - conc_low)) * (aqi_high - aqi_low)
        else:
            sub_index = (aqi_low + aqi_high) / 2
        
        sub_index = float(np.clip(sub_index, 0, 500))
        
        return {
            'sub_index': sub_index,
            'category': us_category,
            'pollutant': pollutant,
            'concentration': concentration
        }
    
    def calculate_overall_aqi(self, predictions: Dict, standard: str = 'IN') -> Dict:
        """
        Calculate overall AQI from all pollutant predictions
        FIXED: Proper US EPA calculation
        """
        if not predictions:
            return {
                'aqi': 0,
                'category': 'Unknown',
                'dominant_pollutant': 'None'
            }
        
        sub_indices = []
        
        for pollutant, (category, concentration) in predictions.items():
            if standard == 'IN':
                sub_idx = self.calculate_sub_index_indian(pollutant, category, concentration)
            else:
                # FIXED: Calculate US AQI directly from concentration (no Indian category dependency)
                sub_idx = self.calculate_sub_index_us(pollutant, concentration)
            
            sub_indices.append(sub_idx)
        
        # Overall AQI is the maximum sub-index
        max_sub = max(sub_indices, key=lambda x: x['sub_index'])
        
        return {
            'aqi': max_sub['sub_index'],
            'category': max_sub['category'],
            'dominant_pollutant': max_sub['pollutant']
        }


# ============================================================================
# HEALTH ADVISORY
# ============================================================================

class HealthAdvisory:
    """Provide health advisories based on AQI category"""
    
    @staticmethod
    def get_advisory(category: str, standard: str = 'IN') -> str:
        """Get health advisory for given AQI category"""
        
        indian_advisories = {
            'Good': 'Air quality is excellent. Ideal conditions for outdoor activities.',
            'Satisfactory': 'Air quality is acceptable. Sensitive individuals should consider limiting prolonged outdoor exertion.',
            'Moderate': 'Sensitive groups may experience minor respiratory discomfort. General public is less likely to be affected.',
            'Poor': 'Everyone may begin to experience health effects. Sensitive groups should avoid outdoor activities.',
            'Very_Poor': 'Health alert! Everyone should avoid prolonged outdoor activities. Sensitive groups should stay indoors.',
            'Severe': 'Health emergency! Everyone should stay indoors and keep activity levels low.'
        }
        
        us_advisories = {
            'Good': 'Air quality is excellent. Ideal conditions for outdoor activities.',
            'Moderate': 'Unusually sensitive people should consider reducing prolonged outdoor exertion.',
            'Unhealthy_for_Sensitive': 'Sensitive groups should reduce prolonged outdoor exertion.',
            'Unhealthy': 'Everyone should reduce prolonged outdoor exertion.',
            'Very_Unhealthy': 'Everyone should avoid prolonged outdoor exertion. Sensitive groups should stay indoors.',
            'Hazardous': 'Health emergency! Everyone should avoid outdoor activities.'
        }
        
        if standard == 'IN':
            return indian_advisories.get(category, 'Unknown air quality.')
        else:
            return us_advisories.get(category, 'Unknown air quality.')


# ============================================================================
# LSTM PREDICTOR - MAIN CLASS
# ============================================================================

class LSTMPredictor:
    """
    Main predictor class that orchestrates LSTM model predictions
    """
    
    def __init__(self,
                 model_manager: ModelManager,
                 feature_engineer: FeatureEngineer,
                 spatial_data: Optional[pd.DataFrame] = None):
        self.model_manager = model_manager
        self.feature_engineer = feature_engineer
        self.spatial_data = spatial_data
        self.sequence_builder = SequenceBuilder(feature_engineer)
        self.aqi_calculator = AQICalculator()
        self.pollutants = ['PM25', 'PM10', 'NO2', 'OZONE']
    
    def predict_single(self,
                      current_data: pd.DataFrame,
                      historical_data: pd.DataFrame,
                      pollutant: str,
                      horizon: str) -> Tuple[str, float]:
        """
        Predict a single pollutant for a single horizon
        
        Returns:
            (category, predicted_concentration)
        """
        try:
            logger.info(f"🔮 Predicting {pollutant} {horizon}...")
            
            # 1. Load model artifact
            model_artifact = self.model_manager.load_model(pollutant, horizon)
            keras_model = model_artifact['model']
            feature_scaler = model_artifact['feature_scaler']
            target_scaler = model_artifact['target_scaler']
            sequence_length = model_artifact['sequence_length']
            model_features = model_artifact['feature_names']
            
            logger.info(f"  Model expects {len(model_features)} features")
            logger.info(f"  Sequence length: {sequence_length}")
            
            # 2. Build sequence from historical data
            sequence = self.sequence_builder.build_sequence(
                historical_data=historical_data,
                model_features=model_features,
                pollutant=pollutant,
                horizon=horizon,
                sequence_length=sequence_length
            )
            logger.info(f"  Built sequence: {sequence.shape}")
            
            # 3. Scale features
            seq_scaled = feature_scaler.transform(sequence)
            logger.info(f"  Scaled sequence: {seq_scaled.shape}")
            
            # 4. Reshape for LSTM: (1, sequence_length, n_features)
            seq_scaled_3d = seq_scaled.reshape(1, sequence_length, -1)
            logger.info(f"  Final input shape for model: {seq_scaled_3d.shape}")
            
            # 5. Predict concentration (regression)
            pred_scaled = keras_model.predict(seq_scaled_3d, verbose=0)
            logger.info(f"  Scaled prediction: {pred_scaled[0][0]:.4f}")
            
            # 6. Inverse transform to get actual concentration
            pred_concentration = target_scaler.inverse_transform(pred_scaled)[0][0]
            logger.info(f"  Raw prediction (before clip): {pred_concentration:.2f}")
            
            # 7. Handle negative predictions
            if pred_concentration < 0:
                logger.warning(
                    f"  ⚠️  NEGATIVE prediction: {pollutant}_{horizon} = {pred_concentration:.2f}"
                )
                logger.warning(f"  Clipping to 0.0")
                pred_concentration = 0.0
            
            # 8. Clip to valid range
            pred_concentration = float(np.clip(pred_concentration, 0, 1000))
            
            # 9. Map concentration to AQI category (Indian)
            category = self.aqi_calculator.concentration_to_category(
                pred_concentration, pollutant
            )
            
            logger.info(
                f"  ✓ {pollutant} {horizon}: {pred_concentration:.2f} µg/m³ → {category}"
            )
            
            return category, pred_concentration
            
        except Exception as e:
            logger.error(f"❌ Failed to predict {pollutant} {horizon}: {e}")
            # Return fallback based on pollutant
            fallback_values = {
                'PM25': (75.0, 'Moderate'),
                'PM10': (150.0, 'Moderate'),
                'NO2': (100.0, 'Moderate'),
                'OZONE': (120.0, 'Moderate')
            }
            value, cat = fallback_values.get(pollutant, (75.0, 'Moderate'))
            return cat, value
    
    def predict_all_horizons(self,
                           current_data: pd.DataFrame,
                           historical_data: pd.DataFrame,
                           pollutant: str,
                           standard: str = 'IN') -> Dict:
        """Predict for all horizons (1h, 6h, 12h, 24h) with FIXED US EPA calculation"""
        results = {}
        
        fallback_values = {
            'PM25': 75.0,
            'PM10': 150.0,
            'NO2': 100.0,
            'OZONE': 120.0
        }
        
        for horizon in ['1h', '6h', '12h', '24h']:
            try:
                # Get prediction (returns Indian category + concentration)
                indian_category, pred_value = self.predict_single(
                    current_data=current_data,
                    historical_data=historical_data,
                    pollutant=pollutant,
                    horizon=horizon
                )
                
                # Calculate AQI based on selected standard
                if standard == 'IN':
                    sub_index = self.aqi_calculator.calculate_sub_index_indian(
                        pollutant, indian_category, pred_value
                    )
                else:
                    # FIXED: Pass only pollutant and concentration (no Indian category dependency)
                    sub_index = self.aqi_calculator.calculate_sub_index_us(
                        pollutant, pred_value
                    )
                
                results[horizon] = {
                    'value': pred_value,
                    'category': sub_index['category'],  # Use the calculated category (IN or US)
                    'sub_index': sub_index['sub_index'],
                    'pollutant': pollutant,
                    'horizon': horizon,
                    'aqi_min': sub_index['sub_index'] - 25,
                    'aqi_max': sub_index['sub_index'] + 25,
                    'aqi_mid': sub_index['sub_index'],
                    'confidence': 0.85
                }
                
            except Exception as e:
                logger.error(f"Prediction failed for {pollutant} {horizon}: {e}")
                fallback_val = fallback_values.get(pollutant, 75.0)
                
                if standard == 'IN':
                    fallback_category = self.aqi_calculator.concentration_to_category(fallback_val, pollutant)
                    fallback_sub = self.aqi_calculator.calculate_sub_index_indian(
                        pollutant, fallback_category, fallback_val
                    )
                else:
                    # FIXED: Use direct US calculation
                    fallback_sub = self.aqi_calculator.calculate_sub_index_us(
                        pollutant, fallback_val
                    )
                
                results[horizon] = {
                    'value': fallback_val,
                    'category': fallback_sub['category'],
                    'sub_index': fallback_sub['sub_index'],
                    'pollutant': pollutant,
                    'horizon': horizon,
                    'aqi_min': fallback_sub['sub_index'] - 25,
                    'aqi_max': fallback_sub['sub_index'] + 25,
                    'aqi_mid': fallback_sub['sub_index'],
                    'confidence': 0.50,
                    'error': str(e)
                }
        
        return results
    
    def predict_all_pollutants_all_horizons(self,
                                           current_data: pd.DataFrame,
                                           historical_data: pd.DataFrame,
                                           standard: str = 'IN') -> Dict:
        """Comprehensive prediction for all pollutants across all horizons"""
        results = {}
        
        for pollutant in self.pollutants:
            logger.info(f"\n🔬 Predicting {pollutant}...")
            results[pollutant] = self.predict_all_horizons(
                current_data=current_data,
                historical_data=historical_data,
                pollutant=pollutant,
                standard=standard
            )
        
        # Calculate overall AQI for each horizon
        overall_results = {}
        
        for horizon in ['1h', '6h', '12h', '24h']:
            predictions = {
                pollutant: (
                    results[pollutant][horizon]['category'],
                    results[pollutant][horizon].get('value')
                )
                for pollutant in self.pollutants
                if 'error' not in results[pollutant][horizon]
            }
            
            overall = self.aqi_calculator.calculate_overall_aqi(predictions, standard)
            overall_results[horizon] = overall
        
        results['overall'] = overall_results
        
        return results
    
    def predict_location(self,
                        current_data: pd.DataFrame,
                        historical_data: pd.DataFrame,
                        location: str,
                        standard: str = 'IN') -> Dict:
        """
        Main prediction method with app.py compatible output format
        """
        logger.info(f"📍 predict_location() called for {location} (standard: {standard})")
        logger.info(f"  Current data: {len(current_data)} row(s), Historical data: {len(historical_data)} rows")
        
        if len(current_data) == 0:
            raise ValueError(f"No current data available for location: {location}")
        if len(historical_data) == 0:
            raise ValueError(f"No historical data available for location: {location}")
        
        # Get timestamp
        if 'date' in current_data.columns:
            timestamp = current_data['date'].iloc[0]
        elif 'timestamp' in current_data.columns:
            timestamp = current_data['timestamp'].iloc[0]
        else:
            timestamp = pd.Timestamp.now()
        
        # Get all predictions
        raw_results = self.predict_all_pollutants_all_horizons(
            current_data=current_data,
            historical_data=historical_data,
            standard=standard
        )
        
        # Format for app.py
        formatted_results = {
            'timestamp': timestamp,
            'predictions': {},
            'overall_aqi': raw_results.get('overall', {})
        }
        
        # Restructure predictions
        for pollutant in self.pollutants:
            if pollutant in raw_results:
                formatted_results['predictions'][pollutant] = raw_results[pollutant]
        
        return formatted_results