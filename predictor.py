"""
Complete LSTM Model Predictor with Comprehensive AQI Calculator
FIXED VERSION: Proper feature alignment, no padding, better error handling
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
        'Good': (0.0, 12.0), 'Moderate': (12.1, 35.4), 'Unhealthy_for_Sensitive': (35.5, 55.4),
        'Unhealthy': (55.5, 150.4), 'Very_Unhealthy': (150.5, 250.4), 'Hazardous': (250.5, 500.4)
    },
    'PM10': {
        'Good': (0, 54), 'Moderate': (55, 154), 'Unhealthy_for_Sensitive': (155, 254),
        'Unhealthy': (255, 354), 'Very_Unhealthy': (355, 424), 'Hazardous': (425, 604)
    },
    'NO2': {
        'Good': (0, 53), 'Moderate': (54, 100), 'Unhealthy_for_Sensitive': (101, 360),
        'Unhealthy': (361, 649), 'Very_Unhealthy': (650, 1249), 'Hazardous': (1250, 2049)
    },
    'OZONE': {
        'Good': (0.000, 0.054), 'Moderate': (0.055, 0.070), 'Unhealthy_for_Sensitive': (0.071, 0.085),
        'Unhealthy': (0.086, 0.105), 'Very_Unhealthy': (0.106, 0.200), 'Hazardous': (0.201, 0.604)
    }
}

US_AQI_INDEX = {
    'Good': (0, 50), 'Moderate': (51, 100), 'Unhealthy_for_Sensitive': (101, 150),
    'Unhealthy': (151, 200), 'Very_Unhealthy': (201, 300), 'Hazardous': (301, 500)
}

CATEGORY_MAPPING = {
    'Good': {'us': 'Good'},
    'Satisfactory': {'us': 'Moderate'},
    'Moderate': {'us': 'Unhealthy_for_Sensitive'},
    'Poor': {'us': 'Unhealthy'},
    'Very_Poor': {'us': 'Very_Unhealthy'},
    'Severe': {'us': 'Hazardous'}
}


# ============================================================================
# MODEL MANAGER - UPDATED FOR LSTM
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
        
        # ⚠️ CRITICAL FIX: Check for inconsistency between pickle and scaler
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
# SEQUENCE BUILDER FOR LSTM - FIXED VERSION
# ============================================================================

class SequenceBuilder:
    """
    Builds sequences from historical data for LSTM input
    LSTM requires 3D input: (batch_size, sequence_length, n_features)
    
    FIXED: Proper feature alignment with NO padding
    """
    
    def create_sequence(self,
                       current_features: pd.DataFrame,
                       historical_data: pd.DataFrame,
                       feature_engineer: FeatureEngineer,
                       feature_aligner: FeatureAligner,
                       pollutant: str,
                       horizon: str,
                       feature_names: List[str],
                       sequence_length: int) -> Tuple[np.ndarray, List[pd.DataFrame]]:
        """
        Create a sequence for LSTM prediction with PROPER FEATURE ALIGNMENT
        
        Args:
            current_features: Engineered features for current timestep (1 row)
            historical_data: Historical data (sorted by timestamp)
            feature_engineer: Feature engineering instance
            feature_aligner: Feature alignment instance
            pollutant: Target pollutant
            horizon: Prediction horizon
            feature_names: Required feature names from model
            sequence_length: Number of timesteps needed
            
        Returns:
            3D array: (1, sequence_length, n_features)
            List of engineered DataFrames for debugging
        """
        sequence_list = []
        engineered_frames = []
        
        logger.info(f"Engineering features for {pollutant} {horizon}...")
        
        # Get enough historical rows (we need sequence_length timesteps)
        if len(historical_data) < sequence_length:
            logger.warning(
                f"Insufficient history: {len(historical_data)} rows, need {sequence_length}. "
                "Will pad with zeros for missing timesteps."
            )
        
        # Create features for each timestep in the sequence
        for i in range(sequence_length - 1, -1, -1):  # Go backwards from current
            if i == 0:
                # Current timestep (already engineered)
                timestep_features = current_features
            else:
                # Historical timestep
                if len(historical_data) >= i:
                    # Get data up to this timestep
                    current_hist = historical_data.iloc[-i]
                    hist_before = historical_data.iloc[:-i] if len(historical_data) > i else pd.DataFrame()
                    
                    # Engineer features for this timestep
                    timestep_features = feature_engineer.engineer_features(
                        current_data=pd.DataFrame([current_hist]),
                        historical_data=hist_before,
                        pollutant=pollutant,
                        horizon=horizon
                    )
                else:
                    # Not enough history, create zero features
                    timestep_features = pd.DataFrame(
                        np.zeros((1, len(feature_names))),
                        columns=feature_names
                    )
            
            # ⚠️ CRITICAL FIX: Use FeatureAligner instead of reindex
            # This ensures EXACT feature matching without padding
            aligned = feature_aligner.align_features(
                timestep_features,
                feature_names
            )
            
            sequence_list.append(aligned.values[0])
            engineered_frames.append(aligned)
        
        # Stack into 3D array: (sequence_length, n_features)
        sequence = np.array(sequence_list, dtype=np.float32)
        
        # Add batch dimension: (1, sequence_length, n_features)
        sequence = np.expand_dims(sequence, axis=0)
        
        logger.info(f"  Created features: {sequence.shape[2]}")
        logger.info(f"  Expected features: {len(feature_names)}")
        
        # Verify shape matches expectations
        if sequence.shape[2] != len(feature_names):
            logger.error(
                f"  ❌ Feature count mismatch! "
                f"Created {sequence.shape[2]}, expected {len(feature_names)}"
            )
            raise ValueError(
                f"Feature alignment failed: got {sequence.shape[2]} features, "
                f"expected {len(feature_names)}"
            )
        
        logger.info(f"  Sequence shape: {sequence.shape}")
        
        return sequence, engineered_frames


# ============================================================================
# AQI CALCULATOR
# ============================================================================

class AQICalculator:
    """Calculate detailed AQI with ranges for both Indian and US standards"""
    
    def __init__(self):
        self.indian_breakpoints = INDIAN_AQI_BREAKPOINTS
        self.us_breakpoints = US_AQI_BREAKPOINTS
        self.indian_index = INDIAN_AQI_INDEX
        self.us_index = US_AQI_INDEX
    
    def concentration_to_category(self, concentration: float, pollutant: str) -> str:
        """
        Map predicted concentration to AQI category
        
        Args:
            concentration: Predicted pollutant concentration (µg/m³)
            pollutant: Pollutant name (PM25, PM10, NO2, OZONE)
            
        Returns:
            Category name (Good, Satisfactory, Moderate, Poor, Very_Poor, Severe)
        """
        breakpoints = self.indian_breakpoints.get(pollutant, {})
        
        for category, (low, high) in breakpoints.items():
            if low <= concentration <= high:
                return category
        
        # If concentration exceeds all breakpoints, return Severe
        return 'Severe'
    
    def calculate_sub_index_indian(self, 
                                   pollutant: str, 
                                   category: str,
                                   concentration: float) -> Dict:
        """Calculate sub-index for Indian AQI standard"""
        breakpoint_range = self.indian_breakpoints[pollutant][category]
        index_range = self.indian_index[category]
        
        # Linear interpolation
        conc_low, conc_high = breakpoint_range
        index_low, index_high = index_range
        
        sub_index = index_low + (
            (concentration - conc_low) / (conc_high - conc_low)
        ) * (index_high - index_low)
        
        return {
            'aqi_range': index_range,
            'aqi_min': index_low,
            'aqi_max': index_high,
            'aqi_mid': (index_low + index_high) / 2,
            'concentration_range': breakpoint_range,
            'sub_index': sub_index
        }
    
    def calculate_sub_index_us(self,
                               pollutant: str,
                               category: str,
                               concentration: float) -> Dict:
        """Calculate sub-index for US AQI standard"""
        breakpoint_range = self.us_breakpoints[pollutant][category]
        index_range = self.us_index[category]
        
        conc_low, conc_high = breakpoint_range
        index_low, index_high = index_range
        
        sub_index = index_low + (
            (concentration - conc_low) / (conc_high - conc_low)
        ) * (index_high - index_low)
        
        return {
            'aqi_range': index_range,
            'aqi_min': index_low,
            'aqi_max': index_high,
            'aqi_mid': (index_low + index_high) / 2,
            'concentration_range': breakpoint_range,
            'sub_index': sub_index
        }
    
    def calculate_overall_aqi(self,
                             pollutant_predictions: Dict[str, Tuple[str, float]],
                             standard: str = 'IN') -> Dict:
        """
        Calculate overall AQI from multiple pollutants
        
        Args:
            pollutant_predictions: {pollutant: (category, concentration)}
            standard: 'IN' or 'US'
        
        Returns:
            {
                'aqi': overall_aqi,
                'category': dominant_category,
                'dominant_pollutant': pollutant_name,
                'sub_indices': {pollutant: sub_index}
            }
        """
        sub_indices = {}
        
        for pollutant, (category, concentration) in pollutant_predictions.items():
            if standard == 'IN':
                result = self.calculate_sub_index_indian(pollutant, category, concentration)
            else:
                result = self.calculate_sub_index_us(pollutant, category, concentration)
            
            sub_indices[pollutant] = result['sub_index']
        
        # Overall AQI is the maximum sub-index
        dominant_pollutant = max(sub_indices, key=sub_indices.get)
        overall_aqi = sub_indices[dominant_pollutant]
        
        # Get category for overall AQI
        if standard == 'IN':
            index_ranges = self.indian_index
        else:
            index_ranges = self.us_index
        
        overall_category = 'Unknown'
        for category, (low, high) in index_ranges.items():
            if low <= overall_aqi <= high:
                overall_category = category
                break
        
        return {
            'aqi': round(overall_aqi, 2),
            'category': overall_category,
            'dominant_pollutant': dominant_pollutant,
            'sub_indices': sub_indices
        }
    
    def get_health_implications(self, category: str, standard: str = 'IN') -> str:
        """Get health implications for a category"""
        # Normalize category name if needed
        if standard == 'US' and category in CATEGORY_MAPPING:
            normalized_category = CATEGORY_MAPPING[category]['us']
        else:
            normalized_category = category
        
        if standard == 'IN':
            implications = {
                'Good': 'Minimal Impact',
                'Satisfactory': 'Minor breathing discomfort to sensitive people',
                'Moderate': 'Breathing discomfort to people with lung, asthma and heart diseases',
                'Poor': 'Breathing discomfort to most people on prolonged exposure',
                'Very_Poor': 'Respiratory illness on prolonged exposure',
                'Severe': 'Affects healthy people and seriously impacts those with existing conditions'
            }
        else:
            implications = {
                'Good': 'Air quality is satisfactory, and air pollution poses little or no risk',
                'Moderate': 'Air quality is acceptable. However, there may be a risk for some people',
                'Unhealthy_for_Sensitive': 'Members of sensitive groups may experience health effects',
                'Unhealthy': 'Some members of the general public may experience health effects',
                'Very_Unhealthy': 'Health alert: The risk of health effects is increased for everyone',
                'Hazardous': 'Health warning of emergency conditions: everyone is more likely to be affected'
            }
        
        return implications.get(normalized_category, 'Unknown')


# ============================================================================
# HEALTH ADVISORY
# ============================================================================

class HealthAdvisory:
    """Provides health advisory based on AQI category"""
    
    @staticmethod
    def get_advisory(category: str, standard: str = 'IN') -> Dict:
        """
        Get health advisory for a given AQI category
        
        Args:
            category: AQI category (Good, Moderate, etc.)
            standard: 'IN' or 'US'
        
        Returns:
            Dictionary with advisory information
        """
        if standard == 'IN':
            advisories = {
                'Good': {
                    'general': 'Air quality is satisfactory',
                    'sensitive': 'No health impacts',
                    'actions': ['Enjoy outdoor activities']
                },
                'Satisfactory': {
                    'general': 'Air quality is acceptable',
                    'sensitive': 'Minor breathing discomfort for sensitive people',
                    'actions': ['Normal outdoor activities', 'Sensitive groups should watch for symptoms']
                },
                'Moderate': {
                    'general': 'May cause breathing discomfort',
                    'sensitive': 'People with lung, asthma and heart diseases should avoid prolonged exposure',
                    'actions': ['Reduce prolonged outdoor exertion', 'Close windows to avoid dirty outdoor air']
                },
                'Poor': {
                    'general': 'Breathing discomfort to most people',
                    'sensitive': 'Significantly affects people with respiratory diseases',
                    'actions': ['Avoid prolonged outdoor activities', 'Use masks when going outdoors', 'Use air purifiers']
                },
                'Very_Poor': {
                    'general': 'Respiratory illness on prolonged exposure',
                    'sensitive': 'Affects even healthy people',
                    'actions': ['Stay indoors as much as possible', 'Use air purifiers', 'Avoid physical activities outdoors']
                },
                'Severe': {
                    'general': 'Health warning of emergency conditions',
                    'sensitive': 'Everyone is more likely to be affected',
                    'actions': ['Emergency - Stay indoors', 'Seek medical attention if experiencing symptoms', 'Use air purifiers on high']
                }
            }
        else:  # US standard
            advisories = {
                'Good': {
                    'general': 'Air quality is satisfactory',
                    'sensitive': 'No health impacts',
                    'actions': ['Enjoy outdoor activities']
                },
                'Moderate': {
                    'general': 'Air quality is acceptable for most',
                    'sensitive': 'Unusually sensitive people should consider limiting prolonged outdoor exertion',
                    'actions': ['Unusually sensitive people should watch for symptoms']
                },
                'Unhealthy_for_Sensitive': {
                    'general': 'General public not likely to be affected',
                    'sensitive': 'Members of sensitive groups may experience health effects',
                    'actions': ['Sensitive groups should limit prolonged outdoor exertion']
                },
                'Unhealthy': {
                    'general': 'Everyone may begin to experience health effects',
                    'sensitive': 'Members of sensitive groups may experience more serious health effects',
                    'actions': ['Everyone should limit prolonged outdoor exertion', 'Sensitive groups should avoid prolonged outdoor exertion']
                },
                'Very_Unhealthy': {
                    'general': 'Health alert: The risk of health effects is increased for everyone',
                    'sensitive': 'Everyone should avoid prolonged outdoor exertion',
                    'actions': ['Everyone should avoid prolonged outdoor exertion', 'Sensitive groups should avoid all outdoor exertion']
                },
                'Hazardous': {
                    'general': 'Health warning of emergency conditions',
                    'sensitive': 'Everyone is more likely to be affected',
                    'actions': ['Everyone should avoid all outdoor exertion', 'Stay indoors', 'Seek medical attention if experiencing symptoms']
                }
            }
        
        return advisories.get(category, advisories['Good'])


# ============================================================================
# POLLUTANT PREDICTOR - FIXED VERSION
# ============================================================================

class LSTMPredictor:
    """Complete LSTM prediction pipeline for all pollutants - FIXED VERSION"""
    
    def __init__(self, model_manager=None, model_path=None, feature_engineer=None, spatial_data=None):
        """
        Initialize predictor
        
        Args:
            model_manager: ModelManager instance (or None to create from model_path)
            model_path: Path to models directory (used if model_manager is None)
            feature_engineer: FeatureEngineer instance (or None to create new)
            spatial_data: Spatial data DataFrame (optional)
        """
        # Handle model manager
        if model_manager is not None:
            self.model_manager = model_manager
        elif model_path is not None:
            self.model_manager = ModelManager(model_path)
        else:
            raise ValueError("Either model_manager or model_path must be provided")
        
        # Handle feature engineer
        if feature_engineer is not None:
            self.feature_engineer = feature_engineer
        else:
            self.feature_engineer = FeatureEngineer()
        
        # Store spatial data (optional)
        self.spatial_data = spatial_data
        
        # Create other components
        self.feature_aligner = FeatureAligner()
        self.aqi_calculator = AQICalculator()
        self.sequence_builder = SequenceBuilder()
        
        self.pollutants = ['PM25', 'PM10', 'NO2', 'OZONE']
    
    def predict_single(self,
                      current_data: pd.DataFrame,
                      historical_data: pd.DataFrame,
                      pollutant: str,
                      horizon: str) -> Tuple[str, float]:
        """
        Predict using LSTM regression model - FIXED VERSION
        
        Args:
            current_data: Current timestamp data (single row)
            historical_data: Historical data sorted by timestamp
            pollutant: Target pollutant
            horizon: Prediction horizon (1h, 6h, 12h, 24h)
            
        Returns:
            (category: str, predicted_value: float)
        """
        try:
            # 1. Load LSTM model and artifacts
            model_artifact = self.model_manager.load_model(pollutant, horizon)
            keras_model = model_artifact['model']
            feature_scaler = model_artifact['feature_scaler']
            target_scaler = model_artifact['target_scaler']
            sequence_length = model_artifact['sequence_length']
            feature_names = model_artifact['feature_names']
            
            # 2. Engineer features for current timestep
            current_features = self.feature_engineer.engineer_features(
                current_data=current_data,
                historical_data=historical_data,
                pollutant=pollutant,
                horizon=horizon
            )
            
            # 3. Build sequence from current + historical with PROPER ALIGNMENT
            sequence, _ = self.sequence_builder.create_sequence(
                current_features=current_features,
                historical_data=historical_data,
                feature_engineer=self.feature_engineer,
                feature_aligner=self.feature_aligner,  # ⚠️ ADDED
                pollutant=pollutant,
                horizon=horizon,
                feature_names=feature_names,
                sequence_length=sequence_length
            )
            
            # 4. Scale features
            # The sequence is already aligned to model features by FeatureAligner
            # Just scale and use directly - NO PADDING NEEDED
            
            seq_2d = sequence[0]  # Remove batch dimension: (sequence_length, n_features)
            
            # Verify shape matches scaler
            expected_features = feature_scaler.n_features_in_
            actual_features = seq_2d.shape[1]
            
            if actual_features != expected_features:
                raise ValueError(
                    f"Feature count mismatch after alignment!\n"
                    f"  FeatureAligner created: {actual_features} features\n"
                    f"  Scaler expects: {expected_features} features\n"
                    f"  This means the feature_names in the pickle file don't match\n"
                    f"  what feature_engineer.py creates. You need to update the pickle files."
                )
            
            # Scale features
            seq_scaled_2d = feature_scaler.transform(seq_2d)
            
            # Reshape back to 3D: (1, sequence_length, n_features)
            seq_scaled_3d = np.expand_dims(seq_scaled_2d, axis=0)
            
            logger.info(f"  Final input shape for model: {seq_scaled_3d.shape}")
            
            # 5. Predict concentration (regression)
            pred_scaled = keras_model.predict(seq_scaled_3d, verbose=0)
            logger.info(f"  Scaled prediction: {pred_scaled[0][0]:.4f}")
            
            # 6. Inverse transform to get actual concentration
            pred_concentration = target_scaler.inverse_transform(pred_scaled)[0][0]
            logger.info(f"  Raw prediction (before clip): {pred_concentration:.2f}")
            
            # 7. Check for negative predictions (indicates model issue)
            if pred_concentration < 0:
                logger.warning(f"  ⚠️  NEGATIVE prediction detected! Model: {pollutant}_{horizon}, Value: {pred_concentration:.2f}")
                logger.warning(f"  This indicates an issue with the model or target scaler for this horizon.")
                logger.error(f"  ❌ CRITICAL: {pollutant}_{horizon} model predicting {pred_concentration:.2f}")
                logger.error(f"  Possible causes:")
                logger.error(f"    1. Target scaler not properly fitted during training")
                logger.error(f"    2. Model not converged during training")
                logger.error(f"    3. Feature distribution mismatch")
                
                # Raise error instead of clipping to 0
                raise ValueError(
                    f"Model {pollutant}_{horizon} is predicting negative values ({pred_concentration:.2f}). "
                    f"This model needs to be retrained with proper target scaling."
                )
            
            # 8. Clip to valid range
            pred_concentration = float(np.clip(pred_concentration, 0, 1000))
            
            # 9. Map concentration to AQI category using breakpoints
            category = self.aqi_calculator.concentration_to_category(
                pred_concentration, pollutant
            )
            
            logger.info(
                f"  ✓ {pollutant} {horizon}: {pred_concentration:.2f} µg/m³ → {category}"
            )
            
            return category, pred_concentration
            
        except Exception as e:
            logger.error(f"❌ Failed to predict {pollutant} {horizon}: {e}")
            logger.error(f"   This model may need to be retrained or checked.")
            raise
    
    def predict_all_horizons(self,
                           current_data: pd.DataFrame,
                           historical_data: pd.DataFrame,
                           pollutant: str,
                           standard: str = 'IN') -> Dict:
        """Predict for all horizons (1h, 6h, 12h, 24h)"""
        results = {}
        
        for horizon in ['1h', '6h', '12h', '24h']:
            try:
                # Get category and predicted concentration
                category, pred_value = self.predict_single(
                    current_data=current_data,
                    historical_data=historical_data,
                    pollutant=pollutant,
                    horizon=horizon
                )
                
                # Calculate AQI details for this pollutant
                if standard == 'IN':
                    sub_index = self.aqi_calculator.calculate_sub_index_indian(
                        pollutant, category, pred_value
                    )
                else:
                    sub_index = self.aqi_calculator.calculate_sub_index_us(
                        pollutant, category, pred_value
                    )
                
                results[horizon] = {
                    'value': pred_value,
                    'category': category,
                    'sub_index': sub_index['sub_index'],
                    'pollutant': pollutant,
                    'horizon': horizon
                }
                
            except Exception as e:
                logger.error(f"Prediction failed for {pollutant} {horizon}: {e}")
                results[horizon] = {
                    'value': 0.0,
                    'category': 'Unknown',
                    'sub_index': 0.0,
                    'pollutant': pollutant,
                    'horizon': horizon,
                    'error': str(e)
                }
        
        return results
    
    def predict_all_pollutants_all_horizons(self,
                                           current_data: pd.DataFrame,
                                           historical_data: pd.DataFrame,
                                           standard: str = 'IN') -> Dict:
        """Comprehensive prediction for all pollutants across all horizons"""
        results = {}
        
        # Predict each pollutant
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
            # Gather predictions for this horizon
            predictions = {
                pollutant: (
                    results[pollutant][horizon]['category'],
                    results[pollutant][horizon].get('value')
                )
                for pollutant in self.pollutants
                if 'error' not in results[pollutant][horizon]
            }
            
            # Calculate overall AQI
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
        BACKWARD COMPATIBILITY METHOD
        
        This method provides the same interface as the old predict_location()
        for compatibility with existing app.py code.
        
        It wraps predict_all_pollutants_all_horizons() and returns results
        in the expected format that app.py expects:
        - result['timestamp']
        - result['predictions'] (nested by pollutant and horizon)
        - result['overall_aqi'] (nested by horizon)
        
        Args:
            current_data: Current data (1 row)
            historical_data: Historical data for the location
            location: Location name (for logging)
            standard: 'IN' for Indian AQI, 'US' for US AQI
            
        Returns:
            Dictionary with predictions formatted for app.py
        """
        logger.info(f"📍 predict_location() called for {location} (standard: {standard})")
        logger.info(f"  Current data: {len(current_data)} row(s), Historical data: {len(historical_data)} rows")
        
        # Validate data
        if len(current_data) == 0:
            raise ValueError(f"No current data available for location: {location}")
        if len(historical_data) == 0:
            raise ValueError(f"No historical data available for location: {location}")
        
        # Get timestamp from current data
        if 'date' in current_data.columns:
            timestamp = current_data['date'].iloc[0]
        elif 'timestamp' in current_data.columns:
            timestamp = current_data['timestamp'].iloc[0]
        else:
            timestamp = pd.Timestamp.now()
        
        # Call the main prediction method
        raw_results = self.predict_all_pollutants_all_horizons(
            current_data=current_data,
            historical_data=historical_data,
            standard=standard
        )
        
        # Transform to app.py's expected format
        formatted_results = {
            'timestamp': timestamp,
            'predictions': {},
            'overall_aqi': raw_results.get('overall', {})
        }
        
        # Restructure predictions by pollutant
        for pollutant in self.pollutants:
            if pollutant in raw_results:
                formatted_results['predictions'][pollutant] = raw_results[pollutant]
        
        return formatted_results