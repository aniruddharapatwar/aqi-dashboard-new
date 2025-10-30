"""
Complete LSTM Model Predictor with Comprehensive AQI Calculator
Handles LSTM regression models with sequence generation
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
        
        # Load artifacts (scalers, sequence_length, feature_names)
        with open(artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        # Combine into single dict
        model_artifact = {
            'model': keras_model,
            'feature_scaler': artifacts['feature_scaler'],
            'target_scaler': artifacts['target_scaler'],
            'sequence_length': artifacts['sequence_length'],
            'feature_names': artifacts.get('feature_names', []),
            'training_history': artifacts.get('training_history', None)
        }
        
        # Cache for future use
        self.models[cache_key] = model_artifact
        
        logger.info(f"  ✓ Loaded LSTM with {len(model_artifact['feature_names'])} features")
        logger.info(f"  ✓ Sequence length: {model_artifact['sequence_length']}")
        
        return model_artifact


# ============================================================================
# SEQUENCE BUILDER FOR LSTM
# ============================================================================

class SequenceBuilder:
    """
    Builds sequences from historical data for LSTM input
    LSTM requires 3D input: (batch_size, sequence_length, n_features)
    """
    
    def create_sequence(self,
                       current_features: pd.DataFrame,
                       historical_data: pd.DataFrame,
                       feature_engineer: FeatureEngineer,
                       pollutant: str,
                       horizon: str,
                       feature_names: List[str],
                       sequence_length: int) -> Tuple[np.ndarray, List[pd.DataFrame]]:
        """
        Create a sequence for LSTM prediction
        
        Args:
            current_features: Engineered features for current timestep (1 row)
            historical_data: Historical data (sorted by timestamp)
            feature_engineer: Feature engineering instance
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
        
        # Get enough historical rows (we need sequence_length timesteps)
        if len(historical_data) < sequence_length:
            logger.warning(
                f"Insufficient history: {len(historical_data)} rows, need {sequence_length}. "
                "Will pad with zeros."
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
            
            # Align to model features
            aligned = timestep_features.reindex(
                columns=feature_names,
                fill_value=0.0
            ).astype(np.float32)
            
            sequence_list.append(aligned.values[0])
            engineered_frames.append(aligned)
        
        # Stack into 3D array: (sequence_length, n_features)
        sequence = np.array(sequence_list, dtype=np.float32)
        
        # Add batch dimension: (1, sequence_length, n_features)
        sequence = np.expand_dims(sequence, axis=0)
        
        logger.debug(f"Created sequence shape: {sequence.shape}")
        
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
        
        # If above all ranges, return Severe
        return 'Severe'
    
    def category_to_aqi_range_indian(self, category: str) -> Tuple[int, int]:
        normalized_category = category.replace(' ', '_')
        if normalized_category not in self.indian_index:
            logger.warning(f"Unknown category: {category}, defaulting to Severe")
            normalized_category = 'Severe'
        return self.indian_index[normalized_category]
    
    def category_to_aqi_range_us(self, category: str) -> Tuple[int, int]:
        normalized_category = category.replace(' ', '_')
        us_category = CATEGORY_MAPPING.get(normalized_category, {}).get('us', 'Hazardous')
        if us_category not in self.us_index:
            logger.warning(f"Unknown US category: {us_category}, defaulting to Hazardous")
            us_category = 'Hazardous'
        return self.us_index[us_category]
    
    def get_concentration_range(self, pollutant: str, category: str, standard: str = 'IN') -> Tuple[float, float]:
        normalized_category = category.replace(' ', '_')
        breakpoints = self.indian_breakpoints if standard == 'IN' else self.us_breakpoints
        
        if pollutant in breakpoints:
            if standard == 'IN' and normalized_category in breakpoints[pollutant]:
                return breakpoints[pollutant][normalized_category]
            elif standard == 'US':
                us_category = CATEGORY_MAPPING.get(normalized_category, {}).get('us', 'Hazardous')
                if us_category in breakpoints[pollutant]:
                    return breakpoints[pollutant][us_category]
        
        return (0, 0)
    
    def calculate_sub_index_indian(self, pollutant: str, category: str, 
                                   predicted_value: Optional[float] = None) -> Dict:
        """
        Calculate AQI sub-index for Indian standard
        
        Args:
            pollutant: Pollutant name
            category: AQI category
            predicted_value: Actual predicted concentration (optional)
        """
        aqi_min, aqi_max = self.category_to_aqi_range_indian(category)
        conc_min, conc_max = self.get_concentration_range(pollutant, category, 'IN')
        
        return {
            'pollutant': pollutant,
            'category': category,
            'predicted_value': predicted_value,
            'aqi_range': (aqi_min, aqi_max),
            'aqi_min': aqi_min,
            'aqi_max': aqi_max,
            'aqi_mid': (aqi_min + aqi_max) / 2,
            'concentration_range': (conc_min, conc_max),
            'confidence': None  # No confidence in regression models
        }
    
    def calculate_sub_index_us(self, pollutant: str, category: str,
                               predicted_value: Optional[float] = None) -> Dict:
        """
        Calculate AQI sub-index for US standard
        """
        normalized_category = category.replace(' ', '_')
        us_category = CATEGORY_MAPPING.get(normalized_category, {}).get('us', 'Hazardous')
        aqi_min, aqi_max = self.category_to_aqi_range_us(category)
        conc_min, conc_max = self.get_concentration_range(pollutant, category, 'US')
        
        return {
            'pollutant': pollutant,
            'category': us_category,
            'predicted_value': predicted_value,
            'aqi_range': (aqi_min, aqi_max),
            'aqi_min': aqi_min,
            'aqi_max': aqi_max,
            'aqi_mid': (aqi_min + aqi_max) / 2,
            'concentration_range': (conc_min, conc_max),
            'confidence': None
        }
    
    def calculate_overall_aqi(self, predictions: Dict[str, Tuple[str, Optional[float]]], 
                             standard: str = 'IN') -> Dict:
        """
        Calculate overall AQI from multiple pollutant predictions
        
        Args:
            predictions: Dict mapping pollutant -> (category, predicted_value)
            standard: 'IN' or 'US'
        """
        if not predictions:
            return {
                'category': 'Unknown',
                'aqi_mid': 0,
                'dominant_pollutant': None,
                'sub_indices': {}
            }
        
        sub_indices = {}
        max_aqi = 0
        dominant_pollutant = None
        
        for pollutant, (category, pred_value) in predictions.items():
            if standard == 'IN':
                sub_index = self.calculate_sub_index_indian(pollutant, category, pred_value)
            else:
                sub_index = self.calculate_sub_index_us(pollutant, category, pred_value)
            
            sub_indices[pollutant] = sub_index
            
            if sub_index['aqi_mid'] > max_aqi:
                max_aqi = sub_index['aqi_mid']
                dominant_pollutant = pollutant
        
        # Overall category is determined by highest AQI
        overall_category = sub_indices[dominant_pollutant]['category'] if dominant_pollutant else 'Unknown'
        
        return {
            'category': overall_category,
            'aqi_mid': max_aqi,
            'dominant_pollutant': dominant_pollutant,
            'sub_indices': sub_indices
        }
    
    def get_health_implications(self, category: str, standard: str = 'IN') -> str:
        """Get health implications for a category"""
        normalized_category = category.replace(' ', '_')
        
        if standard == 'IN':
            implications = {
                'Good': 'Minimal Impact',
                'Satisfactory': 'Minor breathing discomfort to sensitive people',
                'Moderate': 'Breathing discomfort to people with lung/heart disease, children, older adults',
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
# POLLUTANT PREDICTOR - UPDATED FOR LSTM
# ============================================================================

class PollutantPredictor:
    """Complete LSTM prediction pipeline for all pollutants"""
    
    def __init__(self, model_path: Path):
        self.model_manager = ModelManager(model_path)
        self.feature_engineer = FeatureEngineer()
        self.feature_aligner = FeatureAligner()
        self.aqi_calculator = AQICalculator()
        self.sequence_builder = SequenceBuilder()
        
        self.pollutants = ['PM25', 'PM10', 'NO2', 'OZONE']
    
    def predict_single_pollutant(self,
                                current_data: pd.DataFrame,
                                historical_data: pd.DataFrame,
                                pollutant: str,
                                horizon: str) -> Tuple[str, float]:
        """
        Predict using LSTM regression model
        
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
            
            # 3. Build sequence from current + historical
            sequence, _ = self.sequence_builder.create_sequence(
                current_features=current_features,
                historical_data=historical_data,
                feature_engineer=self.feature_engineer,
                pollutant=pollutant,
                horizon=horizon,
                feature_names=feature_names,
                sequence_length=sequence_length
            )
            
            # 4. Scale features
            # Reshape to 2D for scaler: (sequence_length, n_features)
            seq_2d = sequence[0]  # Remove batch dimension
            seq_scaled_2d = feature_scaler.transform(seq_2d)
            
            # Reshape back to 3D: (1, sequence_length, n_features)
            seq_scaled_3d = np.expand_dims(seq_scaled_2d, axis=0)
            
            # 5. Predict concentration (regression)
            pred_scaled = keras_model.predict(seq_scaled_3d, verbose=0)
            
            # 6. Inverse transform to get actual concentration
            pred_concentration = target_scaler.inverse_transform(pred_scaled)[0][0]
            pred_concentration = float(np.clip(pred_concentration, 0, 1000))
            
            # 7. Map concentration to AQI category using breakpoints
            category = self.aqi_calculator.concentration_to_category(
                pred_concentration, pollutant
            )
            
            logger.info(
                f"  ✓ {pollutant} {horizon}: {pred_concentration:.2f} µg/m³ → {category}"
            )
            
            return category, pred_concentration
            
        except Exception as e:
            logger.error(f"LSTM prediction failed for {pollutant} {horizon}: {e}")
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
                category, pred_value = self.predict_single_pollutant(
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
                    'category': category,
                    'predicted_value': pred_value,  # NEW: actual concentration
                    'confidence': None,              # No confidence in regression
                    'aqi_range': sub_index['aqi_range'],
                    'aqi_min': sub_index['aqi_min'],
                    'aqi_max': sub_index['aqi_max'],
                    'aqi_mid': sub_index['aqi_mid'],
                    'concentration_range': sub_index['concentration_range']
                }
                
            except Exception as e:
                logger.error(f"Failed to predict {pollutant} {horizon}: {e}")
                results[horizon] = {
                    'category': 'Unknown',
                    'predicted_value': 0.0,
                    'confidence': None,
                    'aqi_range': (0, 0),
                    'aqi_min': 0,
                    'aqi_max': 0,
                    'aqi_mid': 0,
                    'concentration_range': (0, 0),
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
            logger.info(f"\n🔍 Predicting {pollutant}...")
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
                    results[pollutant][horizon].get('predicted_value')
                )
                for pollutant in self.pollutants
                if 'error' not in results[pollutant][horizon]
            }
            
            # Calculate overall AQI
            overall = self.aqi_calculator.calculate_overall_aqi(predictions, standard)
            
            overall_results[horizon] = overall
        
        results['overall'] = overall_results
        
        return results