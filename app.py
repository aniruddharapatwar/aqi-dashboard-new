"""
AQI Dashboard - FastAPI Backend (LSTM VERSION)
Complete REST API for air quality predictions and AI assistance
UPDATED FOR: LSTM Regression Models with Lat/Long matching
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
import json
import os
import re

# LSTM imports
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import proper feature engineer
from feature_engineer import FeatureEngineer, FeatureAligner

# Import Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Generative AI not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    BASE_DIR = Path(__file__).parent
    MODEL_PATH = BASE_DIR / "Classification_trained_models"
    DATA_PATH = BASE_DIR / "data" / "inference_data.csv"
    WHITELIST_PATH = BASE_DIR / "data" / "region_wise_popular_places_from_inference.csv"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    AQI_COLORS_IN = {
        'Good': '#00E400', 'Satisfactory': '#FFFF00', 'Moderate': '#FF7E00',
        'Poor': '#FF0000', 'Very_Poor': '#8F3F97', 'Severe': '#7E0023'
    }
    
    AQI_COLORS_US = {
        'Good': '#00E400', 'Moderate': '#FFFF00', 'Unhealthy_for_Sensitive': '#FF7E00',
        'Unhealthy': '#FF0000', 'Very_Unhealthy': '#8F3F97', 'Hazardous': '#7E0023'
    }

# ============================================================================
# MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    location: str
    standard: str = "IN"

class ChatRequest(BaseModel):
    message: str
    location: Optional[str] = None
    aqi_data: Optional[Dict] = None
    user_profile: Optional[Dict] = None

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="AQI Dashboard API",
    description="Air Quality Index Prediction and Advisory System (LSTM)",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MANAGER
# ============================================================================

class DataManager:
    def __init__(self):
        self.data = self.load_data()
        self.whitelist = self.load_whitelist()
        self.models = {}
    
    def load_data(self):
        """Load inference data with mixed date format support"""
        try:
            if not os.path.exists(Config.DATA_PATH):
                raise FileNotFoundError(f"Data file not found: {Config.DATA_PATH}")
            
            logger.info(f"Loading data from: {Config.DATA_PATH}")
            df = pd.read_csv(Config.DATA_PATH)
            
            logger.info(f"Data columns: {list(df.columns)}")
            
            # Handle timestamp column
            # Expected columns: timestamp,lat,lon,year,month,day,hour,location,pincode,region,CO,NO2,OZONE,SO2,PM25,PM10,AQI,...
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
            else:
                raise ValueError("Data must have 'timestamp' or 'date' column")
            
            # Verify required columns for lat/lon matching
            if 'lng' in df.columns and 'lon' not in df.columns:
                df['lon'] = df['lng']
            if 'lat' not in df.columns or 'lon' not in df.columns:
                raise ValueError(f"Data must have 'lat' and 'lon' columns. Found: {list(df.columns)}")
            
            # Drop rows with missing coordinates
            initial_count = len(df)
            df = df.dropna(subset=['lat', 'lon', 'timestamp'])
            dropped = initial_count - len(df)
            if dropped > 0:
                logger.warning(f"Dropped {dropped} rows with missing lat/lon/timestamp")
            
            df = df.sort_values(['lat', 'lon', 'timestamp'])
            logger.info(f"✓ Loaded {len(df)} valid data rows")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"Unique coordinates: {len(df.groupby(['lat', 'lon']))}")
            
            # Log sample of available pollutants
            pollutant_cols = ['PM25', 'PM10', 'NO2', 'OZONE', 'CO', 'SO2']
            available_pollutants = [col for col in pollutant_cols if col in df.columns]
            logger.info(f"Available pollutants: {available_pollutants}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            return pd.DataFrame(columns=['lat', 'lon', 'timestamp'])
    
    def load_whitelist(self):
        """Load whitelist ONLY from region_wise_popular_places_from_inference.csv"""
        try:
            whitelist = {}
            
            if not os.path.exists(Config.WHITELIST_PATH):
                raise FileNotFoundError(f"Whitelist file not found: {Config.WHITELIST_PATH}")
            
            logger.info(f"Loading whitelist from: {Config.WHITELIST_PATH}")
            df = pd.read_csv(Config.WHITELIST_PATH)
            
            logger.info(f"Whitelist columns: {list(df.columns)}")
            
            # Expected columns: Region, Place, Area/Locality, PIN Code, Latitude, Longitude
            required_cols = ['Region', 'Place', 'Latitude', 'Longitude']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in whitelist: {missing_cols}. Found: {list(df.columns)}")
            
            for idx, row in df.iterrows():
                try:
                    place_name = str(row['Place']).strip()
                    region = str(row['Region']).strip()
                    lat = float(row['Latitude'])
                    lon = float(row['Longitude'])
                    
                    # Skip invalid entries
                    if not place_name or not region or pd.isna(lat) or pd.isna(lon):
                        logger.warning(f"Skipping invalid row {idx}: {place_name}, {region}, {lat}, {lon}")
                        continue
                    
                    whitelist[place_name] = {
                        'region': region,
                        'lat': lat,
                        'lon': lon,
                        'pin': str(row.get('PIN Code', '')).strip() if pd.notna(row.get('PIN Code')) else '',
                        'area': str(row.get('Area/Locality', place_name)).strip() if pd.notna(row.get('Area/Locality')) else place_name
                    }
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    continue
            
            logger.info(f"✓ Loaded {len(whitelist)} locations from whitelist")
            
            if len(whitelist) == 0:
                logger.error("❌ No locations loaded from whitelist!")
                logger.error(f"Check if the CSV file has valid data at: {Config.WHITELIST_PATH}")
            else:
                # Log sample locations per region
                regions = {}
                for place, info in whitelist.items():
                    region = info['region']
                    if region not in regions:
                        regions[region] = []
                    regions[region].append(place)
                
                logger.info(f"✓ Regions found: {list(regions.keys())}")
                for region, places in list(regions.items())[:5]:  # Show first 5 regions
                    logger.info(f"  • {region}: {len(places)} locations (e.g., {places[0] if places else 'N/A'})")
                
                if len(regions) > 5:
                    logger.info(f"  ... and {len(regions) - 5} more regions")
            
            return whitelist
            
        except Exception as e:
            logger.error(f"Failed to load whitelist: {e}", exc_info=True)
            return {}
    
    def get_regions(self):
        """Get unique regions from whitelist"""
        regions = set(loc['region'] for loc in self.whitelist.values())
        return sorted([r for r in regions if r and str(r).strip()])
    
    def get_locations_by_region(self, region):
        """Get locations for a specific region"""
        locations = [name for name, info in self.whitelist.items() 
                    if info['region'] == region]
        return sorted(locations)
    
    def get_location_data(self, location_name):
        """Get current and historical data for a location using lat/lon matching"""
        if location_name not in self.whitelist:
            raise ValueError(f"Location '{location_name}' not found in whitelist")
        
        loc = self.whitelist[location_name]
        lat, lon = loc['lat'], loc['lon']
        
        logger.info(f"Searching data for {location_name} at ({lat:.6f}, {lon:.6f})")
        
        # Use exact coordinates from whitelist with tolerance
        tolerance = 0.0001  # ~11 meters
        mask = ((np.abs(self.data['lat'] - lat) < tolerance) & 
                (np.abs(self.data['lon'] - lon) < tolerance))
        loc_data = self.data[mask].copy()
        
        # If no exact match, expand search radius progressively
        if len(loc_data) == 0:
            logger.warning(f"No exact match, expanding search radius")
            tolerance = 0.001  # ~111 meters
            mask = ((np.abs(self.data['lat'] - lat) < tolerance) & 
                    (np.abs(self.data['lon'] - lon) < tolerance))
            loc_data = self.data[mask].copy()
        
        if len(loc_data) == 0:
            tolerance = 0.01  # ~1.1 km
            mask = ((np.abs(self.data['lat'] - lat) < tolerance) & 
                    (np.abs(self.data['lon'] - lon) < tolerance))
            loc_data = self.data[mask].copy()
        
        if len(loc_data) == 0:
            tolerance = 0.05  # ~5.5 km
            mask = ((np.abs(self.data['lat'] - lat) < tolerance) & 
                    (np.abs(self.data['lon'] - lon) < tolerance))
            loc_data = self.data[mask].copy()
        
        if len(loc_data) == 0:
            raise ValueError(
                f"No data found for {location_name} at ({lat:.6f}, {lon:.6f})\n"
                f"Searched up to 5.5km radius. Please check if coordinates match inference data."
            )
        
        logger.info(f"✓ Found {len(loc_data)} data rows for {location_name}")
        
        loc_data = loc_data.sort_values('timestamp')
        
        # Return current (last row) and historical (up to 300 rows for sequence building)
        return loc_data.iloc[[-1]].copy(), loc_data.tail(300).copy()
    
    def load_model(self, pollutant: str, horizon: str):
        """Load LSTM model and artifacts for specific pollutant and horizon"""
        cache_key = f"{pollutant}_{horizon}"
        if cache_key in self.models:
            return self.models[cache_key]
        
        # Load from directory structure: POLLUTANT_HORIZON/
        model_dir = Config.MODEL_PATH / f"{pollutant}_{horizon}"
        keras_model_path = model_dir / "lstm_pure_regression.h5"
        artifacts_path = model_dir / "model_artifacts.pkl"
        
        if not keras_model_path.exists():
            raise FileNotFoundError(
                f"LSTM model not found: {keras_model_path}\n"
                f"Expected: {model_dir}/lstm_pure_regression.h5"
            )
        if not artifacts_path.exists():
            raise FileNotFoundError(
                f"Artifacts not found: {artifacts_path}\n"
                f"Expected: {model_dir}/model_artifacts.pkl"
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
            'feature_names': artifacts.get('feature_names', [])
        }
        
        self.models[cache_key] = model_artifact
        logger.info(f"  ✓ Loaded with {len(model_artifact['feature_names'])} features")
        
        return model_artifact

# Initialize data manager
data_manager = DataManager()

# Initialize feature engineer
feature_engineer = FeatureEngineer()
feature_aligner = FeatureAligner()

# ============================================================================
# AQI CALCULATION
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

US_AQI_INDEX = {
    'Good': (0, 50), 'Moderate': (51, 100), 'Unhealthy_for_Sensitive': (101, 150),
    'Unhealthy': (151, 200), 'Very_Unhealthy': (201, 300), 'Hazardous': (301, 500)
}

CATEGORY_MAPPING = {
    'Good': {'us': 'Good'}, 'Satisfactory': {'us': 'Moderate'},
    'Moderate': {'us': 'Unhealthy_for_Sensitive'}, 'Poor': {'us': 'Unhealthy'},
    'Very_Poor': {'us': 'Very_Unhealthy'}, 'Severe': {'us': 'Hazardous'}
}

def concentration_to_category(concentration: float, pollutant: str) -> str:
    """Map predicted concentration to AQI category"""
    breakpoints = INDIAN_AQI_BREAKPOINTS.get(pollutant, {})
    
    for category, (low, high) in breakpoints.items():
        if low <= concentration <= high:
            return category
    
    return 'Severe'  # If above all ranges

class AQICalculator:
    def calculate_sub_index(self, pollutant: str, category: str, 
                           predicted_value: Optional[float], standard: str):
        normalized = category.replace(' ', '_')
        if standard == 'IN':
            aqi_min, aqi_max = INDIAN_AQI_INDEX.get(normalized, (0, 0))
        else:
            us_cat = CATEGORY_MAPPING.get(normalized, {}).get('us', 'Hazardous')
            aqi_min, aqi_max = US_AQI_INDEX.get(us_cat, (0, 0))
        
        conc_min, conc_max = INDIAN_AQI_BREAKPOINTS.get(pollutant, {}).get(normalized, (0, 0))
        
        return {
            'pollutant': pollutant,
            'category': category,
            'predicted_value': predicted_value,
            'aqi_min': aqi_min,
            'aqi_max': aqi_max,
            'aqi_mid': (aqi_min + aqi_max) / 2,
            'concentration_range': (conc_min, conc_max),
            'confidence': None
        }
    
    def calculate_overall(self, predictions: Dict, standard: str):
        sub_indices = []
        for pollutant, (category, pred_value) in predictions.items():
            if pollutant in ['PM25', 'PM10', 'NO2', 'OZONE']:
                sub_idx = self.calculate_sub_index(pollutant, category, pred_value, standard)
                sub_indices.append(sub_idx)
        
        if not sub_indices:
            return {'error': 'No valid predictions'}
        
        max_idx = max(sub_indices, key=lambda x: x['aqi_mid'])
        return {
            'aqi_min': max_idx['aqi_min'],
            'aqi_max': max_idx['aqi_max'],
            'aqi_mid': max_idx['aqi_mid'],
            'category': max_idx['category'],
            'dominant_pollutant': max_idx['pollutant'],
            'confidence': None
        }

aqi_calculator = AQICalculator()

# ============================================================================
# PREDICTION ENGINE (LSTM)
# ============================================================================

def predict_single(current_data: pd.DataFrame, historical_data: pd.DataFrame,
                  pollutant: str, horizon: str):
    """
    Predict using LSTM regression model
    Returns: (category, predicted_concentration)
    """
    # 1. Load LSTM model and artifacts
    model = data_manager.load_model(pollutant, horizon)
    keras_model = model['model']
    feature_scaler = model['feature_scaler']
    target_scaler = model['target_scaler']
    sequence_length = model['sequence_length']
    feature_names = model['feature_names']
    
    # 2. Engineer features for current timestep
    features = feature_engineer.engineer_features(
        current_data=current_data,
        historical_data=historical_data,
        pollutant=pollutant,
        horizon=horizon
    )
    
    # 3. Build sequence from current + historical
    sequence_list = []
    
    for i in range(sequence_length - 1, -1, -1):
        if i == 0:
            # Current timestep
            timestep_features = features
        else:
            # Historical timestep
            if len(historical_data) >= i:
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
                # Not enough history, pad with zeros
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
    
    # Stack into 3D array: (sequence_length, n_features)
    sequence = np.array(sequence_list, dtype=np.float32)
    sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
    
    # 4. Scale features
    seq_2d = sequence[0]  # Remove batch dimension for scaling
    seq_scaled_2d = feature_scaler.transform(seq_2d)
    seq_scaled_3d = np.expand_dims(seq_scaled_2d, axis=0)  # Add batch back
    
    # 5. Predict concentration (regression)
    pred_scaled = keras_model.predict(seq_scaled_3d, verbose=0)
    
    # 6. Inverse transform to get actual concentration
    pred_concentration = target_scaler.inverse_transform(pred_scaled)[0][0]
    pred_concentration = float(np.clip(pred_concentration, 0, 1000))
    
    # 7. Map concentration to AQI category
    category = concentration_to_category(pred_concentration, pollutant)
    
    return category, pred_concentration

def extract_weather_data(current_data: pd.DataFrame) -> Dict:
    """Extract weather parameters from current data row"""
    weather_features = [
        'temperature', 'humidity', 'dewPoint', 'apparentTemperature',
        'precipIntensity', 'pressure', 'surfacePressure',
        'cloudCover', 'windSpeed', 'windBearing', 'windGust'
    ]
    
    weather = {}
    if len(current_data) > 0:
        row = current_data.iloc[0]
        for feature in weather_features:
            if feature in row.index and pd.notna(row[feature]):
                value = float(row[feature])
                original_value = value
                
                # Convert Fahrenheit to Celsius for temperature fields
                if feature in ['temperature', 'dewPoint', 'apparentTemperature'] and value > 50:
                    value = (value - 32) * 5 / 9
                    logger.info(f"Converted {feature}: {original_value:.1f}°F → {value:.1f}°C")
                
                weather[feature] = value
            else:
                weather[feature] = 0.0
    
    logger.info(f"Weather extracted: temperature={weather.get('temperature', 0):.1f}°C, humidity={weather.get('humidity', 0):.1f}%, windSpeed={weather.get('windSpeed', 0):.1f} km/h")
    return weather

def predict_all(current_data: pd.DataFrame, historical_data: pd.DataFrame, standard: str = 'IN'):
    results = {}
    
    logger.info(f"Current data shape: {current_data.shape}, Historical: {historical_data.shape}")
    
    # Extract weather data
    weather_data = extract_weather_data(current_data)
    results['weather'] = weather_data
    
    # Add historical AQI data
    historical_aqi = []
    if len(historical_data) > 0 and 'timestamp' in historical_data.columns:
        historical_subset = historical_data.tail(48).copy()
        
        for _, row in historical_subset.iterrows():
            aqi_value = 0
            timestamp = row.get('timestamp', '')
            
            if 'AQI' in row.index and pd.notna(row['AQI']):
                aqi_value = float(row['AQI'])
            else:
                if 'PM25' in row.index and pd.notna(row['PM25']):
                    pm25 = float(row['PM25'])
                    if pm25 <= 30:
                        aqi_value = pm25 * 50 / 30
                    elif pm25 <= 60:
                        aqi_value = 50 + (pm25 - 30) * 50 / 30
                    elif pm25 <= 90:
                        aqi_value = 100 + (pm25 - 60) * 100 / 30
                    elif pm25 <= 120:
                        aqi_value = 200 + (pm25 - 90) * 100 / 30
                    elif pm25 <= 250:
                        aqi_value = 300 + (pm25 - 120) * 100 / 130
                    else:
                        aqi_value = 400 + (pm25 - 250) * 100 / 130
            
            if aqi_value > 0:
                historical_aqi.append({
                    'timestamp': str(timestamp),
                    'aqi': round(aqi_value, 1)
                })
        
        logger.info(f"✓ Prepared {len(historical_aqi)} historical AQI data points")
    
    results['historical'] = historical_aqi
    
    for pollutant in ['PM25', 'PM10', 'NO2', 'OZONE']:
        results[pollutant] = {}
        
        if pollutant not in historical_data.columns:
            logger.warning(f"{pollutant} not in data columns")
            for horizon in ['1h', '6h', '12h', '24h']:
                results[pollutant][horizon] = {
                    'category': 'Unknown',
                    'predicted_value': 0.0,
                    'confidence': None,
                    'aqi_min': 0,
                    'aqi_max': 0,
                    'aqi_mid': 0,
                    'concentration_range': (0, 0),
                    'error': f'{pollutant} data not available'
                }
            continue
        
        for horizon in ['1h', '6h', '12h', '24h']:
            try:
                category, pred_value = predict_single(current_data, historical_data, pollutant, horizon)
                sub_idx = aqi_calculator.calculate_sub_index(pollutant, category, pred_value, standard)
                results[pollutant][horizon] = {
                    'category': category,
                    'predicted_value': pred_value,
                    'confidence': None,
                    'aqi_min': sub_idx['aqi_min'],
                    'aqi_max': sub_idx['aqi_max'],
                    'aqi_mid': sub_idx['aqi_mid'],
                    'concentration_range': sub_idx['concentration_range']
                }
                logger.info(f"✓ {pollutant} {horizon}: {pred_value:.2f} µg/m³ → {category}")
            except Exception as e:
                logger.error(f"Failed {pollutant} {horizon}: {e}")
                results[pollutant][horizon] = {
                    'category': 'Unknown',
                    'predicted_value': 0.0,
                    'confidence': None,
                    'aqi_min': 0,
                    'aqi_max': 0,
                    'aqi_mid': 0,
                    'concentration_range': (0, 0),
                    'error': str(e)
                }
    
    # Calculate overall AQI
    results['overall'] = {}
    for horizon in ['1h', '6h', '12h', '24h']:
        preds = {p: (results[p][horizon]['category'], 
                     results[p][horizon].get('predicted_value'))
                for p in ['PM25', 'PM10', 'NO2', 'OZONE']
                if 'error' not in results[p][horizon]}
        
        if preds:
            results['overall'][horizon] = aqi_calculator.calculate_overall(preds, standard)
        else:
            results['overall'][horizon] = {
                'aqi_min': 0,
                'aqi_max': 0,
                'aqi_mid': 0,
                'category': 'Unknown',
                'dominant_pollutant': 'None',
                'confidence': None
            }
    
    return results

# ============================================================================
# GEMINI AI ASSISTANT
# ============================================================================

class ResponseValidator:
    """Validates LLM responses for safety, accuracy, and structure"""
    
    FORBIDDEN_PHRASES = [
        'diagnose', 'diagnosis', 'cure', 'treatment', 'medication',
        'prescribe', 'prescription', 'take this medicine',
        'definitely have', 'you have a disease', 'medical condition'
    ]
    
    REQUIRED_DISCLAIMER = "**Important:** This is not medical advice"
    DISCLAIMER_TEXT = "\n\n⚠️ **Important:** This is not medical advice. Please consult a healthcare professional for diagnosis or treatment."
    
    @staticmethod
    def ensure_structure(response: str, context: Dict) -> str:
        has_structure = ('**Current Situation:**' in response or 
                        '**Your Risk Level:**' in response or
                        '**Recommended Actions:**' in response)
        
        if has_structure:
            return response.strip()
        
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        
        if len(paragraphs) >= 2:
            structured = f"**Current Situation:**\n{paragraphs[0]}\n\n"
            
            if len(paragraphs) >= 3:
                structured += f"**Recommended Actions:**\n{paragraphs[1]}\n\n"
                structured += '\n\n'.join(paragraphs[2:])
            else:
                structured += '\n\n'.join(paragraphs[1:])
            
            return structured
        
        return response
    
    @staticmethod
    def validate_response(response: str, context: Dict) -> Dict:
        warnings = []
        valid = True
        
        response_lower = response.lower()
        for phrase in ResponseValidator.FORBIDDEN_PHRASES:
            if phrase in response_lower:
                warnings.append(f"Contains forbidden medical term: '{phrase}'")
                valid = False
        
        if len(response.strip()) < 50:
            warnings.append("Response too short - may be incomplete")
            valid = False
            return {
                'valid': valid,
                'warnings': warnings,
                'modified_response': response
            }
        
        modified_response = ResponseValidator.ensure_structure(response, context)
        
        if context.get('aqi_data'):
            aqi_mid = context['aqi_data'].get('aqi_mid', 0)
            if aqi_mid > 0 and str(int(aqi_mid)) not in modified_response and 'AQI' not in modified_response.upper():
                warnings.append("Response may not be grounded in provided AQI data")
        
        if context.get('weather'):
            temp = context['weather'].get('temperature', 0)
            temp_mentions = re.findall(r'(\d+)\s*°?[Cc]', modified_response)
            if temp_mentions and temp > 0:
                for temp_str in temp_mentions:
                    try:
                        mentioned_temp = int(temp_str)
                        if abs(mentioned_temp - temp) > 10:
                            warnings.append(f"Temperature mismatch: mentioned {mentioned_temp}°C but actual is {temp:.1f}°C")
                    except ValueError:
                        pass
        
        disclaimer_patterns = [
            r'\n*\s*⚠️\s*\*\*Important:\*\*.*?medical advice.*?(?:\n\n|\Z)',
            r'\n*\s*\*\*Important:\*\*\s*This is not medical advice.*?(?:\n\n|\Z)',
            r'\n*\s*This is not medical advice.*?(?:\n\n|\Z)',
            r'\n*\s*⚠.*?medical advice.*?(?:\n\n|\Z)'
        ]
        
        for pattern in disclaimer_patterns:
            modified_response = re.sub(pattern, '', modified_response, flags=re.DOTALL | re.IGNORECASE)
        
        modified_response = modified_response.strip()
        modified_response = modified_response + ResponseValidator.DISCLAIMER_TEXT
        
        if ResponseValidator.REQUIRED_DISCLAIMER not in modified_response:
            warnings.append("Disclaimer could not be added properly")
            modified_response = modified_response + ResponseValidator.DISCLAIMER_TEXT
        
        return {
            'valid': valid,
            'warnings': warnings,
            'modified_response': modified_response
        }

response_validator = ResponseValidator()

class GeminiAssistant:
    def __init__(self):
        self.enabled = False
        self.model = None
        self.model_name = None
        
        if not Config.GEMINI_API_KEY:
            logger.warning("⚠ GEMINI_API_KEY not found in environment")
            return
            
        if not GEMINI_AVAILABLE:
            logger.warning("⚠ google-generativeai package not installed")
            return
        
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            
            models_to_try = [
                'gemini-2.5-pro',
                'gemini-2.5-flash',
                'gemini-2.5-flash-lite'
            ]
            
            for model_name in models_to_try:
                try:
                    self.model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config={
                            'temperature': 0.7,
                            'top_p': 0.95,
                            'top_k': 40,
                            'max_output_tokens': 1024,
                        }
                    )
                    test_response = self.model.generate_content("Hello")
                    if test_response and test_response.text:
                        self.enabled = True
                        self.model_name = model_name
                        logger.info(f"✓ Gemini AI initialized with {model_name}")
                        break
                except Exception as e:
                    logger.debug(f"Model {model_name} not available: {e}")
                    continue
            
            if not self.enabled:
                logger.error("Failed to initialize any Gemini model")
                    
        except Exception as e:
            logger.error(f"⚠ Gemini initialization failed: {e}")
            self.enabled = False
    
    def get_response(self, message: str, context: Dict) -> Dict:
        if not self.enabled:
            return {
                'response': "Gemini AI not available",
                'updated_profile': None,
                'validation': {'valid': True, 'warnings': []}
            }
        
        # Simplified - add full Gemini logic if needed
        return {
            'response': "Air quality advice based on current conditions",
            'updated_profile': None,
            'validation': {'valid': True, 'warnings': []}
        }

gemini_assistant = GeminiAssistant()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "AQI Dashboard API - LSTM Version (Lat/Long Matching)",
        "version": "2.1.0",
        "status": "running",
        "model_type": "LSTM Regression",
        "matching_method": "Latitude/Longitude coordinates"
    }

@app.get("/api/regions")
async def get_regions():
    """Get all available regions"""
    try:
        regions = data_manager.get_regions()
        logger.info(f"Returning {len(regions)} regions")
        return regions
    except Exception as e:
        logger.error(f"Error getting regions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/locations/{region}")
async def get_locations(region: str):
    """Get locations for a specific region"""
    try:
        locations = data_manager.get_locations_by_region(region)
        logger.info(f"Returning {len(locations)} locations for region: {region}")
        return locations
    except Exception as e:
        logger.error(f"Error getting locations for {region}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """Get AQI predictions for a location"""
    try:
        logger.info(f"Prediction request for: {request.location} (standard: {request.standard})")
        current, historical = data_manager.get_location_data(request.location)
        predictions = predict_all(current, historical, request.standard)
        return predictions
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with AI assistant"""
    try:
        context = {
            'location': request.location,
            'aqi_data': request.aqi_data,
            'user_profile': request.user_profile,
            'weather': request.aqi_data.get('weather', {}) if request.aqi_data else {}
        }
        
        result = gemini_assistant.get_response(request.message, context)
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.1.0",
        "model_type": "LSTM",
        "matching_method": "lat/long",
        "data_loaded": len(data_manager.data) > 0,
        "data_rows": len(data_manager.data),
        "locations": len(data_manager.whitelist),
        "regions": len(data_manager.get_regions()),
        "gemini_enabled": gemini_assistant.enabled
    }

@app.get("/api/debug/whitelist")
async def debug_whitelist():
    """Debug endpoint to view whitelist locations"""
    try:
        return {
            "total_locations": len(data_manager.whitelist),
            "regions": data_manager.get_regions(),
            "sample_locations": {
                name: {
                    "region": info['region'],
                    "lat": info['lat'],
                    "lon": info['lon'],
                    "area": info['area']
                }
                for name, info in list(data_manager.whitelist.items())[:10]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debug/location/{location_name}")
async def debug_location(location_name: str):
    """Debug endpoint to check coordinate matching for a specific location"""
    try:
        if location_name not in data_manager.whitelist:
            raise HTTPException(status_code=404, detail=f"Location '{location_name}' not found in whitelist")
        
        loc = data_manager.whitelist[location_name]
        lat, lon = loc['lat'], loc['lon']
        
        # Check data availability at different tolerance levels
        tolerance_checks = []
        for tolerance, description in [
            (0.0001, "~11 meters"),
            (0.001, "~111 meters"),
            (0.01, "~1.1 km"),
            (0.05, "~5.5 km")
        ]:
            mask = ((np.abs(data_manager.data['lat'] - lat) < tolerance) & 
                   (np.abs(data_manager.data['lon'] - lon) < tolerance))
            matched_data = data_manager.data[mask]
            
            tolerance_checks.append({
                "tolerance": tolerance,
                "description": description,
                "rows_found": len(matched_data),
                "unique_timestamps": len(matched_data['timestamp'].unique()) if len(matched_data) > 0 else 0
            })
        
        return {
            "location": location_name,
            "whitelist_info": loc,
            "coordinates": {"lat": lat, "lon": lon},
            "tolerance_checks": tolerance_checks,
            "recommendation": "Use smallest tolerance with sufficient data (>100 rows recommended)"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)