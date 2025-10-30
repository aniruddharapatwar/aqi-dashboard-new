"""
AQI Dashboard - FastAPI Backend (ENHANCED VERSION)
Complete REST API for air quality predictions and AI assistance
IMPROVEMENTS: Better Gemini prompts, validation, guardrails, safety checks
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

# ðŸ”§ Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

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
    WHITELIST_PATH = BASE_DIR / "region_wise_popular_places_from_inference.csv"
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
    description="Air Quality Index Prediction and Advisory System",
    version="2.0.0"
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
        """Load data with mixed date format support"""
        try:
            if not os.path.exists(Config.DATA_PATH):
                raise FileNotFoundError(f"Data file not found: {Config.DATA_PATH}")
            
            logger.info(f"Loading data from: {Config.DATA_PATH}")
            df = pd.read_csv(Config.DATA_PATH)
            
            # Handle mixed date formats
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            else:
                raise ValueError("Data must have 'date' or 'timestamp' column")
            
            # Handle lng -> lon mapping
            if 'lng' in df.columns:
                df['lon'] = df['lng']
            
            # Verify required columns
            if 'lat' not in df.columns or 'lon' not in df.columns:
                raise ValueError("Data must have 'lat' and 'lon' columns")
            
            df = df.sort_values(['lat', 'lon', 'timestamp'])
            logger.info(f"âœ“ Loaded {len(df)} data rows")
            logger.info(f"Available columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return pd.DataFrame(columns=['lat', 'lon', 'timestamp'])
    
    def load_whitelist(self):
        """Load whitelist and augment with actual data locations"""
        try:
            whitelist = {}
            
            # Load original whitelist if exists
            if os.path.exists(Config.WHITELIST_PATH):
                df = pd.read_csv(Config.WHITELIST_PATH)
                for _, row in df.iterrows():
                    whitelist[row['Place']] = {
                        'region': row['Region'],
                        'lat': row['Latitude'],
                        'lon': row['Longitude'],
                        'pin': row.get('PIN Code', ''),
                        'area': row.get('Area/Locality', row['Place'])
                    }
                logger.info(f"âœ“ Loaded {len(whitelist)} locations from whitelist")
            
            # Add locations from actual data
            if len(self.data) > 0 and 'location' in self.data.columns:
                location_groups = self.data.groupby(['lat', 'lon']).agg({
                    'location': 'first',
                    'pincode': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else '',
                    'region': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else 'Unknown'
                }).reset_index()
                
                added = 0
                for _, row in location_groups.iterrows():
                    loc_name = row['location']
                    if pd.notna(loc_name) and str(loc_name).strip():
                        whitelist[loc_name] = {
                            'region': row['region'] if pd.notna(row['region']) else 'Central Delhi',
                            'lat': row['lat'],
                            'lon': row['lon'],
                            'pin': row['pincode'] if pd.notna(row['pincode']) else '',
                            'area': loc_name
                        }
                        added += 1
                
                logger.info(f"âœ“ Added {added} locations from actual data")
            
            if len(whitelist) == 0:
                logger.error("No locations loaded!")
            
            return whitelist
            
        except Exception as e:
            logger.error(f"Failed to load whitelist: {e}")
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
        """Get current and historical data for a location"""
        if location_name not in self.whitelist:
            raise ValueError(f"Location '{location_name}' not found in whitelist")
        
        loc = self.whitelist[location_name]
        lat, lon = loc['lat'], loc['lon']
        
        logger.info(f"Searching data for {location_name} at ({lat}, {lon})")
        
        # Use exact coordinates from whitelist
        mask = ((np.abs(self.data['lat'] - lat) < 0.0001) & 
               (np.abs(self.data['lon'] - lon) < 0.0001))
        loc_data = self.data[mask].copy()
        
        # If no exact match, expand search
        if len(loc_data) == 0:
            logger.warning(f"No exact match, expanding search radius")
            mask = ((np.abs(self.data['lat'] - lat) < 0.01) & 
                   (np.abs(self.data['lon'] - lon) < 0.01))
            loc_data = self.data[mask].copy()
        
        if len(loc_data) == 0:
            mask = ((np.abs(self.data['lat'] - lat) < 0.05) & 
                   (np.abs(self.data['lon'] - lon) < 0.05))
            loc_data = self.data[mask].copy()
        
        if len(loc_data) == 0:
            raise ValueError(f"No data found for {location_name} at ({lat}, {lon})")
        
        logger.info(f"âœ“ Found {len(loc_data)} data rows for {location_name}")
        
        loc_data = loc_data.sort_values('timestamp')
        
        return loc_data.iloc[[-1]].copy(), loc_data.tail(96).copy()
    
    def load_model(self, pollutant: str, horizon: str):
        """Load ML model for specific pollutant and horizon"""
        cache_key = f"{pollutant}_{horizon}"
        if cache_key in self.models:
            return self.models[cache_key]
        
        model_file = Config.MODEL_PATH / f"model_artifacts_{pollutant}_{horizon}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            model_artifact = pickle.load(f)
        
        self.models[cache_key] = model_artifact
        return model_artifact

# Initialize data manager
data_manager = DataManager()

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class SimpleFeatureEngineer:
    WEATHER_FEATURES = [
        'temperature', 'humidity', 'dewPoint', 'apparentTemperature',
        'precipIntensity', 'pressure', 'surfacePressure',
        'cloudCover', 'windSpeed', 'windBearing', 'windGust'
    ]
    
    def engineer_features(self, current_data: pd.DataFrame, historical_data: pd.DataFrame,
                         pollutant: str, horizon: str) -> pd.DataFrame:
        features = current_data.copy()
        
        # Add lag features
        lag_map = {'1h': [1, 2, 3], '6h': [6, 12], '12h': [12, 24], '24h': [24, 48]}
        for lag in lag_map.get(horizon, [1]):
            if len(historical_data) >= lag and pollutant in historical_data.columns:
                features.loc[features.index[0], f'{pollutant}_lag_{lag}h'] = historical_data[pollutant].iloc[-lag]
        
        # Select numeric features only
        exclude = {'location', 'timestamp', 'date', 'lat', 'lng', 'lon', 'region',
                  'PM25', 'PM10', 'NO2', 'OZONE', 'CO', 'SO2', 'AQI', 'pincode', 'loc_key', 'loc_id'}
        numeric_cols = [c for c in features.columns 
                       if c not in exclude and pd.api.types.is_numeric_dtype(features[c])]
        
        return features[numeric_cols].fillna(0).astype(np.float32)
    
    def align_features(self, features: pd.DataFrame, model_features: List[str]) -> pd.DataFrame:
        return features.reindex(columns=model_features, fill_value=0.0).astype(np.float32)

feature_engineer = SimpleFeatureEngineer()

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

class AQICalculator:
    def calculate_sub_index(self, pollutant: str, category: str, confidence: float, standard: str):
        normalized = category.replace(' ', '_')
        if standard == 'IN':
            aqi_min, aqi_max = INDIAN_AQI_INDEX.get(normalized, (0, 0))
        else:
            us_cat = CATEGORY_MAPPING.get(normalized, {}).get('us', 'Hazardous')
            aqi_min, aqi_max = US_AQI_INDEX.get(us_cat, (0, 0))
        
        conc_min, conc_max = INDIAN_AQI_BREAKPOINTS.get(pollutant, {}).get(normalized, (0, 0))
        
        return {
            'pollutant': pollutant, 'category': category,
            'aqi_min': aqi_min, 'aqi_max': aqi_max,
            'aqi_mid': (aqi_min + aqi_max) / 2,
            'concentration_range': (conc_min, conc_max),
            'confidence': confidence
        }
    
    def calculate_overall(self, predictions: Dict, standard: str):
        sub_indices = []
        for pollutant, (category, confidence) in predictions.items():
            if pollutant in ['PM25', 'PM10', 'NO2', 'OZONE']:
                sub_idx = self.calculate_sub_index(pollutant, category, confidence, standard)
                sub_indices.append(sub_idx)
        
        if not sub_indices:
            return {'error': 'No valid predictions'}
        
        max_idx = max(sub_indices, key=lambda x: x['aqi_mid'])
        return {
            'aqi_min': max_idx['aqi_min'], 'aqi_max': max_idx['aqi_max'],
            'aqi_mid': max_idx['aqi_mid'], 'category': max_idx['category'],
            'dominant_pollutant': max_idx['pollutant'], 'confidence': max_idx['confidence']
        }

aqi_calculator = AQICalculator()

# ============================================================================
# PREDICTION ENGINE
# ============================================================================

def predict_single(current_data: pd.DataFrame, historical_data: pd.DataFrame,
                  pollutant: str, horizon: str):
    model = data_manager.load_model(pollutant, horizon)
    features = feature_engineer.engineer_features(current_data, historical_data, pollutant, horizon)
    aligned = feature_engineer.align_features(features, model['feature_names'])
    probs = model['calibrated_model'].predict_proba(aligned)[0]
    pred_idx = np.argmax(probs)
    return model['classes'][pred_idx], probs[pred_idx]

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
                    logger.info(f"Converted {feature}: {original_value:.1f}Â°F â†’ {value:.1f}Â°C")
                
                weather[feature] = value
            else:
                weather[feature] = 0.0
    
    logger.info(f"Weather extracted: temperature={weather.get('temperature', 0):.1f}Â°C, humidity={weather.get('humidity', 0):.1f}%, windSpeed={weather.get('windSpeed', 0):.1f} km/h")
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
        
        logger.info(f"âœ“ Prepared {len(historical_aqi)} historical AQI data points")
    
    results['historical'] = historical_aqi
    
    for pollutant in ['PM25', 'PM10', 'NO2', 'OZONE']:
        results[pollutant] = {}
        
        if pollutant not in historical_data.columns:
            logger.warning(f"{pollutant} not in data columns")
            for horizon in ['1h', '6h', '12h', '24h']:
                results[pollutant][horizon] = {
                    'category': 'Unknown', 'confidence': 0.0,
                    'aqi_min': 0, 'aqi_max': 0, 'aqi_mid': 0,
                    'concentration_range': (0, 0), 'error': f'{pollutant} data not available'
                }
            continue
        
        for horizon in ['1h', '6h', '12h', '24h']:
            try:
                category, confidence = predict_single(current_data, historical_data, pollutant, horizon)
                sub_idx = aqi_calculator.calculate_sub_index(pollutant, category, confidence, standard)
                results[pollutant][horizon] = {
                    'category': category, 'confidence': confidence,
                    'aqi_min': sub_idx['aqi_min'], 'aqi_max': sub_idx['aqi_max'],
                    'aqi_mid': sub_idx['aqi_mid'],
                    'concentration_range': sub_idx['concentration_range']
                }
                logger.info(f"âœ“ {pollutant} {horizon}: {category} ({confidence:.2%})")
            except Exception as e:
                logger.error(f"Failed {pollutant} {horizon}: {e}")
                results[pollutant][horizon] = {
                    'category': 'Unknown', 'confidence': 0.0,
                    'aqi_min': 0, 'aqi_max': 0, 'aqi_mid': 0,
                    'concentration_range': (0, 0), 'error': str(e)
                }
    
    # Calculate overall AQI
    results['overall'] = {}
    for horizon in ['1h', '6h', '12h', '24h']:
        preds = {p: (results[p][horizon]['category'], results[p][horizon]['confidence'])
                for p in ['PM25', 'PM10', 'NO2', 'OZONE'] if 'error' not in results[p][horizon]}
        
        if preds:
            results['overall'][horizon] = aqi_calculator.calculate_overall(preds, standard)
        else:
            results['overall'][horizon] = {
                'aqi_min': 0, 'aqi_max': 0, 'aqi_mid': 0,
                'category': 'Unknown', 'dominant_pollutant': 'None', 'confidence': 0.0
            }
    
    return results

# ============================================================================
# RESPONSE VALIDATOR
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
        """Ensure response has proper structure with sections"""
        
        # Check if response already has good structure (has section headers)
        has_structure = ('**Current Situation:**' in response or 
                        '**Your Risk Level:**' in response or
                        '**Recommended Actions:**' in response)
        
        if has_structure:
            # Response already structured, just clean it up
            return response.strip()
        
        # If not structured, try to add basic structure
        # Split into paragraphs
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        
        if len(paragraphs) >= 2:
            # Try to structure it
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
        """
        Validate LLM response for safety, grounding, and structure
        
        Returns:
            {
                'valid': bool,
                'warnings': List[str],
                'modified_response': str (if modifications needed)
            }
        """
        warnings = []
        valid = True
        
        # 1. Check for forbidden medical terms
        response_lower = response.lower()
        for phrase in ResponseValidator.FORBIDDEN_PHRASES:
            if phrase in response_lower:
                warnings.append(f"Contains forbidden medical term: '{phrase}'")
                valid = False
        
        # 2. Check response length (too short may indicate error)
        if len(response.strip()) < 50:
            warnings.append("Response too short - may be incomplete")
            valid = False
            # For very short responses, don't modify further
            return {
                'valid': valid,
                'warnings': warnings,
                'modified_response': response
            }
        
        # 3. Ensure structure
        modified_response = ResponseValidator.ensure_structure(response, context)
        
        # 4. Check data grounding
        if context.get('aqi_data'):
            aqi_mid = context['aqi_data'].get('aqi_mid', 0)
            if aqi_mid > 0 and str(int(aqi_mid)) not in modified_response and 'AQI' not in modified_response.upper():
                warnings.append("Response may not be grounded in provided AQI data")
        
        # 5. Check for hallucinated temperature data
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
        
        # 6. ALWAYS ensure medical disclaimer is at the end (do this last)
        # Remove any existing disclaimers first to avoid duplicates
        disclaimer_patterns = [
            r'\n*\s*⚠️\s*\*\*Important:\*\*.*?medical advice.*?(?:\n\n|\Z)',
            r'\n*\s*\*\*Important:\*\*\s*This is not medical advice.*?(?:\n\n|\Z)',
            r'\n*\s*This is not medical advice.*?(?:\n\n|\Z)',
            r'\n*\s*⚠.*?medical advice.*?(?:\n\n|\Z)'
        ]
        
        for pattern in disclaimer_patterns:
            modified_response = re.sub(pattern, '', modified_response, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up trailing whitespace
        modified_response = modified_response.strip()
        
        # Add the disclaimer at the end
        modified_response = modified_response + ResponseValidator.DISCLAIMER_TEXT
        
        # 7. Final check: ensure disclaimer is present
        if ResponseValidator.REQUIRED_DISCLAIMER not in modified_response:
            warnings.append("Disclaimer could not be added properly")
            # Force add it again
            modified_response = modified_response + ResponseValidator.DISCLAIMER_TEXT
        
        return {
            'valid': valid,
            'warnings': warnings,
            'modified_response': modified_response
        }

response_validator = ResponseValidator()

# ============================================================================
# GEMINI AI ASSISTANT - ENHANCED VERSION
# ============================================================================

class GeminiAssistant:
    def __init__(self):
        self.enabled = False
        self.model = None
        self.model_name = None
        
        if not Config.GEMINI_API_KEY:
            logger.warning("âš  GEMINI_API_KEY not found in environment")
            return
            
        if not GEMINI_AVAILABLE:
            logger.warning("âš  google-generativeai package not installed")
            return
        
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            
            # FIXED: Try correct Gemini model names
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
                        },
                        safety_settings=[
                            {
                                "category": "HARM_CATEGORY_HARASSMENT",
                                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                            },
                            {
                                "category": "HARM_CATEGORY_HATE_SPEECH",
                                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                            },
                            {
                                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                            },
                            {
                                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                            },
                        ]
                    )
                    # Test the model
                    test_response = self.model.generate_content("Hello")
                    if test_response and test_response.text:
                        self.enabled = True
                        self.model_name = model_name
                        logger.info(f"âœ“ Gemini AI initialized with {model_name}")
                        break
                except Exception as e:
                    logger.debug(f"Model {model_name} not available: {e}")
                    continue
            
            if not self.enabled:
                logger.error("Failed to initialize any Gemini model")
                    
        except Exception as e:
            logger.error(f"âš  Gemini initialization failed: {e}")
            self.enabled = False
    
    def get_response(self, message: str, context: Dict) -> Dict:
        user_profile = context.get('user_profile', {})
        location = context.get('location', 'Unknown')
        aqi_data = context.get('aqi_data', {})
        weather_data = context.get('weather', {})
        
        if not self.enabled:
            logger.warning("Gemini not enabled, using static response")
            return {
                'response': self._static_response(context, user_profile),
                'updated_profile': None,
                'validation': {'valid': True, 'warnings': []}
            }
        
        try:
            # Build comprehensive context
            age_category = user_profile.get('age_category', '')
            health_category = user_profile.get('health_category', '')
            gender = user_profile.get('gender', '')
            profile_label = user_profile.get('profile_label', 'general public')
            
            # IMPROVED PROMPT with strict structure enforcement
            system_instruction = """You are an empathetic, professional Air Quality & Health Advisory Assistant for Delhi NCR, India.

                                    YOUR ROLE:
                                    - Provide personalized, risk-based, and actionable air quality advice
                                    - Be warm, caring, and supportive while remaining professional
                                    - Focus on prevention and practical recommendations

                                    CRITICAL RULES (NEVER BREAK):
                                    1. NEVER diagnose medical conditions or diseases
                                    2. NEVER prescribe medications or treatments
                                    3. ALWAYS base advice ONLY on the provided air quality and weather data
                                    4. DO NOT include a medical disclaimer yourself - it will be added automatically
                                    5. NEVER introduce data not provided in the context
                                    6. Stay focused on air quality, pollution, and health protection
                                    7. If asked about unrelated topics, politely redirect to air quality

                                    MANDATORY OUTPUT FORMAT (FOLLOW EXACTLY):
                                    You MUST structure your response using these EXACT section headers:

                                    **Current Situation:**
                                    [Write 1-2 sentences summarizing the air quality and weather conditions using the provided data]

                                    **Your Risk Level:**
                                    [State the risk level (Low/Moderate/High/Severe) specific to the user's profile]

                                    **Recommended Actions:**
                                    - [Action 1 with emoji - be specific and practical]
                                    - [Action 2 with emoji - be specific and practical]
                                    - [Action 3 with emoji - be specific and practical]
                                    - [Action 4 with emoji - be specific and practical]

                                    DO NOT add any additional sections after Recommended Actions. DO NOT add a disclaimer section.

                                    EXAMPLE RESPONSE:
                                    **Current Situation:**
                                    Air quality is poor with AQI at 180 (Unhealthy for Sensitive Groups). Temperature is 28°C with moderate humidity at 65%.

                                    **Your Risk Level:**
                                    High Risk - As someone with asthma, you are particularly vulnerable to current pollution levels.

                                    **Recommended Actions:**
                                    - 🏠 Stay indoors with windows closed and use air purifiers if available
                                    - 😷 Wear a properly fitted N95 mask if you must go outside
                                    - 💊 Keep your rescue inhaler with you at all times
                                    - 📞 Contact your doctor if you experience any breathing difficulties"""

            # Build detailed profile context
            profile_context = self._build_profile_context(
                age_category, health_category, gender, profile_label
            )
            
            # Build weather context
            weather_context = ""
            if weather_data:
                temp = weather_data.get('temperature', 0)
                humidity = weather_data.get('humidity', 0)
                wind = weather_data.get('windSpeed', 0)
                weather_context = f"""
                                        **Current Weather:**
                                        - Temperature: {temp:.1f}Â°C
                                        - Humidity: {humidity:.1f}%
                                        - Wind Speed: {wind:.1f} km/h
                                    """
            
            # Build AQI context
            aqi_context = ""
            if aqi_data:
                aqi_mid = aqi_data.get('aqi_mid', 0)
                category = aqi_data.get('category', 'Unknown')
                dominant = aqi_data.get('dominant_pollutant', 'Unknown')
                aqi_context = f"""
                                    **Current Air Quality:**
                                    - Location: {location}
                                    - AQI: {aqi_mid:.0f} ({category})
                                    - Dominant Pollutant: {dominant}
                                """
            
            # Construct full prompt
            full_prompt = f"""{system_instruction}

                                {aqi_context}
                                {weather_context}
                                {profile_context}

                                **User Question:** {message}

                                **Instructions:**
                                1. Analyze the current air quality for this specific user profile
                                2. Assess the health risk level (Low/Moderate/High/Severe)
                                3. Provide 3-4 specific, actionable recommendations
                                4. Consider the weather conditions in your advice
                                5. End with: "âš ï¸ **Important:** This is not medical advice. Please consult a healthcare professional for diagnosis or treatment."

                                **Your Response:**
                            """
            
            logger.info(f"Sending request to Gemini for: {location}")
            response = self.model.generate_content(full_prompt)
            
            if response and response.text:
                logger.info("âœ“ Received response from Gemini")
                
                # Validate response
                validation = response_validator.validate_response(
                    response.text, context
                )
                
                final_response = validation['modified_response']
                
                if not validation['valid']:
                    logger.warning(f"Response validation warnings: {validation['warnings']}")
                
                return {
                    'response': final_response,
                    'updated_profile': None,
                    'validation': validation
                }
            else:
                logger.warning("Empty response from Gemini")
                return {
                    'response': self._static_response(context, user_profile),
                    'updated_profile': None,
                    'validation': {'valid': True, 'warnings': ['Used fallback response']}
                }
                
        except Exception as e:
            logger.error(f"âš  Gemini error: {str(e)}")
            logger.exception(e)
            return {
                    'response': self._static_response(context, user_profile),
                    'updated_profile': None,
                    'validation': {'valid': True, 'warnings': [f'Error: {str(e)}, used fallback']}
            }
    
    def _build_profile_context(self, age_category, health_category, gender, profile_label):
        """Build detailed profile context for prompt"""
        profile_context = ""
        profile_parts = []
        
        if age_category or health_category or gender:
            age_guidance = {
                'child': 'Children have developing lungs and breathe more air relative to body weight. They are extra sensitive to pollution and should avoid outdoor activities during poor air quality.',
                'teenager': 'Teenagers are highly active and breathe more air during sports/activities. They should monitor for symptoms like coughing or shortness of breath.',
                'adult': 'Adults should take standard precautions based on AQI levels and limit outdoor exertion during poor air quality.',
                'elderly': 'Elderly persons have weaker immune systems and may have existing health conditions. They are a high-risk group requiring extra caution.'
            }
            
            health_guidance = {
                'asthma': 'Asthma patients are extremely sensitive to air pollution. Even low pollution levels can trigger asthma attacks. Keep rescue inhaler readily available at all times.',
                'heart_condition': 'People with heart conditions are at high risk from air pollution, which can trigger cardiac events. Avoid any physical exertion during poor air quality.',
                'respiratory': 'Those with respiratory issues are highly vulnerable to air pollution. Avoid outdoor exposure during poor AQI and use air purifiers indoors.',
                'copd': 'COPD patients must exercise extreme caution. Air pollution can cause severe exacerbations. Stay indoors during poor air quality and keep medications handy.',
                'diabetes': 'People with diabetes may experience increased inflammation from pollution. Monitor blood sugar levels closely and minimize outdoor exposure during poor AQI.',
                'pregnant': 'Pregnancy requires special care as pollution affects both mother and baby. Minimize outdoor exposure and ensure good indoor air quality.'
            }
            
            if age_category:
                profile_parts.append(age_guidance.get(age_category, ''))
            
            if health_category:
                profile_parts.append(health_guidance.get(health_category, ''))
            
            if profile_parts:
                profile_context = f"\n\n**IMPORTANT - User Profile is {profile_label}:**\n"
                profile_context += "\n".join(f"- {part}" for part in profile_parts if part)
                profile_context += "\n\n**Tailor your advice specifically for this high-risk profile with extra precautions.**"
        
        return profile_context
    
    def _static_response(self, context, user_profile=None):
        """Enhanced static response with better structure"""
        category = context.get('aqi_data', {}).get('category', 'Unknown')
        aqi_mid = context.get('aqi_data', {}).get('aqi_mid', 0)
        weather = context.get('weather', {})
        
        # Base responses by AQI category with structured format
        base_responses = {
            'Good': {
                'situation': f"Air quality is excellent with AQI {aqi_mid:.0f}! This is perfect weather for outdoor activities.",
                'risk': 'Low Risk - Safe for everyone',
                'actions': [
                    "âœ“ Enjoy outdoor activities and exercise freely",
                    "âœ“ Open windows to ventilate your home",
                    "âœ“ Great time for children to play outside",
                    "âœ“ No special precautions needed"
                ]
            },
            'Satisfactory': {
                'situation': f"Air quality is acceptable with AQI {aqi_mid:.0f}. Generally safe for most people.",
                'risk': 'Low to Moderate Risk',
                'actions': [
                    "âœ“ Outdoor activities are generally safe",
                    "âœ“ Sensitive individuals should monitor for symptoms",
                    "âœ“ Consider lighter exercises if you're sensitive",
                    "âœ“ Keep windows open for ventilation"
                ]
            },
            'Moderate': {
                'situation': f"Air quality is moderate with AQI {aqi_mid:.0f}. Sensitive groups should be cautious.",
                'risk': 'Moderate Risk - Caution for sensitive groups',
                'actions': [
                    "âš  Limit prolonged outdoor activities for sensitive groups",
                    "âš  Consider wearing masks for extended outdoor exposure",
                    "âš  Children and elderly should reduce outdoor time",
                    "âš  Use air purifiers indoors if available"
                ]
            },
            'Poor': {
                'situation': f"Poor air quality with AQI {aqi_mid:.0f}. Health effects possible for general public.",
                'risk': 'High Risk - Limit outdoor exposure',
                'actions': [
                    "ðŸš¨ Avoid prolonged outdoor activities",
                    "ðŸš¨ Wear N95 masks when going outside",
                    "ðŸš¨ Keep windows closed and use air purifiers",
                    "ðŸš¨ Those with respiratory conditions: stay indoors and keep medications ready"
                ]
            },
            'Very_Poor': {
                'situation': f"Very poor air quality with AQI {aqi_mid:.0f}! Serious health effects for everyone.",
                'risk': 'Severe Risk - Stay indoors',
                'actions': [
                    "ðŸ›‘ Stay indoors with windows closed",
                    "ðŸ›‘ Use N95 masks if you must go outside",
                    "ðŸ›‘ Run air purifiers on high setting",
                    "ðŸ›‘ High-risk individuals: avoid all outdoor exposure and monitor health closely"
                ]
            },
            'Severe': {
                'situation': f"SEVERE air quality emergency with AQI {aqi_mid:.0f}! Immediate health danger.",
                'risk': 'EXTREME RISK - Do not go outside',
                'actions': [
                    "ðŸ”´ DO NOT go outside under any circumstances",
                    "ðŸ”´ Seal windows and doors, use air purifiers",
                    "ðŸ”´ High-risk groups: Have emergency medications ready",
                    "ðŸ”´ If experiencing breathing difficulties, seek immediate medical help"
                ]
            }
        }
        
        response_data = base_responses.get(category, base_responses['Severe'])
        
        # Build structured response
        response = f"""**Current Situation:**
{response_data['situation']}

**Your Risk Level:**
{response_data['risk']}

**Recommended Actions:**
{chr(10).join(response_data['actions'])}"""
        
        # Add weather context
        if weather and weather.get('temperature', 0) > 0:
            temp = weather.get('temperature', 0)
            humidity = weather.get('humidity', 0)
            response += f"\n\n**Weather:** Temperature {temp:.1f}Â°C, Humidity {humidity:.1f}%"
        
        # Add profile-specific advice
        if user_profile:
            age_category = user_profile.get('age_category', '')
            health_category = user_profile.get('health_category', '')
            
            profile_advice = []
            
            if health_category == 'asthma':
                profile_advice.append("ðŸ« **Asthma Alert:** Keep your rescue inhaler with you at all times. Avoid triggers and outdoor exposure.")
            elif health_category == 'heart_condition':
                profile_advice.append("â¤ï¸ **Heart Condition Alert:** Avoid any physical exertion. Rest indoors and monitor for chest pain or discomfort.")
            elif health_category == 'pregnant':
                profile_advice.append("ðŸ¤° **Pregnancy Alert:** Protect both you and your baby by staying indoors during poor air quality.")
            elif health_category == 'elderly':
                profile_advice.append("ðŸ‘´ **Elderly Alert:** Take extra precautions. Stay indoors and monitor your health closely.")
            
            if age_category == 'child':
                profile_advice.append("ðŸ‘¶ **Children's Alert:** Keep children indoors during poor air quality. Avoid outdoor play.")
            
            if profile_advice:
                response += "\n\n" + "\n".join(profile_advice)
        
        # Add medical disclaimer
        response += "\n\nâš ï¸ **Important:** This is not medical advice. Please consult a healthcare professional for diagnosis or treatment."
        
        return response

gemini_assistant = GeminiAssistant()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "AQI Dashboard API - Enhanced Version",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "AI-powered health advice",
            "Response validation",
            "Safety guardrails",
            "Multilingual support",
            "Personalized recommendations"
        ]
    }

@app.get("/api/regions")
async def get_regions():
    """Get all available regions"""
    try:
        return data_manager.get_regions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/locations/{region}")
async def get_locations(region: str):
    """Get locations for a specific region"""
    try:
        return data_manager.get_locations_by_region(region)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """Get AQI predictions for a location"""
    try:
        logger.info(f"Prediction request for: {request.location}")
        current, historical = data_manager.get_location_data(request.location)
        predictions = predict_all(current, historical, request.standard)
        return predictions
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with AI assistant with validation"""
    try:
        # Build context
        context = {
            'location': request.location,
            'aqi_data': request.aqi_data,
            'user_profile': request.user_profile,
            'weather': request.aqi_data.get('weather', {}) if request.aqi_data else {}
        }
        
        # Get response with validation
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
        "version": "2.0.0",
        "data_loaded": len(data_manager.data) > 0,
        "locations": len(data_manager.whitelist),
        "gemini_enabled": gemini_assistant.enabled,
        "gemini_model": gemini_assistant.model_name if gemini_assistant.enabled else "disabled",
        "features": {
            "response_validation": True,
            "safety_guardrails": True,
            "medical_disclaimer": True,
            "data_grounding": True
        }
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)