"""
AQI Dashboard - FastAPI Backend (COMPLETE FIXED VERSION)
Complete REST API for air quality predictions and AI assistance
FIXES: Gemini integration + Weather data extraction + dotenv loading
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

# 🔧 CRITICAL FIX: Load environment variables from .env file
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
    version="1.0.0"
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
# DATA MANAGER - FIXED VERSION
# ============================================================================

class DataManager:
    def __init__(self):
        self.data = self.load_data()  # Load data FIRST
        self.whitelist = self.load_whitelist()  # Then whitelist
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
            logger.info(f"✓ Loaded {len(df)} data rows")
            
            # Log available columns for debugging
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
                logger.info(f"✓ Loaded {len(whitelist)} locations from whitelist")
            
            # Add locations from actual data
            if len(self.data) > 0 and 'location' in self.data.columns:
                # Get unique locations with their coordinates
                location_groups = self.data.groupby(['lat', 'lon']).agg({
                    'location': 'first',
                    'pincode': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else '',
                    'region': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else 'Unknown'
                }).reset_index()
                
                added = 0
                for _, row in location_groups.iterrows():
                    loc_name = row['location']
                    if pd.notna(loc_name) and str(loc_name).strip():
                        # Use actual coordinates from data
                        whitelist[loc_name] = {
                            'region': row['region'] if pd.notna(row['region']) else 'Central Delhi',
                            'lat': row['lat'],
                            'lon': row['lon'],
                            'pin': row['pincode'] if pd.notna(row['pincode']) else '',
                            'area': loc_name
                        }
                        added += 1
                
                logger.info(f"✓ Added {added} locations from actual data")
            
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
        
        # Use exact coordinates from whitelist (which now matches data)
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
            # Even wider search
            mask = ((np.abs(self.data['lat'] - lat) < 0.05) & 
                   (np.abs(self.data['lon'] - lon) < 0.05))
            loc_data = self.data[mask].copy()
        
        if len(loc_data) == 0:
            raise ValueError(f"No data found for {location_name} at ({lat}, {lon})")
        
        logger.info(f"✓ Found {len(loc_data)} data rows for {location_name}")
        
        loc_data = loc_data.sort_values('timestamp')
        
        # Return most recent row as current, last 96 as historical
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
                # If value > 50, it's likely Fahrenheit (Delhi rarely exceeds 50°C)
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
    logger.info(f"Available columns: {list(current_data.columns)}")
    
    # FIXED: Extract weather data from current row
    weather_data = extract_weather_data(current_data)
    results['weather'] = weather_data
    
    # NEW: Add historical AQI data (last 48 hours)
    historical_aqi = []
    if len(historical_data) > 0 and 'timestamp' in historical_data.columns:
        # Get last 48 hours of data
        historical_subset = historical_data.tail(48).copy()
        
        for _, row in historical_subset.iterrows():
            # Calculate AQI from pollutant values if available
            aqi_value = 0
            timestamp = row.get('timestamp', '')
            
            # Use existing AQI if available
            if 'AQI' in row.index and pd.notna(row['AQI']):
                aqi_value = float(row['AQI'])
            else:
                # Calculate from pollutants (simple PM2.5 based)
                if 'PM25' in row.index and pd.notna(row['PM25']):
                    pm25 = float(row['PM25'])
                    # PM2.5 based AQI approximation
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
            
            if aqi_value > 0:  # Only add if we have valid AQI
                historical_aqi.append({
                    'timestamp': str(timestamp),
                    'aqi': round(aqi_value, 1)
                })
        
        logger.info(f"✓ Prepared {len(historical_aqi)} historical AQI data points")
    
    results['historical'] = historical_aqi
    
    for pollutant in ['PM25', 'PM10', 'NO2', 'OZONE']:
        results[pollutant] = {}
        
        # Check if pollutant data exists
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
                logger.info(f"✓ {pollutant} {horizon}: {category} ({confidence:.2%})")
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
# GEMINI AI ASSISTANT - FIXED VERSION
# ============================================================================

class GeminiAssistant:
    def __init__(self):
        self.enabled = False
        self.model = None
        self.model_name = None  # Track which model we're using
        
        if not Config.GEMINI_API_KEY:
            logger.warning("❌ GEMINI_API_KEY not found in environment")
            return
            
        if not GEMINI_AVAILABLE:
            logger.warning("❌ google-generativeai package not installed")
            return
        
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            
            # Try multiple model names in order of preference
            models_to_try = [
                'gemini-2.5-pro',
                'gemini-2.5-flash',
                'gemini-2.5-flash-lite'
            ]
            
            for model_name in models_to_try:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    # Test the model works
                    test_response = self.model.generate_content("Hello")
                    if test_response:
                        self.enabled = True
                        self.model_name = model_name  # Store the working model name
                        logger.info(f"✓ Gemini AI initialized with {model_name}")
                        break
                except Exception as e:
                    logger.debug(f"Model {model_name} not available: {e}")
                    continue
            
            if not self.enabled:
                logger.error("Failed to initialize any Gemini model")
                self.enabled = False
                    
        except Exception as e:
            logger.error(f"❌ Gemini initialization failed: {e}")
            self.enabled = False
    
    def get_response(self, message: str, context: Dict) -> Dict:
        user_profile = context.get('user_profile', {})
        location = context.get('location', 'Unknown')
        aqi_data = context.get('aqi_data', {})
        
        # FIXED: Add weather data to context
        weather_data = context.get('weather', {})
        
        if not self.enabled:
            logger.warning("Gemini not enabled, using static response")
            return {
                'response': self._static_response(context, user_profile),
                'updated_profile': None
            }
        
        try:
            # Extract user profile information
            age_category = user_profile.get('age_category', '')
            health_category = user_profile.get('health_category', '')
            gender = user_profile.get('gender', '')
            profile_label = user_profile.get('profile_label', 'general public')
            
            # Build profile context with detailed breakdown
            profile_context = ""
            profile_parts = []
            
            if age_category or health_category or gender:
                age_guidance = {
                    'child': 'Children have developing lungs and breathe more air relative to body weight. Extra sensitive to pollution.',
                    'teenager': 'Teenagers are active and breathe more air during sports/activities. Monitor for symptoms.',
                    'adult': 'Adults should take standard precautions based on AQI levels.',
                    'elderly': 'Elderly persons have weaker immune systems and may have existing conditions. High risk group.'
                }
                
                health_guidance = {
                    'asthma': 'Asthma patients extremely sensitive - even low pollution can trigger attacks. Keep rescue inhaler ready.',
                    'heart_condition': 'Heart patients at high risk - pollution can trigger cardiac events. Avoid exertion in poor air.',
                    'respiratory': 'Respiratory issues make them highly vulnerable. Avoid outdoor exposure during poor AQI.',
                    'copd': 'COPD patients must take extreme caution - pollution can cause severe exacerbations.',
                    'diabetes': 'Diabetics may have increased inflammation from pollution. Monitor blood sugar and avoid exposure.',
                    'pregnant': 'Pregnancy requires special care - pollution affects both mother and baby. Minimize exposure.'
                }
                
                gender_context = {
                    'female': 'Women may experience different health impacts from air pollution.',
                    'male': 'Men should be aware of cardiovascular risks from pollution exposure.'
                }
                
                if age_category:
                    profile_parts.append(age_guidance.get(age_category, ''))
                
                if health_category:
                    profile_parts.append(health_guidance.get(health_category, ''))
                
                if gender and gender in gender_context:
                    profile_parts.append(gender_context[gender])
                
                if profile_parts:
                    profile_context = f"\n\nIMPORTANT: User profile is {profile_label}. "
                    profile_context += " ".join(profile_parts)
                    profile_context += " Tailor your advice specifically for this profile."
            
            # FIXED: Add weather context
            weather_context = ""
            if weather_data:
                temp = weather_data.get('temperature', 0)
                humidity = weather_data.get('humidity', 0)
                wind = weather_data.get('windSpeed', 0)
                weather_context = f"\n\nWeather Conditions:\n- Temperature: {temp:.1f}Â°C\n- Humidity: {humidity:.1f}%\n- Wind Speed: {wind:.1f} km/h"
            
            prompt = f"""You are an expert AQI health advisor for Delhi NCR, providing personalized air quality advice.

Current Context:
- Location: {location}
- Current AQI: {aqi_data.get('aqi_mid', 'N/A')} ({aqi_data.get('category', 'Unknown')})
- Dominant Pollutant: {aqi_data.get('dominant_pollutant', 'Unknown')}
{weather_context}
{profile_context}

User Question: {message}

Instructions:
1. Provide personalized health advice based on the user's profile
2. Be specific about precautions for their situation
3. Consider weather conditions in your recommendations
4. Include actionable recommendations
5. Keep response 4-6 sentences
6. Use a warm, caring, but professional tone

Provide your response:"""
            
            logger.info(f"Sending request to Gemini for location: {location}")
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                logger.info("✓ Received response from Gemini")
                return {
                    'response': response.text,
                    'updated_profile': None
                }
            else:
                logger.warning("Empty response from Gemini")
                return {
                    'response': self._static_response(context, user_profile),
                    'updated_profile': None
                }
                
        except Exception as e:
            logger.error(f"❌ Gemini error: {str(e)}")
            logger.exception(e)  # Log full traceback
            return {
                'response': self._static_response(context, user_profile),
                'updated_profile': None
            }
    
    def _static_response(self, context, user_profile=None):
        category = context.get('aqi_data', {}).get('category', 'Unknown')
        weather = context.get('weather', {})
        
        # Base responses by AQI category
        base_responses = {
            'Good': "✅ Air quality is excellent! Safe for all outdoor activities.",
            'Satisfactory': "😊 Air quality is acceptable for most people.",
            'Moderate': "⚠️ Moderate air quality. Sensitive individuals should be cautious.",
            'Poor': "🚨 Poor air quality. Limit outdoor activities.",
            'Very_Poor': "⛔ Very poor air quality! Stay indoors.",
            'Severe': "🔴 SEVERE air quality! Do not go outside."
        }
        
        response = base_responses.get(category, "How can I help you with air quality information?")
        
        # Add weather context if available
        if weather and weather.get('temperature', 0) > 0:
            temp = weather.get('temperature', 0)
            humidity = weather.get('humidity', 0)
            response += f" Current temperature is {temp:.1f}Â°C with {humidity:.1f}% humidity."
        
        # Add profile-specific advice based on new structure
        if user_profile:
            age_category = user_profile.get('age_category', '')
            health_category = user_profile.get('health_category', '')
            
            age_advice = {
                'child': " Children should avoid outdoor play during poor air quality.",
                'elderly': " Elderly persons should take extra precautions and stay indoors.",
                'teenager': " Teenagers should avoid intensive outdoor sports during poor AQI."
            }
            
            health_advice = {
                'pregnant': " Pregnant women should minimize outdoor exposure to protect the baby.",
                'asthma': " Asthma patients should keep rescue inhalers ready and avoid triggers.",
                'heart_condition': " Heart patients should avoid physical exertion outdoors.",
                'respiratory': " Those with respiratory issues should use air purifiers indoors.",
                'copd': " COPD patients must stay indoors and use supplemental oxygen if needed.",
                'diabetes': " Diabetics should monitor blood sugar and limit outdoor exposure."
            }
            
            if age_category in age_advice:
                response += age_advice[age_category]
            
            if health_category in health_advice:
                response += health_advice[health_category]
        
        return response

gemini_assistant = GeminiAssistant()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "AQI Dashboard API",
        "version": "1.0.0",
        "status": "running"
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
    """Chat with AI assistant"""
    try:
        # FIXED: Include weather data in context
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
        "data_loaded": len(data_manager.data) > 0,
        "locations": len(data_manager.whitelist),
        "gemini_enabled": gemini_assistant.enabled,
        "gemini_model": gemini_assistant.model_name if gemini_assistant.enabled else "disabled"
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)