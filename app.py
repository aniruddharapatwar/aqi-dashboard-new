"""
AQI Dashboard - FastAPI Backend (LSTM MODEL VERSION)
Complete REST API for air quality predictions using LSTM models
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import LSTM Predictor
from predictor import LSTMPredictor, ModelManager, AQICalculator, HealthAdvisory
from feature_engineer import FeatureEngineer

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
    description="Air Quality Index Prediction using LSTM Models",
    version="3.0.0-lstm"
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
        
        logger.info(f"✓ Found {len(loc_data)} data rows for {location_name}")
        
        loc_data = loc_data.sort_values('timestamp')
        
        return loc_data.iloc[[-1]].copy(), loc_data.tail(200).copy()

# Initialize data manager
data_manager = DataManager()

# ============================================================================
# LSTM PREDICTOR INITIALIZATION
# ============================================================================

logger.info("Initializing LSTM Predictor...")
model_manager = ModelManager(Config.MODEL_PATH)
feature_engineer = FeatureEngineer()
lstm_predictor = LSTMPredictor(
    model_manager=model_manager,
    feature_engineer=feature_engineer,
    spatial_data=None
)
logger.info("✓ LSTM Predictor initialized")

# ============================================================================
# GEMINI AI ASSISTANT
# ============================================================================

class GeminiAssistant:
    def __init__(self):
        self.enabled = False
        self.model = None
        self.model_name = None
        
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini not available - google.generativeai not installed")
            return
            
        if not Config.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not configured in environment")
            return
        
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            
            model_options = [
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-pro'
            ]
            
            for model_name in model_options:
                try:
                    logger.info(f"Attempting to initialize {model_name}...")
                    self.model = genai.GenerativeModel(model_name)
                    
                    test_response = self.model.generate_content("Say 'OK' if you're working")
                    if test_response and test_response.text:
                        self.model_name = model_name
                        self.enabled = True
                        logger.info(f"✓ Gemini initialized successfully with {model_name}")
                        return
                except Exception as e:
                    logger.warning(f"Failed to initialize {model_name}: {e}")
                    continue
            
            logger.error("All Gemini model options failed")
            self.enabled = False
                    
        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}", exc_info=True)
            self.enabled = False
    
    def get_response(self, message: str, context: Dict) -> Dict:
        """Get AI response with enhanced validation"""
        user_profile = context.get('user_profile', {})
        location = context.get('location', 'Unknown')
        aqi_data = context.get('aqi_data', {})
        weather_data = context.get('weather', {})
        
        if not self.enabled:
            logger.info("Gemini not enabled, using static response")
            return {
                'response': self._static_response(context, user_profile),
                'source': 'static',
                'model': 'none',
                'validation': {'valid': True, 'warnings': []}
            }
        
        try:
            aqi_mid = aqi_data.get('aqi_mid', 0)
            category = aqi_data.get('category', 'Unknown')
            dominant = aqi_data.get('dominant_pollutant', 'Unknown')
            
            temp = weather_data.get('temperature', 0)
            humidity = weather_data.get('humidity', 0)
            
            profile_label = user_profile.get('profile_label', 'general public')
            
            prompt = f"""You are an air quality health advisor for Delhi NCR, India.

Current Situation:
- Location: {location}
- AQI: {aqi_mid:.0f} ({category})
- Dominant Pollutant: {dominant}
- Temperature: {temp:.1f}°C
- Humidity: {humidity:.1f}%
- User Profile: {profile_label}

User Question: {message}

Provide a helpful, accurate response with:
1. Brief assessment of the air quality impact for this user profile
2. Health risk level (specific to their profile)
3. 3-4 specific recommended actions

CRITICAL RULES:
- Be factual and evidence-based
- Tailor advice to the user's profile ({profile_label})
- If user has health conditions, emphasize extra precautions
- Recommend N95 masks for AQI > 150
- Keep response under 250 words
- Use clear, actionable language

Format with sections: **Current Situation**, **Your Risk Level**, **Recommended Actions**"""
            
            logger.info(f"Sending request to {self.model_name}...")
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                logger.info(f"✓ Received response from {self.model_name}")
                
                # Validate response
                validation_result = self._validate_response(
                    response.text, 
                    aqi_mid, 
                    category, 
                    user_profile
                )
                
                return {
                    'response': response.text,
                    'source': 'gemini',
                    'model': self.model_name,
                    'validation': validation_result
                }
            else:
                logger.warning("Empty response from Gemini, using fallback")
                return {
                    'response': self._static_response(context, user_profile),
                    'source': 'static_fallback',
                    'model': self.model_name,
                    'validation': {'valid': True, 'warnings': []}
                }
                
        except Exception as e:
            logger.error(f"Gemini error: {e}", exc_info=True)
            return {
                'response': self._static_response(context, user_profile),
                'source': 'static_error',
                'error': str(e),
                'validation': {'valid': False, 'warnings': ['AI service error']}
            }
    
    def _validate_response(self, response_text: str, aqi: float, category: str, user_profile: Dict) -> Dict:
        """Validate AI response for safety and accuracy"""
        warnings = []
        
        # Check for dangerous advice patterns
        dangerous_patterns = [
            'safe to go outside',
            'no need for mask',
            'air quality is fine'
        ]
        
        if aqi > 150:
            response_lower = response_text.lower()
            for pattern in dangerous_patterns:
                if pattern in response_lower:
                    warnings.append(f"Response may downplay risks (AQI {aqi:.0f})")
                    break
        
        # Check if response addresses user profile
        profile_label = user_profile.get('profile_label', '')
        health_category = user_profile.get('health_category', '')
        
        if health_category and health_category not in response_text.lower():
            warnings.append("Response may not fully address user's health condition")
        
        # Check for mask recommendation at high AQI
        if aqi > 150 and 'mask' not in response_text.lower():
            warnings.append("Response should recommend masks for this AQI level")
        
        # Check response length
        if len(response_text) > 2000:
            warnings.append("Response is very long, may be hard to read")
        
        return {
            'valid': len(warnings) == 0,
            'warnings': warnings
        }
    
    def _static_response(self, context, user_profile=None):
        """Static fallback response"""
        category = context.get('aqi_data', {}).get('category', 'Unknown')
        aqi_mid = context.get('aqi_data', {}).get('aqi_mid', 0)
        
        profile_label = user_profile.get('profile_label', 'general public') if user_profile else 'general public'
        
        responses = {
            'Good': f"✓ Air quality is excellent with AQI {aqi_mid:.0f}! Safe for all outdoor activities, including for {profile_label}.",
            'Satisfactory': f"Air quality is acceptable (AQI {aqi_mid:.0f}). Generally safe for {profile_label}, but sensitive individuals should be cautious during prolonged outdoor activities.",
            'Moderate': f"⚠ Air quality is moderate (AQI {aqi_mid:.0f}). {profile_label.capitalize()} should limit prolonged outdoor exertion. Consider wearing masks during extended outdoor activities.",
            'Poor': f"🚨 Poor air quality (AQI {aqi_mid:.0f}). {profile_label.capitalize()} should limit outdoor exposure. Wear N95 masks if going outside. Keep windows closed.",
            'Very_Poor': f"🛑 Very poor air quality (AQI {aqi_mid:.0f})! {profile_label.capitalize()} should stay indoors. Use air purifiers. Avoid all outdoor activities. N95 masks essential if must go out.",
            'Severe': f"🔴 SEVERE air quality (AQI {aqi_mid:.0f})! URGENT: {profile_label.capitalize()} must stay indoors. Close all windows. Use air purifiers. Avoid all outdoor activities. Emergency health risk."
        }
        
        return responses.get(category, f"Air quality status: {category} (AQI {aqi_mid:.0f}). Please consult local health advisories for {profile_label}.")

gemini_assistant = GeminiAssistant()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "AQI Dashboard API - LSTM Model Version",
        "version": "3.0.0-lstm",
        "status": "running",
        "model_type": "LSTM Deep Learning Models"
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
    """Get AQI predictions for a location using LSTM models"""
    try:
        logger.info(f"=== Prediction request for: {request.location} ===")
        current, historical = data_manager.get_location_data(request.location)
        
        # Use LSTM predictor
        predictions_dict = lstm_predictor.predict_location(
            current_data=current,
            historical_data=historical,
            location=request.location,
            standard=request.standard
        )
        
        # Extract weather data
        weather_data = {}
        if len(current) > 0:
            row = current.iloc[0]
            weather_features = ['temperature', 'humidity', 'dewPoint', 'apparentTemperature',
                              'precipIntensity', 'pressure', 'surfacePressure',
                              'cloudCover', 'windSpeed', 'windBearing', 'windGust']
            
            for feature in weather_features:
                if feature in row.index and pd.notna(row[feature]):
                    value = float(row[feature])
                    # Convert Fahrenheit to Celsius for temperature fields
                    if feature in ['temperature', 'dewPoint', 'apparentTemperature'] and value > 50:
                        value = (value - 32) * 5 / 9
                    weather_data[feature] = value
                else:
                    weather_data[feature] = 0.0
        
        # Add historical AQI data
        historical_aqi = []
        if len(historical) > 0 and 'timestamp' in historical.columns:
            historical_subset = historical.tail(48).copy()
            
            for _, row in historical_subset.iterrows():
                aqi_value = 0
                timestamp = row.get('timestamp', '')
                
                if 'AQI' in row.index and pd.notna(row['AQI']):
                    aqi_value = float(row['AQI'])
                else:
                    # Calculate from PM2.5 if available
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
        
        # Format response to match frontend expectations
        result = {
            'weather': weather_data,
            'historical': historical_aqi,
            'PM25': {},
            'PM10': {},
            'NO2': {},
            'OZONE': {},
            'overall': {}
        }
        
        # Restructure predictions to match frontend format
        for pollutant in ['PM25', 'PM10', 'NO2', 'OZONE']:
            if pollutant in predictions_dict['predictions']:
                for horizon in ['1h', '6h', '12h', '24h']:
                    if horizon in predictions_dict['predictions'][pollutant]:
                        pred = predictions_dict['predictions'][pollutant][horizon]
                        result[pollutant][horizon] = {
                            'category': pred['category'],
                            'confidence': pred.get('confidence', pred.get('sub_index', 0) / 500),
                            'aqi_min': pred.get('aqi_min', pred.get('sub_index', 0) - 25),
                            'aqi_max': pred.get('aqi_max', pred.get('sub_index', 0) + 25),
                            'aqi_mid': pred.get('aqi_mid', pred.get('sub_index', 0)),
                            'concentration_range': (pred.get('value', 0) * 0.9, pred.get('value', 0) * 1.1),
                            'value': pred.get('value', 0)
                        }
        
        # Overall AQI per horizon
        if 'overall_aqi' in predictions_dict:
            for horizon, overall in predictions_dict['overall_aqi'].items():
                result['overall'][horizon] = {
                    'aqi_min': overall.get('aqi', 0) - 25,
                    'aqi_max': overall.get('aqi', 0) + 25,
                    'aqi_mid': overall.get('aqi', 0),
                    'category': overall.get('category', 'Unknown'),
                    'dominant_pollutant': overall.get('dominant_pollutant', 'Unknown'),
                    'confidence': 0.85
                }
        
        # Log summary
        for horizon in ['1h', '6h', '12h', '24h']:
            if horizon in result['overall']:
                overall = result['overall'][horizon]
                logger.info(f"{horizon} forecast: AQI={overall['aqi_mid']:.1f} ({overall['category']})")
        
        return result
        
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
        "version": "3.0.0-lstm",
        "data_loaded": len(data_manager.data) > 0,
        "locations": len(data_manager.whitelist),
        "gemini_enabled": gemini_assistant.enabled,
        "gemini_model": gemini_assistant.model_name if gemini_assistant.enabled else "disabled",
        "model_type": "LSTM Deep Learning",
        "model_path": str(Config.MODEL_PATH)
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("="*80)
    logger.info("Starting AQI Dashboard API (LSTM Version)")
    logger.info(f"Data file: {Config.DATA_PATH}")
    logger.info(f"Model path: {Config.MODEL_PATH}")
    logger.info(f"Gemini enabled: {gemini_assistant.enabled}")
    if gemini_assistant.enabled:
        logger.info(f"Using model: {gemini_assistant.model_name}")
    logger.info("="*80)
    uvicorn.run(app, host="0.0.0.0", port=8000)