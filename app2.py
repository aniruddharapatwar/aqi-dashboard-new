"""
AQI Dashboard - FastAPI Backend (FINAL PRODUCTION VERSION)
Complete REST API for air quality predictions using LSTM models
with Enhanced AI Assistant and Safety Validation
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

# Import Enhanced Gemini AI Assistant
try:
    from gemini_assistant import GeminiAssistant, create_gemini_assistant
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Enhanced Gemini Assistant not available")

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
    description="Air Quality Index Prediction using LSTM Models with Enhanced AI Assistant",
    version="3.1.0-production"
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
        """Load whitelist from CSV ONLY - no augmentation"""
        try:
            whitelist = {}
            
            # Load ONLY from the whitelist CSV
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
                logger.info(f"✓ Loaded {len(whitelist)} locations from whitelist CSV")
            else:
                logger.error(f"Whitelist CSV not found: {Config.WHITELIST_PATH}")
            
            if len(whitelist) == 0:
                logger.error("No locations loaded from whitelist!")
            
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
# ENHANCED GEMINI AI ASSISTANT INITIALIZATION
# ============================================================================

if GEMINI_AVAILABLE:
    logger.info("Initializing Enhanced Gemini AI Assistant...")
    gemini_assistant = create_gemini_assistant(api_key=Config.GEMINI_API_KEY)
    logger.info(f"✓ Gemini Assistant initialized (enabled: {gemini_assistant.enabled})")
else:
    logger.warning("Enhanced Gemini Assistant not available, using fallback")
    
    # Create basic fallback assistant
    class BasicAssistant:
        def __init__(self):
            self.enabled = False
            self.model_name = None
        
        def get_response(self, message: str, context: Dict) -> Dict:
            """Basic fallback response"""
            category = context.get('aqi_data', {}).get('category', 'Unknown')
            aqi_mid = context.get('aqi_data', {}).get('aqi_mid', 0)
            user_profile = context.get('user_profile', {})
            profile_label = user_profile.get('profile_label', 'general public')
            
            responses = {
                'Good': f"**Current Situation**\nAir quality is excellent (AQI {aqi_mid:.0f}).\n\n**Your Risk Level**\nLOW - Safe for all outdoor activities.\n\n**Recommended Actions**\n• Enjoy outdoor activities\n• No special precautions needed",
                'Satisfactory': f"**Current Situation**\nAir quality is acceptable (AQI {aqi_mid:.0f}).\n\n**Your Risk Level**\nLOW - Generally safe for {profile_label}.\n\n**Recommended Actions**\n• Outdoor activities generally safe\n• Monitor for any discomfort",
                'Moderate': f"**Current Situation**\nAir quality is moderate (AQI {aqi_mid:.0f}).\n\n**Your Risk Level**\nMODERATE - Limit prolonged outdoor activities.\n\n**Recommended Actions**\n• Reduce outdoor exertion\n• Consider wearing mask\n• Keep windows closed",
                'Poor': f"**Current Situation**\n🚨 Poor air quality (AQI {aqi_mid:.0f}).\n\n**Your Risk Level**\nHIGH - Significant health risk.\n\n**Recommended Actions**\n• Limit outdoor exposure\n• Wear N95 mask outside\n• Use air purifiers\n• Keep windows closed",
                'Very_Poor': f"**Current Situation**\n🛑 Very poor air quality (AQI {aqi_mid:.0f})!\n\n**Your Risk Level**\nVERY HIGH - Serious health risk.\n\n**Recommended Actions**\n• STAY INDOORS\n• Wear N95 mask if must go out\n• Run air purifiers continuously\n• Avoid all physical exertion",
                'Severe': f"**Current Situation**\n🔴 SEVERE air quality (AQI {aqi_mid:.0f})! HEALTH EMERGENCY.\n\n**Your Risk Level**\nCRITICAL - Life-threatening conditions.\n\n**Recommended Actions**\n• DO NOT GO OUTSIDE\n• Seal windows and doors\n• Multiple air purifiers essential\n• Seek medical advice if symptoms occur"
            }
            
            return {
                'response': responses.get(category, f"Air quality: {category} (AQI {aqi_mid:.0f})"),
                'source': 'static',
                'model': 'none',
                'validation': {'valid': True, 'warnings': []}
            }
    
    gemini_assistant = BasicAssistant()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "AQI Dashboard API - Production Version with Enhanced AI",
        "version": "3.1.0-production",
        "status": "running",
        "model_type": "LSTM Deep Learning Models",
        "ai_assistant": "Enhanced Gemini with Safety Validation" if gemini_assistant.enabled else "Basic Fallback"
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
    """Chat with Enhanced AI assistant with safety validation"""
    try:
        context = {
            'location': request.location,
            'aqi_data': request.aqi_data,
            'user_profile': request.user_profile,
            'weather': request.aqi_data.get('weather', {}) if request.aqi_data else {}
        }
        
        result = gemini_assistant.get_response(request.message, context)
        
        # Log validation warnings if any
        if result.get('validation', {}).get('warnings'):
            logger.warning(f"AI Response Validation Warnings: {result['validation']['warnings']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint with comprehensive system status"""
    return {
        "status": "healthy",
        "version": "3.1.0-production",
        "components": {
            "data_manager": {
                "status": "operational",
                "data_loaded": len(data_manager.data) > 0,
                "locations_count": len(data_manager.whitelist),
                "data_rows": len(data_manager.data)
            },
            "lstm_predictor": {
                "status": "operational",
                "model_type": "LSTM Deep Learning",
                "model_path": str(Config.MODEL_PATH),
                "pollutants": ["PM25", "PM10", "NO2", "OZONE"],
                "horizons": ["1h", "6h", "12h", "24h"]
            },
            "ai_assistant": {
                "status": "operational" if gemini_assistant.enabled else "fallback",
                "enabled": gemini_assistant.enabled,
                "model": gemini_assistant.model_name if gemini_assistant.enabled else "basic_fallback",
                "features": [
                    "Profile-aware responses",
                    "Safety validation",
                    "Risk assessment",
                    "Health condition specific advice"
                ] if gemini_assistant.enabled else ["Basic responses"]
            }
        },
        "api_endpoints": {
            "regions": "/api/regions",
            "locations": "/api/locations/{region}",
            "predict": "/api/predict",
            "chat": "/api/chat",
            "health": "/api/health"
        }
    }

@app.get("/api/assistant/info")
async def assistant_info():
    """Get detailed information about the AI assistant"""
    return {
        "enabled": gemini_assistant.enabled,
        "model": gemini_assistant.model_name if gemini_assistant.enabled else "basic_fallback",
        "type": "Enhanced Gemini with Safety Validation" if gemini_assistant.enabled else "Basic Fallback",
        "capabilities": {
            "profile_awareness": gemini_assistant.enabled,
            "safety_validation": gemini_assistant.enabled,
            "health_condition_specific": gemini_assistant.enabled,
            "risk_assessment": gemini_assistant.enabled,
            "structured_responses": True,
            "markdown_formatting": True
        },
        "supported_profiles": {
            "age_categories": ["child", "teenager", "adult", "elderly"],
            "health_conditions": ["asthma", "heart_condition", "respiratory", "copd", "diabetes", "pregnant"],
            "risk_levels": ["standard", "elevated", "high", "very_high", "critical", "emergency"]
        }
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("="*80)
    logger.info("Starting AQI Dashboard API - Production Version")
    logger.info(f"Data file: {Config.DATA_PATH}")
    logger.info(f"Model path: {Config.MODEL_PATH}")
    logger.info(f"AI Assistant: {'Enhanced Gemini' if gemini_assistant.enabled else 'Basic Fallback'}")
    if gemini_assistant.enabled:
        logger.info(f"Using model: {gemini_assistant.model_name}")
    logger.info("="*80)
    uvicorn.run(app, host="0.0.0.0", port=8000)