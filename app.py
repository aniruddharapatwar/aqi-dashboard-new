"""
AQI Dashboard - FastAPI Backend (UPDATED FOR NEW LSTM MODELS)
Complete REST API for air quality predictions and AI assistance
UPDATED: Works with new feature engineering and model artifacts
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import updated feature engineer and predictor
from feature_engineer import FeatureEngineer, FeatureAligner
from predictor import (
    ModelManager, LSTMPredictor, AQICalculator, HealthAdvisory,
    INDIAN_AQI_BREAKPOINTS, US_AQI_BREAKPOINTS
)

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
    SPATIAL_DATA_PATH = BASE_DIR / "data" / "matched_region_unique_proximity.csv"  # NEW
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
    description="Air Quality Index Prediction and Advisory System (UPDATED LSTM)",
    version="2.2.0"
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
        self.spatial_data = self.load_spatial_data()  # NEW
        self.model_manager = ModelManager(Config.MODEL_PATH)
        self.feature_engineer = FeatureEngineer()
        self.predictor = LSTMPredictor(
            model_manager=self.model_manager,
            feature_engineer=self.feature_engineer,
            spatial_data=self.spatial_data
        )
    
    def load_data(self):
        """Load inference data with mixed date format support"""
        try:
            if not os.path.exists(Config.DATA_PATH):
                raise FileNotFoundError(f"Data file not found: {Config.DATA_PATH}")
            
            logger.info(f"Loading data from: {Config.DATA_PATH}")
            df = pd.read_csv(Config.DATA_PATH)
            
            logger.info(f"Data columns: {list(df.columns)}")
            
            # Handle timestamp/date column
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
            else:
                raise ValueError("Data must have 'timestamp' or 'date' column")
            
            # Handle lng/lon
            if 'lng' in df.columns and 'lon' not in df.columns:
                df['lon'] = df['lng']
            if 'lat' not in df.columns or 'lon' not in df.columns:
                raise ValueError(f"Data must have 'lat' and 'lon' columns. Found: {list(df.columns)}")
            
            # Drop rows with missing coordinates
            initial_count = len(df)
            df = df.dropna(subset=['lat', 'lon', 'date'])
            dropped = initial_count - len(df)
            if dropped > 0:
                logger.warning(f"Dropped {dropped} rows with missing lat/lon/date")
            
            df = df.sort_values(['lat', 'lon', 'date'])
            logger.info(f"✓ Loaded {len(df)} valid data rows")
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"Unique coordinates: {len(df.groupby(['lat', 'lon']))}")
            
            # Log available pollutants
            pollutant_cols = ['PM25', 'PM10', 'NO2', 'OZONE', 'CO', 'SO2']
            available_pollutants = [col for col in pollutant_cols if col in df.columns]
            logger.info(f"Available pollutants: {available_pollutants}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            return pd.DataFrame(columns=['lat', 'lon', 'date'])
    
    def load_whitelist(self):
        """Load whitelist from region_wise_popular_places_from_inference.csv"""
        try:
            whitelist = {}
            
            if not os.path.exists(Config.WHITELIST_PATH):
                raise FileNotFoundError(f"Whitelist file not found: {Config.WHITELIST_PATH}")
            
            logger.info(f"Loading whitelist from: {Config.WHITELIST_PATH}")
            df = pd.read_csv(Config.WHITELIST_PATH)
            
            logger.info(f"Whitelist columns: {list(df.columns)}")
            
            required_cols = ['Region', 'Place', 'Latitude', 'Longitude']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in whitelist: {missing_cols}")
            
            for idx, row in df.iterrows():
                try:
                    place_name = str(row['Place']).strip()
                    region = str(row['Region']).strip()
                    lat = float(row['Latitude'])
                    lon = float(row['Longitude'])
                    
                    if not place_name or not region or pd.isna(lat) or pd.isna(lon):
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
            
            if len(whitelist) > 0:
                regions = {}
                for place, info in whitelist.items():
                    region = info['region']
                    if region not in regions:
                        regions[region] = []
                    regions[region].append(place)
                
                logger.info(f"✓ Regions found: {list(regions.keys())}")
                for region, places in list(regions.items())[:5]:
                    logger.info(f"  • {region}: {len(places)} locations")
            
            return whitelist
            
        except Exception as e:
            logger.error(f"Failed to load whitelist: {e}", exc_info=True)
            return {}
    
    def load_spatial_data(self):
        """Load spatial features data (NEW)"""
        try:
            if not os.path.exists(Config.SPATIAL_DATA_PATH):
                logger.warning(f"Spatial data file not found: {Config.SPATIAL_DATA_PATH}")
                logger.warning("Spatial features will use defaults")
                return None
            
            logger.info(f"Loading spatial data from: {Config.SPATIAL_DATA_PATH}")
            df = pd.read_csv(Config.SPATIAL_DATA_PATH)
            
            # Expected columns: location_id, traffic_density_score, industrial_proximity, etc.
            required_cols = ['location_id']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Spatial data missing required columns")
                return None
            
            logger.info(f"✓ Loaded spatial data for {len(df)} locations")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load spatial data: {e}")
            return None
    
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
        
        loc_data = loc_data.sort_values('date')
        
        # Add location_id for spatial feature lookup
        loc_data['location_id'] = f"{lat:.3f}_{lon:.3f}"
        
        # Return current (last row) and historical (up to 300 rows for sequence building)
        current = loc_data.iloc[[-1]].copy()
        historical = loc_data.tail(300).copy()
        
        return current, historical

# ============================================================================
# GLOBAL DATA MANAGER
# ============================================================================

data_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize data manager on startup"""
    global data_manager
    try:
        logger.info("Initializing Data Manager...")
        data_manager = DataManager()
        logger.info("✓ Data Manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Data Manager: {e}", exc_info=True)
        raise

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "version": "2.2.0",
        "description": "AQI Dashboard API (Updated LSTM Models)",
        "endpoints": {
            "/regions": "Get all regions",
            "/locations": "Get locations by region",
            "/predict": "Get AQI predictions",
            "/chat": "AI chat assistance"
        }
    }

@app.get("/regions")
async def get_regions():
    """Get all available regions"""
    try:
        regions = data_manager.get_regions()
        return {"regions": regions}
    except Exception as e:
        logger.error(f"Error getting regions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/locations")
async def get_locations(region: str):
    """Get locations for a specific region"""
    try:
        locations = data_manager.get_locations_by_region(region)
        return {"region": region, "locations": locations}
    except Exception as e:
        logger.error(f"Error getting locations for {region}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Get AQI predictions for a location
    
    Returns predictions for all pollutants (PM2.5, PM10, NO2, O3) 
    and all horizons (1h, 6h, 12h, 24h)
    """
    try:
        logger.info(f"Prediction request for: {request.location} (standard: {request.standard})")
        
        # Get location data
        current_data, historical_data = data_manager.get_location_data(request.location)
        
        logger.info(f"Using {len(historical_data)} historical rows")
        
        # Make predictions using new predictor (FIXED: pass both current and historical)
        result = data_manager.predictor.predict_location(
            current_data=current_data,
            historical_data=historical_data,
            location=request.location,
            standard=request.standard
        )
        
        # Format response
        response = {
            "location": request.location,
            "timestamp": str(result['timestamp']),
            "standard": request.standard,
            "predictions": result['predictions'],
            "overall_aqi": result['overall_aqi']
        }
        
        # Add health advisory
        primary_horizon = '1h'
        if primary_horizon in result['overall_aqi']:
            aqi_category = result['overall_aqi'][primary_horizon]['category']
            response['health_advisory'] = HealthAdvisory.get_advisory(aqi_category, request.standard)
        
        # Add colors
        if request.standard == "IN":
            response['colors'] = Config.AQI_COLORS_IN
        else:
            response['colors'] = Config.AQI_COLORS_US
        
        logger.info(f"✓ Prediction completed for {request.location}")
        
        return response
        
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    AI chat endpoint for health advisory and AQI questions
    Uses Gemini AI if available
    """
    try:
        if not GEMINI_AVAILABLE or not Config.GEMINI_API_KEY:
            return {
                "response": "AI chat is not configured. Please set GEMINI_API_KEY in environment variables.",
                "source": "error"
            }
        
        # Configure Gemini
        genai.configure(api_key=Config.GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        
        # Build context
        context = "You are an Air Quality Index (AQI) expert assistant. "
        context += "Provide helpful, accurate information about air quality, health impacts, and protective measures. "
        
        if request.location:
            context += f"User is asking about {request.location}. "
        
        if request.aqi_data:
            context += f"\nCurrent AQI data: {json.dumps(request.aqi_data, indent=2)}\n"
        
        if request.user_profile:
            context += f"\nUser profile: {json.dumps(request.user_profile, indent=2)}\n"
        
        prompt = context + f"\nUser question: {request.message}\n\nProvide a helpful, concise response:"
        
        # Generate response
        response = model.generate_content(prompt)
        
        return {
            "response": response.text,
            "source": "gemini-pro"
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        # Fallback response
        return {
            "response": "I'm having trouble connecting to the AI service. Please try again later.",
            "source": "error"
        }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        status = {
            "status": "healthy",
            "data_loaded": len(data_manager.data) > 0,
            "whitelist_loaded": len(data_manager.whitelist) > 0,
            "spatial_data_loaded": data_manager.spatial_data is not None,
            "data_rows": len(data_manager.data),
            "locations": len(data_manager.whitelist),
            "regions": len(data_manager.get_regions()),
            "gemini_available": GEMINI_AVAILABLE and bool(Config.GEMINI_API_KEY)
        }
        
        # Check if models can be loaded
        try:
            test_model = data_manager.model_manager.load_model('PM25', '1h')
            status["models_accessible"] = True
        except:
            status["models_accessible"] = False
        
        return status
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/info")
async def api_info():
    """Get API information"""
    return {
        "version": "2.2.0",
        "pollutants": ["PM25", "PM10", "NO2", "OZONE"],
        "horizons": ["1h", "6h", "12h", "24h"],
        "standards": ["IN", "US"],
        "features": [
            "Multi-pollutant predictions",
            "Multiple forecast horizons",
            "Indian and US AQI standards",
            "Health advisories",
            "AI chat assistance",
            "Location-based predictions"
        ],
        "model_info": {
            "type": "LSTM (Long Short-Term Memory)",
            "architecture": "Pure regression with sequence modeling",
            "features": "Enhanced with temporal, spatial, and rolling statistics"
        }
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": str(exc.detail) if hasattr(exc, 'detail') else "Resource not found",
            "path": request.url.path
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "path": request.url.path
        }
    )

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*80)
    logger.info("AQI DASHBOARD API - UPDATED LSTM MODELS")
    logger.info("="*80)
    logger.info(f"Model path: {Config.MODEL_PATH}")
    logger.info(f"Data path: {Config.DATA_PATH}")
    logger.info(f"Whitelist path: {Config.WHITELIST_PATH}")
    logger.info(f"Spatial data path: {Config.SPATIAL_DATA_PATH}")
    logger.info("="*80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
