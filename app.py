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

"""
Enhanced Gemini AI Assistant with Safety Validation
Production-ready version with comprehensive error handling and response validation
"""

import logging
from typing import Dict, List
import re

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available")


class GeminiAssistant:
    """
    Enhanced AI Assistant with:
    - User profile-aware responses
    - Safety validation
    - Structured output formatting
    - Comprehensive error handling
    """
    
    def __init__(self, api_key: str = None):
        self.enabled = False
        self.model = None
        self.model_name = None
        self.api_key = api_key
        
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini not available - google.generativeai not installed")
            return
            
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not configured")
            return
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Gemini model with fallback options"""
        try:
            genai.configure(api_key=self.api_key)
            
            model_options = [
                            "gemini-2.5-pro",
                            "gemini-2.5-flash",
                            "gemini-2.5-flash-lite",
                            "gemini-2.5-flash-image",
                            "gemini-2.0-flash",
                            "gemini-2.0-flash-lite"
                            ]

            
            for model_name in model_options:
                try:
                    logger.info(f"Attempting to initialize {model_name}...")
                    self.model = genai.GenerativeModel(
                        model_name,
                        generation_config={
                            'temperature': 0.7,
                            'top_p': 0.8,
                            'top_k': 40,
                            'max_output_tokens': 800,
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
                            }
                        ]
                    )
                    
                    # Test the model
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
        """
        Get AI response with comprehensive validation
        
        Args:
            message: User's question/message
            context: Dictionary containing:
                - location: Current location
                - aqi_data: AQI information
                - user_profile: User's health profile
                - weather: Weather data
        
        Returns:
            Dictionary with response, validation, and metadata
        """
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
            # Extract data from context
            aqi_mid = aqi_data.get('aqi_mid', 0)
            category = aqi_data.get('category', 'Unknown')
            dominant = aqi_data.get('dominant_pollutant', 'Unknown')
            
            temp = weather_data.get('temperature', 0)
            humidity = weather_data.get('humidity', 0)
            wind_speed = weather_data.get('windSpeed', 0)
            
            # Build profile description
            profile_label = user_profile.get('profile_label', 'general public')
            age_category = user_profile.get('age_category', '')
            health_category = user_profile.get('health_category', '')
            
            # Construct enhanced prompt
            prompt = self._build_enhanced_prompt(
                message=message,
                location=location,
                aqi_mid=aqi_mid,
                category=category,
                dominant=dominant,
                temp=temp,
                humidity=humidity,
                wind_speed=wind_speed,
                profile_label=profile_label,
                age_category=age_category,
                health_category=health_category
            )
            
            logger.info(f"Sending request to {self.model_name}...")
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                logger.info(f"✓ Received response from {self.model_name}")
                
                # Validate response for safety and accuracy
                validation_result = self._validate_response(
                    response.text, 
                    aqi_mid, 
                    category, 
                    user_profile
                )
                
                # Clean up response
                cleaned_response = self._clean_response(response.text)
                
                return {
                    'response': cleaned_response,
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
                    'validation': {'valid': True, 'warnings': ['Empty AI response']}
                }
                
        except Exception as e:
            logger.error(f"Gemini error: {e}", exc_info=True)
            return {
                'response': self._static_response(context, user_profile),
                'source': 'static_error',
                'error': str(e),
                'validation': {'valid': False, 'warnings': ['AI service error']}
            }
    
    def _build_enhanced_prompt(self, message: str, location: str, aqi_mid: float,
                               category: str, dominant: str, temp: float, 
                               humidity: float, wind_speed: float, profile_label: str,
                               age_category: str, health_category: str) -> str:
        """Build comprehensive prompt with all context"""
        
        # Risk level based on profile
        risk_level = "STANDARD"
        if health_category in ['asthma', 'copd', 'respiratory', 'heart_condition']:
            risk_level = "HIGH RISK"
        elif age_category in ['child', 'elderly']:
            risk_level = "ELEVATED RISK"
        elif health_category == 'pregnant':
            risk_level = "HIGH RISK"
        
        # Weather impact
        weather_impact = ""
        if aqi_mid > 100:
            if humidity > 70:
                weather_impact = "High humidity may increase pollutant adhesion and respiratory effects. "
            if wind_speed < 5:
                weather_impact += "Low wind speeds are preventing pollutant dispersion. "
            elif wind_speed > 20:
                weather_impact += "High winds may resuspend particulate matter. "
        
        prompt = f"""You are Dr. AQI, an expert air quality health advisor for Delhi NCR, India. You provide evidence-based, personalized health recommendations.

CURRENT AIR QUALITY SITUATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Location: {location}
AQI Level: {aqi_mid:.0f} ({category})
Dominant Pollutant: {dominant}
Weather: {temp:.1f}°C, {humidity:.0f}% humidity, {wind_speed:.0f} km/h wind
{weather_impact}

USER PROFILE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Category: {profile_label}
Risk Level: {risk_level}
Age Group: {age_category if age_category else 'Not specified'}
Health Condition: {health_category if health_category else 'None specified'}

USER QUESTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{message}

RESPONSE GUIDELINES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Address the user's specific question directly
2. Tailor advice to their risk level and health profile
3. Be specific and actionable - give concrete steps
4. Use clear, non-technical language
5. Format response with these sections:

**Current Situation & Risk**
- Brief assessment of air quality impact for THIS user
- Specific health risks for their profile

**Your Risk Level**
- Why their risk level is {risk_level}
- What symptoms to watch for

**Recommended Actions**
- 3-4 specific, actionable recommendations
- Prioritize most important actions first
- Include both short-term and preventive measures

CRITICAL SAFETY RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ For AQI > 150: ALWAYS recommend N95 masks, limiting outdoor exposure
✓ For high-risk users: Emphasize staying indoors, using air purifiers
✓ For AQI > 200: Stress URGENT need to avoid outdoor activities
✓ Never downplay serious health risks
✓ Be factual - don't speculate or give unproven advice
✓ Keep response under 250 words
✓ Use markdown formatting (**bold** for emphasis, bullet points for lists)

Provide your response now:"""
        
        return prompt
    
    def _validate_response(self, response_text: str, aqi: float, 
                          category: str, user_profile: Dict) -> Dict:
        """
        Comprehensive response validation for safety and accuracy
        
        Returns:
            Dict with 'valid' (bool) and 'warnings' (list of str)
        """
        warnings = []
        
        # Check for dangerous advice patterns at high AQI
        if aqi > 150:
            dangerous_patterns = [
                (r'\b(safe|fine|okay|ok)\s+to\s+go\s+outside\b', 'Response suggests outdoor activity is safe at dangerous AQI'),
                (r'\bno\s+need\s+(for|to\s+wear)\s+mask\b', 'Response dismisses mask use at high AQI'),
                (r'\bair\s+quality\s+is\s+(fine|good|acceptable)\b', 'Response understates dangerous air quality'),
                (r'\bdon\'?t\s+worry\b', 'Response may be too reassuring for dangerous conditions'),
            ]
            
            response_lower = response_text.lower()
            for pattern, warning in dangerous_patterns:
                if re.search(pattern, response_lower):
                    warnings.append(warning)
        
        # Check for appropriate mask recommendations
        if aqi > 150 and 'mask' not in response_text.lower():
            warnings.append(f"Response should recommend masks (AQI {aqi:.0f})")
        
        if aqi > 150 and 'n95' not in response_text.lower():
            warnings.append("Response should specify N95 masks for this AQI level")
        
        # Check if response addresses user's health condition
        health_category = user_profile.get('health_category', '')
        if health_category and aqi > 100:
            health_keywords = {
                'asthma': ['asthma', 'inhaler', 'breathing', 'respiratory'],
                'heart_condition': ['heart', 'cardiovascular', 'chest'],
                'copd': ['copd', 'lung', 'breathing'],
                'pregnant': ['pregnancy', 'baby', 'fetus'],
                'diabetes': ['diabetes', 'blood sugar'],
                'respiratory': ['respiratory', 'breathing', 'lungs']
            }
            
            keywords = health_keywords.get(health_category, [])
            if keywords and not any(kw in response_text.lower() for kw in keywords):
                warnings.append(f"Response may not address user's {health_category} condition")
        
        # Check for overly long responses
        if len(response_text) > 2500:
            warnings.append("Response is very long (>2500 chars)")
        
        # Check for appropriate urgency at severe AQI
        if aqi > 300:
            urgency_indicators = ['urgent', 'emergency', 'critical', 'immediately', 'serious']
            if not any(word in response_text.lower() for word in urgency_indicators):
                warnings.append(f"Response should convey urgency at AQI {aqi:.0f}")
        
        # Check that response uses proper formatting
        if '**' not in response_text:
            warnings.append("Response lacks formatting (should use **bold**)")
        
        return {
            'valid': len(warnings) == 0,
            'warnings': warnings,
            'severity': 'high' if len(warnings) > 3 else 'medium' if len(warnings) > 0 else 'none'
        }
    
    def _clean_response(self, response_text: str) -> str:
        """Clean and format response text"""
        # Remove any disclaimers
        response_text = re.sub(r'(?i)(disclaimer|note):\s*.{0,200}', '', response_text)
        
        # Ensure proper spacing around markdown
        response_text = re.sub(r'\*\*([^\*]+)\*\*', r'**\1**', response_text)
        
        # Remove excessive newlines
        response_text = re.sub(r'\n{3,}', '\n\n', response_text)
        
        # Trim whitespace
        response_text = response_text.strip()
        
        return response_text
    
    def _static_response(self, context: Dict, user_profile: Dict = None) -> str:
        """
        Enhanced static fallback responses with user profile awareness
        """
        category = context.get('aqi_data', {}).get('category', 'Unknown')
        aqi_mid = context.get('aqi_data', {}).get('aqi_mid', 0)
        dominant = context.get('aqi_data', {}).get('dominant_pollutant', 'PM2.5')
        
        profile_label = 'general public'
        age_category = ''
        health_category = ''
        
        if user_profile:
            profile_label = user_profile.get('profile_label', 'general public')
            age_category = user_profile.get('age_category', '')
            health_category = user_profile.get('health_category', '')
        
        # Determine risk level
        high_risk = health_category in ['asthma', 'copd', 'respiratory', 'heart_condition', 'pregnant']
        elevated_risk = age_category in ['child', 'elderly']
        
        # Build profile-specific advice
        profile_advice = ""
        if high_risk:
            profile_advice = f" As someone with {health_category}, you are at higher risk and should take extra precautions."
        elif elevated_risk:
            profile_advice = f" As a {age_category}, you should be more cautious than the general population."
        
        responses = {
            'Good': f"""**Current Situation & Risk**
                    Air quality is excellent (AQI {aqi_mid:.0f}). {dominant} levels are within safe limits.{profile_advice}

                    **Your Risk Level**
                    LOW - Safe for all outdoor activities and exercise.

                    **Recommended Actions**
                    • Enjoy outdoor activities without restrictions
                    • Good time for exercise and recreation
                    • Open windows to ventilate indoor spaces
                    • No special precautions needed""",
                                
                                'Satisfactory': f"""**Current Situation & Risk**
                    Air quality is acceptable (AQI {aqi_mid:.0f}). {dominant} levels are slightly elevated.{profile_advice}

                    **Your Risk Level**
                    {"MODERATE" if high_risk or elevated_risk else "LOW"} - Generally safe, but sensitive individuals should monitor symptoms.

                    **Recommended Actions**
                    • Outdoor activities generally safe
                    {"• Limit prolonged outdoor exertion" if high_risk or elevated_risk else "• Monitor for any breathing discomfort"}
                    {"• Keep rescue inhaler accessible" if health_category == 'asthma' else ""}
                    • Stay hydrated during outdoor activities""",
                                
                                'Moderate': f"""**Current Situation & Risk**
                    Air quality is moderate (AQI {aqi_mid:.0f}). {dominant} pollution is affecting air quality.{profile_advice}

                    **Your Risk Level**
                    {"HIGH" if high_risk else "ELEVATED" if elevated_risk else "MODERATE"} - {"Avoid prolonged outdoor activities" if high_risk else "Limit extended outdoor exposure"}

                    **Recommended Actions**
                    • {"Stay indoors as much as possible" if high_risk else "Reduce prolonged outdoor activities"}
                    • Wear N95 mask if going outside
                    {"• Use air purifier indoors" if high_risk else "• Keep windows closed during peak pollution hours"}
                    • Monitor symptoms (coughing, breathing difficulty)
                    {"• Have medications readily available" if health_category in ['asthma', 'copd'] else ""}""",
                                
                                'Poor': f"""**Current Situation & Risk**
                    🚨 Poor air quality (AQI {aqi_mid:.0f}). {dominant} pollution is at unhealthy levels.{profile_advice}

                    **Your Risk Level**
                    {"VERY HIGH" if high_risk else "HIGH"} - Significant health risk for {"everyone, especially those with " + health_category if high_risk else "all groups"}

                    **Recommended Actions**
                    • {"STAY INDOORS - Do not go outside" if high_risk else "Limit outdoor exposure to essential activities only"}
                    • WEAR N95 MASK when going outside
                    • Use air purifiers with HEPA filters indoors
                    • Keep all windows and doors closed
                    {"• Monitor symptoms closely - seek medical help if needed" if high_risk else "• Avoid physical exertion"}
                    {"• Keep emergency medications accessible" if health_category in ['asthma', 'copd', 'heart_condition'] else ""}""",
                                
                                'Very_Poor': f"""**Current Situation & Risk**
                    🛑 VERY POOR air quality (AQI {aqi_mid:.0f})! {dominant} pollution at hazardous levels.{profile_advice}

                    **Your Risk Level**
                    {"CRITICAL" if high_risk else "VERY HIGH"} - Emergency health risk for {"vulnerable individuals - medical attention may be needed" if high_risk else "all populations"}

                    **Recommended Actions**
                    • 🚨 STAY INDOORS - Essential trips only
                    • ALWAYS wear N95 mask outdoors
                    • Run air purifiers continuously with HEPA filters
                    • Seal windows/doors - use damp cloths under door gaps
                    • {"URGENT: Monitor symptoms - call doctor if breathing worsens" if high_risk else "Avoid ALL physical exertion"}
                    {"• Have emergency plan ready - know nearest hospital" if high_risk else ""}
                    {"• Ensure adequate medication supply" if health_category in ['asthma', 'copd'] else ""}""",
                                
                                'Severe': f"""**Current Situation & Risk**
                    🔴 SEVERE air quality (AQI {aqi_mid:.0f})! HEALTH EMERGENCY - {dominant} at dangerous levels.{profile_advice}

                    **Your Risk Level**
                    {"EMERGENCY" if high_risk else "CRITICAL"} - Life-threatening conditions for {"high-risk individuals" if high_risk else "all populations"}

                    **Recommended Actions**
                    • 🚨 EMERGENCY: DO NOT GO OUTSIDE
                    • {"SEEK MEDICAL ADVICE IMMEDIATELY if experiencing symptoms" if high_risk else "Stay in sealed indoor environment"}
                    • Multiple air purifiers - HEPA filters essential
                    • Create "clean room" - seal one room completely
                    • N95 masks required even for brief outdoor exposure
                    {"• URGENT: Have emergency medical contact ready" if high_risk else ""}
                    {"• Monitor vital signs - oxygen levels if available" if health_category in ['asthma', 'copd', 'heart_condition'] else ""}
                    • {"Contact healthcare provider for guidance" if high_risk else "No outdoor activities under any circumstances"}"""
                            }
                            
        default_response = f"""**Current Situation & Risk**
                    Air quality status: {category} (AQI {aqi_mid:.0f})
                    {dominant} is the dominant pollutant.{profile_advice}

                    **Your Risk Level**
                    Please consult local health advisories for {profile_label}.

                    **Recommended Actions**
                    • Monitor air quality updates regularly
                    • Follow local health authority guidelines
                    • {"Take extra precautions due to health condition" if high_risk else "Limit outdoor exposure if possible"}
                    • Wear mask when air quality deteriorates"""
        
        return responses.get(category, default_response)


# Helper function for easy integration
def create_gemini_assistant(api_key: str = None) -> GeminiAssistant:
    """
    Factory function to create GeminiAssistant instance
    
    Args:
        api_key: Gemini API key (optional, will use from env if not provided)
    
    Returns:
        GeminiAssistant instance
    """
    return GeminiAssistant(api_key=api_key)

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