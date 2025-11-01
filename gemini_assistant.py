"""
Simplified Gemini AI Assistant - Natural Conversational Style
UPDATED: Support for multiple health conditions
"""

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available")


class GeminiAssistant:
    """
    Simplified AI Assistant with:
    - Natural conversational responses (no rigid JSON schema)
    - Context-aware advice based on AQI and user profile
    - Support for MULTIPLE health conditions
    - Increased token limit for complete responses
    - Minimal validation (only for dangerous advice)
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
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash-image",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-2.5-pro"
            ]
            
            for model_name in model_options:
                try:
                    logger.info(f"Attempting to initialize {model_name}...")
                    self.model = genai.GenerativeModel(
                        model_name,
                        generation_config={
                            'temperature': 0.7,
                            'top_p': 0.9,
                            'top_k': 40,
                            'max_output_tokens': 1500,
                        },
                        safety_settings=[
                            {
                                "category": "HARM_CATEGORY_HARASSMENT",
                                "threshold": "BLOCK_ONLY_HIGH"
                            },
                            {
                                "category": "HARM_CATEGORY_HATE_SPEECH",
                                "threshold": "BLOCK_ONLY_HIGH"
                            },
                            {
                                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                "threshold": "BLOCK_ONLY_HIGH"
                            },
                            {
                                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                "threshold": "BLOCK_ONLY_HIGH"
                            }
                        ]
                    )
                    
                    # Test the model
                    test_response = self.model.generate_content("Say 'OK' if you're working")
                    if test_response and test_response.text:
                        self.model_name = model_name
                        self.enabled = True
                        logger.info(f"[SUCCESS] Gemini initialized successfully with {model_name}")
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
        Get natural AI response without rigid structure
        UPDATED: Handles multiple health conditions from context
        
        Args:
            message: User's question/message
            context: Dictionary containing location, aqi_data, user_profile, weather
        
        Returns:
            Dictionary with response (natural text), validation, and metadata
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
            # Build natural prompt (no JSON schema!)
            prompt = self._build_natural_prompt(
                message=message,
                location=location,
                aqi_data=aqi_data,
                weather_data=weather_data,
                user_profile=user_profile
            )
            
            logger.info(f"Sending request to {self.model_name}...")
            logger.debug(f"Prompt: {prompt[:200]}...")
            
            # Generate natural text response
            response = self.model.generate_content(prompt)
            
            # Check response validity
            if not response:
                logger.warning("No response received from Gemini")
                return self._fallback_response(context, user_profile, "No response from AI")
            
            if not response.candidates or len(response.candidates) == 0:
                logger.warning("Response has no candidates")
                return self._fallback_response(context, user_profile, "No candidates in AI response")
            
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason
            
            # Check if response completed successfully (finish_reason == 1 means STOP/complete)
            if finish_reason != 1:
                finish_reason_names = {
                    0: 'UNSPECIFIED', 1: 'STOP', 2: 'MAX_TOKENS',
                    3: 'SAFETY', 4: 'RECITATION', 5: 'OTHER'
                }
                reason = finish_reason_names.get(finish_reason, f'UNKNOWN({finish_reason})')
                logger.warning(f"Response incomplete. Finish reason: {reason}")
                
                # For MAX_TOKENS, still try to use the response
                if finish_reason == 2:
                    logger.info("Max tokens reached, but using partial response")
                else:
                    warning_msg = f'AI response blocked or incomplete (finish_reason: {reason})'
                    return self._fallback_response(context, user_profile, warning_msg)
            
            # Extract text
            try:
                response_text = response.text.strip()
            except Exception as text_error:
                logger.error(f"Failed to extract text: {text_error}")
                return self._fallback_response(context, user_profile, "Failed to extract text from AI response")
            
            if not response_text:
                logger.warning("Empty response text")
                return self._fallback_response(context, user_profile, "Empty AI response")
            
            logger.info(f"[SUCCESS] Received response from {self.model_name} ({len(response_text)} chars)")
            
            # Light validation (only check for dangerous advice)
            validation_result = self._validate_response(
                response_text,
                aqi_data.get('aqi_mid', 0),
                aqi_data.get('category', 'Unknown'),
                user_profile
            )
            
            # Clean up response
            cleaned_response = self._clean_response(response_text)
            
            return {
                'response': cleaned_response,
                'source': 'gemini',
                'model': self.model_name,
                'validation': validation_result
            }
                
        except Exception as e:
            logger.error(f"Gemini error: {e}", exc_info=True)
            return self._fallback_response(context, user_profile, f"AI service error: {str(e)}")
    
    def _build_natural_prompt(self, message: str, location: str, aqi_data: Dict,
                              weather_data: Dict, user_profile: Dict) -> str:
        """
        Build natural, conversational prompt
        UPDATED: Handles multiple health conditions
        """
        
        # Extract data
        aqi_mid = aqi_data.get('aqi_mid', 0)
        category = aqi_data.get('category', 'Unknown')
        dominant = aqi_data.get('dominant_pollutant', 'Unknown')
        
        temp = weather_data.get('temperature', 0)
        humidity = weather_data.get('humidity', 0)
        wind_speed = weather_data.get('windSpeed', 0)
        
        # Profile info - UPDATED to handle array of conditions
        profile_label = user_profile.get('profile_label', 'general public')
        age_category = user_profile.get('age_category', '')
        health_conditions = user_profile.get('health_conditions', [])  # ARRAY
        
        # Build context - UPDATED for multiple conditions
        profile_context = ""
        if health_conditions and len(health_conditions) > 0:
            if len(health_conditions) == 1:
                profile_context = f"The user has {health_conditions[0]}."
            else:
                conditions_str = ', '.join(health_conditions[:-1]) + f' and {health_conditions[-1]}'
                profile_context = f"The user has multiple health conditions: {conditions_str}."
        
        if age_category:
            profile_context += f" They are in the {age_category} age group."
        
        prompt = f"""You are an air quality health advisor for Delhi NCR. Answer the user's question naturally and conversationally.

CURRENT CONDITIONS:
Location: {location}
Air Quality: {category} (AQI {aqi_mid:.0f})
Main Pollutant: {dominant}
Weather: {temp:.1f}Â°C, {humidity:.0f}% humidity, {wind_speed:.0f} km/h wind

USER INFO:
{profile_context if profile_context else "No specific health conditions mentioned."}

USER QUESTION:
"{message}"

INSTRUCTIONS:
- Answer the question directly and naturally (like a helpful friend, not a robot)
- Keep it conversational and easy to understand
- Be specific about the CURRENT air quality conditions
- Give practical, actionable advice
- If air quality is poor/dangerous, be clear about the risks
- If the user has multiple health conditions, address ALL of them in your response
- Keep response concise (3-5 short paragraphs maximum)
- Use simple language, avoid medical jargon
- DON'T use rigid formats or section headers unless natural
- DON'T add disclaimers about "not being medical advice"

Your response:"""
        
        return prompt
    
    def _validate_response(self, response_text: str, aqi: float, 
                          category: str, user_profile: Dict) -> Dict:
        """
        Light validation - only flag truly dangerous advice
        """
        warnings = []
        
        # Only check for obviously dangerous patterns at high AQI
        if aqi > 200:
            response_lower = response_text.lower()
            
            dangerous_patterns = [
                (r'\b(safe|fine|okay|ok)\s+to\s+go\s+outside', 'Suggests outdoor activity is safe at dangerous AQI'),
                (r'\bno\s+need\s+for\s+mask', 'Dismisses mask use at high AQI'),
                (r'\bdon\'?t\s+worry', 'May be too reassuring for dangerous conditions'),
            ]
            
            for pattern, warning in dangerous_patterns:
                if re.search(pattern, response_lower):
                    warnings.append(warning)
        
        # Check if response is too short (might indicate truncation)
        if len(response_text) < 50:
            warnings.append("Response seems unusually short")
        
        return {
            'valid': len(warnings) == 0,
            'warnings': warnings,
            'severity': 'high' if len(warnings) > 2 else 'medium' if len(warnings) > 0 else 'none'
        }
    
    def _clean_response(self, response_text: str) -> str:
        """
        Minimal cleaning - preserve natural formatting
        """
        # Remove excessive newlines
        response_text = re.sub(r'\n{3,}', '\n\n', response_text)
        
        # Remove any accidental "Assistant:" or "AI:" prefixes
        response_text = re.sub(r'^(Assistant|AI):\s*', '', response_text, flags=re.IGNORECASE)
        
        return response_text.strip()
    
    def _fallback_response(self, context: Dict, user_profile: Dict, error_msg: str) -> Dict:
        """Return static fallback when AI fails"""
        return {
            'response': self._static_response(context, user_profile),
            'source': 'static_fallback',
            'model': self.model_name,
            'validation': {'valid': False, 'warnings': [error_msg]}
        }
    
    def _static_response(self, context: Dict, user_profile: Dict = None) -> str:
        """
        Simple static fallback responses
        UPDATED: Handles multiple health conditions
        """
        aqi_data = context.get('aqi_data', {})
        category = aqi_data.get('category', 'Unknown')
        aqi_mid = aqi_data.get('aqi_mid', 0)
        dominant = aqi_data.get('dominant_pollutant', 'PM2.5')
        location = context.get('location', 'your area')
        
        profile_label = 'general public'
        health_conditions_str = ''
        
        if user_profile:
            profile_label = user_profile.get('profile_label', 'general public')
            health_conditions = user_profile.get('health_conditions', [])
            
            # Build health conditions string
            if health_conditions and len(health_conditions) > 0:
                if len(health_conditions) == 1:
                    health_conditions_str = f" with {health_conditions[0]}"
                else:
                    conditions_friendly = ', '.join(health_conditions[:-1]) + f' and {health_conditions[-1]}'
                    health_conditions_str = f" with {conditions_friendly}"
        
        responses = {
            'Good': f"Air quality in {location} is excellent right now (AQI {aqi_mid:.0f}). It's a great time for outdoor activities!",
            
            'Satisfactory': f"Air quality in {location} is acceptable (AQI {aqi_mid:.0f}). Most people can enjoy outdoor activities normally. If you're sensitive to air pollution{health_conditions_str}, just monitor how you feel.",
            
            'Moderate': f"Air quality in {location} is moderate (AQI {aqi_mid:.0f}), mainly due to {dominant}. For {profile_label}{health_conditions_str}, it's best to limit prolonged outdoor activities. If you go out, consider wearing a mask, especially in high-traffic areas.",
            
            'Poor': f"Air quality in {location} is poor (AQI {aqi_mid:.0f}) due to {dominant} pollution. I'd recommend limiting time outdoors. If you need to go out, wear an N95 mask and avoid strenuous activities. Keep windows closed and use an air purifier indoors if you have one.{' Given your health conditions' + health_conditions_str + ', extra caution is advised.' if health_conditions_str else ''}",
            
            'Very_Poor': f"Air quality in {location} is very poor right now (AQI {aqi_mid:.0f}), with high {dominant} levels. It's best to stay indoors as much as possible. If you must go outside, always wear an N95 mask and keep it brief. Run air purifiers at home and keep windows sealed.{' With your conditions' + health_conditions_str + ', staying indoors is especially important.' if health_conditions_str else ''}",
            
            'Severe': f"âš ï¸ URGENT: Air quality in {location} is at severe levels (AQI {aqi_mid:.0f}). This is a health emergency. Please stay indoors, seal windows and doors, and run air purifiers continuously. Only go outside if absolutely necessary, and wear a properly fitted N95 mask. If you experience breathing difficulties, seek medical attention immediately.{' Your health conditions' + health_conditions_str + ' put you at higher risk - please take extra precautions.' if health_conditions_str else ''}"
        }
        
        return responses.get(category, 
            f"Current air quality in {location}: {category} (AQI {aqi_mid:.0f}). Please check local health advisories for guidance.")


def create_gemini_assistant(api_key: str = None) -> GeminiAssistant:
    """
    Factory function to create GeminiAssistant instance
    """
    return GeminiAssistant(api_key=api_key)