"""
Enhanced Gemini AI Assistant with Safety Validation and Structured Output
Production-ready version with comprehensive error handling and response validation
FINAL FIXED VERSION: Handles safety blocks, finish reasons, and all edge cases betterly
"""

import logging
import json
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
    - Structured JSON output formatting
    - Comprehensive error handling
    - Reduced token usage (300 tokens)
    - Robust safety block handling
    """
    
    def __init__(self, api_key: str = None):
        self.enabled = False
        self.model = None
        self.model_name = None
        self.api_key = api_key
        
        # Define the strict JSON schema for structured output
        self.json_schema = {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING", "description": "A brief, 5-7 word title summarizing the advice."},
                "situation_and_risk": {"type": "STRING", "description": "Brief assessment of air quality impact and specific health risks for the user's profile."},
                "risk_level_summary": {"type": "STRING", "description": "Summary of the user's calculated risk level and key symptoms to watch for."},
                "recommended_actions": {
                    "type": "ARRAY",
                    "description": "3 to 4 specific, prioritized, and actionable recommendations. Each item should be a complete sentence.",
                    "items": {"type": "STRING"}
                }
            },
            "required": ["title", "situation_and_risk", "risk_level_summary", "recommended_actions"]
        }
        
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
                            'top_p': 0.8,
                            'top_k': 40,
                            'max_output_tokens': 300,
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
            Dictionary with response (formatted markdown string), validation, and metadata
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
            
            logger.info(f"Sending structured request to {self.model_name}...")
            
            # CRITICAL: Try to get response with proper error handling
            response = None
            try:
                # Attempt structured output
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=self.json_schema
                    )
                )
                logger.info("✓ Using structured JSON output")
            except (AttributeError, TypeError) as struct_error:
                # Fallback: If structured output not supported, use regular mode
                logger.warning(f"Structured output not supported, using regular mode: {struct_error}")
                response = self.model.generate_content(prompt)
            except Exception as gen_error:
                logger.error(f"Content generation failed: {gen_error}")
                return {
                    'response': self._static_response(context, user_profile),
                    'source': 'static_generation_error',
                    'model': self.model_name,
                    'validation': {'valid': False, 'warnings': ['Content generation failed']}
                }

            # ============================================================================
            # CRITICAL FIX: Check response validity BEFORE accessing .text
            # ============================================================================
            
            # Check 1: Response exists
            if not response:
                logger.warning("No response received from Gemini")
                return {
                    'response': self._static_response(context, user_profile),
                    'source': 'static_no_response',
                    'model': self.model_name,
                    'validation': {'valid': False, 'warnings': ['No response from AI']}
                }
            
            # Check 2: Response has candidates
            if not response.candidates or len(response.candidates) == 0:
                logger.warning("Response has no candidates")
                return {
                    'response': self._static_response(context, user_profile),
                    'source': 'static_no_candidates',
                    'model': self.model_name,
                    'validation': {'valid': False, 'warnings': ['No candidates in AI response']}
                }
            
            # Check 3: Check finish_reason BEFORE accessing .text
            # finish_reason values:
            # 0 = FINISH_REASON_UNSPECIFIED
            # 1 = STOP (normal completion)
            # 2 = MAX_TOKENS
            # 3 = SAFETY (blocked by safety filters)
            # 4 = RECITATION
            # 5 = OTHER
            
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason
            
            # Map finish_reason to readable names
            finish_reason_names = {
                0: 'UNSPECIFIED',
                1: 'STOP',
                2: 'MAX_TOKENS',
                3: 'SAFETY',
                4: 'RECITATION',
                5: 'OTHER'
            }
            finish_reason_name = finish_reason_names.get(finish_reason, f'UNKNOWN({finish_reason})')
            
            # Only proceed if finish_reason is STOP (1)
            if finish_reason != 1:
                logger.warning(f"Response blocked or incomplete. Finish reason: {finish_reason_name}")
                
                # Provide specific warning based on finish reason
                if finish_reason == 3:  # SAFETY
                    warning_msg = 'AI response was blocked by safety filters'
                elif finish_reason == 2:  # MAX_TOKENS
                    warning_msg = 'AI response exceeded maximum tokens'
                elif finish_reason == 4:  # RECITATION
                    warning_msg = 'AI response flagged as potential recitation'
                else:
                    warning_msg = f'AI response incomplete (finish_reason: {finish_reason_name})'
                
                return {
                    'response': self._static_response(context, user_profile),
                    'source': 'static_blocked',
                    'model': self.model_name,
                    'validation': {'valid': False, 'warnings': [warning_msg]}
                }
            
            # Check 4: Candidate has parts with content
            if not candidate.content or not candidate.content.parts:
                logger.warning("Response candidate has no content parts")
                return {
                    'response': self._static_response(context, user_profile),
                    'source': 'static_no_parts',
                    'model': self.model_name,
                    'validation': {'valid': False, 'warnings': ['No content in AI response']}
                }
            
            # NOW it's safe to access response.text
            try:
                response_text = response.text
            except Exception as text_error:
                logger.error(f"Failed to extract text from response: {text_error}")
                return {
                    'response': self._static_response(context, user_profile),
                    'source': 'static_text_extraction_error',
                    'model': self.model_name,
                    'validation': {'valid': False, 'warnings': ['Failed to extract text from AI response']}
                }
            
            logger.info(f"✓ Received valid response from {self.model_name}")
            
            # ============================================================================
            # END CRITICAL FIX
            # ============================================================================
            
            # Parse JSON output
            try:
                json_response = json.loads(response_text)
                logger.info("✓ Successfully parsed JSON response")
            except json.JSONDecodeError as json_err:
                logger.warning(f"Failed to parse AI response as JSON: {json_err}")
                logger.warning(f"Raw response: {response_text[:200]}...")
                
                # Fallback: Try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    try:
                        json_response = json.loads(json_match.group(1))
                        logger.info("✓ Extracted JSON from markdown code block")
                    except json.JSONDecodeError:
                        logger.error("Could not extract valid JSON, using static fallback")
                        return {
                            'response': self._static_response(context, user_profile),
                            'source': 'static_fallback_json_error',
                            'model': self.model_name,
                            'validation': {'valid': False, 'warnings': ['AI output was not valid JSON']}
                        }
                else:
                    # No JSON found, use static response
                    return {
                        'response': self._static_response(context, user_profile),
                        'source': 'static_fallback_json_error',
                        'model': self.model_name,
                        'validation': {'valid': False, 'warnings': ['AI output was not valid JSON']}
                    }

            # Validate response content for safety and accuracy
            validation_result = self._validate_response(
                str(json_response), 
                aqi_mid, 
                category, 
                user_profile
            )
            
            # Format JSON back into readable Markdown string
            cleaned_response = self._format_json_response(json_response)
            
            return {
                'response': cleaned_response,
                'source': 'gemini',
                'model': self.model_name,
                'validation': validation_result
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
        if health_category in ['asthma', 'copd', 'respiratory', 'heart_condition', 'pregnant']:
            risk_level = "HIGH RISK"
        elif age_category in ['child', 'elderly']:
            risk_level = "ELEVATED RISK"
        
        # Weather impact
        weather_impact = ""
        if aqi_mid > 100:
            if humidity > 70:
                weather_impact = "High humidity may increase pollutant adhesion and respiratory effects. "
            if wind_speed < 5:
                weather_impact += "Low wind speeds are preventing pollutant dispersion. "
            elif wind_speed > 20:
                weather_impact += "High winds may resuspend particulate matter. "
        
        prompt = f"""You are the **AQI Assistant**, an expert air quality health advisor for Delhi NCR, India. You provide evidence-based, personalized health recommendations.

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
1. You MUST output a JSON object strictly following this schema:
   {{
     "title": "Brief 5-7 word title",
     "situation_and_risk": "Assessment of air quality and health risks",
     "risk_level_summary": "User's risk level and symptoms to watch",
     "recommended_actions": ["Action 1", "Action 2", "Action 3", "Action 4"]
   }}
2. Address the user's specific question directly within the structured fields.
3. Tailor advice to their risk level and health profile.
4. Be specific, actionable, and use clear, non-technical language.
5. The 'recommended_actions' array must contain 3-4 specific, prioritized, actionable steps.
6. Keep response concise (max 300 tokens).

CRITICAL SAFETY RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ For AQI > 150: ALWAYS recommend N95 masks, limiting outdoor exposure
✓ For high-risk users: Emphasize staying indoors, using air purifiers
✓ For AQI > 200: Stress URGENT need to avoid outdoor activities
✓ Never downplay serious health risks
✓ Be factual - don't speculate or give unproven advice

Provide your response now as a valid JSON object:"""
        
        return prompt
    
    def _format_json_response(self, json_data: Dict) -> str:
        """
        Formats the structured JSON response back into a readable markdown string
        for consistent display on the dashboard.
        """
        if not json_data:
            return "Could not retrieve structured advice."

        # Use the 'title' field as the main header
        markdown_output = f"**{json_data.get('title', 'Air Quality Advice')}**\n\n"

        # Situation and Risk
        markdown_output += "**Current Situation & Risk**\n"
        markdown_output += json_data.get('situation_and_risk', 'N/A') + "\n\n"

        # Risk Level Summary
        markdown_output += "**Your Risk Level**\n"
        markdown_output += json_data.get('risk_level_summary', 'N/A') + "\n\n"

        # Recommended Actions (as a bulleted list)
        markdown_output += "**Recommended Actions**\n"
        actions = json_data.get('recommended_actions', [])
        if actions and isinstance(actions, list):
            markdown_output += "\n".join([f"• {action}" for action in actions])
        else:
            markdown_output += "• No specific actions recommended."
            
        return markdown_output
    
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
            urgency_indicators = ['urgent', 'emergency', 'critical', 'immediately', 'serious', 'stay indoors']
            if not any(word in response_text.lower() for word in urgency_indicators):
                warnings.append(f"Response should convey urgency at AQI {aqi:.0f}")
        
        return {
            'valid': len(warnings) == 0,
            'warnings': warnings,
            'severity': 'high' if len(warnings) > 3 else 'medium' if len(warnings) > 0 else 'none'
        }
    
    def _clean_response(self, response_text: str) -> str:
        """
        Clean response text (used for static/fallback responses)
        """
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