"""
Complete Feature Engineering Pipeline - PRODUCTION VERSION
Proper feature alignment with comprehensive feature creation
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates features matching training pipeline for PM2.5, PM10, NO2, OZONE
    
    PRODUCTION VERSION: Comprehensive feature engineering with proper defaults
    """
    
    # Weather features (raw)
    WEATHER_FEATURES = [
        'temperature', 'humidity', 'dewPoint', 'apparentTemperature',
        'precipIntensity', 'pressure', 'surfacePressure',
        'cloudCover', 'windSpeed', 'windBearing', 'windGust'
    ]
    
    # Lag windows
    LAG_WINDOWS = {
        '1h': [1, 2, 3, 6, 12, 24],
        '6h': [6, 12, 24, 48],
        '12h': [12, 24, 48, 72],
        '24h': [24, 48, 72, 96, 168]
    }
    
    # Rolling windows
    ROLLING_WINDOWS = {
        '1h': [6, 12, 24, 48],
        '6h': [6, 12],
        '12h': [12, 24],
        '24h': [24, 48]
    }
    
    WEATHER_ROLLING_WINDOWS = [6, 12, 24]
    
    def __init__(self):
        self.feature_cache = {}
    
    def engineer_features(self, 
                         current_data: pd.DataFrame, 
                         historical_data: pd.DataFrame,
                         pollutant: str,
                         horizon: str) -> pd.DataFrame:
        """
        Create features for prediction
        
        Args:
            current_data: Current timestamp data (single row)
            historical_data: Historical data sorted by timestamp (for lags)
            pollutant: 'PM25' | 'PM10' | 'NO2' | 'OZONE'
            horizon: '1h' | '6h' | '12h' | '24h'
        
        Returns:
            DataFrame with engineered features (single row)
        """
        try:
            # Start with current data
            features = current_data.copy()
            
            # Ensure we have a clean copy
            if len(features) > 1:
                features = features.iloc[[-1]].copy()
            
            # 0. Add missing base features
            features = self._add_missing_base_features(features)
            
            # 1. Create lag features
            features = self._add_lag_features(
                features, historical_data, pollutant, horizon
            )
            
            # 2. Create pollutant rolling features
            features = self._add_pollutant_rolling(
                features, historical_data, pollutant, horizon
            )
            
            # 3. Create weather rolling features
            features = self._add_weather_rolling(
                features, historical_data, horizon
            )
            
            # 4. Add enhanced interaction features
            features = self._add_enhanced_features(features, pollutant)
            
            # 5. Select features based on horizon
            features = self._select_horizon_features(
                features, pollutant, horizon
            )
            
            # 6. Fill nulls and cast to float32
            features = self._finalize_features(features)
            
            logger.info(f"Engineered {len(features.columns)} features for {pollutant} {horizon}")
            
            return features
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def _add_missing_base_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add any missing base features"""
        
        # 1. Add weekday if missing
        if 'weekday' not in features.columns:
            if 'date' in features.columns:
                features['weekday'] = pd.to_datetime(features['date']).dt.weekday
            elif all(col in features.columns for col in ['year', 'month', 'day']):
                date_str = f"{int(features['year'].iloc[0])}-{int(features['month'].iloc[0]):02d}-{int(features['day'].iloc[0]):02d}"
                date = pd.to_datetime(date_str)
                features['weekday'] = date.weekday()
            else:
                features['weekday'] = 0
        
        # 2. Add cyclical temporal encodings
        features = self._add_cyclical_encodings(features)
        
        # 3. Add spatial features
        features = self._add_spatial_features(features)
        
        return features
    
    def _add_cyclical_encodings(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical temporal encodings"""
        
        # Hour encodings (0-23)
        if 'hour' in features.columns:
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # Day of week encodings (0-6)
        if 'weekday' in features.columns:
            features['dow_sin'] = np.sin(2 * np.pi * features['weekday'] / 7)
            features['dow_cos'] = np.cos(2 * np.pi * features['weekday'] / 7)
            features['is_weekend'] = (features['weekday'] >= 5).astype(int)
        
        # Week of year (1-52)
        if 'date' in features.columns:
            date = pd.to_datetime(features['date'])
            week_of_year = date.dt.isocalendar().week.iloc[0]
            features['week_sin'] = np.sin(2 * np.pi * week_of_year / 52)
            features['week_cos'] = np.cos(2 * np.pi * week_of_year / 52)
        elif all(col in features.columns for col in ['year', 'month', 'day']):
            date_str = f"{int(features['year'].iloc[0])}-{int(features['month'].iloc[0]):02d}-{int(features['day'].iloc[0]):02d}"
            date = pd.to_datetime(date_str)
            week_of_year = date.isocalendar().week
            features['week_sin'] = np.sin(2 * np.pi * week_of_year / 52)
            features['week_cos'] = np.cos(2 * np.pi * week_of_year / 52)
        else:
            features['week_sin'] = 0.0
            features['week_cos'] = 1.0
        
        # Month encodings (1-12)
        if 'month' in features.columns:
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Day of year (1-365)
        if 'date' in features.columns:
            date = pd.to_datetime(features['date'])
            day_of_year = date.dt.dayofyear.iloc[0]
            features['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365)
            features['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        elif all(col in features.columns for col in ['year', 'month', 'day']):
            date_str = f"{int(features['year'].iloc[0])}-{int(features['month'].iloc[0]):02d}-{int(features['day'].iloc[0]):02d}"
            date = pd.to_datetime(date_str)
            day_of_year = date.dayofyear
            features['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365)
            features['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        else:
            features['doy_sin'] = 0.0
            features['doy_cos'] = 0.0
        
        return features
    
    def _add_spatial_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add spatial features with defaults"""
        
        spatial_defaults = {
            'lat': 28.6139,
            'lon': 77.2090,
            'elevation': 216.0,
            'population_density': 11000.0,
            'distance_to_coast': 1000.0,
            'distance_to_road': 0.5,
            'distance_to_industrial': 5.0
        }
        
        critical_spatial = {
            'traffic_density_score': 0.0,
            'industrial_proximity': 10.0,
            'industrial_density_score': 0.0,
            'near_industrial_1km': 0.0,
            'near_industrial_3km': 0.0,
            'near_industrial_5km': 0.0,
            'exposure_index': 0.0
        }
        
        all_spatial = {**spatial_defaults, **critical_spatial}
        
        for feature, default_value in all_spatial.items():
            if feature not in features.columns:
                features[feature] = default_value
        
        # Ensure 'lon' exists even if we have 'lng'
        if 'lon' not in features.columns and 'lng' in features.columns:
            features['lon'] = features['lng']
        
        return features
    
    def _add_lag_features(self,
                         current: pd.DataFrame,
                         historical: pd.DataFrame,
                         pollutant: str,
                         horizon: str) -> pd.DataFrame:
        """Add lag features for the pollutant"""
        lag_windows = self.LAG_WINDOWS[horizon]
        
        for window in lag_windows:
            if pollutant in historical.columns and len(historical) >= window:
                lag_value = historical[pollutant].iloc[-window]
                current.loc[current.index[0], f'{pollutant}_lag_{window}h'] = lag_value
            else:
                current.loc[current.index[0], f'{pollutant}_lag_{window}h'] = np.nan
        
        return current
    
    def _add_pollutant_rolling(self,
                              current: pd.DataFrame,
                              historical: pd.DataFrame,
                              pollutant: str,
                              horizon: str) -> pd.DataFrame:
        """Add pollutant rolling statistics"""
        rolling_windows = self.ROLLING_WINDOWS[horizon]
        
        for window in rolling_windows:
            if pollutant in historical.columns and len(historical) >= window:
                recent = historical[pollutant].tail(window)
                
                if horizon == '1h':
                    current.loc[current.index[0], f'{pollutant}_rolling_mean_{window}h'] = recent.mean()
                    current.loc[current.index[0], f'{pollutant}_rolling_std_{window}h'] = recent.std()
                else:
                    current.loc[current.index[0], f'{pollutant}_rolling_mean_{window}h'] = recent.mean()
                    current.loc[current.index[0], f'{pollutant}_rolling_std_{window}h'] = recent.std()
                    current.loc[current.index[0], f'{pollutant}_rolling_min_{window}h'] = recent.min()
                    current.loc[current.index[0], f'{pollutant}_rolling_max_{window}h'] = recent.max()
            else:
                if horizon == '1h':
                    current.loc[current.index[0], f'{pollutant}_rolling_mean_{window}h'] = np.nan
                    current.loc[current.index[0], f'{pollutant}_rolling_std_{window}h'] = np.nan
                else:
                    current.loc[current.index[0], f'{pollutant}_rolling_mean_{window}h'] = np.nan
                    current.loc[current.index[0], f'{pollutant}_rolling_std_{window}h'] = np.nan
                    current.loc[current.index[0], f'{pollutant}_rolling_min_{window}h'] = np.nan
                    current.loc[current.index[0], f'{pollutant}_rolling_max_{window}h'] = np.nan
        
        return current
    
    def _add_weather_rolling(self,
                            current: pd.DataFrame,
                            historical: pd.DataFrame,
                            horizon: str) -> pd.DataFrame:
        """Add weather rolling statistics"""
        window_map = {
            '1h': None,
            '6h': 6,
            '12h': 12,
            '24h': 24
        }
        
        window = window_map[horizon]
        
        if window is not None:
            for feature in self.WEATHER_FEATURES:
                if feature in historical.columns:
                    if len(historical) >= window:
                        recent = historical[feature].tail(window)
                        current.loc[current.index[0], f'{feature}_rolling_mean_{horizon}'] = recent.mean()
                        current.loc[current.index[0], f'{feature}_rolling_std_{horizon}'] = recent.std()
                    else:
                        current.loc[current.index[0], f'{feature}_rolling_mean_{horizon}'] = np.nan
                        current.loc[current.index[0], f'{feature}_rolling_std_{horizon}'] = np.nan
        
        return current
    
    def _add_enhanced_features(self,
                              features: pd.DataFrame,
                              pollutant: str) -> pd.DataFrame:
        """Add interaction features"""
        
        # Short/long change from lags
        lag_cols = [col for col in features.columns if f'{pollutant}_lag_' in col]
        
        if len(lag_cols) >= 2:
            features.loc[features.index[0], f'{pollutant}_short_change'] = (
                features[lag_cols[0]].iloc[0] - features[lag_cols[1]].iloc[0]
            )
            
            if len(lag_cols) > 2:
                features.loc[features.index[0], f'{pollutant}_long_change'] = (
                    features[lag_cols[0]].iloc[0] - features[lag_cols[-1]].iloc[0]
                )
        
        # Weather deviations
        weather_vars = ['temperature', 'humidity', 'pressure', 'windSpeed']
        
        for var in weather_vars:
            if var in features.columns:
                rolling_cols = [col for col in features.columns 
                               if f'{var}_rolling_' in col and '_mean' in col]
                
                if rolling_cols:
                    features.loc[features.index[0], f'{var}_deviation'] = (
                        features[var].iloc[0] - features[rolling_cols[0]].iloc[0]
                    )
        
        # Wind-humidity interaction
        if 'windSpeed' in features.columns and 'humidity' in features.columns:
            features.loc[features.index[0], 'wind_humidity_interaction'] = (
                features['windSpeed'].iloc[0] * features['humidity'].iloc[0]
            )
        
        # Pollutant-specific interactions
        if pollutant == 'OZONE':
            if 'temperature' in features.columns and 'humidity' in features.columns:
                features.loc[features.index[0], 'temp_humidity_interaction'] = (
                    features['temperature'].iloc[0] * (100 - features['humidity'].iloc[0])
                )
            
            if 'windSpeed' in features.columns and 'temperature' in features.columns:
                features.loc[features.index[0], 'wind_temp_interaction'] = (
                    features['windSpeed'].iloc[0] * features['temperature'].iloc[0]
                )
            
            if 'cloudCover' in features.columns and 'temperature' in features.columns:
                features.loc[features.index[0], 'cloud_temp_interaction'] = (
                    (1 - features['cloudCover'].iloc[0]) * features['temperature'].iloc[0]
                )
        
        elif pollutant == 'NO2':
            if 'windSpeed' in features.columns and 'temperature' in features.columns:
                features.loc[features.index[0], 'wind_temp_interaction'] = (
                    features['windSpeed'].iloc[0] * features['temperature'].iloc[0]
                )
        
        return features
    
    def _select_horizon_features(self,
                                features: pd.DataFrame,
                                pollutant: str,
                                horizon: str) -> pd.DataFrame:
        """Select only features used for this horizon"""
        
        exclude_cols = {
            'location', 'location_aq', 'exposure_category',
            'region', 'season', 'timestamp', 'location_id',
            'date', 'pincode', 'loc_key', 'loc_id',
            'PM25', 'PM10', 'NO2', 'OZONE', 'CO', 'SO2', 'AQI',
            'target_1h', 'target_6h', 'target_12h', 'target_24h'
        }
        
        numeric_cols = [
            col for col in features.columns
            if col not in exclude_cols and 
            pd.api.types.is_numeric_dtype(features[col])
        ]
        
        return features[numeric_cols]
    
    def _finalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Final processing: fill nulls + cast to float32"""
        
        # Fill nulls with column median
        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                features[col].fillna(median_val, inplace=True)
        
        # If still nulls, fill with 0
        features.fillna(0, inplace=True)
        
        # Cast to float32
        features = features.astype(np.float32)
        
        return features


class FeatureAligner:
    """
    Aligns features to match model's expected input
    """
    
    def align_features(self,
                      features: pd.DataFrame,
                      model_features: List[str]) -> pd.DataFrame:
        """
        Reindex features to match model's training features
        
        Args:
            features: Engineered features
            model_features: Feature names from model artifact
        
        Returns:
            Aligned DataFrame with exact columns in exact order
        """
        # Reindex with fill_value=0.0 for missing features
        aligned = features.reindex(columns=model_features, fill_value=0.0)
        
        # Ensure float32
        aligned = aligned.astype(np.float32)
        
        # Log any missing features
        missing_features = set(model_features) - set(features.columns)
        if missing_features:
            logger.debug(f"Filled {len(missing_features)} missing features with zeros")
        
        # Log any extra features
        extra_features = set(features.columns) - set(model_features)
        if extra_features:
            logger.debug(f"Dropped {len(extra_features)} extra features not in model")
        
        return aligned