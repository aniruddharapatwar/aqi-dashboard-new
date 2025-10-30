"""
Complete Feature Engineering Pipeline - CORRECTED VERSION
Matches training code exactly with all required features
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates features matching training pipeline for PM2.5, PM10, NO2, OZONE
    CORRECTED: Now matches the exact features used during model training
    """
    
    # Weather features (raw)
    WEATHER_FEATURES = [
        'temperature', 'humidity', 'dewPoint', 'apparentTemperature',
        'precipIntensity', 'pressure', 'surfacePressure',
        'cloudCover', 'windSpeed', 'windBearing', 'windGust'
    ]
    
    # CORRECTED: All horizons use the same lags matching training
    LAG_HOURS = [1, 3, 6, 12, 24, 48, 72, 168]
    
    # CORRECTED: All horizons use the same rolling windows
    ROLLING_WINDOWS = [6, 12, 24]
    
    def __init__(self):
        self.feature_cache = {}
    
    def engineer_features(self, 
                         current_data: pd.DataFrame, 
                         historical_data: pd.DataFrame,
                         pollutant: str,
                         horizon: str) -> pd.DataFrame:
        """
        Create features for prediction - CORRECTED VERSION
        
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
            
            # 1. Add temporal features (CYCLICAL ENCODING)
            features = self._add_temporal_features(features)
            
            # 2. Add pollutant-specific features
            if pollutant == 'OZONE':
                features = self._add_ozone_specific_features(features)
            
            # 3. Create lag features (ALL 8 lags)
            features = self._add_lag_features(
                features, historical_data, pollutant
            )
            
            # 4. Create pollutant rolling features (mean and max only)
            features = self._add_pollutant_rolling(
                features, historical_data, pollutant
            )
            
            # 5. Select features (remove excluded columns)
            features = self._select_features(features, pollutant)
            
            # 6. Fill nulls and cast to float32
            features = self._finalize_features(features)
            
            logger.info(f"Engineered {len(features.columns)} features for {pollutant} {horizon}")
            
            return features
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def _add_temporal_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical time encoding - MATCHES TRAINING
        """
        try:
            # Try to extract from timestamp
            if 'timestamp' in features.columns:
                ts = pd.to_datetime(features['timestamp'].iloc[0])
                hour = ts.hour
                dow = ts.dayofweek
                month = ts.month
                day_of_year = ts.dayofyear
            # Try to use existing temporal columns
            elif 'hour' in features.columns:
                hour = int(features['hour'].iloc[0])
                dow = int(features.get('day_of_week', 0).iloc[0]) if 'day_of_week' in features.columns else 0
                month = int(features.get('month', 1).iloc[0]) if 'month' in features.columns else 1
                day_of_year = int(features.get('day_of_year', 1).iloc[0]) if 'day_of_year' in features.columns else 1
            else:
                # Default values if nothing available
                logger.warning("No timestamp or hour column found, using defaults")
                hour = 12
                dow = 0
                month = 1
                day_of_year = 1
            
            # Cyclical encoding - MATCHES TRAINING
            features.loc[features.index[0], 'hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features.loc[features.index[0], 'hour_cos'] = np.cos(2 * np.pi * hour / 24)
            features.loc[features.index[0], 'dow_sin'] = np.sin(2 * np.pi * dow / 7)
            features.loc[features.index[0], 'dow_cos'] = np.cos(2 * np.pi * dow / 7)
            features.loc[features.index[0], 'month_sin'] = np.sin(2 * np.pi * month / 12)
            features.loc[features.index[0], 'month_cos'] = np.cos(2 * np.pi * month / 12)
            
            # Store for OZONE features
            features.loc[features.index[0], '_hour'] = hour
            features.loc[features.index[0], '_dow'] = dow
            features.loc[features.index[0], '_month'] = month
            features.loc[features.index[0], '_day_of_year'] = day_of_year
            
        except Exception as e:
            logger.warning(f"Error adding temporal features: {e}")
            # Add zeros if extraction fails
            features.loc[features.index[0], 'hour_sin'] = 0.0
            features.loc[features.index[0], 'hour_cos'] = 1.0
            features.loc[features.index[0], 'dow_sin'] = 0.0
            features.loc[features.index[0], 'dow_cos'] = 1.0
            features.loc[features.index[0], 'month_sin'] = 0.0
            features.loc[features.index[0], 'month_cos'] = 1.0
            features.loc[features.index[0], '_hour'] = 12
            features.loc[features.index[0], '_dow'] = 0
            features.loc[features.index[0], '_month'] = 1
            features.loc[features.index[0], '_day_of_year'] = 1
        
        return features
    
    def _add_ozone_specific_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Add OZONE-specific features - MATCHES TRAINING
        """
        try:
            # Get temporal values (set by _add_temporal_features)
            hour = int(features['_hour'].iloc[0])
            dow = int(features['_dow'].iloc[0])
            month = int(features['_month'].iloc[0])
            day_of_year = int(features['_day_of_year'].iloc[0])
            
            # OZONE-specific features from training
            features.loc[features.index[0], 'is_peak_hour'] = float(12 <= hour <= 16)
            features.loc[features.index[0], 'is_weekend'] = float(dow >= 5)
            features.loc[features.index[0], 'is_summer'] = float(3 <= month <= 10)
            
            # Solar intensity proxy - MATCHES TRAINING EXACTLY
            solar_intensity = (
                np.cos(np.pi * (hour - 12) / 12) *
                np.cos(2 * np.pi * (day_of_year - 172) / 365)
            )
            solar_intensity = np.clip(solar_intensity, 0, 1)
            features.loc[features.index[0], 'solar_intensity'] = float(solar_intensity)
            
        except Exception as e:
            logger.warning(f"Error adding OZONE features: {e}")
            # Add defaults if extraction fails
            features.loc[features.index[0], 'is_peak_hour'] = 0.0
            features.loc[features.index[0], 'is_weekend'] = 0.0
            features.loc[features.index[0], 'is_summer'] = 0.0
            features.loc[features.index[0], 'solar_intensity'] = 0.5
        
        return features
    
    def _add_lag_features(self, 
                         current: pd.DataFrame,
                         historical: pd.DataFrame,
                         pollutant: str) -> pd.DataFrame:
        """
        Add ALL lag features - CORRECTED: 1, 3, 6, 12, 24, 48, 72, 168
        These match the training code exactly
        """
        # CORRECTED: Use all 8 lags for all horizons
        lags = self.LAG_HOURS
        
        # Ensure pollutant column exists
        if pollutant not in historical.columns:
            logger.warning(f"{pollutant} not found in historical data")
            for lag in lags:
                current.loc[current.index[0], f'lag_{lag}h'] = np.nan
            return current
        
        for lag in lags:
            lag_col = f'lag_{lag}h'
            if len(historical) >= lag:
                current.loc[current.index[0], lag_col] = historical[pollutant].iloc[-lag]
            else:
                current.loc[current.index[0], lag_col] = np.nan
        
        return current
    
    def _add_pollutant_rolling(self,
                              current: pd.DataFrame,
                              historical: pd.DataFrame,
                              pollutant: str) -> pd.DataFrame:
        """
        Add rolling statistics for pollutant
        CORRECTED: Use [6, 12, 24] windows with mean and max only (matches training)
        """
        # CORRECTED: All horizons use same windows
        windows = self.ROLLING_WINDOWS
        
        if pollutant not in historical.columns:
            for window in windows:
                current.loc[current.index[0], f'rolling_mean_{window}h'] = np.nan
                current.loc[current.index[0], f'rolling_max_{window}h'] = np.nan
            return current
        
        for window in windows:
            if len(historical) >= window:
                recent = historical[pollutant].tail(window)
                
                # CORRECTED: Only mean and max (matches training)
                current.loc[current.index[0], f'rolling_mean_{window}h'] = recent.mean()
                current.loc[current.index[0], f'rolling_max_{window}h'] = recent.max()
            else:
                current.loc[current.index[0], f'rolling_mean_{window}h'] = np.nan
                current.loc[current.index[0], f'rolling_max_{window}h'] = np.nan
        
        return current
    
    def _select_features(self,
                        features: pd.DataFrame,
                        pollutant: str) -> pd.DataFrame:
        """
        Remove columns that should not be in model input
        """
        # Columns to exclude
        exclude_cols = {
            'location', 'location_aq', 'exposure_category',
            'region', 'season', 'timestamp', 'location_id',
            'date', 'lat', 'lng', 'lon', 'pincode', 'loc_key', 'loc_id',
            # Exclude contemporaneous pollutants (data leakage)
            'PM25', 'PM10', 'NO2', 'OZONE', 'CO', 'SO2', 'AQI',
            # Exclude all targets
            'target_1h', 'target_6h', 'target_12h', 'target_24h',
            # Exclude temporary columns used for feature creation
            '_hour', '_dow', '_month', '_day_of_year',
            'hour', 'day_of_week', 'month', 'day_of_year', 'year'
        }
        
        # Keep only numeric columns not in exclude list
        numeric_cols = [
            col for col in features.columns
            if col not in exclude_cols and 
            pd.api.types.is_numeric_dtype(features[col])
        ]
        
        return features[numeric_cols]
    
    def _finalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Final processing: fill nulls + cast to float32
        """
        # Fill nulls with column median
        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                # CORRECTED: Use proper assignment instead of inplace
                features[col] = features[col].fillna(median_val)
        
        # If still nulls, fill with 0
        features = features.fillna(0)
        
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
        
        logger.info(f"Aligned to {len(aligned.columns)} model features")
        
        return aligned