"""
OPTIMIZED Feature Engineering Pipeline for LSTM Sequences
Handles batch feature engineering to avoid repeated calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    OPTIMIZED: Creates features for LSTM models with efficient sequence building
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
    
    def __init__(self):
        self.feature_cache = {}
    
    def engineer_features_batch(self,
                               historical_data: pd.DataFrame,
                               pollutant: str,
                               horizon: str,
                               sequence_length: int) -> pd.DataFrame:
        """
        OPTIMIZED: Engineer features for entire sequence at once
        
        Args:
            historical_data: Historical data (last N rows for sequence)
            pollutant: Target pollutant
            horizon: Forecast horizon
            sequence_length: Number of timesteps needed
            
        Returns:
            DataFrame with all timesteps' features (sequence_length rows)
        """
        try:
            # Ensure we have enough data
            if len(historical_data) < sequence_length:
                logger.warning(f"Not enough historical data: {len(historical_data)} < {sequence_length}")
                # Pad with first row if needed
                padding_rows = sequence_length - len(historical_data)
                if padding_rows > 0:
                    first_row = historical_data.iloc[[0]]
                    padding = pd.concat([first_row] * padding_rows, ignore_index=True)
                    historical_data = pd.concat([padding, historical_data], ignore_index=True)
            
            # Take last sequence_length rows
            sequence_data = historical_data.tail(sequence_length).copy().reset_index(drop=True)
            
            # 1. Add missing base features (ONCE for entire dataframe)
            sequence_data = self._add_missing_base_features_batch(sequence_data)
            
            # 2. Add lag features (vectorized)
            sequence_data = self._add_lag_features_batch(sequence_data, pollutant, horizon)
            
            # 3. Add rolling features (vectorized)
            sequence_data = self._add_pollutant_rolling_batch(sequence_data, pollutant, horizon)
            
            # 4. Add weather rolling (vectorized)
            sequence_data = self._add_weather_rolling_batch(sequence_data, horizon)
            
            # 5. Select features
            sequence_data = self._select_horizon_features(sequence_data, pollutant, horizon)
            
            # 6. Finalize
            sequence_data = self._finalize_features(sequence_data)
            
            logger.info(f"Batch engineered {len(sequence_data)} rows × {len(sequence_data.columns)} features for {pollutant} {horizon}")
            
            return sequence_data
            
        except Exception as e:
            logger.error(f"Batch feature engineering failed: {e}", exc_info=True)
            raise
    
    def _add_missing_base_features_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add base features for entire batch at once"""
        
        # Add weekday if missing
        if 'weekday' not in data.columns:
            if 'date' in data.columns:
                data['weekday'] = pd.to_datetime(data['date']).dt.weekday
            elif all(col in data.columns for col in ['year', 'month', 'day']):
                dates = pd.to_datetime(data[['year', 'month', 'day']])
                data['weekday'] = dates.dt.weekday
            else:
                data['weekday'] = 0
        
        # Add cyclical encodings (vectorized)
        data = self._add_cyclical_encodings_batch(data)
        
        # Add spatial features
        data = self._add_spatial_features_batch(data)
        
        return data
    
    def _add_cyclical_encodings_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized cyclical encoding for entire batch"""
        
        # Hour encodings
        if 'hour' in data.columns:
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        # Day of week encodings
        if 'weekday' in data.columns:
            data['dow_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
            data['dow_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)
            data['is_weekend'] = (data['weekday'] >= 5).astype(int)
        
        # Week of year
        if 'date' in data.columns:
            dates = pd.to_datetime(data['date'])
            week_of_year = dates.dt.isocalendar().week
            data['week_sin'] = np.sin(2 * np.pi * week_of_year / 52)
            data['week_cos'] = np.cos(2 * np.pi * week_of_year / 52)
        elif all(col in data.columns for col in ['year', 'month', 'day']):
            dates = pd.to_datetime(data[['year', 'month', 'day']])
            week_of_year = dates.dt.isocalendar().week
            data['week_sin'] = np.sin(2 * np.pi * week_of_year / 52)
            data['week_cos'] = np.cos(2 * np.pi * week_of_year / 52)
        else:
            data['week_sin'] = 0.0
            data['week_cos'] = 1.0
        
        # Month encodings
        if 'month' in data.columns:
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Day of year
        if 'date' in data.columns:
            dates = pd.to_datetime(data['date'])
            day_of_year = dates.dt.dayofyear
            data['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365)
            data['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        elif all(col in data.columns for col in ['year', 'month', 'day']):
            dates = pd.to_datetime(data[['year', 'month', 'day']])
            day_of_year = dates.dt.dayofyear
            data['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365)
            data['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        else:
            data['doy_sin'] = 0.0
            data['doy_cos'] = 1.0
        
        return data
    
    def _add_spatial_features_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add spatial features (defaults if not present)"""
        
        spatial_defaults = {
            'traffic_density_score': 0.0,
            'industrial_proximity': 10.0,
            'industrial_density_score': 0.0,
            'near_industrial_1km': 0,
            'near_industrial_3km': 0,
            'near_industrial_5km': 0,
            'exposure_index': 0.0
        }
        
        for feat_name, default_value in spatial_defaults.items():
            if feat_name not in data.columns:
                data[feat_name] = default_value
        
        return data
    
    def _add_lag_features_batch(self,
                                data: pd.DataFrame,
                                pollutant: str,
                                horizon: str) -> pd.DataFrame:
        """Vectorized lag feature creation"""
        
        lags = self.LAG_WINDOWS[horizon]
        
        if pollutant not in data.columns:
            for lag in lags:
                data[f'{pollutant}_lag_{lag}h'] = np.nan
            return data
        
        # Use shift() for vectorized lag creation
        for lag in lags:
            data[f'{pollutant}_lag_{lag}h'] = data[pollutant].shift(lag)
        
        return data
    
    def _add_pollutant_rolling_batch(self,
                                     data: pd.DataFrame,
                                     pollutant: str,
                                     horizon: str) -> pd.DataFrame:
        """Vectorized rolling statistics"""
        
        windows = self.ROLLING_WINDOWS[horizon]
        
        if pollutant not in data.columns:
            for window in windows:
                if horizon == '1h':
                    data[f'{pollutant}_rolling_mean_{window}h'] = np.nan
                    data[f'{pollutant}_rolling_std_{window}h'] = np.nan
                else:
                    data[f'{pollutant}_rolling_mean_{window}h'] = np.nan
                    data[f'{pollutant}_rolling_std_{window}h'] = np.nan
                    data[f'{pollutant}_rolling_min_{window}h'] = np.nan
                    data[f'{pollutant}_rolling_max_{window}h'] = np.nan
            return data
        
        # Use rolling() for vectorized computation
        for window in windows:
            rolling = data[pollutant].rolling(window=window, min_periods=1)
            
            if horizon == '1h':
                data[f'{pollutant}_rolling_mean_{window}h'] = rolling.mean()
                data[f'{pollutant}_rolling_std_{window}h'] = rolling.std()
            else:
                data[f'{pollutant}_rolling_mean_{window}h'] = rolling.mean()
                data[f'{pollutant}_rolling_std_{window}h'] = rolling.std()
                data[f'{pollutant}_rolling_min_{window}h'] = rolling.min()
                data[f'{pollutant}_rolling_max_{window}h'] = rolling.max()
        
        return data
    
    def _add_weather_rolling_batch(self,
                                   data: pd.DataFrame,
                                   horizon: str) -> pd.DataFrame:
        """Vectorized weather rolling statistics"""
        
        window_map = {
            '1h': None,
            '6h': 6,
            '12h': 12,
            '24h': 24
        }
        
        window = window_map[horizon]
        
        if window is not None:
            for feature in self.WEATHER_FEATURES:
                if feature in data.columns:
                    rolling = data[feature].rolling(window=window, min_periods=1)
                    data[f'{feature}_rolling_mean_{horizon}'] = rolling.mean()
                    data[f'{feature}_rolling_std_{horizon}'] = rolling.std()
        
        return data
    
    def _select_horizon_features(self,
                                data: pd.DataFrame,
                                pollutant: str,
                                horizon: str) -> pd.DataFrame:
        """Select only relevant features"""
        
        exclude_cols = {
            'location', 'location_aq', 'exposure_category',
            'region', 'season', 'timestamp', 'location_id',
            'date', 'pincode', 'loc_key', 'loc_id',
            'PM25', 'PM10', 'NO2', 'OZONE', 'CO', 'SO2', 'AQI',
            'target_1h', 'target_6h', 'target_12h', 'target_24h'
        }
        
        numeric_cols = [
            col for col in data.columns
            if col not in exclude_cols and 
            pd.api.types.is_numeric_dtype(data[col])
        ]
        
        return data[numeric_cols]
    
    def _finalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill nulls and cast to float32"""
        
        # Fill nulls with forward fill first, then backward fill, then 0
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Cast to float32
        data = data.astype(np.float32)
        
        return data


class FeatureAligner:
    """Aligns features to match model's expected input"""
    
    def align_features(self,
                      features: pd.DataFrame,
                      model_features: List[str]) -> pd.DataFrame:
        """
        Reindex features to match model's training features
        
        Args:
            features: Engineered features (can be multiple rows for sequence)
            model_features: Feature names from model artifact
        
        Returns:
            Aligned DataFrame with exact columns in exact order
        """
        # Reindex with fill_value=0.0 for missing features
        aligned = features.reindex(columns=model_features, fill_value=0.0)
        
        # Ensure float32
        aligned = aligned.astype(np.float32)
        
        logger.info(f"Aligned {len(aligned)} rows to {len(aligned.columns)} model features")
        
        return aligned