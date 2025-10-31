# Updated Dashboard Files for New LSTM Models

## 📋 Overview

These three files have been completely rewritten to work with your updated LSTM models that use the new feature engineering approach. The key changes align with the training code you provided.

## 🆕 What Changed

### 1. **feature_engineer.py**
**Complete rewrite to match your training pipeline exactly:**

- ✅ **Cyclical Temporal Features (11 features):**
  - `hour_sin`, `hour_cos`
  - `dow_sin`, `dow_cos` 
  - `week_sin`, `week_cos`
  - `month_sin`, `month_cos`
  - `doy_sin`, `doy_cos`
  - `is_weekend`

- ✅ **Spatial Features (7 features):**
  - `traffic_density_score`
  - `industrial_proximity`
  - `industrial_density_score`
  - `near_industrial_1km`, `near_industrial_3km`, `near_industrial_5km`
  - `exposure_index`

- ✅ **Pollutant Lag Features (horizon-specific):**
  - 1h: [1, 2, 3, 6, 12, 24] hours
  - 6h: [6, 12, 24, 48] hours
  - 12h: [12, 24, 48, 72] hours
  - 24h: [24, 48, 72, 96, 168] hours

- ✅ **Pollutant Rolling Features:**
  - 1h: mean & std only for [6, 12, 24, 48]h windows
  - 6h/12h/24h: mean, std, min, max for windows

- ✅ **Weather Rolling Features (ONLY for 6h, 12h, 24h):**
  - Creates `{weather_var}_rolling_mean_{horizon}`
  - Creates `{weather_var}_rolling_std_{horizon}`

### 2. **predictor.py**
**Complete rewrite for new model artifacts:**

- ✅ **Model Loading:**
  - Loads from `model_artifacts.pkl` (contains Keras model + scalers)
  - Reads `feature_names` from artifacts
  - Uses `FeatureAligner` to match expected features

- ✅ **Sequence Generation:**
  - Creates sequences of length specified in artifacts
  - Handles padding for insufficient history

- ✅ **Pure Regression:**
  - Uses both `feature_scaler` and `target_scaler`
  - Converts continuous predictions to categories
  - Clips predictions to reasonable ranges (0-1000)

- ✅ **Multi-Pollutant Support:**
  - Predicts PM2.5, PM10, NO2, Ozone
  - All four horizons: 1h, 6h, 12h, 24h
  - Calculates overall AQI from sub-indices

### 3. **app.py**
**Updated to use new predictor and feature engineer:**

- ✅ **Data Loading:**
  - Loads spatial features from `matched_region_unique_proximity.csv`
  - Passes spatial data to predictor

- ✅ **Prediction Flow:**
  - Uses new `LSTMPredictor` class
  - Handles location matching with lat/lon
  - Returns predictions for all pollutants & horizons

- ✅ **API Endpoints:**
  - All endpoints remain unchanged
  - Dashboard design completely preserved
  - Health advisories integrated

## 📁 File Structure

Your models should be organized as:
```
Classification_trained_models/
├── NO2_1h/
│   └── model_artifacts.pkl  (contains everything)
├── NO2_6h/
│   └── model_artifacts.pkl
├── NO2_12h/
│   └── model_artifacts.pkl
├── NO2_24h/
│   └── model_artifacts.pkl
├── PM25_1h/
│   └── model_artifacts.pkl
├── PM25_6h/
│   └── model_artifacts.pkl
... (and so on for PM10 and OZONE)
```

## 📊 Required Data Files

Place these in your `data/` folder:

1. **inference_data.csv** (your uploaded file)
   - Contains current weather and pollutant data
   - Required columns: date, lat, lng, year, month, day, hour, PM25, PM10, NO2, OZONE, temperature, humidity, etc.

2. **region_wise_popular_places_from_inference.csv** (your uploaded file)
   - Contains location whitelist
   - Required columns: Region, Place, Latitude, Longitude

3. **matched_region_unique_proximity.csv** (NEW - your uploaded file)
   - Contains spatial features for each location
   - Required columns: location_id, traffic_density_score, industrial_proximity, etc.

## 🚀 Setup Instructions

### 1. Replace Files
```bash
# Backup your old files first!
mv app.py app.py.backup
mv predictor.py predictor.py.backup
mv feature_engineer.py feature_engineer.py.backup

# Copy new files
cp /path/to/new/app.py .
cp /path/to/new/predictor.py .
cp /path/to/new/feature_engineer.py .
```

### 2. Verify Data Files
```bash
data/
├── inference_data.csv
├── region_wise_popular_places_from_inference.csv
└── matched_region_unique_proximity.csv
```

### 3. Verify Model Files
```bash
Classification_trained_models/
├── NO2_1h/model_artifacts.pkl
├── NO2_6h/model_artifacts.pkl
├── NO2_12h/model_artifacts.pkl
├── NO2_24h/model_artifacts.pkl
├── PM25_1h/model_artifacts.pkl
├── PM25_6h/model_artifacts.pkl
├── PM25_12h/model_artifacts.pkl
├── PM25_24h/model_artifacts.pkl
├── PM10_1h/model_artifacts.pkl
├── PM10_6h/model_artifacts.pkl
├── PM10_12h/model_artifacts.pkl
├── PM10_24h/model_artifacts.pkl
├── OZONE_1h/model_artifacts.pkl
├── OZONE_6h/model_artifacts.pkl
├── OZONE_12h/model_artifacts.pkl
└── OZONE_24h/model_artifacts.pkl
```

### 4. Install Dependencies
```bash
pip install fastapi uvicorn pandas numpy tensorflow scikit-learn python-dotenv google-generativeai
```

### 5. Run the Dashboard
```bash
python app.py
```

The API will start on `http://localhost:8000`

## 🔍 Testing

### 1. Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "data_loaded": true,
  "whitelist_loaded": true,
  "spatial_data_loaded": true,
  "models_accessible": true,
  ...
}
```

### 2. Get Regions
```bash
curl http://localhost:8000/regions
```

### 3. Get Locations
```bash
curl "http://localhost:8000/locations?region=Central%20Delhi"
```

### 4. Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"location": "Connaught Place (Rajiv Chowk)", "standard": "IN"}'
```

## ⚠️ Important Notes

### Feature Engineering
- The code automatically creates all required features
- If spatial data is missing, it uses safe defaults
- Feature alignment ensures model compatibility

### Model Artifacts
- Each `model_artifacts.pkl` must contain:
  - `model`: Keras LSTM model
  - `feature_scaler`: StandardScaler for features
  - `target_scaler`: StandardScaler for target
  - `sequence_length`: Number of timesteps
  - `feature_names`: List of expected features (optional but recommended)

### Location Filtering
- Models were trained on 25 specific locations
- The code handles location matching via lat/lon
- Spatial features are loaded from `matched_region_unique_proximity.csv`

### Prediction Flow
1. Get current and historical data for location
2. Engineer features for each historical timestep
3. Align features to model's expected features
4. Scale features using `feature_scaler`
5. Create sequence (e.g., last 48, 96, 144, or 240 timesteps)
6. Predict using LSTM model
7. Inverse transform using `target_scaler`
8. Convert to category and calculate AQI

## 🐛 Troubleshooting

### Error: "Model artifacts not found"
- Check that `model_artifacts.pkl` exists in each model folder
- Verify the path in `Config.MODEL_PATH`

### Error: "Missing features"
- The `FeatureAligner` will add missing features with defaults
- Check logs for which features are being added

### Error: "No data found for location"
- Verify lat/lon coordinates match between whitelist and inference data
- The code searches with progressively larger radius (up to 5.5km)

### Error: "Spatial data file not found"
- Place `matched_region_unique_proximity.csv` in `data/` folder
- Or the code will use default spatial feature values

### Low Accuracy
- Ensure historical data has at least 240 rows
- Check that all pollutants (PM25, PM10, NO2, OZONE) are present
- Verify date format is consistent

## 📊 Model Information

- **Type:** LSTM (Long Short-Term Memory)
- **Architecture:** Pure regression with sequence modeling
- **Pollutants:** PM2.5, PM10, NO2, Ozone
- **Horizons:** 1h, 6h, 12h, 24h
- **Features:** 55 (1h) to 83 (6h/12h/24h)
- **Training:** Year-based split (2021-2024 train, 2025 test)
- **Locations:** 25 locations in Delhi NCR

## 📝 Key Differences from Old Code

| Aspect | Old Code | New Code |
|--------|----------|----------|
| Feature Engineering | Basic | Enhanced with rolling stats, multi-lags |
| Cyclical Features | None | 11 cyclical temporal features |
| Spatial Features | Missing | 7 spatial exposure features |
| Model Loading | Separate .h5 and .pkl | Single model_artifacts.pkl |
| Sequence Creation | Simple | Handles padding and alignment |
| Scalers | Feature scaler only | Both feature and target scalers |
| Weather Rolling | All horizons | Only 6h, 12h, 24h |

## ✅ Validation Checklist

- [ ] All 16 model files exist (4 pollutants × 4 horizons)
- [ ] Each model_artifacts.pkl contains required keys
- [ ] Data files are in the correct location
- [ ] Spatial data file is present
- [ ] Dependencies are installed
- [ ] Health check returns "healthy"
- [ ] Can fetch regions and locations
- [ ] Predictions work for at least one location
- [ ] Dashboard frontend connects successfully

## 🎯 Next Steps

1. Test predictions for all 25 locations
2. Verify AQI calculations are correct
3. Check that health advisories display properly
4. Test all API endpoints
5. Monitor logs for any warnings
6. Compare predictions with actual values (if available)

## 📧 Support

If you encounter issues:
1. Check the logs (`tail -f logs/app.log`)
2. Verify all file paths are correct
3. Test each component independently
4. Check model artifacts have all required keys

---

**Version:** 2.2.0  
**Last Updated:** October 2025  
**Compatibility:** Works with LSTM models trained using the pure regression approach with enhanced feature engineering
