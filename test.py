from app import data_manager, predict_all

# Load data for one location
current, historical = data_manager.get_location_data("Connaught Place (Rajiv Chowk)")
print(f"Current: {len(current)} rows")
print(f"Historical: {len(historical)} rows")

# Make prediction
predictions = predict_all(current, historical, standard='IN')
print(predictions['overall']['6h'])