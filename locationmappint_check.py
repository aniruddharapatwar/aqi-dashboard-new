#!/usr/bin/env python3
"""
Location Matching Diagnostic Tool
Identifies mismatches between whitelist locations and data file locations
"""

import pandas as pd
import numpy as np
from pathlib import Path

def print_header(text):
    print(f"\n{'=' * 80}")
    print(f"{text.center(80)}")
    print('=' * 80)

def check_location_matching():
    """Check if whitelist locations match data locations"""
    
    print_header("LOCATION MATCHING DIAGNOSTIC")
    
    # Load whitelist
    try:
        whitelist = pd.read_csv('region_wise_popular_places_from_inference.csv')
        print(f"\n✓ Whitelist loaded: {len(whitelist)} locations")
    except Exception as e:
        print(f"\n❌ Failed to load whitelist: {e}")
        return
    
    # Load data
    try:
        data = pd.read_csv('data/inference_data.csv')
        print(f"✓ Data loaded: {len(data)} rows, {data['location'].nunique()} unique locations")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Check column names
    print(f"\n📋 Whitelist columns: {list(whitelist.columns)}")
    print(f"📋 Data columns: {list(data.columns)}")
    
    # Get location names
    whitelist_locations = set(whitelist['Place'].unique())
    data_locations = set(data['location'].unique())
    
    print(f"\n📍 Locations in whitelist: {len(whitelist_locations)}")
    print(f"📍 Locations in data: {len(data_locations)}")
    
    # Check exact matches
    exact_matches = whitelist_locations & data_locations
    print(f"\n✓ Exact name matches: {len(exact_matches)}")
    
    if exact_matches:
        print("  Matched locations:")
        for loc in sorted(list(exact_matches)[:10]):
            print(f"    • {loc}")
        if len(exact_matches) > 10:
            print(f"    ... and {len(exact_matches) - 10} more")
    
    # Check missing from data
    missing_in_data = whitelist_locations - data_locations
    if missing_in_data:
        print(f"\n❌ Whitelist locations NOT in data: {len(missing_in_data)}")
        print("  These locations are in whitelist but have NO data:")
        for loc in sorted(list(missing_in_data)[:10]):
            print(f"    • {loc}")
        if len(missing_in_data) > 10:
            print(f"    ... and {len(missing_in_data) - 10} more")
    
    # Check locations in data but not in whitelist
    extra_in_data = data_locations - whitelist_locations
    if extra_in_data:
        print(f"\n⚠️  Data locations NOT in whitelist: {len(extra_in_data)}")
        print("  These locations have data but are NOT in whitelist:")
        for loc in sorted(list(extra_in_data)[:10]):
            print(f"    • {loc}")
        if len(extra_in_data) > 10:
            print(f"    ... and {len(extra_in_data) - 10} more")
    
    # Check coordinate matching for Connaught Place specifically
    print_header("CHECKING 'CONNAUGHT PLACE (RAJIV CHOWK)'")
    
    cp_whitelist = whitelist[whitelist['Place'] == 'Connaught Place (Rajiv Chowk)']
    
    if len(cp_whitelist) > 0:
        wl_lat = cp_whitelist['Latitude'].iloc[0]
        wl_lon = cp_whitelist['Longitude'].iloc[0]
        print(f"\n✓ Found in whitelist:")
        print(f"  Coordinates: ({wl_lat}, {wl_lon})")
        
        # Search for matching data by coordinates
        print(f"\n🔍 Searching data for coordinates ({wl_lat}, {wl_lon})...")
        
        # Exact match
        exact_match = data[(data['lat'] == wl_lat) & (data['lon'] == wl_lon)]
        if len(exact_match) > 0:
            print(f"  ✓ Exact coordinate match: {len(exact_match)} rows")
            print(f"    Location name in data: '{exact_match['location'].iloc[0]}'")
        else:
            print(f"  ❌ No exact coordinate match")
            
            # Search with tolerance
            tolerances = [0.0001, 0.001, 0.01, 0.05]
            for tol in tolerances:
                nearby = data[
                    (np.abs(data['lat'] - wl_lat) < tol) & 
                    (np.abs(data['lon'] - wl_lon) < tol)
                ]
                if len(nearby) > 0:
                    print(f"\n  ⚠️  Found {len(nearby)} rows within ±{tol} degrees:")
                    unique_locs = nearby['location'].unique()
                    for loc in unique_locs[:5]:
                        loc_data = nearby[nearby['location'] == loc].iloc[0]
                        print(f"    • '{loc}'")
                        print(f"      Coords: ({loc_data['lat']}, {loc_data['lon']})")
                        print(f"      Distance: lat_diff={abs(loc_data['lat'] - wl_lat):.6f}, lon_diff={abs(loc_data['lon'] - wl_lon):.6f}")
                    break
            else:
                print(f"  ❌ No data found even with ±0.05 degree tolerance")
        
        # Check by location name
        print(f"\n🔍 Searching data by location name 'Connaught Place (Rajiv Chowk)'...")
        name_match = data[data['location'] == 'Connaught Place (Rajiv Chowk)']
        if len(name_match) > 0:
            print(f"  ✓ Found by name: {len(name_match)} rows")
            print(f"    Coordinates in data: ({name_match['lat'].iloc[0]}, {name_match['lon'].iloc[0]})")
        else:
            print(f"  ❌ No data with this exact location name")
            
            # Try partial matches
            partial = data[data['location'].str.contains('Connaught', case=False, na=False)]
            if len(partial) > 0:
                print(f"\n  ⚠️  Found {len(partial)} rows with 'Connaught' in name:")
                for loc in partial['location'].unique()[:5]:
                    print(f"    • '{loc}'")
            else:
                print(f"  ❌ No partial matches found")
    else:
        print("\n❌ 'Connaught Place (Rajiv Chowk)' not found in whitelist!")
    
    # Provide solution
    print_header("DIAGNOSIS & SOLUTION")
    
    if len(exact_matches) == 0:
        print("\n❌ CRITICAL ISSUE: No location names match between whitelist and data!")
        print("\nROOT CAUSE:")
        print("  The 'location' column in your data file uses different names than")
        print("  the 'Place' column in your whitelist CSV.")
        print("\nSOLUTIONS:")
        print("\n  Option 1: Update whitelist to use data location names")
        print("    • Regenerate whitelist from your data file")
        print("    • Use actual location names from data['location']")
        print("\n  Option 2: Update data location names to match whitelist")
        print("    • Standardize location names in your data file")
        print("    • Ensure exact string matching")
        print("\n  Option 3: Use coordinate-based matching")
        print("    • Modify app.py to match by coordinates only")
        print("    • Ignore location name differences")
        
    elif len(missing_in_data) > 0:
        print(f"\n⚠️  PARTIAL ISSUE: {len(missing_in_data)} whitelist locations have no data")
        print("\nSOLUTION:")
        print("  Remove these locations from whitelist OR add their data to the data file")
    
    else:
        print("\n✓ All whitelist locations found in data!")
        print("\nIf you're still getting zeros, the issue is likely:")
        print("  1. Server not restarted after adding data/models")
        print("  2. Wrong location name being used in frontend")
        print("  3. Bug in prediction logic (check server logs)")

def generate_matching_whitelist():
    """Generate a new whitelist from actual data locations"""
    
    print_header("GENERATING NEW WHITELIST FROM DATA")
    
    try:
        data = pd.read_csv('data/inference_data.csv')
        print(f"✓ Loaded data: {len(data)} rows")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Get unique locations with their coordinates
    location_summary = data.groupby('location').agg({
        'lat': 'first',
        'lon': 'first',
        'region': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
        'pincode': lambda x: x.mode()[0] if len(x.mode()) > 0 else ''
    }).reset_index()
    
    # Add Region if not present
    if 'region' not in location_summary.columns:
        location_summary['region'] = 'Delhi'
    
    # Rename columns to match whitelist format
    new_whitelist = pd.DataFrame({
        'Region': location_summary['region'],
        'Place': location_summary['location'],
        'Area/Locality': location_summary['location'],
        'PIN Code': location_summary['pincode'],
        'Latitude': location_summary['lat'],
        'Longitude': location_summary['lon']
    })
    
    # Save
    output_path = 'generated_whitelist.csv'
    new_whitelist.to_csv(output_path, index=False)
    
    print(f"\n✓ Generated new whitelist: {output_path}")
    print(f"  Contains {len(new_whitelist)} locations from your data")
    print("\nNext steps:")
    print(f"  1. Review {output_path}")
    print("  2. If looks good, replace: cp {output_path} region_wise_popular_places_from_inference.csv")
    print("  3. Restart server: uvicorn app:app --reload")
    
    print(f"\nPreview of generated whitelist:")
    print(new_whitelist.head(10).to_string())

def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("LOCATION MATCHING DIAGNOSTIC TOOL".center(80))
    print("=" * 80)
    
    check_location_matching()
    
    print("\n" + "=" * 80)
    response = input("\nWould you like to generate a new whitelist from your data? (y/n): ").strip().lower()
    
    if response == 'y':
        generate_matching_whitelist()
    
    print("\n")

if __name__ == "__main__":
    main()