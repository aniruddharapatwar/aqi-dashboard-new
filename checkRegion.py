#!/usr/bin/env python3
"""
Quick API Region Check
Shows what your API is currently returning
"""

import requests
import sys

def check_api_regions():
    """Check what regions the API is returning"""
    
    print("=" * 80)
    print("API REGION CHECK")
    print("=" * 80)
    
    api_url = "http://localhost:8000"
    
    # Test 1: Check if server is running
    print(f"\n1️⃣  Testing connection to {api_url}...")
    try:
        response = requests.get(f"{api_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("   ✓ Server is running")
        else:
            print(f"   ⚠️  Server responded with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to server!")
        print("   Make sure server is running: uvicorn app:app --reload")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Get regions from API
    print(f"\n2️⃣  Fetching regions from API...")
    try:
        response = requests.get(f"{api_url}/api/regions", timeout=5)
        if response.status_code == 200:
            regions = response.json()
            print(f"   ✓ API returned {len(regions)} regions:")
            print()
            for i, region in enumerate(regions, 1):
                print(f"      {i}. {region}")
            
            # Check if they're correct
            expected = ['Central Delhi', 'East Delhi', 'West Delhi', 'South Delhi', 'North Delhi']
            unexpected = [r for r in regions if r not in expected]
            
            if len(unexpected) > 0:
                print(f"\n   ⚠️  UNEXPECTED REGIONS FOUND:")
                for region in unexpected:
                    print(f"      • {region}")
                print(f"\n   These should not be there!")
                return regions, unexpected
            else:
                print(f"\n   ✓ All regions are correct!")
                return regions, []
        else:
            print(f"   ❌ API returned status {response.status_code}")
            return None, None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, None
    
    return True

def check_api_locations(region):
    """Check locations for a specific region"""
    
    api_url = "http://localhost:8000"
    
    print(f"\n3️⃣  Fetching locations for '{region}'...")
    try:
        response = requests.get(
            f"{api_url}/api/locations/{region}", 
            timeout=5
        )
        if response.status_code == 200:
            locations = response.json()
            print(f"   ✓ API returned {len(locations)} locations:")
            print()
            for i, location in enumerate(locations[:10], 1):
                print(f"      {i}. {location}")
            if len(locations) > 10:
                print(f"      ... and {len(locations) - 10} more")
            return locations
        else:
            print(f"   ❌ API returned status {response.status_code}")
            return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    """Main execution"""
    
    print("\n" + "=" * 80)
    print("QUICK API REGION CHECK")
    print("=" * 80)
    print("\nChecking what your API is currently returning...\n")
    
    # Check regions
    regions, unexpected = check_api_regions()
    
    if regions is None:
        print("\n❌ Could not fetch regions from API")
        print("\nMake sure:")
        print("  1. Server is running: uvicorn app:app --reload")
        print("  2. Server is on port 8000")
        print("  3. No firewall blocking")
        return 1
    
    # If unexpected regions found
    if unexpected and len(unexpected) > 0:
        print("\n" + "=" * 80)
        print("DIAGNOSIS")
        print("=" * 80)
        print(f"\n❌ Your API is returning unexpected regions: {unexpected}")
        print("\n🔍 Root cause:")
        print("   These regions are in your whitelist CSV file:")
        print("   region_wise_popular_places_from_inference.csv")
        print("\n💡 Solution:")
        print("   Run: python diagnose_regions.py")
        print("   This will help you fix the regions in your whitelist.")
        
        # Sample one unexpected region
        if len(unexpected) > 0:
            sample_region = unexpected[0]
            check_api_locations(sample_region)
    
    else:
        print("\n" + "=" * 80)
        print("✅ API IS CORRECT")
        print("=" * 80)
        print("\nYour API is returning the correct regions!")
        
        # Sample locations
        if regions and len(regions) > 0:
            check_api_locations(regions[0])
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if unexpected and len(unexpected) > 0:
        print("\n⚠️  Issue found: Unexpected regions in API")
        print("\n📋 Action items:")
        print("   1. Run: python diagnose_regions.py")
        print("   2. Fix regions in whitelist CSV")
        print("   3. Restart server: uvicorn app:app --reload")
        print("   4. Run this script again to verify")
    else:
        print("\n✓ Everything looks good!")
        print("\nIf dashboard still shows wrong regions:")
        print("   1. Clear browser cache (Ctrl+Shift+R)")
        print("   2. Check browser console (F12) for errors")
        print("   3. Verify API URL in frontend is correct")
    
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())