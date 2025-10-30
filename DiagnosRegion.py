#!/usr/bin/env python3
"""
Region Diagnostic Tool
Checks what regions are in your whitelist CSV and fixes them
"""

import pandas as pd
from pathlib import Path
import sys

def diagnose_regions():
    """Check what regions are actually in the whitelist"""
    
    print("=" * 80)
    print("REGION DIAGNOSTIC TOOL")
    print("=" * 80)
    
    whitelist_path = Path("region_wise_popular_places_from_inference.csv")
    
    if not whitelist_path.exists():
        print(f"\n❌ Whitelist not found: {whitelist_path}")
        return
    
    # Load whitelist
    try:
        df = pd.read_csv(whitelist_path)
        print(f"\n✓ Loaded whitelist: {len(df)} locations")
    except Exception as e:
        print(f"\n❌ Error reading whitelist: {e}")
        return
    
    # Check columns
    print(f"\n📋 Columns in file: {list(df.columns)}")
    
    if 'Region' not in df.columns:
        print("\n❌ 'Region' column not found!")
        print("Available columns:", list(df.columns))
        return
    
    # Get unique regions
    regions = df['Region'].unique()
    print(f"\n📍 Found {len(regions)} unique regions in whitelist:")
    print("-" * 80)
    
    for i, region in enumerate(sorted(regions), 1):
        count = len(df[df['Region'] == region])
        print(f"  {i}. {region:30s} ({count} locations)")
    
    # Show sample locations per region
    print("\n" + "=" * 80)
    print("LOCATIONS BY REGION")
    print("=" * 80)
    
    for region in sorted(regions):
        region_locs = df[df['Region'] == region]
        print(f"\n📍 {region} ({len(region_locs)} locations):")
        for i, (idx, row) in enumerate(region_locs.head(5).iterrows(), 1):
            print(f"    {i}. {row['Place']}")
        if len(region_locs) > 5:
            print(f"    ... and {len(region_locs) - 5} more")
    
    # Check for problematic regions
    print("\n" + "=" * 80)
    print("REGION ANALYSIS")
    print("=" * 80)
    
    expected_regions = [
        'Central Delhi',
        'East Delhi', 
        'West Delhi',
        'South Delhi',
        'North Delhi'
    ]
    
    print("\n✅ Expected regions:")
    for region in expected_regions:
        print(f"    • {region}")
    
    print("\n📊 Actual regions in file:")
    unexpected = []
    missing = []
    
    for region in sorted(regions):
        if region in expected_regions:
            print(f"    ✓ {region} (correct)")
        else:
            print(f"    ⚠️  {region} (UNEXPECTED)")
            unexpected.append(region)
    
    for region in expected_regions:
        if region not in regions:
            missing.append(region)
            print(f"    ❌ {region} (MISSING)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if len(unexpected) > 0:
        print(f"\n⚠️  Found {len(unexpected)} unexpected regions:")
        for region in unexpected:
            print(f"    • {region}")
        print("\n💡 These regions are showing up in your API but shouldn't be there.")
    
    if len(missing) > 0:
        print(f"\n❌ Missing {len(missing)} expected regions:")
        for region in missing:
            print(f"    • {region}")
    
    if len(unexpected) == 0 and len(missing) == 0:
        print("\n✓ All regions are correct!")
    
    return df, unexpected, missing

def fix_regions(df, unexpected, missing):
    """Fix incorrect regions in the whitelist"""
    
    if len(unexpected) == 0:
        print("\n✓ No regions to fix!")
        return
    
    print("\n" + "=" * 80)
    print("REGION FIXING")
    print("=" * 80)
    
    print("\n🔧 Let's fix the unexpected regions...")
    print("\nOptions:")
    print("  1. Map unexpected regions to correct ones")
    print("  2. Delete locations with unexpected regions")
    print("  3. Keep as is (cancel)")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == '1':
        # Map regions
        region_mapping = {}
        
        correct_regions = [
            'Central Delhi', 'East Delhi', 'West Delhi', 
            'South Delhi', 'North Delhi'
        ]
        
        print("\n📝 Map each unexpected region to a correct region:\n")
        
        for unexpected_region in unexpected:
            print(f"\nUnexpected region: '{unexpected_region}'")
            print("Map to:")
            for i, region in enumerate(correct_regions, 1):
                print(f"  {i}. {region}")
            print(f"  {len(correct_regions) + 1}. Delete locations with this region")
            
            selection = input(f"\nSelect (1-{len(correct_regions) + 1}): ").strip()
            
            try:
                selection = int(selection)
                if 1 <= selection <= len(correct_regions):
                    region_mapping[unexpected_region] = correct_regions[selection - 1]
                    print(f"  ✓ Will map '{unexpected_region}' → '{correct_regions[selection - 1]}'")
                elif selection == len(correct_regions) + 1:
                    region_mapping[unexpected_region] = None
                    print(f"  ✓ Will delete locations in '{unexpected_region}'")
            except ValueError:
                print(f"  ⚠️  Invalid input, skipping '{unexpected_region}'")
        
        # Apply mapping
        df_fixed = df.copy()
        deleted_count = 0
        
        for old_region, new_region in region_mapping.items():
            if new_region is None:
                # Delete
                count = len(df_fixed[df_fixed['Region'] == old_region])
                df_fixed = df_fixed[df_fixed['Region'] != old_region]
                deleted_count += count
                print(f"\n  🗑️  Deleted {count} locations from '{old_region}'")
            else:
                # Map
                count = len(df_fixed[df_fixed['Region'] == old_region])
                df_fixed.loc[df_fixed['Region'] == old_region, 'Region'] = new_region
                print(f"\n  ✓ Mapped {count} locations: '{old_region}' → '{new_region}'")
        
        # Save
        backup_path = "region_wise_popular_places_BACKUP.csv"
        df.to_csv(backup_path, index=False)
        print(f"\n💾 Original backed up to: {backup_path}")
        
        output_path = "region_wise_popular_places_from_inference.csv"
        df_fixed.to_csv(output_path, index=False)
        print(f"✓ Fixed file saved to: {output_path}")
        
        print(f"\n📊 Summary:")
        print(f"  • Original locations: {len(df)}")
        print(f"  • Updated locations: {len(df_fixed)}")
        print(f"  • Deleted: {deleted_count}")
        
        print("\n✅ Regions fixed!")
        print("\n⚠️  IMPORTANT: Restart your server for changes to take effect:")
        print("   uvicorn app:app --reload")
    
    elif choice == '2':
        # Delete unexpected regions
        df_fixed = df.copy()
        deleted_count = 0
        
        for region in unexpected:
            count = len(df_fixed[df_fixed['Region'] == region])
            df_fixed = df_fixed[df_fixed['Region'] != region]
            deleted_count += count
            print(f"\n  🗑️  Deleted {count} locations from '{region}'")
        
        # Save
        backup_path = "region_wise_popular_places_BACKUP.csv"
        df.to_csv(backup_path, index=False)
        print(f"\n💾 Original backed up to: {backup_path}")
        
        output_path = "region_wise_popular_places_from_inference.csv"
        df_fixed.to_csv(output_path, index=False)
        print(f"✓ Fixed file saved to: {output_path}")
        
        print(f"\n📊 Summary:")
        print(f"  • Original locations: {len(df)}")
        print(f"  • Updated locations: {len(df_fixed)}")
        print(f"  • Deleted: {deleted_count}")
        
        print("\n✅ Regions cleaned!")
        print("\n⚠️  IMPORTANT: Restart your server for changes to take effect:")
        print("   uvicorn app:app --reload")
    
    else:
        print("\n❌ Cancelled - no changes made")

def check_data_regions():
    """Check what regions are in the data file"""
    
    print("\n" + "=" * 80)
    print("CHECKING DATA FILE REGIONS")
    print("=" * 80)
    
    data_path = Path("data/inference_data.csv")
    
    if not data_path.exists():
        print(f"\n⚠️  Data file not found: {data_path}")
        return
    
    try:
        df = pd.read_csv(data_path)
        print(f"\n✓ Loaded data file: {len(df)} rows")
    except Exception as e:
        print(f"\n❌ Error reading data: {e}")
        return
    
    if 'region' not in df.columns:
        print("\n⚠️  'region' column not found in data")
        return
    
    regions = df['region'].unique()
    print(f"\n📍 Regions in data file ({len(regions)} unique):")
    
    for region in sorted(regions):
        count = len(df[df['region'] == region])
        print(f"  • {region:30s} ({count} rows)")

def main():
    """Main execution"""
    
    print("\n" + "=" * 80)
    print("REGION DIAGNOSTIC & FIX TOOL")
    print("=" * 80)
    print("\nThis tool will:")
    print("  1. Show what regions are in your whitelist CSV")
    print("  2. Identify unexpected regions")
    print("  3. Help you fix them")
    print()
    
    # Diagnose
    result = diagnose_regions()
    
    if result is None:
        return 1
    
    df, unexpected, missing = result
    
    # Check data file too
    check_data_regions()
    
    # Offer to fix
    if len(unexpected) > 0:
        print("\n" + "=" * 80)
        fix = input("\nWould you like to fix the unexpected regions? (y/n): ").strip().lower()
        
        if fix == 'y':
            fix_regions(df, unexpected, missing)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. If you made changes, restart server:
   uvicorn app:app --reload

2. Test API:
   curl http://localhost:8000/api/regions

3. Verify in browser:
   - Clear cache (Ctrl+Shift+R)
   - Check dropdown shows correct regions
    """)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())