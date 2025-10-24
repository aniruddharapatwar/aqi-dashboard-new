#!/usr/bin/env python3
"""
Automated Patch Script for index.html
Applies weather data fixes to the frontend
"""

import re
import sys
from pathlib import Path

def apply_patches(html_content):
    """Apply all necessary patches to index.html"""
    
    patches_applied = []
    
    # PATCH 1: Fix loadPredictions() to extract weather from API
    old_pattern_1 = r"// Extract weather data from current_data if available\s+if \(data\.current_data\) \{\s+this\.weather = \{\s+temperature: Math\.round\(data\.current_data\.temperature \|\| 0\),\s+humidity: Math\.round\(data\.current_data\.humidity \|\| 0\),\s+windSpeed: Math\.round\(data\.current_data\.windSpeed \|\| 0\)\s+\};\s+\}"
    
    new_code_1 = """// FIXED: Extract weather data from API response
                if (data.weather) {
                    this.weather = {
                        temperature: Math.round(data.weather.temperature || 0),
                        humidity: Math.round(data.weather.humidity || 0),
                        windSpeed: Math.round(data.weather.windSpeed || 0)
                    };
                    console.log('Weather data loaded:', this.weather);
                } else {
                    console.warn('No weather data in API response');
                }"""
    
    if re.search(old_pattern_1, html_content, re.DOTALL):
        html_content = re.sub(old_pattern_1, new_code_1, html_content, flags=re.DOTALL)
        patches_applied.append("✓ PATCH 1: Fixed loadPredictions() weather extraction")
    else:
        # Try alternative pattern
        old_pattern_1_alt = r"if \(data\.current_data\) \{\s+this\.weather = \{[^}]+\};\s+\}"
        if re.search(old_pattern_1_alt, html_content, re.DOTALL):
            html_content = re.sub(old_pattern_1_alt, new_code_1, html_content, flags=re.DOTALL)
            patches_applied.append("✓ PATCH 1: Fixed loadPredictions() weather extraction (alt)")
    
    # PATCH 2: Fix sendMessage() to include weather in chat context
    old_pattern_2 = r"aqi_data: this\.predictions\?\.overall\?\.\[this\.horizon\],"
    
    new_code_2 = """aqi_data: {
                            ...this.predictions?.overall?.[this.horizon],
                            weather: this.weather  // Include weather data
                        },"""
    
    if re.search(old_pattern_2, html_content):
        html_content = re.sub(old_pattern_2, new_code_2, html_content)
        patches_applied.append("✓ PATCH 2: Fixed sendMessage() to include weather data")
    
    return html_content, patches_applied


def main():
    """Main patch application function"""
    
    print("=" * 70)
    print("AQI DASHBOARD - INDEX.HTML PATCHER")
    print("=" * 70)
    print()
    
    # Find index.html
    possible_paths = [
        Path("index.html"),
        Path("../index.html"),
        Path("/mnt/project/index.html"),
        Path.cwd() / "index.html"
    ]
    
    index_path = None
    for path in possible_paths:
        if path.exists():
            index_path = path
            break
    
    if not index_path:
        print("❌ ERROR: index.html not found!")
        print("   Searched locations:")
        for path in possible_paths:
            print(f"   - {path}")
        print()
        print("   Please run this script from the project directory")
        print("   or provide the path to index.html")
        sys.exit(1)
    
    print(f"📄 Found index.html at: {index_path}")
    print()
    
    # Backup original
    backup_path = index_path.parent / f"{index_path.name}.backup"
    print(f"💾 Creating backup: {backup_path}")
    
    try:
        content = index_path.read_text(encoding='utf-8')
        backup_path.write_text(content, encoding='utf-8')
        print("✓ Backup created successfully")
        print()
    except Exception as e:
        print(f"❌ ERROR creating backup: {e}")
        sys.exit(1)
    
    # Apply patches
    print("🔧 Applying patches...")
    print()
    
    try:
        patched_content, patches = apply_patches(content)
        
        if not patches:
            print("⚠️  WARNING: No patches were applied!")
            print("   This could mean:")
            print("   1. Patches were already applied")
            print("   2. Code structure has changed")
            print("   3. Manual updates are needed")
            print()
            print("   Please review INDEX_HTML_UPDATES.txt for manual instructions")
            sys.exit(0)
        
        # Show applied patches
        for patch in patches:
            print(f"  {patch}")
        print()
        
        # Write patched file
        print(f"💾 Writing patched file to: {index_path}")
        index_path.write_text(patched_content, encoding='utf-8')
        print("✓ File updated successfully")
        print()
        
    except Exception as e:
        print(f"❌ ERROR applying patches: {e}")
        print(f"   Restoring from backup...")
        try:
            content = backup_path.read_text(encoding='utf-8')
            index_path.write_text(content, encoding='utf-8')
            print("✓ Original file restored")
        except:
            print("❌ ERROR: Could not restore backup!")
        sys.exit(1)
    
    # Success message
    print("=" * 70)
    print("✅ PATCHING COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Restart your application")
    print("2. Test weather data display")
    print("3. Test Gemini chat integration")
    print()
    print("If you encounter issues:")
    print("- Check FIX_SUMMARY.txt for troubleshooting")
    print("- Restore from backup: index.html.backup")
    print("- Review INDEX_HTML_UPDATES.txt for manual updates")
    print()


if __name__ == "__main__":
    main()