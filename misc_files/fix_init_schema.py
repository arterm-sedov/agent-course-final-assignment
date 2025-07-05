#!/usr/bin/env python3
"""
Script to fix schema mismatch in init files by adding missing init_summary_json field.
"""

import json
import os
from pathlib import Path

def fix_init_files():
    """Add missing init_summary_json field to older init files."""
    dataset_dir = Path("dataset")
    
    # Files that need the field added (older files without init_summary_json)
    files_to_fix = [
        "init-20250703_122618.jsonl",
        "init-20250703_123454.jsonl", 
        "init-20250703_124712.jsonl",
        "init-20250703_153105.jsonl",
        "init-20250705_130855.jsonl",
        "init-20250705_131128.jsonl",
        "init-20250705_131406.jsonl",
        "init-20250705_131525.jsonl",
        "init-20250705_131702.jsonl",
        "init-20250705_131903.jsonl",
        "init-20250705_132104.jsonl",
        "init-20250705_132209.jsonl"
    ]
    
    for filename in files_to_fix:
        filepath = dataset_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue
            
        print(f"Processing {filename}...")
        
        # Read the file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Parse JSON
        data = json.loads(content)
        
        # Check if init_summary_json already exists
        if 'init_summary_json' in data:
            print(f"  {filename} already has init_summary_json field, skipping...")
            continue
            
        # Add the missing field with empty JSON string
        data['init_summary_json'] = "{}"
        
        # Write back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            
        print(f"  Added init_summary_json field to {filename}")
    
    print("Schema fix completed!")

if __name__ == "__main__":
    fix_init_files() 