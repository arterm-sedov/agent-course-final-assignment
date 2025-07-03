#!/usr/bin/env python3
"""
Test script to verify timestamp extraction from filenames.
"""

import os
import datetime
import sys

# Add parent directory to path to import from app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the function from app.py
from app import extract_timestamp_from_filename

def test_timestamp_extraction():
    """Test the timestamp extraction function with actual filenames."""
    
    # Test filenames from the logs directory
    test_filenames = [
        "INIT_20250704_000343.log",
        "INIT_20250703_153105.log", 
        "INIT_20250703_124712.log",
        "INIT_20250703_123454.log",
        "INIT_20250703_122618.log",
        "20250703_204903.score.txt",
        "20250703_204903.results.csv",
        "20250703_183225.score.txt",
        "20250703_183225.results.csv",
        "20250703_172226.score.txt",
        "20250703_135654.score.txt",
        "20250703_135654.results.csv",
        "20250703_135654.log",
        "20250703_094440.Score.csv",
        "20250703_094440.log",
        "20250702_202757.Score.csv",
        "20250702_202757.log",
        "20250702_192352.log",
        "20250702_021625.log",
        "20250630215711.md",
        "20250630043602.md",
        "20250630041113.md",
        "20250630023654.md",
        "LOG20250629_1.yaml",
        "LOG20250628_2.yaml",
        "LOG202506281412.md",
        "LOG20250628.yaml",
        "Score 60.log.csv",
        "Leaderboard 100pts 2025-07-02 090007.png"
    ]
    
    print("Testing timestamp extraction from filenames:")
    print("=" * 80)
    
    extracted_files = []
    non_extracted_files = []
    
    for filename in test_filenames:
        timestamp, dt = extract_timestamp_from_filename(filename)
        if timestamp and dt:
            extracted_files.append((filename, timestamp, dt))
            print(f"✅ {filename:<40} → {timestamp:<20} → {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            non_extracted_files.append(filename)
            print(f"❌ {filename:<40} → No timestamp found")
    
    print("\n" + "=" * 80)
    print(f"Successfully extracted timestamps from {len(extracted_files)} files")
    print(f"Failed to extract timestamps from {len(non_extracted_files)} files")
    
    if non_extracted_files:
        print("\nFiles without extractable timestamps:")
        for filename in non_extracted_files:
            print(f"  - {filename}")
    
    # Test sorting
    print("\n" + "=" * 80)
    print("Testing sorting by extracted timestamps:")
    
    # Sort by timestamp (newest first)
    extracted_files.sort(key=lambda x: x[2], reverse=True)
    
    for i, (filename, timestamp, dt) in enumerate(extracted_files, 1):
        print(f"{i:2d}. {dt.strftime('%Y-%m-%d %H:%M:%S')} → {filename}")
    
    return extracted_files, non_extracted_files

if __name__ == "__main__":
    test_timestamp_extraction() 