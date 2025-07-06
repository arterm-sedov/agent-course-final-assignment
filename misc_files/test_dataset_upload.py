#!/usr/bin/env python3
"""
Test script for dataset upload functionality.
Uploads selected log files from logs/ directory to HuggingFace datasets.
Validates data against schema before uploading.

This script is located in misc_files/ and should be run from the parent directory
or with proper path setup to access the main project files.
"""

import os
import json
from pathlib import Path
import sys
import os
# Add parent directory to path to import file_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import upload_init_summary, upload_run_data, TRACES_DIR
from dotenv import load_dotenv
load_dotenv()



def load_schema():
    """Load the dataset schema from dataset_config.json."""
    # Try multiple possible locations for the config file
    possible_paths = [
        Path("../dataset_config.json"),  # When run from misc_files/
        Path("dataset_config.json"),     # When run from root directory
        Path("./dataset_config.json"),   # When run from root directory
    ]
    
    config_path = None
    for path in possible_paths:
        if path.exists():
            config_path = path
            break
    
    if not config_path:
        print("❌ dataset_config.json not found in any expected location")
        print("   Tried:", [str(p) for p in possible_paths])
        return None
    if not config_path.exists():
        print("❌ dataset_config.json not found")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract features schema
        if 'features' in config:
            return config['features']
        
        print("❌ No features schema found in dataset_config.json")
        return None
    except Exception as e:
        print(f"❌ Error loading schema: {e}")
        return None

def validate_init_data(data, schema):
    """Validate init data against schema."""
    if not schema or 'init' not in schema:
        print("❌ No init schema found")
        return False
    
    init_schema = schema['init']
    required_fields = list(init_schema.keys())
    
    # Check for required fields
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        return False
    
    # Check data types
    type_errors = []
    for field, value in data.items():
        if field not in init_schema:
            continue
            
        expected_type = init_schema[field]['dtype']
        actual_type = type(value).__name__
        
        # Type validation
        if expected_type == 'string' and not isinstance(value, str):
            type_errors.append(f"{field}: expected string, got {actual_type}")
        elif expected_type == 'int64' and not isinstance(value, int):
            type_errors.append(f"{field}: expected int, got {actual_type}")
    
    if type_errors:
        print(f"❌ Type validation errors: {type_errors}")
        return False
    
    print("✅ Init data validation passed")
    return True

def validate_runs_data(data, schema):
    """Validate runs data against schema."""
    if not schema or 'runs' not in schema:
        print("❌ No runs schema found")
        return False
    
    runs_schema = schema['runs']
    required_fields = list(runs_schema.keys())
    
    # Check for required fields
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        return False
    
    # Check data types
    type_errors = []
    for field, value in data.items():
        if field not in runs_schema:
            continue
            
        expected_type = runs_schema[field]['dtype']
        actual_type = type(value).__name__
        
        # Type validation
        if expected_type == 'string' and not isinstance(value, str):
            type_errors.append(f"{field}: expected string, got {actual_type}")
        elif expected_type == 'int64' and not isinstance(value, int):
            type_errors.append(f"{field}: expected int, got {actual_type}")
    
    if type_errors:
        print(f"❌ Type validation errors: {type_errors}")
        return False
    
    print("✅ Runs data validation passed")
    return True

# Hardcoded file paths for testing (try multiple possible locations)
def find_log_file(filename):
    """Find log file in multiple possible locations."""
    possible_paths = [
        Path(f"../logs/{filename}"),  # When run from misc_files/
        Path(f"logs/{filename}"),     # When run from root directory
        Path(f"./logs/{filename}"),   # When run from root directory
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    return Path(f"logs/{filename}")  # Return default path for error messages

INIT_FILE = find_log_file("20250705_132104_init.log")
LLM_TRACE_FILE = find_log_file("20250703_094440.log")
SCORE_FILE = find_log_file("20250703_135654.score.txt")
RESULTS_FILE = find_log_file("20250703_135654.results.csv")
TIMESTAMP = "20250703_135654"

def read_log_file(file_path):
    """Read log file content."""
    if not file_path or not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        return None

def test_init_upload():
    """Test uploading init log to dataset."""
    print(f"\n📤 Testing init upload for: {INIT_FILE}")
    
    # Load schema for validation
    schema = load_schema()
    if not schema:
        print("❌ Cannot validate without schema")
        return False
    
    init_content = read_log_file(INIT_FILE)
    if not init_content:
        print("❌ Could not read init file")
        return False
    
    # Create structured init data (serialized as strings to match schema)
    init_data = {
        "timestamp": TIMESTAMP,
        "init_summary": init_content,
        "debug_output": init_content,
        "llm_config": json.dumps({"test": "Test configuration"}),
        "available_models": json.dumps({"test": "Test models"}),
        "tool_support": json.dumps({"test": "Test tool support"})
    }
    
    # Validate data before upload
    if not validate_init_data(init_data, schema):
        print("❌ Init data validation failed")
        return False
    
    success = upload_init_summary(init_data)
    if success:
        print(f"✅ Init upload successful for {INIT_FILE}")
    else:
        print(f"❌ Init upload failed for {INIT_FILE}")
    return success

def test_evaluation_upload():
    """Test uploading evaluation run to dataset."""
    print(f"\n📤 Testing evaluation upload for: {LLM_TRACE_FILE}, {SCORE_FILE}, {RESULTS_FILE}")
    
    # Load schema for validation
    schema = load_schema()
    if not schema:
        print("❌ Cannot validate without schema")
        return False
    
    llm_content = read_log_file(LLM_TRACE_FILE)
    score_content = read_log_file(SCORE_FILE)
    results_content = read_log_file(RESULTS_FILE)
    
    if not llm_content:
        print("❌ Could not read LLM trace file")
        return False
    
    # Parse LLM trace as JSON if possible
    try:
        llm_data = json.loads(llm_content)
    except json.JSONDecodeError:
        llm_data = llm_content
    
    run_data = {
        "run_id": f"test_run_{TIMESTAMP}",
        "timestamp": TIMESTAMP,
        "questions_count": len(llm_data) if isinstance(llm_data, list) else 1,
        "results_log": json.dumps(llm_data if isinstance(llm_data, list) else [llm_data]),
        "results_df": json.dumps(llm_data if isinstance(llm_data, list) else [llm_data]),
        "username": "test_user",
        "final_status": score_content if score_content else "Test status",
        "score_path": str(SCORE_FILE) if SCORE_FILE else "test_score.txt"
    }
    
    # Validate data before upload
    if not validate_runs_data(run_data, schema):
        print("❌ Runs data validation failed")
        return False
    
    success = upload_run_data(run_data)
    if success:
        print(f"✅ Evaluation upload successful for {LLM_TRACE_FILE}")
    else:
        print(f"❌ Evaluation upload failed for {LLM_TRACE_FILE}")
    return success

def main():
    print("🧪 Testing Dataset Upload Functionality (Hardcoded Files)")
    print("=" * 50)
    
    # Load and validate schema first
    schema = load_schema()
    if not schema:
        print("❌ Cannot proceed without valid schema")
        return
    
    print("✅ Schema loaded successfully")
    print(f"   Available splits: {list(schema.keys())}")
    
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        print("❌ No HuggingFace token found in environment variables")
        print("   Please set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN")
        return
    
    print(f"✅ HuggingFace token found")
    
    success_count = 0
    total_count = 0
    
    if INIT_FILE.exists():
        if test_init_upload():
            success_count += 1
        total_count += 1
    
    if LLM_TRACE_FILE.exists():
        if test_evaluation_upload():
            success_count += 1
        total_count += 1
    
    print(f"\n📊 Test Summary")
    print("=" * 50)
    print(f"Total uploads attempted: {total_count}")
    print(f"Successful uploads: {success_count}")
    print(f"Success rate: {success_count/total_count*100:.1f}%" if total_count > 0 else "N/A")
    
    if success_count > 0:
        print(f"\n✅ Dataset upload functionality is working!")
        print(f"   Check your HuggingFace dataset:")
        print(f"   - arterm-sedov/agent-course-final-assignment")
        print(f"   - Init data goes to 'init' split")
        print(f"   - Evaluation data goes to 'runs' split")
    else:
        print(f"\n❌ Dataset upload functionality failed")
        print(f"   Check your HuggingFace token and dataset permissions")

if __name__ == "__main__":
    main() 