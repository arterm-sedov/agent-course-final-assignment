#!/usr/bin/env python3
"""
Test script to validate runs_new dataset schema
"""

import json
import sys
from utils import load_dataset_schema, validate_data_structure

def test_runs_new_schema():
    """Test the runs_new schema with mock data"""
    
    # Load the schema
    schema = load_dataset_schema()
    if not schema:
        print("❌ Failed to load dataset schema")
        return False
    
    # Check if runs_new split exists
    if "runs_new" not in schema.get("features", {}):
        print("❌ runs_new split not found in schema")
        return False
    
    # Get the expected features for runs_new
    expected_features = schema["features"]["runs_new"]
    print(f"✅ Found runs_new schema with {len(expected_features)} fields:")
    for field, config in expected_features.items():
        print(f"   - {field}: {config.get('dtype', 'unknown')}")
    
    # Create mock data
    mock_data = {
        "run_id": "20250705_180645_q01",
        "questions_count": "1/1",
        "input_data": json.dumps([{
            "task_id": "task_001",
            "question": "What is the capital of France?",
            "file_name": ""
        }]),
        "reference_answer": "Paris is the capital of France",
        "final_answer": "Paris",
        "reference_similarity": 0.95,
        "question": "What is the capital of France?",
        "file_name": "",
        "llm_used": "Google Gemini",
        "llm_stats_json": json.dumps({
            "models_used": ["Google Gemini"],
            "total_tokens": 150,
            "total_cost": 0.002
        }),
        "total_score": "85% (17/20 correct)",
        "error": "",
        "username": "arterm-sedov"
    }
    
    print(f"\n📋 Testing mock data structure...")
    
    # Validate the data structure
    is_valid = validate_data_structure(mock_data, "runs_new")
    
    if is_valid:
        print("✅ Mock data validates against runs_new schema")
        
        # Test JSON parsing of complex fields
        try:
            input_data = json.loads(mock_data["input_data"])
            llm_stats_json = json.loads(mock_data["llm_stats_json"])
            
            print("✅ JSON parsing successful for complex fields:")
            print(f"   - input_data: {len(input_data)} items")
            print(f"   - llm_stats_json: {len(llm_stats_json)} fields")
            
            # Test specific field content
            if input_data and len(input_data) > 0:
                first_input = input_data[0]
                print(f"   - task_id: {first_input.get('task_id')}")
                print(f"   - question: {first_input.get('question')}")
                print(f"   - file_name: {first_input.get('file_name')}")
            
            print(f"   - total_score: {mock_data.get('total_score')}")
            print(f"   - reference_similarity: {mock_data.get('reference_similarity')}")
            print(f"   - reference_answer: {mock_data.get('reference_answer')}")
            print(f"   - final_answer: {mock_data.get('final_answer')}")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            return False
            
        return True
    else:
        print("❌ Mock data failed validation against runs_new schema")
        return False

def test_file_upload():
    """Test uploading the mock data to the dataset"""
    from utils import upload_run_data
    
    mock_data = {
        "run_id": "20250705_180645_q01",
        "questions_count": "1/1",
        "input_data": json.dumps([{
            "task_id": "task_001",
            "question": "What is the capital of France?",
            "file_name": ""
        }]),
        "reference_answer": "Paris is the capital of France",
        "final_answer": "Paris",
        "reference_similarity": 0.95,
        "question": "What is the capital of France?",
        "file_name": "",
        "llm_used": "Google Gemini",
        "llm_stats_json": json.dumps({
            "models_used": ["Google Gemini"],
            "total_tokens": 150,
            "total_cost": 0.002
        }),
        "total_score": "85% (17/20 correct)",
        "error": "",
        "username": "arterm-sedov"
    }
    
    print(f"\n🚀 Testing file upload to runs_new split...")
    
    try:
        success = upload_run_data(mock_data, split="runs_new")
        if success:
            print("✅ Mock data uploaded successfully to runs_new split")
            return True
        else:
            print("❌ Mock data upload failed")
            return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing runs_new dataset schema and upload functionality")
    print("=" * 60)
    
    # Test schema validation
    schema_ok = test_runs_new_schema()
    
    # Test file upload (only if schema is valid)
    if schema_ok:
        upload_ok = test_file_upload()
    else:
        upload_ok = False
    
    print("\n" + "=" * 60)
    if schema_ok and upload_ok:
        print("🎉 All tests passed! runs_new schema is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above for details.")
        sys.exit(1) 