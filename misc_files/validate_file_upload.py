#!/usr/bin/env python3
"""
Validation script for file uploading functionality in agent.py and app.py
Tests data structure compatibility and upload functions.
"""

import sys
import os
import datetime
import json
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_helper import (
    upload_init_summary, 
    upload_evaluation_run, 
    validate_data_structure,
    get_dataset_features,
    print_dataset_schema
)

def find_file(filename):
    # Try current directory, then misc_files/../
    candidates = [Path(filename), Path(__file__).parent / ".." / filename]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None

def test_init_data_structure():
    """Test init data structure from agent.py"""
    print("🧪 Testing Init Data Structure (agent.py)")
    print("=" * 50)
    
    # Get expected features
    init_features = get_dataset_features('init')
    if not init_features:
        print("❌ No init features found in schema")
        return False
    
    print(f"✅ Expected init features: {list(init_features.keys())}")
    
    # Create sample init data (matching agent.py structure)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_init_data = {
        "timestamp": timestamp,
        "init_summary": "Test initialization summary",
        "debug_output": "Test debug output",
        "llm_config": {"test": "config"},
        "available_models": {"test": "models"},
        "tool_support": {"test": "support"}
    }
    
    # Validate structure
    is_valid = validate_data_structure(sample_init_data, 'init')
    print(f"✅ Init data structure validation: {'PASS' if is_valid else 'FAIL'}")
    
    return is_valid

def test_runs_data_structure():
    """Test runs data structure from app.py"""
    print("\n🧪 Testing Runs Data Structure (app.py)")
    print("=" * 50)
    
    # Get expected features
    runs_features = get_dataset_features('runs')
    if not runs_features:
        print("❌ No runs features found in schema")
        return False
    
    print(f"✅ Expected runs features: {list(runs_features.keys())}")
    
    # Create sample runs data (matching app.py structure)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    sample_runs_data = {
        "run_id": run_id,
        "timestamp": timestamp,
        "questions_count": 5,
        "results_log": [{"test": "log"}],
        "results_df": [{"test": "df"}],
        "username": "test_user",
        "final_status": "Test status",
        "score_path": "test/path"
    }
    
    # Validate structure
    is_valid = validate_data_structure(sample_runs_data, 'runs')
    print(f"✅ Runs data structure validation: {'PASS' if is_valid else 'FAIL'}")
    
    return is_valid

def test_upload_functions():
    """Test upload functions availability and basic functionality"""
    print("\n🧪 Testing Upload Functions")
    print("=" * 50)
    
    # Test function availability
    functions_available = all([
        upload_init_summary is not None,
        upload_evaluation_run is not None
    ])
    print(f"✅ Upload functions available: {'PASS' if functions_available else 'FAIL'}")
    
    # Test function signatures
    try:
        import inspect
        init_sig = inspect.signature(upload_init_summary)
        runs_sig = inspect.signature(upload_evaluation_run)
        print(f"✅ upload_init_summary signature: {init_sig}")
        print(f"✅ upload_evaluation_run signature: {runs_sig}")
        signature_ok = True
    except Exception as e:
        print(f"❌ Error checking function signatures: {e}")
        signature_ok = False
    
    return functions_available and signature_ok

def test_agent_imports():
    """Test that agent.py can import upload functions"""
    print("\n🧪 Testing Agent.py Imports")
    print("=" * 50)
    
    try:
        agent_path = find_file("agent.py")
        if not agent_path:
            print("❌ agent.py not found in any expected location")
            return False
        agent_source = agent_path.read_text()
        if "upload_init_summary" in agent_source:
            print("✅ agent.py uses upload_init_summary")
        else:
            print("❌ agent.py does not use upload_init_summary")
        if "from file_helper import" in agent_source:
            print("✅ agent.py imports from file_helper")
        else:
            print("❌ agent.py does not import from file_helper")
        return True
    except Exception as e:
        print(f"❌ Error checking agent.py: {e}")
        return False

def test_app_imports():
    """Test that app.py can import upload functions"""
    print("\n🧪 Testing App.py Imports")
    print("=" * 50)
    
    try:
        app_path = find_file("app.py")
        if not app_path:
            print("❌ app.py not found in any expected location")
            return False
        app_source = app_path.read_text()
        if "upload_evaluation_run" in app_source:
            print("✅ app.py uses upload_evaluation_run")
        else:
            print("❌ app.py does not use upload_evaluation_run")
        if "from file_helper import" in app_source:
            print("✅ app.py imports from file_helper")
        else:
            print("❌ app.py does not import from file_helper")
        return True
    except Exception as e:
        print(f"❌ Error checking app.py: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 File Upload Validation Test")
    print("=" * 60)
    
    # Print dataset schema for reference
    print_dataset_schema()
    
    # Run all tests
    tests = [
        test_init_data_structure,
        test_runs_data_structure,
        test_upload_functions,
        test_agent_imports,
        test_app_imports
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed! File uploading is ready.")
        return True
    else:
        print("⚠️ Some validation tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 