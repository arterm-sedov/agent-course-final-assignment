#!/usr/bin/env python3
"""
Test script for HuggingFace Hub API file operations using CommitOperationAdd.
This script demonstrates the new API-based file upload functionality.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_hf_api_availability():
    """Test if huggingface_hub is available and working"""
    print("🔍 Testing HuggingFace Hub API availability...")
    
    try:
        from huggingface_hub import HfApi, CommitOperationAdd
        print("✅ huggingface_hub imports successful")
        return True
    except ImportError as e:
        print(f"❌ Failed to import huggingface_hub: {e}")
        return False

def test_api_client():
    """Test API client creation and authentication"""
    print("\n🔍 Testing API client creation...")
    
    try:
        from git_file_helper import get_hf_api_client, get_repo_info
        
        # Test API client creation
        api = get_hf_api_client()
        if api:
            print("✅ API client created successfully")
        else:
            print("❌ Failed to create API client")
            return False
            
        # Test repository info
        repo_id, repo_type = get_repo_info()
        if repo_id:
            print(f"✅ Repository info: {repo_id} ({repo_type})")
        else:
            print("❌ No repository info found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing API client: {e}")
        return False

def test_single_file_upload():
    """Test single file upload using CommitOperationAdd"""
    print("\n🔍 Testing single file upload via API...")
    
    try:
        from git_file_helper import upload_file_via_api
        
        # Test content
        test_content = f"Test file created at {datetime.datetime.now()}\nThis is a test of the CommitOperationAdd functionality."
        test_path = "test_files/api_test.txt"
        
        # Upload file
        success = upload_file_via_api(
            file_path=test_path,
            content=test_content,
            commit_message="Test: Single file upload via API"
        )
        
        if success:
            print("✅ Single file upload successful")
            return True
        else:
            print("❌ Single file upload failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in single file upload test: {e}")
        return False

def test_batch_file_upload():
    """Test batch file upload using multiple CommitOperationAdd operations"""
    print("\n🔍 Testing batch file upload via API...")
    
    try:
        from git_file_helper import batch_upload_files
        
        # Test files
        files_data = {
            "test_files/batch_test_1.txt": f"Batch test file 1 created at {datetime.datetime.now()}",
            "test_files/batch_test_2.txt": f"Batch test file 2 created at {datetime.datetime.now()}",
            "test_files/batch_test_3.json": '{"test": "data", "timestamp": "' + str(datetime.datetime.now()) + '"}'
        }
        
        # Upload files
        results = batch_upload_files(
            files_data=files_data,
            commit_message="Test: Batch file upload via API"
        )
        
        success_count = sum(results.values())
        total_count = len(results)
        
        print(f"✅ Batch upload completed: {success_count}/{total_count} files successful")
        
        for file_path, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {file_path}")
            
        return success_count == total_count
        
    except Exception as e:
        print(f"❌ Error in batch file upload test: {e}")
        return False

def test_log_file_upload():
    """Test log file upload functionality"""
    print("\n🔍 Testing log file upload...")
    
    try:
        from git_file_helper import upload_file_via_api
        
        # Test log file upload
        log_content = f"""Log Entry
Timestamp: {datetime.datetime.now()}
Level: INFO
Message: Test log file upload via API
Status: Success
"""
        test_path = "test_files/test_log.txt"
        
        print("📤 Uploading test log file...")
        upload_success = upload_file_via_api(
            file_path=test_path,
            content=log_content,
            commit_message="Test: Log file upload"
        )
        
        if upload_success:
            print("✅ Log file upload test successful")
        else:
            print("❌ Log file upload test failed")
            
        return upload_success
        
    except Exception as e:
        print(f"❌ Error in log file upload test: {e}")
        return False

def test_api_performance():
    """Test API upload performance"""
    print("\n🔍 Testing API upload performance...")
    
    try:
        from git_file_helper import upload_file_via_api
        import time
        
        test_content = f"Performance test at {datetime.datetime.now()}"
        
        # Test API upload
        print("📤 Testing API upload performance...")
        start_time = time.time()
        api_success = upload_file_via_api(
            file_path="test_files/performance_test.txt",
            content=test_content,
            commit_message="Test: API upload performance"
        )
        api_time = time.time() - start_time
        
        print(f"✅ Performance test results:")
        print(f"   API upload: {'✅' if api_success else '❌'} ({api_time:.2f}s)")
        
        return api_success
        
    except Exception as e:
        print(f"❌ Error in performance test: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting HuggingFace Hub API tests...")
    print("=" * 50)
    
    tests = [
        ("API Availability", test_hf_api_availability),
        ("API Client", test_api_client),
        ("Single File Upload", test_single_file_upload),
        ("Batch File Upload", test_batch_file_upload),
        ("Log File Upload", test_log_file_upload),
        ("API Performance", test_api_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running test: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CommitOperationAdd integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 