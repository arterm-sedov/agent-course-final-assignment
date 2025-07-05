#!/usr/bin/env python3
"""
Example script demonstrating HuggingFace Hub API file operations.
This script shows how to use CommitOperationAdd and related functionality.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def example_single_file_upload():
    """Example: Upload a single file using CommitOperationAdd"""
    print("📤 Example: Single file upload")
    
    try:
        from git_file_helper import upload_file_via_api
        
        # Create some test content
        content = f"""# Test File
Created at: {datetime.datetime.now()}

This is a test file uploaded using the HuggingFace Hub API
with CommitOperationAdd functionality.

## Features Demonstrated:
- Direct API upload (no git operations)
- Automatic commit message generation
- Error handling and logging
- Token-based authentication
"""
        
        # Upload the file
        success = upload_file_via_api(
            file_path="examples/single_upload_demo.md",
            content=content,
            commit_message="Example: Single file upload via API"
        )
        
        if success:
            print("✅ Single file upload successful!")
        else:
            print("❌ Single file upload failed")
            
        return success
        
    except Exception as e:
        print(f"❌ Error in single file upload example: {e}")
        return False

def example_batch_upload():
    """Example: Upload multiple files in a single commit"""
    print("\n📦 Example: Batch file upload")
    
    try:
        from git_file_helper import batch_upload_files
        
        # Prepare multiple files with different content types
        files_data = {
            "examples/batch_demo_1.txt": f"Text file created at {datetime.datetime.now()}",
            "examples/batch_demo_2.json": f'{{"type": "demo", "timestamp": "{datetime.datetime.now()}", "status": "success"}}',
            "examples/batch_demo_3.py": f'''# Demo Python File
# Created at: {datetime.datetime.now()}

def demo_function():
    """Demo function for batch upload example"""
    return "Hello from batch upload!"

if __name__ == "__main__":
    print(demo_function())
''',
            "examples/batch_demo_4.yaml": f"""# Demo YAML File
created_at: {datetime.datetime.now()}
type: demo
features:
  - api_upload
  - batch_operations
  - commit_operation_add
status: active
"""
        }
        
        # Upload all files in one commit
        results = batch_upload_files(
            files_data=files_data,
            commit_message="Example: Batch file upload via API"
        )
        
        # Show results
        success_count = sum(results.values())
        total_count = len(results)
        
        print(f"✅ Batch upload completed: {success_count}/{total_count} files")
        
        for file_path, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {file_path}")
            
        return success_count == total_count
        
    except Exception as e:
        print(f"❌ Error in batch upload example: {e}")
        return False

def example_log_file_upload():
    """Example: Demonstrate log file upload functionality"""
    print("\n📝 Example: Log file upload")
    
    try:
        from git_file_helper import upload_file_via_api
        
        # Create a sample log entry
        log_content = f"""Log Entry
Timestamp: {datetime.datetime.now()}
Level: INFO
Message: Example log file upload via API
Status: Success
Details: This demonstrates how to upload log files using CommitOperationAdd
"""
        
        # Upload the log file
        print("📤 Uploading log file...")
        upload_success = upload_file_via_api(
            file_path="examples/sample_log.txt",
            content=log_content,
            commit_message="Example: Log file upload"
        )
        
        if upload_success:
            print("✅ Log file upload successful")
        else:
            print("❌ Log file upload failed")
            
        return upload_success
        
    except Exception as e:
        print(f"❌ Error in log file upload example: {e}")
        return False

def example_enhanced_save_and_commit():
    """Example: Using the API-based save_and_commit_file function"""
    print("\n💾 Example: API-based save_and_commit_file")
    
    try:
        from git_file_helper import save_and_commit_file
        
        # Test content
        content = f"""API-based Save and Commit Demo
Created at: {datetime.datetime.now()}

This demonstrates the API-based save_and_commit_file function
which uses CommitOperationAdd for efficient file uploads.
"""
        
        # Use the API-based function
        success = save_and_commit_file(
            file_path="examples/api_demo.txt",
            content=content,
            commit_message="Example: API-based save_and_commit_file"
        )
        
        if success:
            print("✅ API-based save_and_commit_file successful!")
        else:
            print("❌ API-based save_and_commit_file failed")
            
        return success
        
    except Exception as e:
        print(f"❌ Error in API-based save_and_commit example: {e}")
        return False

def example_error_handling():
    """Example: Error handling with API operations"""
    print("\n⚠️ Example: Error handling")
    
    try:
        from git_file_helper import upload_file_via_api
        
        # Test with invalid repository (should fail gracefully)
        print("Testing error handling with invalid repo...")
        
        success = upload_file_via_api(
            file_path="test/error_handling.txt",
            content="This should fail gracefully",
            repo_id="invalid/repo/that/does/not/exist",
            commit_message="This should fail"
        )
        
        if not success:
            print("✅ Error handling working correctly - operation failed gracefully")
            return True
        else:
            print("⚠️ Unexpected success with invalid repo")
            return False
            
    except Exception as e:
        print(f"❌ Unexpected exception in error handling example: {e}")
        return False

def main():
    """Run all examples"""
    print("🚀 HuggingFace Hub API Examples")
    print("=" * 50)
    
    examples = [
        ("Single File Upload", example_single_file_upload),
        ("Batch File Upload", example_batch_upload),
        ("Log File Upload", example_log_file_upload),
        ("API-based Save and Commit", example_enhanced_save_and_commit),
        ("Error Handling", example_error_handling)
    ]
    
    results = []
    for example_name, example_func in examples:
        print(f"\n🧪 Running example: {example_name}")
        try:
            success = example_func()
            results.append((example_name, success))
        except Exception as e:
            print(f"❌ Example {example_name} failed with exception: {e}")
            results.append((example_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Example Results Summary:")
    print("=" * 50)
    
    passed = 0
    for example_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {example_name}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} examples passed")
    
    if passed == total:
        print("🎉 All examples passed! CommitOperationAdd integration is working correctly.")
    else:
        print("⚠️ Some examples failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 