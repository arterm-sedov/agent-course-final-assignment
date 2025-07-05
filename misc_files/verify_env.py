#!/usr/bin/env python3
"""
Environment variable verification script for HuggingFace Hub API testing.
Run this script to check if your environment variables are set correctly.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
from dotenv import load_dotenv

def verify_environment():
    """Verify that all required environment variables are set correctly."""
    print("🔍 Verifying environment variables for HuggingFace Hub API...")
    print("=" * 60)
    
    # Load .env file if it exists
    load_dotenv()
    
    # Check HF_TOKEN
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        if hf_token.startswith("hf_"):
            print("✅ HF_TOKEN: Set correctly")
            print(f"   Token starts with: {hf_token[:10]}...")
        else:
            print("⚠️ HF_TOKEN: Set but doesn't start with 'hf_'")
            print(f"   Token: {hf_token[:20]}...")
    else:
        print("❌ HF_TOKEN: Not set")
        print("   Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN environment variable")
    
    # Check SPACE_ID
    space_id = os.getenv("SPACE_ID")
    if space_id:
        if "/" in space_id:
            print("✅ SPACE_ID: Set correctly")
            print(f"   Repository: {space_id}")
        else:
            print("⚠️ SPACE_ID: Set but format may be incorrect")
            print(f"   Expected format: username/repository-name")
            print(f"   Current value: {space_id}")
    else:
        print("❌ SPACE_ID: Not set")
        print("   Set SPACE_ID environment variable")
    
    # Check REPO_TYPE
    repo_type = os.getenv("REPO_TYPE", "space")
    print(f"✅ REPO_TYPE: {repo_type}")
    
    # Check if huggingface_hub is available
    try:
        from huggingface_hub import HfApi
        print("✅ huggingface_hub: Available")
    except ImportError:
        print("❌ huggingface_hub: Not installed")
        print("   Install with: pip install huggingface_hub")
    
    print("=" * 60)
    
    # Summary
    if hf_token and space_id and "/" in space_id:
        print("🎉 All required environment variables are set correctly!")
        print("   You can now run the test scripts:")
        print("   - python test_hf_api_upload.py")
        print("   - python example_api_usage.py")
        return True
    else:
        print("⚠️ Some environment variables are missing or incorrect.")
        print("   Please check the setup guide: LOCAL_TESTING_SETUP.md")
        return False

def test_api_connection():
    """Test the API connection with current environment variables."""
    print("\n🔍 Testing API connection...")
    
    try:
        from git_file_helper import get_hf_api_client, get_repo_info
        
        # Test API client
        api = get_hf_api_client()
        if not api:
            print("❌ Failed to create API client")
            return False
        
        # Test repository info
        repo_id, repo_type = get_repo_info()
        if not repo_id:
            print("❌ Failed to get repository info")
            return False
        
        print(f"✅ API connection successful")
        print(f"   Repository: {repo_id}")
        print(f"   Type: {repo_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def main():
    """Run environment verification and API connection test."""
    print("🚀 HuggingFace Hub API Environment Verification")
    print("=" * 60)
    
    # Verify environment variables
    env_ok = verify_environment()
    
    if env_ok:
        # Test API connection
        api_ok = test_api_connection()
        
        if api_ok:
            print("\n🎉 Everything is set up correctly!")
            print("   You can now use the HuggingFace Hub API functions.")
        else:
            print("\n⚠️ Environment variables are set but API connection failed.")
            print("   Check your token permissions and repository access.")
    else:
        print("\n❌ Please fix the environment variables before testing.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 