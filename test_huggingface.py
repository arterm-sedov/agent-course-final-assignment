#!/usr/bin/env python3
"""
Test script for HuggingFace LLM configuration
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_huggingface_config():
    """Test HuggingFace configuration and connectivity"""
    
    print("🔍 Testing HuggingFace Configuration...")
    
    # Check environment variables
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if hf_token:
        print("✅ HuggingFace API token found")
        print(f"   Token starts with: {hf_token[:10]}...")
    else:
        print("❌ No HuggingFace API token found")
        print("   Set HF_TOKEN or HUGGINGFACE_API_KEY in your .env file")
        return False
    
    # Test imports
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        print("✅ LangChain HuggingFace imports successful")
    except ImportError as e:
        print(f"❌ Failed to import LangChain HuggingFace: {e}")
        return False
    
    # Test basic endpoint connectivity
    try:
        import requests
        
        # Test the inference API endpoint
        headers = {"Authorization": f"Bearer {hf_token}"}
        response = requests.get(
            "https://api-inference.huggingface.co/models/gpt2",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ HuggingFace API connectivity successful")
        else:
            print(f"⚠️ HuggingFace API returned status {response.status_code}")
            
    except Exception as e:
        print(f"❌ HuggingFace API connectivity test failed: {e}")
        return False
    
    # Test LLM initialization
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        
        # Try with a simple model first
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="gpt2",
                task="text-generation",
                max_new_tokens=50,
                do_sample=False,
                temperature=0,
            ),
            verbose=True,
        )
        print("✅ HuggingFace LLM initialization successful")
        
        # Test a simple inference
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content="Hello, world!")])
        print("✅ HuggingFace LLM inference successful")
        print(f"   Response: {response.content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ HuggingFace LLM test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 HuggingFace Configuration Test")
    print("=" * 40)
    
    success = test_huggingface_config()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ All tests passed! HuggingFace should work correctly.")
    else:
        print("❌ Some tests failed. Check the configuration above.")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have a valid HuggingFace API token")
        print("2. Check your internet connection")
        print("3. Try using a different model or endpoint")
        print("4. Consider using Google Gemini or Groq as alternatives")

if __name__ == "__main__":
    main() 