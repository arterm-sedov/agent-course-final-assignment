#!/usr/bin/env python3
"""
Test script for HuggingFace LLM configuration
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_huggingface_config():
    """Test HuggingFace configuration and connectivity"""
    
    print("🔍 Testing HuggingFace Configuration...")
    
    # Check environment variables
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
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
        from langchain_core.messages import HumanMessage
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
    
    # Test LLM initialization with improved configuration
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        
        # Try with the improved configuration (more reliable models first)
        models_to_try = [
            {
                "repo_id": "microsoft/DialoGPT-medium",
                "task": "text-generation",
                "max_new_tokens": 100,  # Very short for testing
                "do_sample": False,
                "temperature": 0
            },
            {
                "repo_id": "gpt2",
                "task": "text-generation", 
                "max_new_tokens": 50,
                "do_sample": False,
                "temperature": 0
            }
        ]
        
        for i, model_config in enumerate(models_to_try):
            try:
                print(f"\n🔄 Testing model {i+1}: {model_config['repo_id']}")
                
                endpoint = HuggingFaceEndpoint(**model_config)
                
                llm = ChatHuggingFace(
                    llm=endpoint,
                    verbose=True,
                )
                
                # Test with a simple request
                test_message = [HumanMessage(content="Hello")]
                print(f"📤 Sending test message to {model_config['repo_id']}...")
                
                start_time = time.time()
                response = llm.invoke(test_message)
                end_time = time.time()
                
                if response and hasattr(response, 'content') and response.content:
                    print(f"✅ {model_config['repo_id']} test successful!")
                    print(f"   Response time: {end_time - start_time:.2f}s")
                    print(f"   Response: {response.content[:100]}...")
                    return True
                else:
                    print(f"⚠️ {model_config['repo_id']} returned empty response")
                    
            except Exception as e:
                error_str = str(e)
                if "500 Server Error" in error_str and "router.huggingface.co" in error_str:
                    print(f"⚠️ {model_config['repo_id']} router error (500): This is a known HuggingFace issue")
                    print("💡 Router errors are common with HuggingFace. Consider using Google Gemini or Groq instead.")
                elif "timeout" in error_str.lower():
                    print(f"⚠️ {model_config['repo_id']} timeout error: Model may be overloaded")
                else:
                    print(f"❌ {model_config['repo_id']} failed: {e}")
                continue
        
        print("❌ All HuggingFace models failed to initialize")
        return False
        
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
        print("3. HuggingFace router errors (500) are common - this is normal")
        print("4. Consider using Google Gemini or Groq as more reliable alternatives")
        print("5. Try again later - HuggingFace services can be temporarily overloaded")

if __name__ == "__main__":
    main() 