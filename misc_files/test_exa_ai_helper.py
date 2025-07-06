#!/usr/bin/env python3
"""
Test script for exa_ai_helper from tools.py

This script allows you to interactively test the exa_ai_helper function
by prompting for questions and displaying the results.

Requirements:
- EXA_API_KEY environment variable must be set
- exa-py package must be installed: pip install exa-py
"""

import os
import sys
import json
from typing import Optional

# Try to load dotenv for .env file support
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("💡 Or set EXA_API_KEY directly: export EXA_API_KEY='your_key'")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")

def check_environment():
    """Check if the required environment is set up."""
    print("🔍 Checking environment...")
    
    # Check if EXA_API_KEY is set
    exa_key = os.environ.get("EXA_API_KEY")
    if not exa_key:
        print("❌ EXA_API_KEY not found in environment variables")
        print("💡 Please set it in your .env file or export it:")
        print("   export EXA_API_KEY='your_api_key_here'")
        return False
    
    print("✅ EXA_API_KEY found")
    
    # Check if exa-py is available
    try:
        from exa_py import Exa
        print("✅ exa-py package is available")
        return True
    except ImportError:
        print("❌ exa-py package not available")
        print("💡 Install it with: pip install exa-py")
        return False

def import_tools():
    """Import the exa_ai_helper function from tools.py."""
    try:
        # Add current directory to path to import tools
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tools import exa_ai_helper
        return exa_ai_helper
    except ImportError as e:
        print(f"❌ Failed to import exa_ai_helper: {e}")
        return None

def parse_exa_response(response: str) -> dict:
    """Parse the JSON response from exa_ai_helper."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # If it's not valid JSON, return as error
        return {
            "type": "tool_response",
            "tool_name": "exa_ai_helper",
            "error": f"Invalid JSON response: {response}"
        }

def display_result(result: dict):
    """Display the result in a formatted way."""
    print("\n" + "="*60)
    print("📋 EXA AI HELPER RESULT")
    print("="*60)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    elif "answer" in result:
        print(f"✅ Answer: {result['answer']}")
    else:
        print(f"⚠️  Unexpected response format: {result}")
    
    print("="*60)

def main():
    """Main function to run the interactive test."""
    print("🤖 EXA AI Helper Test Script")
    print("="*40)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return
    
    # Import the function
    exa_ai_helper = import_tools()
    if not exa_ai_helper:
        print("\n❌ Failed to import exa_ai_helper function.")
        return
    
    print("\n✅ Ready to test exa_ai_helper!")
    print("💡 Type 'quit' or 'exit' to stop")
    print("💡 Type 'help' for example questions")
    
    while True:
        try:
            # Get user input
            question = input("\n❓ Enter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'help':
                print("\n💡 Example questions you can try:")
                print("   - What is the capital of France?")
                print("   - What is the latest version of Python?")
                print("   - What are the benefits of using Docker?")
                print("   - What is the current state of AI in healthcare?")
                print("   - Who won the 2023 Nobel Prize in Physics?")
                continue
            
            if not question:
                print("⚠️  Please enter a question.")
                continue
            
            print(f"\n🔄 Querying Exa AI Helper: '{question}'")
            
            # Call the function
            response = exa_ai_helper(question)
            
            # Parse and display result
            result = parse_exa_response(response)
            display_result(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("💡 Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main() 