#!/usr/bin/env python3
"""
Test script for web_search_deep_research_exa_ai from tools.py

This script allows you to interactively test the web_search_deep_research_exa_ai function
by prompting for research questions and displaying the results.

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
    """Import the web_search_deep_research_exa_ai function from tools.py."""
    try:
        # Add current directory to path to import tools
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tools import web_search_deep_research_exa_ai
        return web_search_deep_research_exa_ai
    except ImportError as e:
        print(f"❌ Failed to import web_search_deep_research_exa_ai: {e}")
        return None

def parse_exa_response(response: str) -> dict:
    """Parse the JSON response from web_search_deep_research_exa_ai."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # If it's not valid JSON, return as error
        return {
            "type": "tool_response",
            "tool_name": "web_search_deep_research_exa_ai",
            "error": f"Invalid JSON response: {response}"
        }

def display_result(result: dict):
    """Display the result in a formatted way."""
    print("\n" + "="*60)
    print("🔬 EXA RESEARCH TOOL RESULT")
    print("="*60)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    elif "result" in result:
        print(f"✅ Research Result: {result['result']}")
    else:
        print(f"⚠️  Unexpected response format: {result}")
    
    print("="*60)

def main():
    """Main function to run the interactive test."""
    print("🔬 EXA Research Tool Test Script")
    print("="*40)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return
    
    # Import the function
    web_search_deep_research_exa_ai = import_tools()
    if not web_search_deep_research_exa_ai:
        print("\n❌ Failed to import web_search_deep_research_exa_ai function.")
        return
    
    print("\n✅ Ready to test web_search_deep_research_exa_ai!")
    print("💡 Type 'quit' or 'exit' to stop")
    print("💡 Type 'help' for example questions")
    print("💡 Type 'demo' to run the Olympics example")
    
    while True:
        try:
            # Get user input
            question = input("\n❓ Enter your research question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'help':
                print("\n💡 Example research questions you can try:")
                print("   - What country had the least number of athletes at the 1928 Summer Olympics?")
                print("   - What is the current state of quantum computing research?")
                print("   - Who are the top 5 AI researchers in 2024?")
                print("   - What are the latest developments in renewable energy?")
                print("   - How has COVID-19 affected global supply chains?")
                continue
            
            if question.lower() == 'demo':
                question = "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer."
                print(f"\n🔄 Running demo with question: '{question}'")
            
            if not question:
                print("⚠️  Please enter a question.")
                continue
            
            print(f"\n🔄 Querying Exa Research Tool: '{question}'")
            print("⏳ This may take a moment as Exa researches the web...")
            
            # Call the function
            response = web_search_deep_research_exa_ai(question)
            
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