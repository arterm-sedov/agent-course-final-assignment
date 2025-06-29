#!/usr/bin/env python3
"""
Test script to verify the improved agent can handle longer reasoning
without premature exits due to the 5-step limit.
"""

import os
import sys
from agent import GaiaAgent

def test_improved_agent():
    """Test the improved agent with a complex question that requires multiple steps."""
    
    print("🧪 Testing improved agent with complex reasoning...")
    
    # Initialize the agent
    agent = GaiaAgent(provider="groq")
    
    # Test question that requires multiple tool calls and reasoning
    test_question = "Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?"
    
    print(f"📝 Test question: {test_question}")
    print("🔄 Starting agent processing...")
    
    try:
        # Process the question
        result = agent(test_question)
        
        print(f"\n✅ Agent completed successfully!")
        print(f"📄 Final answer: {result}")
        
        # Check if we got a meaningful result
        if result and len(result) > 10:
            if "Error:" in result:
                print(f"⚠️ Agent returned error: {result}")
                return False
            else:
                print("✅ Result appears meaningful (not empty or too short)")
        else:
            print("⚠️ Result may be too short or empty")
            return False
            
    except Exception as e:
        print(f"❌ Agent failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Set up environment variables if needed
    if not os.environ.get("GROQ_API_KEY"):
        print("⚠️ GROQ_API_KEY not set. Please set it before running this test.")
        sys.exit(1)
    
    success = test_improved_agent()
    
    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1) 