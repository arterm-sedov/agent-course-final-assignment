#!/usr/bin/env python3
"""
Test script to verify the _extract_final_answer method fix.
"""

import sys
import os

# Add the current directory to the path so we can import agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import GaiaAgent

def test_extraction():
    """Test the _extract_final_answer method with various inputs."""
    
    # Create a minimal agent instance (we don't need full initialization for this test)
    agent = GaiaAgent.__new__(GaiaAgent)
    
    # Test cases
    test_cases = [
        {
            "input": "FINAL ANSWER: 3",
            "expected": "3"
        },
        {
            "input": "FINAL ANSWER: John Smith",
            "expected": "John Smith"
        },
        {
            "input": "Here is my reasoning...\nFINAL ANSWER: 42\nMore text...",
            "expected": "42"
        },
        {
            "input": "FINAL ANSWER: Alice and Bob",
            "expected": "Alice and Bob"
        },
        {
            "input": "No final answer here",
            "expected": None
        },
        {
            "input": "final answer: lowercase test",
            "expected": "lowercase test"
        },
        {
            "input": "FINAL ANSWER 33",  # No colon
            "expected": "33"
        }
    ]
    
    print("🧪 Testing _extract_final_answer method...")
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"  Input: '{test_case['input']}'")
        
        # Create a mock response object
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        mock_response = MockResponse(test_case['input'])
        
        # Test the extraction
        result = agent._extract_final_answer(mock_response)
        expected = test_case['expected']
        
        print(f"  Expected: '{expected}'")
        print(f"  Got: '{result}'")
        
        if result == expected:
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL")
            return False
    
    print("\n🎉 All tests passed!")
    return True

def test_has_marker():
    """Test the _has_final_answer_marker method."""
    
    # Create a minimal agent instance
    agent = GaiaAgent.__new__(GaiaAgent)
    
    # Test cases
    test_cases = [
        {
            "input": "FINAL ANSWER: 3",
            "expected": True
        },
        {
            "input": "Here is my reasoning...\nFINAL ANSWER: 42\nMore text...",
            "expected": True
        },
        {
            "input": "No final answer here",
            "expected": False
        },
        {
            "input": "final answer: lowercase test",
            "expected": True
        },
        {
            "input": "FINAL ANSWER 33",  # No colon
            "expected": True
        }
    ]
    
    print("\n🧪 Testing _has_final_answer_marker method...")
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"  Input: '{test_case['input']}'")
        
        # Create a mock response object
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        mock_response = MockResponse(test_case['input'])
        
        # Test the marker detection
        result = agent._has_final_answer_marker(mock_response)
        expected = test_case['expected']
        
        print(f"  Expected: {expected}")
        print(f"  Got: {result}")
        
        if result == expected:
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL")
            return False
    
    print("\n🎉 All marker tests passed!")
    return True

if __name__ == "__main__":
    print("🚀 Starting extraction method tests...")
    
    success1 = test_extraction()
    success2 = test_has_marker()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The extraction fix is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 