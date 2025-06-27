#!/usr/bin/env python3
"""
Test script to verify the formatting improvements work correctly.
"""

import os
import sys
from agent import GaiaAgent

def test_answer_extraction():
    """Test the intelligent answer extraction and post-processing."""
    agent = GaiaAgent()
    
    # Test cases
    test_cases = [
        {
            "question": "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?",
            "response": "Based on the video, the highest number of bird species on camera simultaneously is **three**.\n\nThis occurs between 1:31 and 1:36, when you can see:\n1.  **Emperor Penguin chicks**\n2.  A **Giant Petrel**\n3.  An **Adelie Penguin**",
            "expected": "3"
        },
        {
            "question": "What is the capital of France?",
            "response": "The capital of France is Paris, which is a beautiful city known for its culture and history.",
            "expected": "Paris"
        },
        {
            "question": "How many colors are in a rainbow?",
            "response": "FINAL ANSWER: 7",
            "expected": "7"
        }
    ]
    
    print("🧪 Testing answer extraction and formatting...")
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test {i+1} ---")
        print(f"Question: {test_case['question']}")
        print(f"Response: {test_case['response']}")
        print(f"Expected: {test_case['expected']}")
        
        # Test intelligent extraction
        extracted = agent._intelligent_answer_extraction(test_case['response'], test_case['question'])
        print(f"Intelligent extraction: {extracted}")
        
        # Test post-processing
        processed = agent._post_process_answer(extracted, test_case['question'])
        print(f"Post-processed: {processed}")
        
        # Check if it matches expected
        if processed == test_case['expected']:
            print("✅ PASS")
        else:
            print("❌ FAIL")
    
    print("\n🎯 Testing complete!")

if __name__ == "__main__":
    test_answer_extraction() 