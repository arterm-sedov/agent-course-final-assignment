#!/usr/bin/env python3
"""
Test the ensure_valid_answer helper function
"""

from app import ensure_valid_answer

def test_ensure_valid_answer():
    """Test the helper function with various inputs"""
    
    print("🧪 Testing ensure_valid_answer helper function")
    print("=" * 50)
    
    test_cases = [
        (None, "No answer provided"),
        ("", "No answer provided"),
        ("   ", "No answer provided"),
        ("test", "test"),
        ("FINAL ANSWER: 42", "FINAL ANSWER: 42"),
        (42, "42"),
        (0, "0"),
        (True, "True"),
        (False, "False"),
        ([1, 2, 3], "[1, 2, 3]"),
        ({"key": "value"}, "{'key': 'value'}"),
    ]
    
    all_passed = True
    
    for input_val, expected in test_cases:
        result = ensure_valid_answer(input_val)
        status = "✅" if result == expected else "❌"
        print(f"{status} {repr(input_val)} -> {repr(result)} (expected: {repr(expected)})")
        
        if result != expected:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("💥 Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    test_ensure_valid_answer() 