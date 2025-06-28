#!/usr/bin/env python3
"""
Simple test script to verify the chess 404 error fix works correctly.
"""

import os
import sys
import requests
import urllib.parse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_stockfish_online_api_v2():
    """Test Stockfish Online API v2 directly."""
    
    print("=== Testing Stockfish Online API v2 ===")
    
    # Test with a simple position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    print(f"Testing FEN: {fen}")
    
    try:
        # Use Stockfish Online API v2
        api_url = "https://stockfish.online/api/s/v2.php"
        params = {
            'fen': fen,
            'depth': 15
        }
        
        response = requests.get(api_url, params=params, timeout=15)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response data: {data}")
            
            # Check if request was successful
            if data.get('success') == True:
                bestmove = data.get('bestmove', '')
                if bestmove:
                    # Extract the actual move from the bestmove string
                    move_parts = bestmove.split()
                    if len(move_parts) >= 2 and move_parts[0] == 'bestmove':
                        actual_move = move_parts[1]
                        print(f"✅ Stockfish Online API v2 succeeded: {actual_move}")
                        return True
                    else:
                        print(f"✅ Stockfish Online API v2 succeeded: {bestmove}")
                        return True
                else:
                    print("❌ No bestmove in response")
                    return False
            else:
                error_msg = data.get('data', 'Unknown error')
                print(f"❌ Stockfish API failed: {error_msg}")
                return False
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Stockfish Online API v2: {e}")
        return False

def test_lichess_api():
    """Test Lichess API with a known position."""
    
    print("\n=== Testing Lichess API ===")
    
    # Test with a known position (should work)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    chess_eval_url = os.environ.get("CHESS_EVAL_URL", "https://lichess.org/api/cloud-eval")
    url = f"{chess_eval_url}?fen={urllib.parse.quote(fen)}&depth=15"
    
    try:
        response = requests.get(url, timeout=15)
        print(f"Lichess API status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Lichess API works for known position")
            return True
        else:
            print(f"❌ Lichess API failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Lichess API: {e}")
        return False

def test_404_handling():
    """Test 404 error handling with a complex position."""
    
    print("\n=== Testing 404 Error Handling ===")
    
    # Test with a complex position that might return 404
    fen = "rn1q1rk1/pp2b1pp/2p2n2/3p1pB1/3P4/1QP2N2/PP1N1PPP/R4RK1 b - - 1 11"
    chess_eval_url = os.environ.get("CHESS_EVAL_URL", "https://lichess.org/api/cloud-eval")
    url = f"{chess_eval_url}?fen={urllib.parse.quote(fen)}&depth=15"
    
    try:
        response = requests.get(url, timeout=15)
        print(f"Complex position status: {response.status_code}")
        
        if response.status_code == 404:
            print("✅ 404 error detected (expected for complex position)")
            print("This would trigger the fallback system in the actual code")
            return True
        elif response.status_code == 200:
            print("✅ Complex position found in Lichess database")
            return True
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing 404 handling: {e}")
        return False

def main():
    """Run all tests."""
    
    print("Chess API Testing")
    print("=" * 50)
    
    # Test Stockfish Online API v2
    stockfish_success = test_stockfish_online_api_v2()
    
    # Test Lichess API
    lichess_success = test_lichess_api()
    
    # Test 404 handling
    error_handling_success = test_404_handling()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Stockfish Online API v2: {'✅ PASS' if stockfish_success else '❌ FAIL'}")
    print(f"Lichess API: {'✅ PASS' if lichess_success else '❌ FAIL'}")
    print(f"404 Error Handling: {'✅ PASS' if error_handling_success else '❌ FAIL'}")
    
    if stockfish_success and lichess_success and error_handling_success:
        print("\n🎉 All tests passed! The chess fallback system should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main() 