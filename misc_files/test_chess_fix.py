#!/usr/bin/env python3
"""
Test script to verify the improved chess functionality handles 404 errors properly.
"""

import os
import sys
import requests
import urllib.parse
from dotenv import load_dotenv

# Add parent directory to Python path to import tools module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

def test_lichess_api():
    """Test Lichess API with a known position and a position that might return 404."""
    
    print("=== Testing Lichess API ===")
    
    # Test 1: Known position (should work)
    print("\nTest 1: Known position (starting position)")
    fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    chess_eval_url = os.environ.get("CHESS_EVAL_URL", "https://lichess.org/api/cloud-eval")
    url1 = f"{chess_eval_url}?fen={urllib.parse.quote(fen1)}&depth=15"
    
    response1 = requests.get(url1, timeout=15)
    print(f"Status: {response1.status_code}")
    if response1.status_code == 200:
        print("✅ Known position works")
    else:
        print(f"❌ Known position failed: {response1.text}")
    
    # Test 2: Complex position (might return 404)
    print("\nTest 2: Complex position (might return 404)")
    fen2 = "rn1q1rk1/pp2b1pp/2p2n2/3p1pB1/3P4/1QP2N2/PP1N1PPP/R4RK1 b - - 1 11"
    url2 = f"{chess_eval_url}?fen={urllib.parse.quote(fen2)}&depth=15"
    
    response2 = requests.get(url2, timeout=15)
    print(f"Status: {response2.status_code}")
    if response2.status_code == 200:
        print("✅ Complex position found in database")
        data = response2.json()
        if 'pvs' in data and len(data['pvs']) > 0:
            moves = data['pvs'][0].get('moves', '')
            if moves:
                first_move = moves.split()[0]
                print(f"Best move: {first_move}")
    elif response2.status_code == 404:
        print("❌ Complex position not found in database (404)")
        print("This is expected for some positions - fallback should be used")
    else:
        print(f"❌ Unexpected response: {response2.text}")

def test_stockfish_online_api_v2():
    """Test Stockfish Online API v2."""
    
    print("\n=== Testing Stockfish Online API v2 ===")
    
    try:
        from tools import _try_stockfish_online_api_v2
        
        # Test with a simple position
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        print(f"Testing FEN: {fen}")
        result = _try_stockfish_online_api_v2(fen)
        print(f"Stockfish Online API v2 result: {result}")
        
        if result.startswith("Error"):
            print("❌ Stockfish Online API v2 failed")
        else:
            print("✅ Stockfish Online API v2 succeeded")
            
    except ImportError as e:
        print(f"❌ Could not import Stockfish Online API v2 function: {e}")
    except Exception as e:
        print(f"❌ Error testing Stockfish Online API v2: {e}")

def test_fallback_function():
    """Test the fallback function directly."""
    
    print("\n=== Testing Fallback Function ===")
    
    try:
        from tools import _get_best_move_fallback
        
        # Test with a complex position that might not be in Lichess database
        fen = "rn1q1rk1/pp2b1pp/2p2n2/3p1pB1/3P4/1QP2N2/PP1N1PPP/R4RK1 b - - 1 11"
        
        print(f"Testing FEN: {fen}")
        result = _get_best_move_fallback(fen)
        print(f"Fallback result: {result}")
        
        if result.startswith("Error"):
            print("❌ Fallback failed")
        else:
            print("✅ Fallback succeeded")
            
    except ImportError as e:
        print(f"❌ Could not import fallback function: {e}")
    except Exception as e:
        print(f"❌ Error testing fallback: {e}")

def test_simple_heuristic():
    """Test the simple heuristic function."""
    
    print("\n=== Testing Simple Heuristic ===")
    
    try:
        from tools import _get_best_move_simple_heuristic
        
        # Test with a simple position
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        print(f"Testing FEN: {fen}")
        result = _get_best_move_simple_heuristic(fen)
        print(f"Heuristic result: {result}")
        
        if result.startswith("Error"):
            print("❌ Heuristic failed")
        else:
            print("✅ Heuristic succeeded")
            
    except ImportError as e:
        print(f"❌ Could not import heuristic function: {e}")
    except Exception as e:
        print(f"❌ Error testing heuristic: {e}")

def main():
    """Run all tests."""
    
    print("Chess API and Fallback Testing")
    print("=" * 50)
    
    # Test Lichess API
    test_lichess_api()
    
    # Test Stockfish Online API v2
    test_stockfish_online_api_v2()
    
    # Test fallback function
    test_fallback_function()
    
    # Test simple heuristic
    test_simple_heuristic()
    
    print("\n" + "=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    main() 