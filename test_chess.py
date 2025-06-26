#!/usr/bin/env python3
"""
Test script for chess position solving functionality.
This demonstrates how to use the chess tools to solve the specific question:
"Review the chess position provided in the image. It is black's turn. 
Provide the correct next move for black which guarantees a win. 
Please provide your response in algebraic notation."
"""

import os
import sys
from tools import solve_chess_position, get_chess_board_fen, get_best_chess_move, convert_chess_move

def test_chess_question():
    """
    Test the chess question from the metadata:
    Task ID: cca530fc-4052-43b2-b130-b30968d8aa44
    Expected answer: "Rd5"
    """
    
    # Test parameters
    task_id = "cca530fc-4052-43b2-b130-b30968d8aa44"
    file_name = "cca530fc-4052-43b2-b130-b30968d8aa44.png"
    player_turn = "black"
    question = "Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation."
    expected_answer = "Rd5"
    
    print("=== Chess Position Solver Test ===")
    print(f"Task ID: {task_id}")
    print(f"File: {file_name}")
    print(f"Player to move: {player_turn}")
    print(f"Question: {question}")
    print(f"Expected answer: {expected_answer}")
    print()
    
    # Check if the image file exists
    image_path = os.path.join("files", file_name)
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        print("Please ensure the chess board image is available in the 'files' directory.")
        return False
    
    try:
        # Method 1: Use the comprehensive solve_chess_position function
        print("=== Method 1: Comprehensive Solution ===")
        result = solve_chess_position(image_path, player_turn, question)
        print(result)
        print()
        
        # Method 2: Step-by-step approach (for debugging)
        print("=== Method 2: Step-by-Step Analysis ===")
        
        # Step 1: Get FEN from image
        print("Step 1: Converting image to FEN...")
        fen = get_chess_board_fen(image_path, player_turn)
        print(f"FEN: {fen}")
        
        if fen.startswith("Error"):
            print(f"Error in FEN conversion: {fen}")
            return False
        
        # Step 2: Get best move
        print("\nStep 2: Getting best move...")
        best_move_coord = get_best_chess_move(fen)
        print(f"Best move (coordinate): {best_move_coord}")
        
        if best_move_coord.startswith("Error"):
            print(f"Error getting best move: {best_move_coord}")
            return False
        
        # Step 3: Convert to algebraic notation
        print("\nStep 3: Converting to algebraic notation...")
        piece_placement = f"FEN: {fen}"
        algebraic_move = convert_chess_move(piece_placement, best_move_coord)
        print(f"Best move (algebraic): {algebraic_move}")
        
        if algebraic_move.startswith("Error"):
            print(f"Error converting move: {algebraic_move}")
            return False
        
        # Step 4: Compare with expected answer
        print(f"\n=== Result Comparison ===")
        print(f"Expected answer: {expected_answer}")
        print(f"Computed answer: {algebraic_move}")
        
        # Simple comparison (case-insensitive, strip whitespace)
        if algebraic_move.strip().lower() == expected_answer.strip().lower():
            print("✅ SUCCESS: Answer matches expected result!")
            return True
        else:
            print("❌ FAILURE: Answer does not match expected result.")
            print("This could be due to:")
            print("- Different chess engine evaluation")
            print("- Board orientation differences")
            print("- Alternative winning moves")
            return False
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False

def test_environment_setup():
    """Test if all required environment variables and dependencies are available."""
    print("=== Environment Setup Test ===")
    
    # Check required environment variables
    required_vars = [
        "GEMINI_KEY",
        "OPENROUTER_API_KEY", 
        "CHESS_EVAL_URL"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file or environment:")
        for var in missing_vars:
            print(f"  {var}=your_api_key_here")
        return False
    else:
        print("✅ All required environment variables are set")
    
    # Check if required packages are available
    try:
        from board_to_fen.predict import get_fen_from_image_path
        print("✅ board-to-fen package is available")
    except ImportError:
        print("❌ board-to-fen package is not available")
        return False
    
    try:
        from litellm import completion
        print("✅ litellm package is available")
    except ImportError:
        print("❌ litellm package is not available")
        return False
    
    try:
        from google import genai
        print("✅ google-genai package is available")
    except ImportError:
        print("❌ google-genai package is not available")
        return False
    
    return True

if __name__ == "__main__":
    print("Chess Position Solver Test")
    print("=" * 50)
    
    # First check environment setup
    if not test_environment_setup():
        print("\n❌ Environment setup failed. Please fix the issues above.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    # Then test the chess functionality
    success = test_chess_question()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the output above for details.")
        sys.exit(1) 