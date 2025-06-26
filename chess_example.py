#!/usr/bin/env python3
"""
Example script showing how to integrate chess position solving into an agent workflow.
This demonstrates the complete pipeline for solving chess questions like the GAIA benchmark.
"""

import os
import json
from tools import solve_chess_position, get_task_file

def solve_chess_question_example():
    """
    Example workflow for solving a chess question from the GAIA benchmark.
    This mimics how an agent would process a chess question.
    """
    
    # Example question data (from the metadata)
    question_data = {
        "task_id": "cca530fc-4052-43b2-b130-b30968d8aa44",
        "Question": "Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation.",
        "file_name": "cca530fc-4052-43b2-b130-b30968d8aa44.png",
        "Level": 1,
        "Final answer": "Rd5"
    }
    
    print("=== Chess Question Solver Example ===")
    print(f"Task ID: {question_data['task_id']}")
    print(f"Question: {question_data['Question']}")
    print(f"Expected Answer: {question_data['Final answer']}")
    print()
    
    try:
        # Step 1: Download/get the chess board image
        print("Step 1: Getting chess board image...")
        image_path = get_task_file(question_data['task_id'], question_data['file_name'])
        
        if image_path.startswith("Error"):
            print(f"Error getting image: {image_path}")
            return None
        
        print(f"Image downloaded to: {image_path}")
        
        # Step 2: Extract information from the question
        print("\nStep 2: Analyzing question...")
        
        # Parse the question to extract key information
        question_text = question_data['Question']
        
        # Determine player turn (look for "black's turn" or "white's turn")
        if "black's turn" in question_text.lower():
            player_turn = "black"
        elif "white's turn" in question_text.lower():
            player_turn = "white"
        else:
            # Default to black if not specified
            player_turn = "black"
            print("Warning: Player turn not specified, defaulting to black")
        
        print(f"Player to move: {player_turn}")
        
        # Extract the specific question about the position
        # Look for phrases like "guarantees a win", "best move", etc.
        if "guarantees a win" in question_text.lower():
            position_question = "guarantees a win"
        elif "best move" in question_text.lower():
            position_question = "best move"
        else:
            position_question = "best move"
        
        print(f"Position question: {position_question}")
        
        # Step 3: Solve the chess position
        print("\nStep 3: Solving chess position...")
        result = solve_chess_position(image_path, player_turn, position_question)
        
        if result.startswith("Error"):
            print(f"Error solving position: {result}")
            return None
        
        print("Solution found:")
        print(result)
        
        # Step 4: Extract the final answer
        print("\nStep 4: Extracting final answer...")
        
        # Parse the result to get the algebraic move
        lines = result.split('\n')
        algebraic_move = None
        
        for line in lines:
            if "Best move (algebraic):" in line:
                algebraic_move = line.split(":")[1].strip()
                break
            elif "Answer:" in line:
                algebraic_move = line.split(":")[1].strip()
                break
        
        if not algebraic_move:
            print("Could not extract algebraic move from result")
            return None
        
        print(f"Final answer: {algebraic_move}")
        
        # Step 5: Validate against expected answer
        print(f"\nStep 5: Validation...")
        expected = question_data['Final answer'].strip()
        computed = algebraic_move.strip()
        
        if computed.lower() == expected.lower():
            print("✅ SUCCESS: Answer matches expected result!")
            return algebraic_move
        else:
            print(f"❌ MISMATCH: Expected '{expected}', got '{computed}'")
            print("This could be due to:")
            print("- Different chess engine evaluation")
            print("- Board orientation differences") 
            print("- Alternative winning moves")
            return algebraic_move
            
    except Exception as e:
        print(f"Error in chess question solving: {str(e)}")
        return None

def agent_workflow_example():
    """
    Example of how this would fit into a complete agent workflow.
    """
    print("=== Agent Workflow Example ===")
    
    # Simulate agent receiving a question
    question = {
        "task_id": "cca530fc-4052-43b2-b130-b30968d8aa44",
        "Question": "Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation.",
        "file_name": "cca530fc-4052-43b2-b130-b30968d8aa44.png"
    }
    
    print("Agent receives question:")
    print(f"  Task ID: {question['task_id']}")
    print(f"  Question: {question['Question']}")
    print(f"  File: {question['file_name']}")
    print()
    
    # Agent reasoning steps
    print("Agent reasoning:")
    print("1. This is a chess position analysis question")
    print("2. Need to download the chess board image")
    print("3. Convert image to FEN notation")
    print("4. Find the best move using chess engine")
    print("5. Convert move to algebraic notation")
    print("6. Verify the move guarantees a win")
    print()
    
    # Execute the solution
    answer = solve_chess_question_example()
    
    if answer:
        print("Agent final response:")
        print(f"  Answer: {answer}")
        print("  Reasoning: Analyzed the chess position using computer vision")
        print("  and chess engine evaluation to find the winning move.")
    else:
        print("Agent failed to solve the question")

if __name__ == "__main__":
    print("Chess Question Solver - Agent Integration Example")
    print("=" * 60)
    
    # Check if we have the required environment
    required_vars = ["GEMINI_KEY", "OPENROUTER_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your environment before running.")
        exit(1)
    
    # Run the examples
    print("\n1. Basic chess question solving:")
    solve_chess_question_example()
    
    print("\n" + "=" * 60)
    
    print("\n2. Agent workflow integration:")
    agent_workflow_example()
    
    print("\n" + "=" * 60)
    print("Example completed!") 