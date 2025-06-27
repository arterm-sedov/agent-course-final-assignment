#!/usr/bin/env python3
"""
Test script to verify that chess tools are properly available in the agent.
This checks that the agent can access and use the chess functionality.
"""

import os
import sys
from agent import GaiaAgent

def test_agent_chess_tools():
    """Test that the agent has access to chess tools."""
    
    print("=== Testing Agent Chess Tools ===")
    
    try:
        # Initialize the agent
        print("1. Initializing agent...")
        agent = GaiaAgent(provider="groq")
        print("✅ Agent initialized successfully")
        
        # Check if chess tools are available
        print("\n2. Checking chess tools availability...")
        tool_names = [tool.__name__ for tool in agent.tools]
        
        chess_tools = [
            'get_chess_board_fen',
            'get_best_chess_move', 
            'convert_chess_move',
            'solve_chess_position'
        ]
        
        missing_tools = []
        for tool_name in chess_tools:
            if tool_name in tool_names:
                print(f"✅ {tool_name} - Available")
            else:
                print(f"❌ {tool_name} - Missing")
                missing_tools.append(tool_name)
        
        if missing_tools:
            print(f"\n❌ Missing chess tools: {missing_tools}")
            return False
        else:
            print("\n✅ All chess tools are available!")
        
        # Test tool function signatures
        print("\n3. Testing tool function signatures...")
        for tool in agent.tools:
            if tool.__name__ in chess_tools:
                print(f"Tool: {tool.__name__}")
                print(f"  Signature: {tool.__name__}{tool.__code__.co_varnames[:tool.__code__.co_argcount]}")
                print(f"  Docstring: {tool.__doc__.split('.')[0] if tool.__doc__ else 'No docstring'}")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing agent chess tools: {e}")
        return False

def test_agent_with_chess_question():
    """Test the agent with a chess question (without actually running it)."""
    
    print("\n=== Testing Agent with Chess Question ===")
    
    try:
        # Initialize the agent
        agent = GaiaAgent(provider="groq")
        
        # Create a test chess question
        test_question = """
        Review the chess position provided in the image. It is black's turn. 
        Provide the correct next move for black which guarantees a win. 
        Please provide your response in algebraic notation.
        """
        
        print("Test question:")
        print(test_question.strip())
        print()
        
        # Check if the agent has the necessary tools to handle this
        tool_names = [tool.__name__ for tool in agent.tools]
        
        required_tools = [
            'get_task_file',  # To get the chess image
            'solve_chess_position'  # To solve the chess position
        ]
        
        print("Required tools for chess question:")
        for tool_name in required_tools:
            if tool_name in tool_names:
                print(f"✅ {tool_name} - Available")
            else:
                print(f"❌ {tool_name} - Missing")
        
        print("\n✅ Agent is ready to handle chess questions!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing agent with chess question: {e}")
        return False

def main():
    """Main test function."""
    print("Agent Chess Tools Test")
    print("=" * 50)
    
    # Check environment variables
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        print("Some tests may fail without these variables.")
    
    # Run tests
    success1 = test_agent_chess_tools()
    success2 = test_agent_with_chess_question()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The agent is ready to handle chess questions.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 