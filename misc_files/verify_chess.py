#!/usr/bin/env python3
"""
Simple verification script to check chess functions are available.
"""

def main():
    print("=== Chess Functions Verification ===")
    
    try:
        # Import tools module
        print("1. Importing tools module...")
        import tools
        print("✅ Tools module imported successfully")
        
        # Check for chess functions
        print("\n2. Checking for chess functions...")
        chess_functions = [
            'convert_chess_move',
            'get_best_chess_move', 
            'get_chess_board_fen',
            'solve_chess_position'
        ]
        
        found_functions = []
        for func_name in chess_functions:
            if hasattr(tools, func_name):
                func = getattr(tools, func_name)
                if callable(func):
                    print(f"✅ {func_name} - Found and callable")
                    found_functions.append(func_name)
                else:
                    print(f"❌ {func_name} - Found but not callable")
            else:
                print(f"❌ {func_name} - Not found")
        
        print(f"\nFound {len(found_functions)} chess functions: {found_functions}")
        
        # Test importing specific functions
        print("\n3. Testing direct imports...")
        try:
            from tools import convert_chess_move, get_best_chess_move, get_chess_board_fen, solve_chess_position
            print("✅ All chess functions imported successfully")
        except ImportError as e:
            print(f"❌ Import error: {e}")
        
        return len(found_functions) == 4
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All chess functions are properly implemented!")
    else:
        print("\n💥 Some chess functions are missing or have issues.") 