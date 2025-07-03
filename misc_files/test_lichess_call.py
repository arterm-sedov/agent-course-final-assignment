# test_lichess_call.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools import get_best_chess_move

def test_lichess_api():
    # Example FEN for starting position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print("Testing Lichess API via get_best_chess_move...")
    result = get_best_chess_move(fen)
    print("Result:")
    print(result)

if __name__ == "__main__":
    test_lichess_api()