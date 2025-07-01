import sys
import os
from pathlib import Path

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    print('python-dotenv not installed. Environment variables may not be loaded from .env file.')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import _get_best_chess_move_internal, _get_best_move_fallback, _try_stockfish_online_api_v2

# FEN from browser test
FEN = "rn1q1rk1/pp2b1pp/2p2n2/3p1pB1/3P4/1QP2N2/PP1N1PPP/R4RK1 b - - 1 11"

print('Testing Stockfish Online API (direct, depth=12):')
result_direct = _try_stockfish_online_api_v2(FEN, depth=12)
print(result_direct)

print('\nTesting Stockfish Online API (internal):')
result_stockfish = _get_best_chess_move_internal(FEN)
print(result_stockfish)

print('\nTesting fallback (local or heuristic):')
result_fallback = _get_best_move_fallback(FEN)
print(result_fallback) 