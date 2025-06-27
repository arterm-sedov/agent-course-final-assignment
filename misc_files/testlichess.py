import os
import requests
import urllib.parse
from dotenv import load_dotenv
load_dotenv()

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
chess_eval_url = os.environ.get("CHESS_EVAL_URL", "https://lichess.org/api/cloud-eval")
url = f"{chess_eval_url}?fen={urllib.parse.quote(fen)}&depth=15"
headers = {}
lichess_key = os.environ.get("LICHESS_KEY")
if lichess_key:
    headers["Authorization"] = f"Bearer {lichess_key}"

response = requests.get(url, headers=headers)
print(response.status_code)
print(response.text)