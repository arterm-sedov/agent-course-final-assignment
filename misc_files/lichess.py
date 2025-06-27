import requests
import re

def lichess_scan_image_to_fen(image_path):
    url = "https://lichess.org/scan"
    headers = {"User-Agent": "Mozilla/5.0"}
    with open(image_path, "rb") as f:
        files = {"image": (image_path, f, "image/png")}
        response = requests.post(url, files=files, headers=headers)
    print(response.text)  # Debug: see what Lichess returns
    if response.ok:
        match = re.search(r'<code>(.*?)</code>', response.text)
        if match:
            return match.group(1)
        # fallback: look for FEN-like string
        for line in response.text.splitlines():
            if "/" in line and len(line.split("/")) == 8:
                return line.strip()
    return None

lichess_scan_image_to_fen("./chessboard.png")