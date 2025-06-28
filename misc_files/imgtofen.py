import requests
import base64

def image_to_fen(image_path):
    api_url = "https://DerekLiu35-ImageToFen.hf.space/api/predict"
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {"data": [img_b64]}
    response = requests.post(api_url, json=payload, timeout=60)
    if response.ok:
        result = response.json()
        # Print for debugging
        print("Full API response:", result)
        # The FEN is the last line of the 'data' field (after the base64 image)
        # Sometimes the response is {'data': [<base64>, FEN]}
        data = result.get("data", [])
        if data:
            # FEN is usually the last string in the list
            fen_candidate = data[-1]
            # FENs are typically 8 ranks separated by '/'
            if isinstance(fen_candidate, str) and fen_candidate.count('/') == 7:
                return fen_candidate
            # Fallback: search for a line with 7 slashes
            for item in data:
                if isinstance(item, str) and item.count('/') == 7:
                    return item
        raise Exception(f"FEN not found in API response: {result}")
    else:
        raise Exception(f"API call failed: {response.text}")

# Usage
fen = image_to_fen("chessboard-recognizer/cca530fc-4052-43b2-b130-b30968d8aa44.png")
print(fen)