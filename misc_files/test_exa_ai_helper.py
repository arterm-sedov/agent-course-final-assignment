import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Ensure tools.py is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import exa_ai_helper

def main():
    if 'EXA_API_KEY' not in os.environ:
        print("Error: EXA_API_KEY environment variable is not set. Please set it in your .env file or environment.")
        sys.exit(1)
    if len(sys.argv) < 2:
        print("Usage: python test_exa_ai_helper.py 'your question here'")
        sys.exit(1)
    question = ' '.join(sys.argv[1:])
    print(f"Question: {question}\n")
    # Use invoke to avoid LangChainDeprecationWarning
    result = exa_ai_helper.invoke({"question": question})
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception:
        print(result)

if __name__ == "__main__":
    main() 