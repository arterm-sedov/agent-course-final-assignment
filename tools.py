# tools.py - Consolidated tools
# Dependencies are included

import os
import io
import re
import json
import uuid
import base64
import shutil
import requests
import tempfile
import contextlib
import logging
import urllib.parse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from typing import Any, Dict, List, Optional, Union

# LangChain imports for search tools
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("Warning: TavilySearchResults not available. Install with: pip install langchain-tavily")

# ========== IMAGE PROCESSING HELPERS ==========
def encode_image(image_path: str) -> str:
    """
    Convert an image file to a base64-encoded string.

    Args:
        image_path (str): The path to the image file to encode.

    Returns:
        str: The base64-encoded string representation of the image file.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def decode_image(base64_string: str) -> Any:
    """
    Convert a base64-encoded string to a PIL Image object.

    Args:
        base64_string (str): The base64-encoded string representing the image.

    Returns:
        Any: The decoded PIL Image object.
    """
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def save_image(image: Any, directory: str = "image_outputs") -> str:
    """
    Save a PIL Image object to disk in the specified directory and return the file path.

    Args:
        image (Any): The PIL Image object to save.
        directory (str, optional): The directory to save the image in. Defaults to "image_outputs".

    Returns:
        str: The file path where the image was saved.
    """
    os.makedirs(directory, exist_ok=True)
    image_id = str(uuid.uuid4())
    image_path = os.path.join(directory, f"{image_id}.png")
    image.save(image_path)
    return image_path

# ========== CODE INTERPRETER ==========
class CodeInterpreter:
    """
    A code interpreter for executing code in various languages (Python, Bash, SQL, C, Java) with safety and resource controls.

    Args:
        allowed_modules (list, optional): List of allowed module names for Python execution.
        max_execution_time (int, optional): Maximum execution time in seconds for code blocks.
        working_directory (str, optional): Directory for temporary files and execution context.

    Attributes:
        globals (dict): Global variables for code execution.
        temp_sqlite_db (str): Path to a temporary SQLite database for SQL code.
    """
    def __init__(self, allowed_modules=None, max_execution_time=30, working_directory=None):
        self.allowed_modules = allowed_modules or [
            "numpy", "pandas", "matplotlib", "scipy", "sklearn", 
            "math", "random", "statistics", "datetime", "collections",
            "itertools", "functools", "operator", "re", "json",
            "sympy", "networkx", "nltk", "PIL", "pytesseract", 
            "cmath", "uuid", "tempfile", "requests", "urllib"
        ]
        self.max_execution_time = max_execution_time
        self.working_directory = working_directory or os.path.join(os.getcwd()) 
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from PIL import Image
        self.globals = {
            "__builtins__": __builtins__,
            "np": np,
            "pd": pd,
            "plt": plt,
            "Image": Image,
        }
        self.temp_sqlite_db = os.path.join(tempfile.gettempdir(), "code_exec.db")
    # ... (methods omitted for brevity, see fisherman611/code_interpreter.py)

# ========== MATH TOOLS ==========
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers and return the result.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The product of a and b.
    """
    return a * b

def add(a: float, b: float) -> float:
    """
    Add two numbers and return the result.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The sum of a and b.
    """
    return a + b

def subtract(a: float, b: float) -> float:
    """
    Subtract the second number from the first and return the result.

    Args:
        a (float): The number to subtract from.
        b (float): The number to subtract.

    Returns:
        float: The result of a - b.
    """
    return a - b

def divide(a: float, b: float) -> float:
    """
    Divide the first number by the second and return the result.

    Args:
        a (float): The numerator.
        b (float): The denominator. Must not be zero.

    Returns:
        float: The result of a / b.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

def modulus(a: int, b: int) -> int:
    """
    Compute the modulus (remainder) of two integers.

    Args:
        a (int): The dividend.
        b (int): The divisor.

    Returns:
        int: The remainder when a is divided by b.
    """
    return a % b

def power(a: float, b: float) -> float:
    """
    Raise the first number to the power of the second and return the result.

    Args:
        a (float): The base number.
        b (float): The exponent.

    Returns:
        float: The result of a raised to the power of b.
    """
    return a ** b

def square_root(a: float) -> float:
    """
    Compute the square root of a number. Returns a complex number if input is negative.

    Args:
        a (float): The number to compute the square root of.

    Returns:
        float or complex: The square root of a. If a < 0, returns a complex number.
    """
    import cmath
    if a >= 0:
        return a ** 0.5
    return cmath.sqrt(a)

# ========== WEB/SEARCH TOOLS ==========
def wiki_search(query: str) -> str:
    """
    Search Wikipedia for a query and return up to 2 results as formatted text.

    Args:
        query (str): The search query string.

    Returns:
        str: Formatted search results from Wikipedia with source information and content.
    """
    try:
        search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
        formatted_results = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}'
                for doc in search_docs
            ]
        )
        return {"wiki_results": formatted_results}
    except Exception as e:
        return f"Error in Wikipedia search: {str(e)}"

def web_search(query: str) -> str:
    """
    Search the web using Tavily for a query and return up to 3 results as formatted text.
    
    Tavily is a search API that provides real-time web search results. This tool is useful for:
    - Finding current information about recent events
    - Searching for specific facts, statistics, or data
    - Getting up-to-date information from various websites
    - Researching topics that may not be covered in Wikipedia or academic papers

    Args:
        query (str): The search query string to search for on the web.

    Returns:
        str: Formatted search results from Tavily with source URLs and content snippets.
             Returns an error message if Tavily is not available or if the search fails.

    Example:
        >>> web_search("latest SpaceX launch date")
        Returns formatted results from recent web searches about SpaceX launches.

    Note:
        Requires TAVILY_API_KEY environment variable to be set.
        Install with: pip install langchain-tavily
    """
    if not TAVILY_AVAILABLE:
        return "Tavily search not available. Install with: pip install langchain-tavily"
    
    try:
        # Check if API key is available
        if not os.environ.get("TAVILY_API_KEY"):
            return "TAVILY_API_KEY not found in environment variables. Please set it in your .env file."
        
        # Perform the search
        search_docs = TavilySearchResults(max_results=3).invoke(query=query)
        
        # Format the results
        formatted_results = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}'
                for doc in search_docs
            ]
        )
        
        return {"web_results": formatted_results}
        
    except Exception as e:
        return f"Error in web search: {str(e)}"

def arxiv_search(query: str) -> str:
    """
    Search Arxiv for academic papers and return up to 3 results as formatted text.

    Args:
        query (str): The search query string for academic papers.

    Returns:
        str: Formatted search results from Arxiv with paper metadata and abstracts.
    """
    try:
        search_docs = ArxivLoader(query=query, load_max_docs=3).load()
        formatted_results = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}'
                for doc in search_docs
            ]
        )
        return {"arxiv_results": formatted_results}
    except Exception as e:
        return f"Error in Arxiv search: {str(e)}"

# ========== FILE/DATA TOOLS (from fisherman611) ==========
def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save the provided content to a file and return the file path.

    Args:
        content (str): The content to write to the file.
        filename (str, optional): The name of the file. If not provided, a random file name is generated.

    Returns:
        str: The file path where the content was saved.
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)
    with open(filepath, "w") as f:
        f.write(content)
    return f"File saved to {filepath}. You can read this file to process its contents."

def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location. Returns the file path.

    Args:
        url (str): The URL of the file to download.
        filename (str, optional): The name of the file. If not provided, a name is inferred or generated.

    Returns:
        str: The file path where the file was downloaded.
    """
    try:
        if not filename:
            from urllib.parse import urlparse
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return f"File downloaded to {filepath}. You can read this file to process its contents."
    except Exception as e:
        return f"Error downloading file: {str(e)}"

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image file using OCR (pytesseract) and return the extracted text.

    Args:
        image_path (str): The path to the image file to process.

    Returns:
        str: The extracted text, or an error message if extraction fails.
    """
    try:
        image = Image.open(image_path)
        import pytesseract
        text = pytesseract.image_to_string(image)
        return f"Extracted text from image:\n\n{text}"
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyze a CSV file using pandas and return summary statistics and column info.

    Args:
        file_path (str): The path to the CSV file.
        query (str): A question or description of the analysis to perform (currently unused).

    Returns:
        str: Summary statistics and column information, or an error message if analysis fails.
    """
    try:
        df = pd.read_csv(file_path)
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        result += "Summary statistics:\n"
        result += str(df.describe())
        return result
    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"

def analyze_excel_file(file_path: str, query: str) -> str:
    """
    Analyze an Excel file using pandas and return summary statistics and column info.

    Args:
        file_path (str): The path to the Excel file.
        query (str): A question or description of the analysis to perform (currently unused).

    Returns:
        str: Summary statistics and column information, or an error message if analysis fails.
    """
    try:
        df = pd.read_excel(file_path)
        result = f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        result += "Summary statistics:\n"
        result += str(df.describe())
        return result
    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"

# ========== IMAGE ANALYSIS/GENERATION TOOLS (from fisherman611) ==========
def analyze_image(image_base64: str) -> str:
    """
    Analyze basic properties of an image (size, mode, color analysis, thumbnail preview) from a base64-encoded image string.

    Args:
        image_base64 (str): The base64-encoded string of the image to analyze.

    Returns:
        str: JSON string with analysis results including dimensions, mode, color_analysis, and thumbnail.
    """
    try:
        img = decode_image(image_base64)
        width, height = img.size
        mode = img.mode
        if mode in ("RGB", "RGBA"):
            arr = np.array(img)
            avg_colors = arr.mean(axis=(0, 1))
            dominant = ["Red", "Green", "Blue"][np.argmax(avg_colors[:3])]
            brightness = avg_colors.mean()
            color_analysis = {
                "average_rgb": avg_colors.tolist(),
                "brightness": brightness,
                "dominant_color": dominant,
            }
        else:
            color_analysis = {"note": f"No color analysis for mode {mode}"}
        thumbnail = img.copy()
        thumbnail.thumbnail((100, 100))
        thumb_path = save_image(thumbnail, "thumbnails")
        thumbnail_base64 = encode_image(thumb_path)
        result = {
            "dimensions": (width, height),
            "mode": mode,
            "color_analysis": color_analysis,
            "thumbnail": thumbnail_base64,
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

# ... (other image tools, chess tools, and civerson916 custom tools can be added here as needed)

# ========== END OF TOOLS.PY ========== 