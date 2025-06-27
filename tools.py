# tools.py - Consolidated tools
# Dependencies are included

import os
import io
import json
import uuid
import base64
import shutil
import requests
import tempfile
import urllib.parse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from typing import Any, Dict, List, Optional, Union
import subprocess

# Always import the tool decorator - it's essential
from langchain_core.tools import tool

# LangChain imports for search tools
try:
    from langchain_tavily import TavilySearchResults
    from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("Warning: TavilySearchResults not available. Install with: pip install langchain-tavily")

# Google Gemini imports for video/audio understanding
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Gemini not available. Install with: pip install google-genai")


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
    
    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Execute code in the specified language with safety controls.
        
        Args:
            code (str): The source code to execute
            language (str): The programming language
            
        Returns:
            Dict containing execution results, status, and outputs
        """
        try:
            if language.lower() == "python":
                return self._execute_python(code)
            elif language.lower() == "bash":
                return self._execute_bash(code)
            elif language.lower() == "sql":
                return self._execute_sql(code)
            elif language.lower() == "c":
                return self._execute_c(code)
            elif language.lower() == "java":
                return self._execute_java(code)
            else:
                return {"status": "error", "stderr": f"Unsupported language: {language}"}
        except Exception as e:
            return {"status": "error", "stderr": str(e)}
    
    def _execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code with safety controls."""
        try:
            # Create a copy of globals for this execution
            local_globals = self.globals.copy()
            local_globals['__name__'] = '__main__'
            
            # Execute the code
            exec(code, local_globals)
            
            # Capture any variables that might be dataframes or plots
            result = {"status": "success", "stdout": "", "stderr": "", "result": None}
            
            # Check for dataframes
            dataframes = []
            for name, value in local_globals.items():
                if isinstance(value, pd.DataFrame):
                    dataframes.append({
                        "name": name,
                        "shape": value.shape,
                        "head": value.head().to_dict('records')
                    })
            if dataframes:
                result["dataframes"] = dataframes
            
            # Check for plots
            plots = []
            if 'plt' in local_globals:
                # Save any current plots
                if plt.get_fignums():
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        plot_path = os.path.join(self.working_directory, f"plot_{fig_num}.png")
                        fig.savefig(plot_path)
                        plots.append(plot_path)
                        plt.close(fig)
            if plots:
                result["plots"] = plots
            
            return result
            
        except Exception as e:
            return {"status": "error", "stderr": str(e)}
    
    def _execute_bash(self, code: str) -> Dict[str, Any]:
        """Execute Bash code."""
        try:
            import subprocess
            result = subprocess.run(
                code, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=self.max_execution_time
            )
            return {
                "status": "success" if result.returncode == 0 else "error",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "stderr": "Execution timed out"}
        except Exception as e:
            return {"status": "error", "stderr": str(e)}
    
    def _execute_sql(self, code: str) -> Dict[str, Any]:
        """Execute SQL code using SQLite."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.temp_sqlite_db)
            cursor = conn.cursor()
            
            # Execute SQL
            cursor.execute(code)
            
            # Fetch results if it's a SELECT
            if code.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                result = {"status": "success", "results": results, "columns": columns}
            else:
                conn.commit()
                result = {"status": "success", "message": f"Executed: {code}"}
            
            conn.close()
            return result
            
        except Exception as e:
            return {"status": "error", "stderr": str(e)}
    
    def _execute_c(self, code: str) -> Dict[str, Any]:
        """Execute C code by compiling and running."""
        try:
            import subprocess
            
            # Create temporary C file
            c_file = os.path.join(self.working_directory, "temp_code.c")
            with open(c_file, 'w') as f:
                f.write(code)
            
            # Compile
            compile_result = subprocess.run(
                ["gcc", "-o", os.path.join(self.working_directory, "temp_program"), c_file],
                capture_output=True,
                text=True
            )
            
            if compile_result.returncode != 0:
                return {"status": "error", "stderr": f"Compilation failed: {compile_result.stderr}"}
            
            # Run
            run_result = subprocess.run(
                [os.path.join(self.working_directory, "temp_program")],
                capture_output=True,
                text=True,
                timeout=self.max_execution_time
            )
            
            return {
                "status": "success",
                "stdout": run_result.stdout,
                "stderr": run_result.stderr,
                "returncode": run_result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {"status": "error", "stderr": "Execution timed out"}
        except Exception as e:
            return {"status": "error", "stderr": str(e)}
    
    def _execute_java(self, code: str) -> Dict[str, Any]:
        """Execute Java code by compiling and running."""
        try:
            import subprocess
            
            # Create temporary Java file
            java_file = os.path.join(self.working_directory, "TempCode.java")
            with open(java_file, 'w') as f:
                f.write(code)
            
            # Compile
            compile_result = subprocess.run(
                ["javac", java_file],
                capture_output=True,
                text=True
            )
            
            if compile_result.returncode != 0:
                return {"status": "error", "stderr": f"Compilation failed: {compile_result.stderr}"}
            
            # Run
            run_result = subprocess.run(
                ["java", "-cp", self.working_directory, "TempCode"],
                capture_output=True,
                text=True,
                timeout=self.max_execution_time
            )
            
            return {
                "status": "success",
                "stdout": run_result.stdout,
                "stderr": run_result.stderr,
                "returncode": run_result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {"status": "error", "stderr": "Execution timed out"}
        except Exception as e:
            return {"status": "error", "stderr": str(e)}

# Create a global instance for use by tools
interpreter_instance = CodeInterpreter()

@tool
def execute_code_multilang(code: str, language: str = "python") -> str:
    """Execute code in multiple languages (Python, Bash, SQL, C, Java) and return results.

    Args:
        code (str): The source code to execute.
        language (str): The language of the code. Supported: "python", "bash", "sql", "c", "java".

    Returns:
        A string summarizing the execution results (stdout, stderr, errors, plots, dataframes if any).
    """
    supported_languages = ["python", "bash", "sql", "c", "java"]
    language = language.lower()

    if language not in supported_languages:
        return f"❌ Unsupported language: {language}. Supported languages are: {', '.join(supported_languages)}"

    result = interpreter_instance.execute_code(code, language=language)

    response = []

    if result["status"] == "success":
        response.append(f"✅ Code executed successfully in **{language.upper()}**")

        if result.get("stdout"):
            response.append(
                "\n**Standard Output:**\n```\n" + result["stdout"].strip() + "\n```"
            )

        if result.get("stderr"):
            response.append(
                "\n**Standard Error (if any):**\n```\n"
                + result["stderr"].strip()
                + "\n```"
            )

        if result.get("result") is not None:
            response.append(
                "\n**Execution Result:**\n```\n"
                + str(result["result"]).strip()
                + "\n```"
            )

        if result.get("dataframes"):
            for df_info in result["dataframes"]:
                response.append(
                    f"\n**DataFrame `{df_info['name']}` (Shape: {df_info['shape']})**"
                )
                df_preview = pd.DataFrame(df_info["head"])
                response.append("First 5 rows:\n```\n" + str(df_preview) + "\n```")

        if result.get("plots"):
            response.append(
                f"\n**Generated {len(result['plots'])} plot(s)** (Image data returned separately)"
            )

    else:
        response.append(f"❌ Code execution failed in **{language.upper()}**")
        if result.get("stderr"):
            response.append(
                "\n**Error Log:**\n```\n" + result["stderr"].strip() + "\n```"
            )

    return "\n".join(response)

# ========== MATH TOOLS ==========
@tool
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

@tool
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

@tool
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

@tool
def divide(a: float, b: float) -> float:
    """
    Divide the first number by the second and return the result.

    Args:
        a (float): The numerator.
        b (float): The denominator. Must not be zero.

    Returns:
        float: The quotient of a and b.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """
    Compute the modulus (remainder) of two integers.

    Args:
        a (int): The dividend.
        b (int): The divisor.

    Returns:
        int: The remainder when a is divided by b.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a % b

@tool
def power(a: float, b: float) -> float:
    """
    Raise the first number to the power of the second and return the result.

    Args:
        a (float): The base number.
        b (float): The exponent.

    Returns:
        float: a raised to the power of b.
    """
    return a ** b

@tool
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
@tool
def wiki_search(query: str) -> str:
    """
    Search Wikipedia for a query and return up to 3 results as formatted text.

    Args:
        query (str): The search query string for Wikipedia.

    Returns:
        str: Formatted search results from Wikipedia with source information and content.
    """
    try:
        search_docs = WikipediaLoader(query=query, load_max_docs=3).load()
        formatted_results = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}'
                for doc in search_docs
            ]
        )
        return {"wiki_results": formatted_results}
    except Exception as e:
        return f"Error in Wikipedia search: {str(e)}"

@tool
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

@tool
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
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}'
                for doc in search_docs
            ]
        )
        return {"arxiv_results": formatted_results}
    except Exception as e:
        return f"Error in Arxiv search: {str(e)}"

# ========== FILE/DATA TOOLS ==========
@tool
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

@tool
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

@tool
def get_task_file(task_id: str, file_name: str) -> str:
    """
    Download a file associated with a given task_id from the evaluation API, with a local fallback.
    
    This tool is used to download files that are part of GAIA benchmark tasks.
    It first tries to download from the evaluation API, and if that fails
    (e.g., due to network issues or rate limits),
    it falls back to local files in the 'files' directory.
    The file is always saved to a 'downloads' directory.

    Args:
        task_id (str): The task ID for the file to download.
        file_name (str): The name of the file to download.

    Returns:
        str: The absolute file path where the file was downloaded, or an error message if not found.
    """
    directory_name = "downloads"
    os.makedirs(directory_name, exist_ok=True)
    try:
        # Try to download from evaluation API
        evaluation_api_base_url = os.environ.get("EVALUATION_API_BASE_URL", "https://api.gaia-benchmark.com")
        response = requests.get(f"{evaluation_api_base_url}/files/{task_id}", timeout=15)
        response.raise_for_status()
        filepath = os.path.join(directory_name, file_name)
        with open(filepath, 'wb') as file:
            file.write(response.content)
        return os.path.abspath(filepath)
    except Exception as e:
        # Fallback to local files
        try:
            local_filepath = os.path.join("files", file_name)
            if os.path.exists(local_filepath):
                filepath = os.path.join(directory_name, file_name)
                shutil.copy2(local_filepath, filepath)
                return os.path.abspath(filepath)
            else:
                return f"Error: File {file_name} not found locally or via API"
        except Exception as local_error:
            return f"Error downloading file: {str(e)}. Local fallback also failed: {str(local_error)}"

@tool
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

@tool
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

@tool
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

# ========== IMAGE ANALYSIS/GENERATION TOOLS ==========
@tool
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

@tool
def transform_image(image_base64: str, operation: str, params: Optional[Dict[str, Any]] = None) -> str:
    """
    Transform an image using various operations like resize, rotate, filter, etc.

    Args:
        image_base64 (str): The base64-encoded string of the image to transform.
        operation (str): The transformation operation to apply.
        params (Dict[str, Any], optional): Parameters for the transformation.

    Returns:
        str: JSON string with the transformed image as base64 or error message.
    """
    try:
        img = decode_image(image_base64)
        params = params or {}

        if operation == "resize":
            width = params.get("width", img.width)
            height = params.get("height", img.height)
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        elif operation == "rotate":
            angle = params.get("angle", 0)
            img = img.rotate(angle, expand=True)
        elif operation == "flip":
            direction = params.get("direction", "horizontal")
            if direction == "horizontal":
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif operation == "blur":
            radius = params.get("radius", 2)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        elif operation == "sharpen":
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        elif operation == "brightness":
            factor = params.get("factor", 1.0)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        elif operation == "contrast":
            factor = params.get("factor", 1.0)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        else:
            return json.dumps({"error": f"Unsupported operation: {operation}"}, indent=2)

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return json.dumps({"transformed_image": result_base64}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@tool
def draw_on_image(image_base64: str, drawing_type: str, params: Dict[str, Any]) -> str:
    """
    Draw shapes, text, or other elements on an image.

    Args:
        image_base64 (str): The base64-encoded string of the image to draw on.
        drawing_type (str): The type of drawing to perform.
        params (Dict[str, Any]): Parameters for the drawing operation.

    Returns:
        str: JSON string with the modified image as base64 or error message.
    """
    try:
        img = decode_image(image_base64)
        draw = ImageDraw.Draw(img)

        if drawing_type == "text":
            text = params.get("text", "")
            position = params.get("position", (10, 10))
            color = params.get("color", "black")
            size = params.get("size", 20)
            
            try:
                font = ImageFont.truetype("arial.ttf", size)
            except:
                font = ImageFont.load_default()
            
            draw.text(position, text, fill=color, font=font)
        elif drawing_type == "rectangle":
            coords = params.get("coords", [10, 10, 100, 100])
            color = params.get("color", "red")
            width = params.get("width", 2)
            draw.rectangle(coords, outline=color, width=width)
        elif drawing_type == "circle":
            center = params.get("center", (50, 50))
            radius = params.get("radius", 30)
            color = params.get("color", "blue")
            width = params.get("width", 2)
            
            bbox = [center[0] - radius, center[1] - radius, 
                   center[0] + radius, center[1] + radius]
            draw.ellipse(bbox, outline=color, width=width)
        elif drawing_type == "line":
            start = params.get("start", (10, 10))
            end = params.get("end", (100, 100))
            color = params.get("color", "green")
            width = params.get("width", 2)
            draw.line([start, end], fill=color, width=width)
        else:
            return json.dumps({"error": f"Unsupported drawing type: {drawing_type}"}, indent=2)

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return json.dumps({"modified_image": result_base64}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@tool
def generate_simple_image(image_type: str, width: int = 500, height: int = 500, 
                         params: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate simple images like gradients, solid colors, checkerboard, or noise patterns.

    Args:
        image_type (str): The type of image to generate.
        width (int): The width of the generated image.
        height (int): The height of the generated image.
        params (Dict[str, Any], optional): Additional parameters for image generation.

    Returns:
        str: JSON string with the generated image as base64 or error message.
    """
    try:
        params = params or {}
        
        if image_type == "solid":
            color = params.get("color", (255, 255, 255))
            img = Image.new("RGB", (width, height), color)
        elif image_type == "gradient":
            start_color = params.get("start_color", (255, 0, 0))
            end_color = params.get("end_color", (0, 0, 255))
            direction = params.get("direction", "horizontal")
            
            img = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(img)
            
            if direction == "horizontal":
                for x in range(width):
                    r = int(start_color[0] + (end_color[0] - start_color[0]) * x / width)
                    g = int(start_color[1] + (end_color[1] - start_color[1]) * x / width)
                    b = int(start_color[2] + (end_color[2] - start_color[2]) * x / width)
                    draw.line([(x, 0), (x, height)], fill=(r, g, b))
            else:
                for y in range(height):
                    r = int(start_color[0] + (end_color[0] - start_color[0]) * y / height)
                    g = int(start_color[1] + (end_color[1] - start_color[1]) * y / height)
                    b = int(start_color[2] + (end_color[2] - start_color[2]) * y / height)
                    draw.line([(0, y), (width, y)], fill=(r, g, b))
        elif image_type == "noise":
            noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(noise_array, "RGB")
        elif image_type == "checkerboard":
            square_size = params.get("square_size", 50)
            color1 = params.get("color1", "white")
            color2 = params.get("color2", "black")
            img = Image.new("RGB", (width, height))
            for y in range(0, height, square_size):
                for x in range(0, width, square_size):
                    color = color1 if ((x // square_size) + (y // square_size)) % 2 == 0 else color2
                    for dy in range(square_size):
                        for dx in range(square_size):
                            if x + dx < width and y + dy < height:
                                img.putpixel((x + dx, y + dy), color)
        else:
            return json.dumps({"error": f"Unsupported image_type {image_type}"}, indent=2)

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return json.dumps({"generated_image": result_base64}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@tool
def combine_images(images_base64: List[str], operation: str, 
                  params: Optional[Dict[str, Any]] = None) -> str:
    """
    Combine multiple images using various operations (collage, stack, blend, horizontal, vertical, overlay, etc.).

    Args:
        images_base64 (List[str]): List of base64-encoded image strings.
        operation (str): The combination operation to perform.
        params (Dict[str, Any], optional): Parameters for the combination.

    Returns:
        str: JSON string with the combined image as base64 or error message.
    """
    try:
        if len(images_base64) < 2:
            return json.dumps({"error": "At least 2 images required for combination"}, indent=2)
        
        images = [decode_image(b64) for b64 in images_base64]
        params = params or {}
        
        if operation == "horizontal":
            # Combine images side by side
            total_width = sum(img.width for img in images)
            max_height = max(img.height for img in images)
            result = Image.new("RGB", (total_width, max_height))
            x_offset = 0
            for img in images:
                result.paste(img, (x_offset, 0))
                x_offset += img.width
        elif operation == "vertical":
            # Stack images vertically
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images)
            result = Image.new("RGB", (max_width, total_height))
            y_offset = 0
            for img in images:
                result.paste(img, (0, y_offset))
                y_offset += img.height
        elif operation == "overlay":
            # Overlay images on top of each other
            base_img = images[0]
            for overlay_img in images[1:]:
                if overlay_img.size != base_img.size:
                    overlay_img = overlay_img.resize(base_img.size, Image.Resampling.LANCZOS)
                base_img = Image.alpha_composite(base_img.convert("RGBA"), overlay_img.convert("RGBA"))
            result = base_img.convert("RGB")
        elif operation == "stack":
            # Original stack operation with direction parameter
            direction = params.get("direction", "horizontal")
            if direction == "horizontal":
                total_width = sum(img.width for img in images)
                max_height = max(img.height for img in images)
                result = Image.new("RGB", (total_width, max_height))
                x = 0
                for img in images:
                    result.paste(img, (x, 0))
                    x += img.width
            else:
                max_width = max(img.width for img in images)
                total_height = sum(img.height for img in images)
                result = Image.new("RGB", (max_width, total_height))
                y = 0
                for img in images:
                    result.paste(img, (0, y))
                    y += img.height
        else:
            return json.dumps({"error": f"Unsupported combination operation: {operation}"}, indent=2)

        result_path = save_image(result)
        result_base64 = encode_image(result_path)
        return json.dumps({"combined_image": result_base64}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

# ========== VIDEO/AUDIO UNDERSTANDING TOOLS ==========
@tool
def understand_video(youtube_url: str, prompt: str) -> str:
    """
    Analyze a YouTube video using Google Gemini's video understanding capabilities.
    
    This tool can understand video content, extract information, and answer questions
    about what happens in the video.
    It uses the Gemini API and requires the GEMINI_KEY environment variable to be set.
    
    Args:
        youtube_url (str): The URL of the YouTube video to analyze.
        prompt (str): A question or request regarding the video content.
    
    Returns:
        str: Analysis of the video content based on the prompt, or error message.
    
    Note:
        Requires GEMINI_KEY environment variable to be set.
        Install with: pip install google-genai
    """
    if not GEMINI_AVAILABLE:
        return "Google Gemini not available. Install with: pip install google-genai"
    try:
        gemini_key = os.environ.get("GEMINI_KEY")
        if not gemini_key:
            return "GEMINI_KEY not found in environment variables."
        client = genai.Client(api_key=gemini_key)
        video_description = client.models.generate_content(
            model="gemini-2.5-pro",  # Use same model as agent for consistency
            contents=types.Content(
                parts=[
                    types.Part(file_data=types.FileData(file_uri=youtube_url)),
                    types.Part(text=prompt)
                ]
            )
        )
        return video_description.text
    except Exception as e:
        return f"Error understanding video: {str(e)}"

@tool
def understand_audio(file_path: str, prompt: str) -> str:
    """
    Analyze an audio file using Google Gemini's audio understanding capabilities.
    
    This tool can transcribe audio, understand spoken content, and answer questions
    about the audio content.
    It uses the Gemini API and requires the GEMINI_KEY environment variable to be set.
    The audio file is uploaded to Gemini and then analyzed with the provided prompt.
    
    Args:
        file_path (str): The path to the local audio file to analyze, or base64 encoded audio data.
        prompt (str): A question or request regarding the audio content.
    
    Returns:
        str: Analysis of the audio content based on the prompt, or error message.
    
    Note:
        Requires GEMINI_KEY environment variable to be set.
        Install with: pip install google-genai
    """
    if not GEMINI_AVAILABLE:
        return "Google Gemini not available. Install with: pip install google-genai"
    try:
        gemini_key = os.environ.get("GEMINI_KEY")
        if not gemini_key:
            return "GEMINI_KEY not found in environment variables."
        
        client = genai.Client(api_key=gemini_key)
        
        # Check if file_path is base64 data or actual file path
        if file_path.startswith('/') or os.path.exists(file_path):
            # It's a file path
            mp3_file = client.files.upload(file=file_path)
        else:
            # Assume it's base64 data
            import base64
            import tempfile
            
            try:
                # Decode base64 and create temporary file
                audio_data = base64.b64decode(file_path)
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file_path = temp_file.name
                
                try:
                    mp3_file = client.files.upload(file=temp_file_path)
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)
            except Exception as decode_error:
                return f"Error processing audio data: {str(decode_error)}. Expected base64 encoded audio data or valid file path."
        
        audio_description = client.models.generate_content(
            model="gemini-2.5-pro",  # Use same model as agent for consistency
            contents=[prompt, mp3_file]
        )
        return audio_description.text
    except Exception as e:
        return f"Error understanding audio: {str(e)}"

# ========== CHESS TOOLS ==========
def _convert_chess_move_internal(piece_placement: str, move: str) -> str:
    """
    Internal function to convert chess moves from coordinate notation to algebraic notation.
    Uses Google Gemini to convert chess moves between different notations.
    Coordinate notation uses square names (e.g., "e2e4"), while algebraic notation
    uses piece symbols and square names (e.g., "e4", "Nf3", "O-O").
    The function constructs a prompt for Gemini and expects 
    only the algebraic notation as output, with no extra commentary.
    """
    try:
        # Use Google Gemini to convert coordinate notation to algebraic notation
        if not GEMINI_AVAILABLE:
            return "Google Gemini not available for chess move conversion"
        
        genai.configure(api_key=os.environ.get("GEMINI_KEY"))
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        prompt = f"""
        Convert this chess move from coordinate notation to algebraic notation.
        
        Piece placement: {piece_placement}
        Move in coordinate notation: {move}
        
        Return only the algebraic notation (e.g., "e4", "Nf3", "O-O", "Qxd5", etc.)
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        return f"Error converting chess move: {str(e)}"

@tool
def convert_chess_move(piece_placement: str, move: str) -> str:
    """
    Convert a chess move from coordinate notation to algebraic notation using Google Gemini.
    
    This tool uses Google Gemini to convert chess moves between different notations.
    Coordinate notation uses square names (e.g., "e2e4"), while algebraic notation
    uses piece symbols and square names (e.g., "e4", "Nf3", "O-O").
    The function constructs a prompt for Gemini and expects 
    only the algebraic notation as output, with no extra commentary.
    
    Args:
        piece_placement (str): The chess piece placement in plain text or FEN format.
        move (str): The move in coordinate notation (e.g., "e2e4").

    Returns:
        str: The move in algebraic notation, or error message.
    
    Note:
        Requires GEMINI_KEY environment variable to be set.
        Install with: pip install google-genai
    """
    if not GEMINI_AVAILABLE:
        return "Google Gemini not available. Install with: pip install google-genai"
    try:
        gemini_key = os.environ.get("GEMINI_KEY")
        if not gemini_key:
            return "GEMINI_KEY not found in environment variables."
        
        client = genai.Client(api_key=GEMINI_KEY)
        move_message = (
            f"Convert this chess move from coordinate notation to algebraic "
            f"notation: {move}. Use the following piece placement: {piece_placement}. "
            f"Do not provide any additional thinking or commentary in the response, "
            f"just the algebraic notation only."
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",  # Use same model as agent for consistency
            contents=move_message
        )
        return response.text
    except Exception as e:
        return f"Error converting chess move: {str(e)}"

def _get_best_chess_move_internal(fen: str) -> str:
    """
    Internal function to get the best chess move for a given FEN position.
    """
    try:
        chess_eval_url = os.environ.get("CHESS_EVAL_URL", "https://lichess.org/api/cloud-eval")
        url = f"{chess_eval_url}?fen={urllib.parse.quote(fen)}&depth=15"
        lichess_key = os.environ.get("LICHESS_KEY")
        headers = {}
        if lichess_key:
            headers["Authorization"] = f"Bearer {lichess_key}"
        response = requests.get(url, timeout=15, headers=headers)
        if response.status_code == 200:
            data = json.loads(response.text)
            if data.get('success') == True:
                return data['bestmove'].split()[1]
            else:
                return f"Error getting chess evaluation: {data.get('error', 'Unknown error')}"
        else:
            return f"Error getting chess evaluation: HTTP {response.status_code}"
    except Exception as e:
        return f"Error getting chess evaluation: {str(e)}"


# ========== FEN HELPER FUNCTIONS ==========

@tool
def get_best_chess_move(fen: str) -> str:
    """
    Get the best chess move in coordinate notation based on a FEN representation
    using a chess evaluation API.
    
    This tool uses a chess evaluation API (default: Lichess cloud eval) 
    to find the best move for a given position.
    The FEN (Forsyth-Edwards Notation) describes the current chess position.
    Eg. rn1q1rk1/pp2b1pp/2p2n2/3p1pB1/3P4/1QP2N2/PP1N1PPP/R4RK1 b - - 1 11

    Args:
        fen (str): The chess position in FEN (Forsyth-Edwards Notation) format.

    Returns:
        str: The best move in coordinate notation (e.g., "e2e4"), or an error message.
    
    Note:
        Requires CHESS_EVAL_URL environment variable to be set.
    """
    return _get_best_chess_move_internal(fen)

# ========== FEN PROCESSING HELPERS ==========
def _add_fen_game_state(board_placement,
                    side_to_move,
                    castling="-",
                    en_passant="-",
                    halfmove_clock=0,
                    fullmove_number=1):
    """
    Appends standard game state information to a FEN board placement string.

    Args:
        board_placement (str): The board layout part of the FEN string
                            (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR").
        side_to_move (str): The active color ('w' for White, 'b' for Black).
                            Case-insensitive, will be converted to lowercase.
        castling (str, optional): Castling availability string (e.g., "KQkq", "-").
                                Defaults to "-".
        en_passant (str, optional): En passant target square string (e.g., "e3", "-").
                                    Defaults to "-".
        halfmove_clock (int, optional): The number of halfmoves since the last
                                    capture or pawn advance. Defaults to 0.
        fullmove_number (int, optional): The number of the full move. Starts at 1
                                    and increments after Black's move. Defaults to 1.

    Returns:
        str: The complete FEN string including the game state,
            or an error message string if inputs are invalid.
    """
    # Validate side_to_move
    side_to_move_lower = str(side_to_move).lower()
    if side_to_move_lower not in ['w', 'b']:
        return f"Error: side_to_move must be 'w' or 'b', received '{side_to_move}'"

    # Validate clock values (should be non-negative integers, fullmove >= 1)
    try:
        halfmove_clock = int(halfmove_clock)
        fullmove_number = int(fullmove_number)
        if halfmove_clock < 0:
            raise ValueError("halfmove_clock cannot be negative.")
        if fullmove_number < 1:
            raise ValueError("fullmove_number must be 1 or greater.")
    except (ValueError, TypeError):
        return (f"Error: halfmove_clock ('{halfmove_clock}') and "
                f"fullmove_number ('{fullmove_number}') must be valid integers "
                f"(non-negative and positive respectively).")

    # Assemble the full FEN string using the validated/defaulted values
    # Note: castling and en_passant strings are used directly as passed or defaulted.
    # More complex validation could be added for them if needed.
    full_fen = (f"{board_placement} {side_to_move_lower} {castling} "
                f"{en_passant} {halfmove_clock} {fullmove_number}")

    return full_fen

def _get_chess_board_fen_internal(image_path: str, player_turn: str) -> str:
    """
    Internal function to get the FEN representation from an image of a chess board.
    Uses linrock-chessboard-recognizer's recognize.py script via subprocess.
    Args:
        image_path (str): Path to the chessboard image file.
        player_turn (str): The player with the next turn ("black" or "white").
    Returns:
        str: The FEN string predicted by the recognizer, or an error message.
    """
    try:
        result = subprocess.run([
            "python3", "chessboard-recognizer/recognize.py", image_path, "-q"
        ], capture_output=True, text=True, check=True)
        # The recognizer prints the FEN on the last line of stdout
        lines = result.stdout.strip().splitlines()
        for line in reversed(lines):
            if "/" in line and len(line.split("/")) == 8:
                return line.strip()
        return f"Error: FEN not found in recognizer output. Output was: {result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Error running chessboard recognizer: {e.stderr or e.stdout}"
    except Exception as e:
        return f"Error running chessboard recognizer: {str(e)}"


@tool
def get_chess_board_fen(image_path: str, player_turn: str) -> str:
    """
    Get the FEN representation from an image of a chess board.
    This tool uses computer vision to analyze a chess board image and convert it
    to FEN (Forsyth-Edwards Notation) format.
    Args:
        image_path (str): The path to the chess board image file.
        player_turn (str): The player with the next turn ("black" or "white").
    Returns:
        str: The FEN representation of the chess position, or error message.
    """
    return _get_chess_board_fen_internal(image_path, player_turn)

@tool
def solve_chess_position(image_path: str, player_turn: str, question: str = "") -> str:
    """
    Solve a chess position by analyzing the board image and finding the best move.
    This comprehensive tool:
    1. Converts the chess board image to FEN notation
    2. Gets the best move from a chess evaluation API
    3. Converts the coordinate notation to algebraic notation
    4. Returns the solution with analysis
    Args:
        image_path (str): The path to the chess board image file.
        player_turn (str): The player with the next turn ("black" or "white").
        question (str): Optional question about the position (e.g., "guarantees a win").
    Returns:
        str: The best move in algebraic notation with analysis, or error message.
    Note:
        Requires image-to-FEN function, chess evaluation API, and Google Gemini to be available.
    """
    try:
        # Step 1: Get FEN from image (using internal function to avoid deprecation warning)
        fen = _get_chess_board_fen_internal(image_path, player_turn)
        if isinstance(fen, str) and fen.startswith("Error"):
            return f"Error getting FEN: {fen}"
        # Step 2: Get best move in coordinate notation (using internal function)
        best_move_coord = _get_best_chess_move_internal(fen)
        if best_move_coord.startswith("Error"):
            return f"Error getting best move: {best_move_coord}"
        # Step 3: Convert to algebraic notation (using internal function)
        # Create a simple piece placement description for the LLM
        piece_placement = f"FEN: {fen}"
        algebraic_move = _convert_chess_move_internal(piece_placement, best_move_coord)
        if algebraic_move.startswith("Error"):
            return f"Error converting move: {algebraic_move}"
        # Step 4: Format the response
        result = f"Chess Position Analysis:\n"
        result += f"FEN: {fen}\n"
        result += f"Player to move: {player_turn}\n"
        result += f"Best move (coordinate): {best_move_coord}\n"
        result += f"Best move (algebraic): {algebraic_move}\n"
        if question:
            result += f"\nQuestion: {question}\n"
            result += f"Answer: {algebraic_move}"
        return result
    except AttributeError as e:
        # Handle AttributeError specifically (like parent_run_id issues)
        error_msg = f"Tool execution error (AttributeError): {str(e)}"
        print(f"[Chess Tool] {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error solving chess position: {str(e)}"
        print(f"[Chess Tool] {error_msg}")
        return error_msg

# ========== END OF TOOLS.PY ========== 