# arterm-sedov Setup Instructions

## Overview

This guide provides comprehensive setup instructions for the arterm-sedov GAIA Unit 4 agent project. The setup is designed to work on both Windows and Linux/macOS systems using platform-specific requirements files.

## Prerequisites

- **Python 3.8 or higher**
- **Git** (for cloning the repository)
- **Internet connection** (for downloading dependencies)

## Quick Start

### Option 1: Automated Setup (Recommended)

The easiest way to set up the project is using the automated setup script:

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd arterm-sedov

# Run the automated setup script
python setup_venv.py
```

This script will:
- Check Python version compatibility
- Create a virtual environment
- Automatically detect your platform (Windows/Linux/macOS)
- Use the appropriate requirements file for your platform
- Install all dependencies in the correct order
- Verify the installation
- Provide next steps

### Option 2: Manual Setup

If you prefer manual setup or encounter issues with the automated script:

#### Step 1: Create Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 2: Install Dependencies

**For Windows:**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install build tools
pip install wheel setuptools

# Install dependencies using Windows-specific requirements
pip install -r requirements.win.txt
```

**For Linux/macOS:**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies using main requirements
pip install -r requirements.txt
```

## Requirements Files

The project uses platform-specific requirements files to handle different installation needs:

### `requirements.txt` (Linux/macOS/Hugging Face Space)
- Optimized for Linux, macOS, and Hugging Face Space deployment
- Uses flexible version constraints for maximum compatibility
- No Windows-specific build constraints

### `requirements.win.txt` (Windows)
- Contains Windows-specific version constraints
- Avoids problematic versions (like pandas 2.2.2)
- Includes all necessary version pins for Windows compatibility

The setup script automatically detects your platform and uses the appropriate file.

## Environment Variables Setup

Create a `.env` file in the project root with the following variables:

```env
# Required for Google Gemini integration
GEMINI_KEY=your_gemini_api_key_here

# Required for Supabase vector store
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here

# Optional: For HuggingFace integration
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Optional: For OpenRouter (chess move conversion)
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Getting API Keys

1. **Google Gemini API Key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key to your `.env` file

2. **Supabase Credentials:**
   - Create a Supabase project at [supabase.com](https://supabase.com)
   - Go to Settings > API
   - Copy the URL and anon key to your `.env` file

3. **HuggingFace API Key (Optional):**
   - Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - Create a new token
   - Copy to your `.env` file

## Vector Store Setup

After setting up the environment, you need to populate the vector store with reference data:

```bash
# Run the vector store setup
python setup_vector_store.py
```

This will:
- Load the metadata.jsonl file
- Connect to your Supabase instance
- Populate the vector store with reference Q&A data
- Test the similarity search functionality

## Running the Agent

### Development Mode

```bash
# Start the Gradio interface
python app.py
```

This will launch a web interface where you can:
- Test individual questions
- Run the full evaluation
- Submit answers to the GAIA benchmark

### Production Mode (Hugging Face Space)

The project is configured for Hugging Face Space deployment. The main `requirements.txt` is optimized for the HF environment.

## Troubleshooting

### Common Issues

#### 1. Platform Detection Issues

**Problem:** Wrong requirements file is used
**Solution:** The setup script automatically detects your platform. If you need to force a specific file:
```bash
# For Windows
pip install -r requirements.win.txt

# For Linux/macOS
pip install -r requirements.txt
```

#### 2. Virtual Environment Issues

**Problem:** Virtual environment creation fails
**Solution:** 
```bash
# Remove existing venv and recreate
rm -rf venv  # Linux/macOS
# OR
rmdir /s /q venv  # Windows
python setup_venv.py
```

#### 3. Permission Errors

**Problem:** Permission denied when installing packages
**Solution:**
```bash
# Use --user flag
pip install --user -r requirements.txt
```

#### 4. Missing Dependencies

**Problem:** Import errors after installation
**Solution:**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

#### 5. API Key Issues

**Problem:** "Missing API key" errors
**Solution:**
- Check that your `.env` file exists and has the correct format
- Verify API keys are valid and have proper permissions
- Ensure no extra spaces or quotes around the values

### Platform-Specific Issues

#### Windows

- **PowerShell Execution Policy:** If you get execution policy errors:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

- **Visual Studio Build Tools:** If you encounter build errors:
  - Install Visual Studio Build Tools 2019 or later
  - Or use conda instead of pip:
    ```cmd
    conda install pandas numpy
    pip install -r requirements.win.txt
    ```

#### Linux/macOS

- **Missing system dependencies:** Install required system packages:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install python3-dev build-essential
  
  # macOS
  xcode-select --install
  ```

## Verification

After setup, verify everything works:

```python
# Test basic imports
import numpy as np
import pandas as pd
import langchain
import supabase
import gradio

print("✅ All core packages imported successfully!")
print(f"Pandas version: {pd.__version__}")
```

## Project Structure

```
arterm-sedov/
├── agent.py              # Main agent implementation
├── app.py                # Gradio web interface
├── tools.py              # Tool functions for the agent
├── setup_venv.py         # Cross-platform setup script
├── setup_vector_store.py # Vector store initialization
├── requirements.txt      # Dependencies (Linux/macOS/HF Space)
├── requirements.win.txt  # Dependencies (Windows)
├── system_prompt.txt     # Agent system prompt
├── metadata.jsonl        # Reference Q&A data
├── supabase_docs.csv     # Vector store backup
└── .env                  # Environment variables (create this)
```

## Advanced Configuration

### Custom Model Providers

The agent supports multiple LLM providers. You can modify `agent.py` to use different providers:

- **Google Gemini** (default): Requires `GEMINI_KEY`
- **Groq**: Requires `GROQ_API_KEY`
- **HuggingFace**: Requires `HUGGINGFACE_API_KEY`

### Vector Store Configuration

The vector store uses Supabase with the following configuration:
- **Table:** `agent_course_reference`
- **Embedding Model:** `sentence-transformers/all-mpnet-base-v2`
- **Similarity Search:** Cosine similarity

### Tool Configuration

The agent includes comprehensive tools for:
- **Math operations:** Basic arithmetic, calculus, statistics
- **Web search:** Google search, Wikipedia, arXiv
- **File operations:** Download, read, analyze files
- **Image processing:** OCR, analysis, transformation
- **Chess analysis:** Position solving, move calculation
- **Code execution:** Python code interpreter

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the error logs in the console
3. Verify your environment variables are set correctly
4. Ensure all dependencies are installed properly

## Next Steps

After successful setup:

1. **Test the agent** with sample questions
2. **Run the evaluation** to see performance metrics
3. **Submit to GAIA benchmark** for official scoring
4. **Customize the agent** for your specific needs

The agent is now ready for the GAIA Unit 4 benchmark! 🚀