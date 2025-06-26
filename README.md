---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# arterm-sedov GAIA Agent

A comprehensive agent for the GAIA Unit 4 benchmark, combining tools from multiple reference implementations.

## Requirements

The project uses two requirements files to handle platform differences:

- **`requirements.txt`**: For Hugging Face Spaces and Linux/macOS (no TensorFlow needed)
- **`requirements.win.txt`**: For Windows local development (includes TensorFlow)

## Installation

### Quick Setup (Recommended)
```bash
python setup_venv.py
```
The setup script automatically selects the appropriate requirements file based on your platform.

### Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt      # For Hugging Face/Linux/macOS
# OR
pip install -r requirements.win.txt  # For Windows local development
```

## Environment Variables

Create a `.env` file with:
```
GEMINI_KEY=your_gemini_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## Usage

```bash
python app.py
```

## Features

- **Multi-LLM Support**: Google Gemini, Groq, HuggingFace
- **Comprehensive Tools**: Math, code, file, image, web, chess
- **Supabase Integration**: Vector search for similar Q/A
- **Robust Fallbacks**: Multiple LLM providers and embedding models
- **Cross-Platform**: Optimized for both Hugging Face Spaces and local development

## Architecture

- `agent.py`: Main agent logic with LLM integration
- `tools.py`: Comprehensive tool collection
- `app.py`: Gradio interface for Hugging Face Spaces
- `setup_venv.py`: Cross-platform setup script

## Platform-Specific Dependencies

### Hugging Face Spaces / Linux / macOS
- All core dependencies included
- TensorFlow is pre-installed on Hugging Face Spaces
- No additional setup needed

### Windows Local Development
- Same core dependencies as other platforms
- Includes `tensorflow-cpu` for local sentence-transformers support
- May require Visual Studio build tools for TensorFlow installation