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

A robust, multi-LLM agent for the GAIA Unit 4 benchmark, blending advanced tool use, model fallback, and vector search for real-world reliability.

## Requirements

- **`requirements.txt`**: For Hugging Face Spaces and Linux/macOS

## Installation

### Quick Setup (Recommended)
```bash
python setup_venv.py
```
The script auto-selects the right requirements file for your OS.

### Manual Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with:
```
GEMINI_KEY=your_gemini_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
# Optional for OpenRouter, Groq, HuggingFace
OPENROUTER_API_KEY=your_openrouter_key
GROQ_API_KEY=your_groq_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

## Usage

```bash
python app.py
```

## Agent Behavior & Tooling

- **Multi-LLM Orchestration**: The agent dynamically selects from Google Gemini, Groq, OpenRouter, and HuggingFace models. Each model is tested for both plain and tool-calling support at startup.
- **Model-Level Tool Support**: The agent binds tools to each model if supported. Google Gemini is always bound with tools for maximum capability, even if the tool test returns empty (tool-calling works in practice; a warning is logged).
- **Automatic Fallbacks**: If a model fails or does not support a required feature, the agent automatically falls back to the next available model, ensuring robust and uninterrupted operation.
- **Comprehensive Tool Suite**: The agent can perform math, code execution, file and image analysis, web and vector search, chess analysis, and more. Tools are modular and extensible. Some tools are themselves AI callers—such as web search, Wikipedia, arXiv, and code execution—enabling the agent to chain LLMs and tools for advanced, multi-step reasoning.
- **Contextual Vector Search**: The agent uses Supabase vector search acting as a baseline to decide if an LLM call succeeded and calculates success score for each model's answer for a question. Reference answers are not submitted, they are used for internal evaluation of LLMs.
- **Structured Initialization Summary**: After startup, a clear table shows which models/providers are available, with/without tools, and any errors.
- **Transparent Reasoning**: The agent logs its reasoning, tool usage, and fallback decisions for full traceability.

## Architecture

- `agent.py`: Main agent logic, LLM/model orchestration, tool binding, and summary reporting
- `tools.py`: Modular tool collection
- `app.py`: Gradio interface
- `setup_venv.py`: Cross-platform setup

## Platform-Specific Dependencies

- All core dependencies included in `requirements.txt`
- TensorFlow pre-installed on Hugging Face Spaces

## LLM Initialization & Tool Support

- Each LLM/model is tested for plain and tool-calling support
- Gemini (Google) is always bound with tools if enabled, even if tool test returns empty (tool-calling works in real use; warning is logged)
- OpenRouter, Groq, and HuggingFace are supported with model-level tool-calling detection
- After initialization, a summary table is printed showing provider, model, plain/tools status, and errors

## Support & Next Steps

- See `SETUP_INSTRUCTIONS.md` for troubleshooting and advanced config
- After setup, test the agent, run evaluation, and submit to GAIA benchmark

The agent is ready for the GAIA Unit 4 benchmark—battle-tested, transparent, and extensible. 🚀