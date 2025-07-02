---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# arterm-sedov GAIA Agent

> **For setup, installation, and troubleshooting, see [SETUP_INSTRUCTIONS.md](./SETUP_INSTRUCTIONS.md).**

---

## 🚀 The Ultimate Multi-LLM GAIA Agent

Behold arterm-sedov's GAIA Unit 4 Agent — a robust, transparent, and extensible system designed for real-world reliability and benchmark performance. This agent is the result of a creative collaboration between seasoned ML engineers, systems analysts, and technical writers who know how to make complex things simple, powerful, and fun to use.

### What Makes This Agent Stand Out?

- **Multi-LLM Orchestration:** Dynamically selects from Google Gemini, Groq, OpenRouter, and HuggingFace models. Each model is tested for both plain and tool-calling support at startup, ensuring maximum coverage and reliability.
- **Model-Level Tool Support:** Binds tools to each model if supported. Google Gemini is always bound with tools for maximum capability—even if the tool test returns empty (tool-calling works in practice; a warning is logged for transparency).
- **Automatic Fallbacks:** If a model fails or lacks a required feature, the agent automatically falls back to the next available model, ensuring robust and uninterrupted operation.
- **Comprehensive Tool Suite:** Math, code execution, file and image analysis, web and vector search, chess analysis, and more. Tools are modular and extensible. Some tools are themselves AI callers—such as web search, Wikipedia, arXiv, and code execution—enabling the agent to chain LLMs and tools for advanced, multi-step reasoning.
- **Contextual Vector Search:** Uses Supabase vector search as a baseline to decide if an LLM call succeeded and calculates a success score for each model's answer. Reference answers are used for internal evaluation, not submission.
- **Structured Initialization Summary:** After startup, a clear table shows which models/providers are available, with/without tools, and any errors—so you always know your agent's capabilities.
- **Transparent Reasoning:** Logs its reasoning, tool usage, and fallback decisions for full traceability. You see not just the answer, but how it was reached.

---

## 🏗️ Architecture at a Glance

- **`agent.py`**: Main agent logic, LLM/model orchestration, tool binding, and summary reporting
- **`tools.py`**: Modular tool collection—math, code, web, file, image, chess, and more
- **`app.py`**: Gradio interface for interactive use
- **`setup_venv.py`**: Cross-platform setup script

### Platform-Specific Dependencies
- All core dependencies included in `requirements.txt`
- TensorFlow pre-installed on Hugging Face Spaces

---

## 🧠 LLM Initialization & Tool Support

- Each LLM/model is tested for plain and tool-calling support
- Gemini (Google) is always bound with tools if enabled, even if tool test returns empty (tool-calling works in real use; warning is logged)
- OpenRouter, Groq, and HuggingFace are supported with model-level tool-calling detection
- After initialization, a summary table is printed showing provider, model, plain/tools status, and errors

---

## 🛠️ For Setup & Troubleshooting

See [SETUP_INSTRUCTIONS.md](./SETUP_INSTRUCTIONS.md) for:
- Installation and environment setup
- Requirements and dependencies
- Environment variable configuration
- Vector store setup
- Platform-specific tips (Windows, Linux/macOS, Hugging Face Spaces)
- Troubleshooting and advanced configuration

---

The agent is ready for the GAIA Unit 4 benchmark—battle-tested, transparent, and extensible. If you want to know how it works, read on. If you want to get started, [check the setup instructions](./SETUP_INSTRUCTIONS.md). Happy hacking! 🕵🏻‍♂️