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

## The result dataset

https://huggingface.co/datasets/arterm-sedov/agent-course-final-assignment

---

## 🚀 The Ultimate Multi-LLM GAIA Agent

Behold arterm-sedov's GAIA Unit 4 Agent — a robust and extensible system designed for real-world reliability and benchmark performance. This agent is the result of a creative collaboration between Arterm and Cursor IDE to make complex things simple, powerful, and fun to use.

### What Makes This Agent Stand Out?

- **Multi-LLM Orchestration:** Dynamically selects from Google Gemini, Groq, OpenRouter, and HuggingFace models. Each model is tested for both plain and tool-calling support at startup, ensuring maximum coverage and reliability.
- **Model-Level Tool Support:** Binds tools to each model if supported. Google Gemini is always bound with tools for maximum capability—even if the tool test returns empty (tool-calling works in practice; a warning is logged for transparency).
- **Automatic Fallbacks:** If a model fails or lacks a required feature, the agent automatically falls back to the next available model, ensuring robust and uninterrupted operation.
- **Comprehensive Tool Suite:** Math, code execution, file and image analysis, web and vector search, chess analysis, and more. Tools are modular and extensible. Some tools are themselves AI callers—such as web search, Wikipedia, arXiv, and code execution—enabling the agent to chain LLMs and tools for advanced, multi-step reasoning.
- **Contextual Vector Search:** Uses Supabase vector search as a baseline to decide if an LLM call succeeded and calculates a success score for each model's answer. Reference answers are used for internal evaluation, not submission.
- **Structured Initialization Summary:** After startup, a clear table shows which models/providers are available, with/without tools, and any errors—so you always know your agent's capabilities.
- **Transparent Reasoning:** Logs its reasoning, tool usage, and fallback decisions for full traceability. You see not just the answer, but how it was reached.

---

## 🎯 Usage

1. Log in to your Hugging Face account using the login button
2. Click "Run Evaluation & Submit All Answers" to start the evaluation
3. Monitor progress and view results in the interface
4. Download logs and results from the LOGS tab

---

The agent is ready for the GAIA Unit 4 benchmark — battle-tested, transparent, and extensible.

If you want to know how it works, read on. If you want to get started, [check the setup instructions](./SETUP_INSTRUCTIONS.md). Happy hacking! 🕵🏻‍♂️

## 🏗️ Architecture at a Glance

- **`agent.py`**: Main agent logic, LLM/model orchestration, tool binding, and summary reporting
- **`tools.py`**: Modular tool collection—math, code, web, file, image, chess, and more
- **`app.py`**: Gradio interface for interactive use
- **`git_file_helper.py`**: File operations with HuggingFace Hub API (CommitOperationAdd)
- **`setup_venv.py`**: Cross-platform setup script

### Platform-Specific Dependencies

- All core dependencies included in `requirements.txt`
- TensorFlow pre-installed on Hugging Face Spaces
- File operations with `huggingface_hub` for API-based uploads

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

## 📊 Dataset Upload System

The project includes a comprehensive dataset upload system for tracking agent performance and initialization:

### 🚀 Features

- **Structured dataset uploads** to HuggingFace datasets
- **Schema validation** against `dataset_config.json`
- **Two data splits**: `init` (initialization) and `runs` (evaluation results)
- **Automatic data serialization** for complex objects
- **Robust error handling** with fallback mechanisms

### 📚 Documentation

- **`dataset_config.json`**: Schema definition for dataset structure
- **`file_helper.py`**: Core upload functions with validation
- **`misc_files/validate_file_upload.py`**: Validation script for upload functionality
- **`misc_files/test_dataset_upload.py`**: Test suite for dataset uploads

### 🔧 Usage Examples

```python
# Upload initialization data
from file_helper import upload_init_summary
init_data = {
    "timestamp": "20250705_123456",
    "init_summary": "LLM initialization results...",
    "debug_output": "Debug information...",
    "llm_config": {"models": [...]},
    "available_models": {"gemini": {...}},
    "tool_support": {"gemini": True}
}
success = upload_init_summary(init_data)

# Upload evaluation run data
from file_helper import upload_evaluation_run
run_data = {
    "run_id": "run_20250705_123456",
    "timestamp": "20250705_123456",
    "questions_count": 10,
    "results_log": [...],
    "results_df": [...],
    "username": "user123",
    "final_status": "Success: 80% score",
    "score_path": "logs/score.txt"
}
success = upload_evaluation_run(run_data)
```

---

## 📋 Data Upload System

The evaluation automatically uploads structured data to the HuggingFace dataset:

### 🔄 Initialization Data (`init` split)
- **Timestamp**: When the agent was initialized
- **Init Summary**: LLM initialization results and model status
- **Debug Output**: Detailed initialization logs
- **LLM Config**: Configuration for all available models
- **Available Models**: List of successfully initialized models
- **Tool Support**: Tool support status for each model

### 📊 Evaluation Data (`runs` split)
- **Run ID**: Unique identifier for each evaluation run
- **Timestamp**: When the evaluation was completed
- **Questions Count**: Number of questions processed
- **Results Log**: Detailed log of all questions and answers
- **Results DF**: Structured data table of results
- **Username**: User who ran the evaluation
- **Final Status**: Success/failure status and score
- **Score Path**: Path to detailed score file

All data is automatically validated against the schema and uploaded to the HuggingFace dataset for analysis and tracking.

---

