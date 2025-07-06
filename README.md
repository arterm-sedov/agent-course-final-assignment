---
title: Template Final Assignment
author: Arte(r)m Sedov
author_github: https://github.com/arterm-sedov/
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

# GAIA Agent

---

## 🚀 Arterm Sedov's Ultimate Multi-LLM GAIA Agent

Behold Arte(r)m's GAIA Unit 4 Agent — a robust and extensible system designed for real-world reliability and benchmark performance. This agent is the result of a creative collaboration between Arterm and Cursor IDE to make complex things simple, powerful, and fun to use.

This is Arterm's graduation work for The Agents Course:

<https://huggingface.co/learn/agents-course/en/>

## The result dataset

<https://huggingface.co/datasets/arterm-sedov/agent-course-final-assignment>

Arterm's github <https://github.com/arterm-sedov/>

> **For agent setup, installation, and troubleshooting, see [SETUP_INSTRUCTIONS.md](./SETUP_INSTRUCTIONS.md).**

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

Dataset: https://huggingface.co/datasets/arterm-sedov/agent-course-final-assignment

### 🚀 Features

- **Structured dataset uploads** to HuggingFace datasets
- **Schema validation** against `dataset_config.json`
- **Three data splits**: `init` (initialization), `runs` (legacy aggregated results), and `runs_new` (granular per-question results)
- **Automatic data serialization** for complex objects
- **Robust error handling** with fallback mechanisms

### 📚 Documentation

- **`dataset_config.json`**: Schema definition for dataset structure
- **`dataset/README.md`**: Detailed dataset documentation and usage examples
- **`file_helper.py`**: Core upload functions with validation
- **`misc_files/validate_file_upload.py`**: Validation script for upload functionality
- **`misc_files/test_dataset_upload.py`**: Test suite for dataset uploads

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

### 📊 Evaluation Data
- **`runs` split (Legacy)**: Aggregated evaluation results with multiple questions per record
- **`runs_new` split (Current)**: Granular per-question results with detailed trace information, similarity scores, LLM usage tracking, and comprehensive trace data

### 🔍 Trace Data in Dataset

The `runs_new` split includes comprehensive trace data for each question:

**Essential Metadata:**

- `file_size`: Length of attached file data (if any)
- `start_time`: ISO timestamp when processing started
- `end_time`: ISO timestamp when processing ended
- `total_execution_time`: Total execution time in seconds
- `tokens_total`: Total tokens used across all LLM calls

**Complete Trace Data:**

- `llm_traces_json`: Complete LLM traces with input/output/timing/token usage
- `logs_json`: Question-level logs and execution context
- `per_llm_stdout_json`: Captured stdout for each LLM attempt

All data is automatically validated against the schema and uploaded to the HuggingFace dataset for analysis and tracking. See `dataset/README.md` for detailed schema documentation and usage examples.

---

## 🔍 Trace Data Model

The agent returns comprehensive trace data for every question, enabling detailed analysis and debugging. The trace is included in the agent's response under the `trace` key.

### 📊 Trace Structure

```python
{
    # === ROOT LEVEL FIELDS ===
    "question": str,                    # Original question text
    "file_name": str,                   # Name of attached file (if any)
    "file_size": int,            # Length of base64 file data (if any)
    "start_time": str,                  # ISO format timestamp when processing started
    "end_time": str,                    # ISO format timestamp when processing ended
    "total_execution_time": float,      # Total execution time in seconds
    "tokens_total": int,                # Total tokens used across all LLM calls
    
    # === LLM TRACES ===
    "llm_traces": {
        "llm_type": [                   # e.g., "gemini", "groq", "huggingface"
            {
                "call_id": str,         # e.g., "gemini_call_1"
                "llm_name": str,        # e.g., "gemini-2.5-pro" or "Google Gemini"
                "timestamp": str,       # ISO format timestamp
                
                # === LLM CALL INPUT ===
                "input": {
                    "messages": List,   # Input messages (trimmed for base64)
                    "use_tools": bool,  # Whether tools were used
                    "llm_type": str     # LLM type
                },
                
                # === LLM CALL OUTPUT ===
                "output": {
                    "content": str,     # Response content
                    "tool_calls": List, # Tool calls from response
                    "response_metadata": dict,  # Response metadata
                    "raw_response": dict # Full response object (trimmed for base64)
                },
                
                # === TOOL EXECUTIONS ===
                "tool_executions": [
                    {
                        "tool_name": str,      # Name of the tool
                        "args": dict,          # Tool arguments (trimmed for base64)
                        "result": str,         # Tool result (trimmed for base64)
                        "execution_time": float, # Time taken for tool execution
                        "timestamp": str,      # ISO format timestamp
                        "logs": List           # Optional: logs during tool execution
                    }
                ],
                
                # === TOOL LOOP DATA ===
                "tool_loop_data": [
                    {
                        "step": int,           # Current step number
                        "tool_calls_detected": int,  # Number of tool calls detected
                        "consecutive_no_progress": int,  # Steps without progress
                        "timestamp": str,      # ISO format timestamp
                        "logs": List           # Optional: logs during this step
                    }
                ],
                
                # === EXECUTION METRICS ===
                "execution_time": float,       # Time taken for this LLM call
                "total_tokens": int,           # Estimated token count (fallback)
                
                # === TOKEN USAGE TRACKING ===
                "token_usage": {               # Detailed token usage data
                    "prompt_tokens": int,      # Total prompt tokens across all calls
                    "completion_tokens": int,  # Total completion tokens across all calls
                    "total_tokens": int,       # Total tokens across all calls
                    "call_count": int,         # Number of calls made
                    "calls": [                 # Individual call details
                        {
                            "call_id": str,   # Unique call identifier
                            "timestamp": str,  # ISO format timestamp
                            "prompt_tokens": int,     # This call's prompt tokens
                            "completion_tokens": int, # This call's completion tokens
                            "total_tokens": int,      # This call's total tokens
                            "finish_reason": str,     # How the call finished (optional)
                            "system_fingerprint": str, # System fingerprint (optional)
                            "input_token_details": dict,  # Detailed input breakdown (optional)
                            "output_token_details": dict  # Detailed output breakdown (optional)
                        }
                    ]
                },
                
                # === ERROR INFORMATION ===
                "error": {                     # Only present if error occurred
                    "type": str,              # Exception type name
                    "message": str,           # Error message
                    "timestamp": str          # ISO format timestamp
                },
                
                # === LLM-SPECIFIC LOGS ===
                "logs": List,                 # Logs specific to this LLM call
                
                # === FINAL ANSWER ENFORCEMENT ===
                "final_answer_enforcement": [  # Optional: logs from _force_final_answer for this LLM call
                    {
                        "timestamp": str,     # ISO format timestamp
                        "message": str,       # Log message
                        "function": str       # Function that generated the log (always "_force_final_answer")
                    }
                ]
            }
        ]
    },
    
    # === PER-LLM STDOUT CAPTURE ===
    "per_llm_stdout": [
        {
            "llm_type": str,            # LLM type
            "llm_name": str,            # LLM name (model ID or provider name)
            "call_id": str,             # Call ID
            "timestamp": str,           # ISO format timestamp
            "stdout": str               # Captured stdout content
        }
    ],
    
    # === QUESTION-LEVEL LOGS ===
    "logs": [
        {
            "timestamp": str,           # ISO format timestamp
            "message": str,             # Log message
            "function": str             # Function that generated the log
        }
    ],
    

    
    # === FINAL RESULTS ===
    "final_result": {
        "answer": str,                 # Final answer
        "similarity_score": float,     # Similarity score (0.0-1.0)
        "llm_used": str,              # LLM that provided the answer
        "reference": str,              # Reference answer used
        "question": str,               # Original question
        "file_name": str,              # File name (if any)
        "error": str                   # Error message (if any)
    }
}
```

### 🔑 Key Features

- **Hierarchical Structure**: Root-level metadata, LLM traces, tool executions, and contextual logs
- **Comprehensive Coverage**: Complete input/output data, tool usage, error handling, and timing
- **Data Preservation**: Full data preserved in traces, with base64 truncation only for logs
- **Multi-Level Logging**: Question-level, LLM-level, tool-level, and loop-level logs
- **Stdout Capture**: Per-LLM stdout capture for debugging and analysis
- **Token Usage Tracking**: Detailed token consumption per LLM call with provider-specific data
- **Cost Analysis**: Total token usage across all LLM calls for cost optimization

### 📈 Usage

The trace data is automatically included in every agent response and can be used for:
- **Debugging**: Complete visibility into execution flow
- **Performance Analysis**: Detailed timing and token usage metrics
- **Error Analysis**: Comprehensive error information with context
- **Tool Usage Analysis**: Complete tool execution history
- **LLM Comparison**: Detailed comparison of different LLM behaviors
- **Cost Optimization**: Token usage analysis for cost management

---

## 🔧 Recent Enhancements

### Trace System Improvements (Latest)

The agent's tracing system has been significantly enhanced to provide complete visibility into execution:

- **Complete LLM Trace Capture**: Every LLM call is captured with input, output, timing, and error information
- **Tool Execution Tracking**: All tool executions are logged with arguments, results, and timing
- **Stdout Capture**: Print statements are captured per LLM attempt for debugging
- **Error Context**: Comprehensive error information with full context
- **Data Truncation**: Smart truncation preserves full data in traces while keeping logs readable
- **Helper Functions**: Encapsulated LLM naming logic for consistency across the codebase

### Key Improvements Made

1. **Recursive JSON Truncation**: Separate methods for base64 and max-length truncation
2. **Decorator-Based Print Capture**: Captures all print statements into trace data
3. **Multilevel Contextual Logging**: Logs tied to specific execution contexts
4. **Per-LLM Stdout Arrays**: Stdout captured separately for each LLM attempt
5. **Consistent LLM Naming**: Helper function for consistent model identification
6. **Complete Trace Model**: Hierarchical structure with comprehensive coverage

The trace system now provides complete visibility into the agent's execution, making debugging, analysis, and evaluation much more effective.

---

HF Spaces configuration reference at https://huggingface.co/docs/hub/spaces-config-reference