---
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

**Author:** Arte(r)m Sedov

**Github:** <https://github.com/arterm-sedov/>

**This repo:** <https://github.com/arterm-sedov/agent-course-final-assignment>

## 🚀 Arterm Sedov's Ultimate Multi-LLM GAIA Agent

Behold the GAIA Unit 4 Agent by Arte(r)m Sedov — a robust and extensible system designed for real-world reliability and benchmark performance.

This project represents what I learned at HuggingFace Agents Course, eg. to build sophisticated AI agents.

This is Arterm's graduation work for The [Agents Course](https://huggingface.co/learn/agents-course/en/).

## The results dataset

Running this agent takes a lot of time due to it complex reasoning and deep research nature.

See previous run details in the dataset:

<https://huggingface.co/datasets/arterm-sedov/agent-course-final-assignment>

## The agent to play with

> [!NOTE]
> The interesting things happen in the **Logs** section in the **HuggingFace space** while the evaluation runs.
> The agent works under the hood so you can only see its behavior in the logs.
> To skip the wait, see some of the previous at the **Log files** tab.
> The log files are more linear but less structured than the dataset above.

HuggingFace space:

<https://huggingface.co/spaces/arterm-sedov/agent-course-final-assignment>

> **For agent setup, installation, and troubleshooting, see [SETUP_INSTRUCTIONS.md](./SETUP_INSTRUCTIONS.md).**

## 🕵🏻‍♂️ What is this project?

This is an **experimental multi-LLM agent** that demonstrates advanced AI agent capabilities. 

I have developed the project to explore and showcase:

- **Input**: HuggingFace supplies curated GAIA questions and optional file attachments
- **Task**: Create an agent that gets a score of at least 30% on the GAIA questions
- **Challenge**: Process complex queries requiring multi-step reasoning, tool usage, and external data access
- **Solution**: Use multiple LLM providers with intelligent fallback and tool orchestration
- **Results**: The agent can get up to 80% score depending on the available LLMs. Typically it gets 50-65% score because I often 
run out of inference limits

## 🎯 Project Goals

- **Multi-LLM Orchestration**: Intelligent sequencing through multiple LLM providers (OpenRouter, Google Gemini, Groq, HuggingFace)
- **Comprehensive Tool Suite**: Math, code execution, AI research, AI video & audio analysis, web search, file analysis, image processing, chess analysis, and more
- **Robust Fallback System**: Automatic retry and switching with different LLMs when one fails
- **Transparency**: Detailed structured execution traces and logs for every question processed (datasets and human-readable)
- **Reliability**: Rate limiting, error handling, and graceful degradation

## ❓ Why This Project?

This experimental system is based on current AI agent technology and demonstrates:

- **Advanced Tool Usage**: Seamless integration of 20+ specialized tools including AI-powered tools and third-party AI engines
- **Multi-Provider Resilience**: Automatic testing and switching between different LLM providers
- **Comprehensive Tracing**: Complete visibility into the agent's decision-making process
- **Real-World Performance**: Designed for actual benchmark evaluation scenarios, balancing speed, accuracy, logging verbosity and cost across multiple models
- **Contextual Vector Search:** Uses Supabase vector search as a baseline to decide if an LLM call succeeded and calculates a
success score for each model's answer. Reference answers are used for internal evaluation, not submission.
- **Structured Initialization Summary:** After startup, a clear table shows which models/providers are available, with/without
tools, and any errors—so you always know your agent's capabilities.


## 📊 What You'll Find Here

- **Live Demo**: [Interactive Gradio interface](https://huggingface.co/datasets/arterm-sedov/agent-course-final-assignment) for testing the agent against the GAIA Unit 4 questions
- **Complete Source Code**: [Full implementation](https://github.com/arterm-sedov/agent-course-final-assignment) with detailed comments
- **Dataset Tracking**: Comprehensive evaluation results and execution traces: timing, token usage, success rates, and more
- **Complete Traces**: See exactly how the agent thinks and uses tools
- **Documentation**: Detailed technical specifications and usage guides

## 🏗️ Technical Architecture

### LLM Configuration

The agent uses a sophisticated multi-LLM approach with the following providers in sequence:

1. **OpenRouter** (Primary)
   - Models: `deepseek/deepseek-chat-v3-0324:free`, `mistralai/mistral-small-3.2-24b-instruct:free`, `openrouter/cypher-alpha:free`
   - Token Limits: 100K-1M tokens
   - Tool Support: ✅ Full tool-calling capabilities

2. **Google Gemini** (Fallback)
   - Model: `gemini-2.5-pro`
   - Token Limit: 2M tokens (virtually unlimited)
   - Tool Support: ✅ Full tool-calling capabilities

3. **Groq** (Second Fallback)
   - Model: `qwen-qwq-32b`
   - Token Limit: 3K tokens
   - Tool Support: ✅ Full tool-calling capabilities

4. **HuggingFace** (Final Fallback)
   - Models: `Qwen/Qwen2.5-Coder-32B-Instruct`, `microsoft/DialoGPT-medium`, `gpt2`
   - Token Limits: 1K tokens
   - Tool Support: ❌ No tool-calling (text-only responses)

### Tool Suite

The agent includes 20+ specialized tools:

- **Math & Computation**: `multiply`, `add`, `subtract`, `divide`, `modulus`, `power`, `square_root`
- **Web & Research**: `wiki_search`, `web_search`, `arxiv_search`, `exa_ai_helper`
- **File Operations**: `save_and_read_file`, `download_file_from_url`, `get_task_file`
- **Image Processing**: `extract_text_from_image`, `analyze_image`, `transform_image`, `draw_on_image`, `generate_simple_image`, `combine_images`
- **Data Analysis**: `analyze_csv_file`, `analyze_excel_file`
- **Media Understanding**: `understand_video`, `understand_audio`
- **Chess**: `convert_chess_move`, `get_best_chess_move`, `get_chess_board_fen`, `solve_chess_position`
- **Code Execution**: `execute_code_multilang`

### Performance Expectations

- **Success Rate**: 50-65% on complex benchmark questions
- **Response Time**: 30-300 seconds per question (depending on complexity and LLM)
- **Tool Usage**: 2-8 tool calls per question on average
- **Fallback Rate**: 20-40% of questions require LLM switching for fallback

## Dataset Structure

The output trace facilitates:

- **Debugging**: Complete visibility into execution flow
- **Performance Analysis**: Detailed timing and token usage metrics
- **Error Analysis**: Comprehensive error information with context
- **Tool Usage Analysis**: Complete tool execution history
- **LLM Comparison**: Detailed comparison of different LLM behaviors
- **Cost Optimization**: Token usage analysis for cost management


Each question trace is uploaded to a HuggingFace dataset.

The dataset contains comprehensive execution traces with the following structure:

### Root Level Fields

```python
{
    "question": str,                    # Original question text
    "file_name": str,                   # Name of attached file (if any)
    "file_size": int,                   # Length of base64 file data (if any)
    "start_time": str,                  # ISO format timestamp when processing started
    "end_time": str,                    # ISO format timestamp when processing ended
    "total_execution_time": float,      # Total execution time in seconds
    "tokens_total": int,                # Total tokens used across all LLM calls
    "debug_output": str,                # Comprehensive debug output as text
}
```

### LLM Traces

```python
"llm_traces": {
    "llm_type": [                      # e.g., "openrouter", "gemini", "groq", "huggingface"
        {
            "call_id": str,             # e.g., "openrouter_call_1"
            "llm_name": str,            # e.g., "deepseek-chat-v3-0324" or "Google Gemini"
            "timestamp": str,           # ISO format timestamp
            
            # === LLM CALL INPUT ===
            "input": {
                "messages": List,       # Input messages (trimmed for base64)
                "use_tools": bool,      # Whether tools were used
                "llm_type": str         # LLM type
            },
            
            # === LLM CALL OUTPUT ===
            "output": {
                "content": str,         # Response content
                "tool_calls": List,     # Tool calls from response
                "response_metadata": dict,  # Response metadata
                "raw_response": dict    # Full response object (trimmed for base64)
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
}
```

### Per-LLM Stdout Capture

```python
"per_llm_stdout": [
    {
        "llm_type": str,            # LLM type
        "llm_name": str,            # LLM name (model ID or provider name)
        "call_id": str,             # Call ID
        "timestamp": str,           # ISO format timestamp
        "stdout": str               # Captured stdout content
    }
]
```

### Question-Level Logs

```python
"logs": [
    {
        "timestamp": str,           # ISO format timestamp
        "message": str,             # Log message
        "function": str             # Function that generated the log
    }
]
```

### Final Results

```python
"final_result": {
    "submitted_answer": str,        # Final answer (consistent with code)
    "similarity_score": float,      # Similarity score (0.0-1.0)
    "llm_used": str,               # LLM that provided the answer
    "reference": str,               # Reference answer used
    "question": str,                # Original question
    "file_name": str,               # File name (if any)
    "error": str                    # Error message (if any)
}
```

## Key Features

### Intelligent Fallback System

The agent automatically tries multiple LLM providers in sequence:

- **OpenRouter** (Primary): Fast, reliable, good tool support, has tight daily limits on free tiers
- **Google Gemini** (Fallback): High token limits, excellent reasoning
- **Groq** (Second Fallback): Fast inference, good for simple tasks, has tight token limits per request
- **HuggingFace** (Final Fallback): Local models, no API costs, does not support tools typically

### Advanced Tool Management

- **Automatic Tool Selection**: LLM chooses appropriate tools based on question
- **Tool Deduplication**: Prevents duplicate tool calls using vector similarity
- **Usage Limits**: Prevents excessive tool usage (e.g., max 3 web searches per question)
- **Error Handling**: Graceful degradation when tools fail

### Sophisticated implementations

- **Recursive Truncation**: Separate methods for base64 and max-length truncation
- **Recursive JSON Serialization**: Ensures the complex objects ar passable as HuggingFace JSON dataset
- **Decorator-Based Print Capture**: Captures all print statements into trace data
- **Multilevel Contextual Logging**: Logs tied to specific execution contexts
- **Per-LLM Stdout Traces**: Stdout captured separately for each LLM attempt in a human-readable form
- **Consistent LLM Schema**: Data structures for consistent model identification, configuring and calling
- **Complete Trace Model**: Hierarchical structure with comprehensive coverage
- **Structured dataset uploads** to HuggingFace datasets
- **Schema validation** against `dataset_config.json`
- **Three data splits**: `init` (initialization), `runs` (legacy aggregated results), and `runs_new` (granular per-question results)
- **Robust error handling** with fallback mechanisms

### Comprehensive Tracing

Every question generates a complete execution trace including:

- **LLM Interactions**: All input/output for each LLM attempt
- **Tool Executions**: Detailed logs of every tool call
- **Performance Metrics**: Token usage, execution times, success rates
- **Error Information**: Complete error context and fallback decisions
- **Stdout Capture**: All debug output from each LLM attempt

### Rate Limiting & Reliability

- **Smart Rate Limiting**: Different intervals for different providers
- **Token Management**: Automatic truncation and summarization
- **Error Recovery**: Automatic retry with different LLMs
- **Graceful Degradation**: Continues processing even if some components fail

## Usage

### Live Demo

Visit the Gradio interface to test the agent interactively:

<https://huggingface.co/spaces/arterm-sedov/agent-course-final-assignment>

### Programmatic Usage

```python
from agent import GaiaAgent

# Initialize the agent
agent = GaiaAgent()

# Process a question
result = agent("What is the capital of France?")

# Access the results
print(f"Answer: {result['submitted_answer']}")
print(f"Similarity: {result['similarity_score']}")
print(f"LLM Used: {result['llm_used']}")
```

### Dataset Access

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("arterm-sedov/agent-course-final-assignment")

# Access initialization data
init_data = dataset["init"]["train"]

# Access evaluation results
runs_data = dataset["runs_new"]["train"]
```

## File Structure

The main agent runtime files are:

```
gaia-agent/
├── agent.py              # Main agent implementation
├── app.py                # Gradio web interface
├── tools.py              # Tool definitions and implementations
├── utils.py              # Core upload functions with validation
├── system_prompt.json    # System prompt configuration
└── logs/               # Execution logs and results
```

There are other files in the root directory, but they are not used at the runtime, rather for setting up the Supabase vector store.

## Performance Statistics

The agent has been evaluated on complex benchmark questions with the following results:

- **Overall Success Rate**: 50-65%, up to 80% with all four LLMs available
- **Tool Usage**: Average 2-8 tools per question
- **LLM Fallback Rate**: 20-40% of questions require multiple LLMs
- **Response Time**: 30-120 seconds per question
- **Token Usage**: 1K-100K tokens per question (depending on complexity)

## Contributing

This is an experimental research project. Contributions are welcome in the form of:

- **Bug Reports**: Issues with the agent's reasoning or tool usage
- **Feature Requests**: New tools or capabilities
- **Performance Improvements**: Optimizations for speed or accuracy
- **Documentation**: Improvements to this README or code comments

## License

This project is part of the Hugging Face Agents Course final assignment. See the course materials for licensing information.

---

**Built with ❤️ by Arte(r)m Sedov using Cursor IDE**
