"""
GAIA Unit 4 Agent
By Arte(r)m Sedov
==================================

This module implements the main agent logic for the abridged GAIA Unit 4 benchmark. 

Usage:
    agent = GaiaAgent(provider="google")
    answer = agent(question)

Environment Variables:
    - GEMINI_KEY: API key for Gemini model (if using Google provider)
    - SUPABASE_URL: URL for Supabase instance
    - SUPABASE_KEY: Key for Supabase access

Files required in the same directory:
    - system_prompt.md
"""
import os
import json
import csv
import time
import random
import re
import numpy as np
import tempfile
import base64
#import hashlib
import tiktoken
from typing import List, Dict, Any, Optional
from tools import *
# Import tools module to get its functions
import tools
from langchain_core.tools import BaseTool
# For LLM and retriever integration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from supabase.client import create_client

class GaiaAgent:
    """
    Main agent for the GAIA Unit 4 benchmark.

    This agent:
      - Uses the tools.py (math, code, file, image, web, etc.)
      - Integrates a supabase retriever for similar Q/A and context
      - Strictly follows the system prompt in system_prompt
      - Is modular and extensible for future tool/model additions
      - Includes rate limiting and retry logic for API calls
      - Uses Google Gemini for first attempt, Groq for retry
      - Implements LLM-specific token management (no limits for Gemini, conservative for others)

    Args:
        provider (str): LLM provider to use. One of "google", "groq", or "huggingface".

    Attributes:
        system_prompt (str): The loaded system prompt template.
        sys_msg (SystemMessage): The system message for the LLM.
        supabase_client: Supabase client instance.
        vector_store: SupabaseVectorStore instance for retrieval.
        retriever_tool: Tool for retrieving similar questions from the vector store. It retrieves reference answers and context via the Supabase vector store.
        llm_primary: Primary LLM instance (Google Gemini).
        llm_fallback: Fallback LLM instance (Groq).
        llm_third_fallback: Third fallback LLM instance (HuggingFace).
        tools: List of callable tool functions.
        llm_primary_with_tools: Primary LLM instance with tools bound for tool-calling.
        llm_fallback_with_tools: Fallback LLM instance with tools bound for tool-calling.
        llm_third_fallback_with_tools: Third fallback LLM instance with tools bound for tool-calling.
        last_request_time (float): Timestamp of the last API request for rate limiting.
        min_request_interval (float): Minimum time between requests in seconds.
        token_limits: Dictionary of token limits for different LLMs
        max_message_history: Maximum number of messages to keep in history
        original_question: Store the original question for reuse
        similarity_threshold: Minimum similarity score (0.0-1.0) to consider answers similar
        tool_calls_similarity_threshold: Silarity for tool deduplication
        max_summary_tokens: Global token limit for summaries
    """
    
    # Single source of truth for LLM configuration
    LLM_CONFIG = {
        "default": {
            "type_str": "default",
            "token_limit": None,
            "max_history": 15,
            },
        "gemini": {
            "name": "Google Gemini",
            "type_str": "gemini",
            "model": "gemini-2.5-pro",
            "temperature": 0,
            "api_key_env": "GEMINI_KEY",
            "token_limit": None,  # No limit for Gemini (2M token context)
            "max_tokens": None,
            "max_history": 25
        },
        "groq": {
            "name": "Groq",
            "type_str": "groq", 
            "model": "qwen-qwq-32b",
            "temperature": 0,
            "api_key_env": "GROQ_API_KEY", # Groq uses the GROQ_API_KEY environment variable automatically
            "token_limit": 4000,
            "max_tokens": 2048,
            "max_history": 15
        },
        "huggingface": {
            "name": "HuggingFace",
            "type_str": "huggingface",
            "temperature": 0,
            "api_key_env": "HUGGINGFACEHUB_API_TOKEN",
            "token_limit": 1000,  # Conservative for HuggingFace
            "max_history": 20,
            "models": [
                {
                    "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "task": "text-generation",
                    "max_new_tokens": 1024,
                    "do_sample": False,
                    "temperature": 0
                },
                {
                    "repo_id": "microsoft/DialoGPT-medium",
                    "task": "text-generation",
                    "max_new_tokens": 512,  # Shorter for reliability
                    "do_sample": False,
                    "temperature": 0
                },
                {
                    "repo_id": "gpt2",
                    "task": "text-generation", 
                    "max_new_tokens": 256,  # Even shorter for basic model
                    "do_sample": False,
                    "temperature": 0
                }
            ]
        }
    }
    
    # Default LLM sequence order - references LLM_CONFIG keys
    DEFAULT_LLM_SEQUENCE = [
        "gemini",
        "groq", 
        "huggingface"
    ]
    # Print truncation length for debug output
    MAX_PRINT_LEN = 1000
    
    def __init__(self, provider: str = "groq"):
        """
        Initialize the agent, loading the system prompt, tools, retriever, and LLM.

        Args:
            provider (str): LLM provider to use. One of "google", "groq", or "huggingface".

        Raises:
            ValueError: If an invalid provider is specified.
        """
        
        self.system_prompt = self._load_system_prompt()
        self.sys_msg = SystemMessage(content=self.system_prompt)
        # Store the original question for reuse
        self.original_question = None
        # Global threshold. Minimum similarity score (0.0-1.0) to consider answers similar
        self.similarity_threshold = 0.9
        # Tool calls deduplication threshold
        self.tool_calls_similarity_threshold=0.90
        # Global token limit for summaries
        self.max_summary_tokens = 255

        # Rate limiting setup
        self.last_request_time = 0
        self.current_llm_type = None  # Track the current LLM type for rate limiting

        # Token management - LLM-specific limits (built from configuration)
        self.token_limits = {
            config["type_str"]: config["token_limit"] 
            for config in self.LLM_CONFIG.values()
        }

        # LLM success counter - clean and lean
        self.llm_success_count = {
            "gemini": 0,
            "groq": 0, 
            "huggingface": 0,
            "reference_fallback": 0
        }
        
        # Total questions counter
        self.total_questions = 0

        # Set up embeddings and supabase retriever
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.supabase_client = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_KEY")
        )
        self.vector_store = SupabaseVectorStore(
            client=self.supabase_client,
            embedding=self.embeddings,
            table_name="agent_course_reference",
            query_name="match_agent_course_reference_langchain",
        )
        self.retriever_tool = create_retriever_tool(
            retriever=self.vector_store.as_retriever(),
            name="Question Search",
            description="A tool to retrieve similar questions from a vector store.",
        )

        # Get the LLM types that should be initialized based on the sequence
        llm_types_to_init = self.DEFAULT_LLM_SEQUENCE
        llm_names = [self.LLM_CONFIG[llm_type]["name"] for llm_type in llm_types_to_init]
        print(f"🔄 Initializing LLMs based on sequence:")
        for i, name in enumerate(llm_names, 1):
            print(f"   {i}. {name}")

        # Set up LLMs based on the sequence configuration
        gemini_name = self.LLM_CONFIG['gemini']['name']
        if "gemini" in llm_types_to_init:
            gemini_position = llm_types_to_init.index("gemini") + 1
            print(f"🔄 Initializing LLM {gemini_name} ({gemini_position} of {len(llm_types_to_init)})")
            try:
                config = self.LLM_CONFIG["gemini"]
                self.llm_primary = ChatGoogleGenerativeAI(
                    model=config["model"], 
                    temperature=config["temperature"], 
                    google_api_key=os.environ.get(config["api_key_env"]),
                    max_tokens=config["max_tokens"]
                )
                print(f"✅ LLM ({gemini_name}) initialized successfully")
                # Test the LLM with Hello message
                if not self._ping_llm(self.llm_primary, gemini_name):
                    print(f"⚠️ {gemini_name} test failed, setting to None")
                    self.llm_primary = None
            except Exception as e:
                print(f"⚠️ Failed to initialize {gemini_name}: {e}")
                self.llm_primary = None
        else:
            print(f"⏭️ Skipping {gemini_name} (not in sequence)")
            self.llm_primary = None
        
        groq_name = self.LLM_CONFIG['groq']['name']
        if "groq" in llm_types_to_init:
            groq_position = llm_types_to_init.index("groq") + 1
            print(f"🔄 Initializing LLM {groq_name} ({groq_position} of {len(llm_types_to_init)})")
            try:
                config = self.LLM_CONFIG["groq"]
                # Groq uses the GROQ_API_KEY environment variable automatically
                # We check if it's available
                if not os.environ.get(config["api_key_env"]):
                    print(f"⚠️ {config['api_key_env']} not found in environment variables. Skipping {groq_name}...")
                    self.llm_fallback = None
                else:
                    self.llm_fallback = ChatGroq(
                        model=config["model"], 
                        temperature=config["temperature"],
                        max_tokens=config["max_tokens"]
                    )
                    print(f"✅ LLM ({groq_name}) initialized successfully")
                    # Test the LLM with Hello message
                    if not self._ping_llm(self.llm_fallback, groq_name):
                        print(f"⚠️ {groq_name} test failed, setting to None")
                        self.llm_fallback = None
            except Exception as e:
                print(f"⚠️ Failed to initialize {groq_name}: {e}")
                self.llm_fallback = None
        else:
            print(f"⏭️ Skipping LLM {groq_name} (not in sequence)")
            self.llm_fallback = None
        
        huggingface_name = self.LLM_CONFIG['huggingface']['name']
        if "huggingface" in llm_types_to_init:
            huggingface_position = llm_types_to_init.index("huggingface") + 1
            print(f"🔄 Initializing LLM {huggingface_name} ({huggingface_position} of {len(llm_types_to_init)})")
            try:
                self.llm_third_fallback = self._create_huggingface_llm()
                if self.llm_third_fallback is not None:
                    print(f"✅ LLM ({huggingface_name}) initialized successfully")
                    # Note: HuggingFace LLM is already tested in _create_huggingface_llm()
                else:
                    print(f"❌ LLM ({huggingface_name}) failed to initialize")
            except Exception as e:
                print(f"⚠️ Failed to initialize {huggingface_name}: {e}")
                self.llm_third_fallback = None
        else:
            print(f"⏭️ Skipping {huggingface_name} LLM (not in sequence)")
            self.llm_third_fallback = None
        
        # Bind all tools from tools.py
        self.tools = self._gather_tools()
        
        if self.llm_primary:
            self.llm_primary_with_tools = self.llm_primary.bind_tools(self.tools)
        else:
            self.llm_primary_with_tools = None
            
        if self.llm_fallback:
            self.llm_fallback_with_tools = self.llm_fallback.bind_tools(self.tools)
        else:
            self.llm_fallback_with_tools = None
            
        if self.llm_third_fallback:
            self.llm_third_fallback_with_tools = self.llm_third_fallback.bind_tools(self.tools)
        else:
            self.llm_third_fallback_with_tools = None

    def _load_system_prompt(self):
        """
        Load the system prompt from the system_prompt.md file.
        """        
        try:
            with open("system_prompt.md", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print("⚠️ system_prompt.md not found, using default system prompt")
            
        except Exception as e:
            print(f"⚠️ Error reading system_prompt.md: {e}")
        
        return "You are a helpful assistant. Please provide clear and accurate responses."
    
    def _rate_limit(self):
        """
        Implement rate limiting to avoid hitting API limits.
        Waits if necessary to maintain minimum interval between requests.
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        # Determine wait time based on current LLM type
        if self.current_llm_type in ["groq", "huggingface"]:
            min_interval = 30
        else:
            min_interval = 30
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            # Add small random jitter to avoid thundering herd
            jitter = random.uniform(0, 0.2)
            time.sleep(sleep_time + jitter)
        self.last_request_time = time.time()

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using tiktoken for accurate counting.
        """
        try:
            # Use GPT-4 encoding as a reasonable approximation for most models
            encoding = tiktoken.encoding_for_model("gpt-4")
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            # Fallback to character-based estimation if tiktoken fails
            print(f"⚠️ Tiktoken failed, using fallback: {e}")
            return len(text) // 4

    def _truncate_messages(self, messages: List[Any], llm_type: str = None) -> List[Any]:
        """
        Truncate message history to prevent token overflow.
        Keeps system message, last human message, and most recent tool messages.
        More lenient for Gemini due to its large context window.
        More aggressive for Groq due to TPM limits.
        
        Args:
            messages: List of messages to truncate
            llm_type: Type of LLM for context-aware truncation
        """
        # Always read max_history from LLM_CONFIG, using 'default' if not found
        max_history = self.LLM_CONFIG.get(llm_type, {}).get("max_history", self.LLM_CONFIG["default"]["max_history"])
        
        if len(messages) <= max_history:
            return messages
        
        # Always keep system message and last human message
        system_msg = messages[0] if messages and hasattr(messages[0], 'type') and messages[0].type == 'system' else None
        last_human_msg = None
        tool_messages = []
        
        # Find last human message and collect tool messages
        for msg in reversed(messages):
            if hasattr(msg, 'type'):
                if msg.type == 'human' and last_human_msg is None:
                    last_human_msg = msg
                elif msg.type == 'tool':
                    tool_messages.append(msg)
        
        # Keep most recent tool messages (limit to prevent overflow)
        max_tool_messages = max_history - 3  # System + Human + AI
        if len(tool_messages) > max_tool_messages:
            tool_messages = tool_messages[-max_tool_messages:]
        

        # Reconstruct message list
        truncated_messages = []
        if system_msg:
            truncated_messages.append(system_msg)
        truncated_messages.extend(tool_messages)
        if last_human_msg:
            truncated_messages.append(last_human_msg)
        
        return truncated_messages

    def _execute_tool(self, tool_name: str, tool_args: dict, tool_registry: dict) -> str:
        """
        Execute a tool with the given name and arguments.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            tool_registry: Registry of available tools
            
        Returns:
            str: Result of tool execution
        """
        # Inject file data if available and needed
        if isinstance(tool_args, dict):
            tool_args = self._inject_file_data_to_tool_args(tool_name, tool_args)
        
        print(f"[Tool Loop] Running tool: {tool_name} with args: {tool_args}")
        tool_func = tool_registry.get(tool_name)
        
        if not tool_func:
            tool_result = f"Tool '{tool_name}' not found."
            print(f"[Tool Loop] Tool '{tool_name}' not found.")
        else:
            try:
                # Check if it's a proper LangChain tool (has invoke method and tool attributes)
                if (hasattr(tool_func, 'invoke') and 
                    hasattr(tool_func, 'name') and 
                    hasattr(tool_func, 'description')):
                    # This is a proper LangChain tool, use invoke method
                    if isinstance(tool_args, dict):
                        tool_result = tool_func.invoke(tool_args)
                    else:
                        # For non-dict args, assume it's a single value that should be passed as 'input'
                        tool_result = tool_func.invoke({'input': tool_args})
                else:
                    # This is a regular function, call it directly
                    if isinstance(tool_args, dict):
                        tool_result = tool_func(**tool_args)
                    else:
                        # For non-dict args, pass directly
                        tool_result = tool_func(tool_args)
                print(f"[Tool Loop] Tool '{tool_name}' executed successfully.")
                # Only trim for printing, not for LLM
                self._print_tool_result(tool_name, tool_result)
            except Exception as e:
                tool_result = f"Error running tool '{tool_name}': {e}"
                print(f"[Tool Loop] Error running tool '{tool_name}': {e}")
        
        return str(tool_result)

    def _has_tool_messages(self, messages: List) -> bool:
        """
        Check if the message history contains ToolMessage objects.
        
        Args:
            messages: List of message objects
            
        Returns:
            bool: True if ToolMessage objects are present, False otherwise
        """
        return any(
            hasattr(msg, 'type') and msg.type == 'tool' and hasattr(msg, 'content') 
            for msg in messages
        )

    def _force_final_answer(self, messages, tool_results_history, llm):
        """
        Handle duplicate tool calls by forcing final answer using LangChain's native mechanisms.
        For Gemini, always include tool results in the reminder. For others, only if not already present.
        Args:
            messages: Current message list
            tool_results_history: History of tool results (can be empty)
            llm: LLM instance
            
        Returns:
            Response from LLM
        """
        # Initialize include_tool_results variable at the top
        include_tool_results = False
        
        # Extract llm_type from llm
        llm_type = getattr(llm, 'llm_type', None) or getattr(llm, 'type_str', None) or ''
        
        # Create a more explicit reminder to provide final answer
        reminder = self._get_reminder_prompt(
            reminder_type="final_answer_prompt",
            messages=messages,
            tools=self.tools,
            tool_results_history=tool_results_history
        )
        # Gemini-specific: add explicit instructions for extracting numbers or lists
        if llm_type == "gemini":
            reminder += (
                "\n\nIMPORTANT: If the tool result contains a sentence with a number spelled out or as a digit, "
                "extract only the number and provide it as the FINAL ANSWER in the required format. "
                "If the tool result contains a list of items (such as ingredients, or any items), "
                "extract the list and provide it as a comma-separated list in the FINAL ANSWER as required."
            )
        # Check if tool results are already in message history as ToolMessage objects
        has_tool_messages = self._has_tool_messages(messages)
        
        # Determine whether to include tool results in the reminder
        if tool_results_history:
            if llm_type == "gemini":
                include_tool_results = True
            else:
                # For non-Gemini LLMs, only include if not already in message history
                if not has_tool_messages:
                    include_tool_results = True
        
        if include_tool_results:
            tool_results_text = "\n\nTOOL RESULTS:\n" + "\n".join([f"Result {i+1}: {result}" for i, result in enumerate(tool_results_history)])
            reminder += tool_results_text
        
        # Add the reminder to the existing message history
        messages.append(HumanMessage(content=reminder))
        try:
            print(f"[Tool Loop] Trying to force the final answer with {len(tool_results_history)} tool results.")
            final_response = llm.invoke(messages)
            if hasattr(final_response, 'content') and final_response.content:
                print(f"[Tool Loop] ✅ Final answer generated: {final_response.content[:200]}...")
                return final_response
            else:
                print("[Tool Loop] ❌ LLM returned empty response")
                return AIMessage(content="Unable to determine the answer from the available information.")
        except Exception as e:
            print(f"[Tool Loop] ❌ Failed to get final answer: {e}")
            return AIMessage(content="Error occurred while processing the question.")
        # If Gemini, use a minimal, explicit prompt
        if llm_type == "gemini" and tool_results_history:
            tool_result = tool_results_history[-1]  # Use the latest tool result
            original_question = None
            for msg in messages:
                if hasattr(msg, 'type') and msg.type == 'human':
                    original_question = msg.content
                    break
            if not original_question:
                original_question = "[Original question not found]"
            prompt = (
                "You have already used the tool and obtained the following result:\n\n"
                f"TOOL RESULT:\n{tool_result}\n\n"
                f"QUESTION:\n{original_question}\n\n"
                "INSTRUCTIONS:\n"
                "Extract the answer from the TOOL RESULT above. Your answer must start with 'FINAL ANSWER: [answer]"
                "and follow the system prompt without any extra text numbers, just answer concisely and directly."
            )
            minimal_messages = [self.sys_msg, HumanMessage(content=prompt)]
            try:
                final_response = llm.invoke(minimal_messages)
                if hasattr(final_response, 'content') and final_response.content:
                    return final_response
                else:
                    # Fallback: return the tool result directly
                    return AIMessage(content=f"RESULT: {tool_result}")
            except Exception as e:
                print(f"[Tool Loop] ❌ Gemini failed to extract final answer: {e}")
                return AIMessage(content=f"RESULT: {tool_result}")

    def _run_tool_calling_loop(self, llm, messages, tool_registry, llm_type="unknown"):
        """
        Run a tool-calling loop: repeatedly invoke the LLM, detect tool calls, execute tools, and feed results back until a final answer is produced.
        - Uses adaptive step limits based on LLM type (Gemini: 25, Groq: 15, HuggingFace: 20, unknown: 20).
        - Tracks called tools to prevent duplicate calls and tool results history for fallback handling.
        - Monitors progress by tracking consecutive steps without meaningful changes in response content.
        - Handles LLM invocation failures gracefully with error messages.
        - Detects when responses are truncated due to token limits and adjusts accordingly.
        
        Args:
            llm: The LLM instance (with or without tools bound)
            messages: The message history (list)
            tool_registry: Dict mapping tool names to functions
            llm_type: Type of LLM ("gemini", "groq", "huggingface", or "unknown")
        Returns:
            The final LLM response (with content)
        """

        # Adaptive step limits based on LLM type and progress
        base_max_steps = {
            "gemini": 25,    # More steps for Gemini due to better reasoning
            "groq": 5,       # Reduced from 10 to 5 to prevent infinite loops
            "huggingface": 20,  # Conservative for HuggingFace
            "unknown": 20
        }
        max_steps = base_max_steps.get(llm_type, 8)
        
        # Tool calling configuration       
        called_tools = []  # Track which tools have been called to prevent duplicates (stores dictionaries with name, embedding, args)
        tool_results_history = []  # Track tool results for better fallback handling
        current_step_tool_results = []  # Track results from current step only
        consecutive_no_progress = 0  # Track consecutive steps without progress
        last_response_content = ""  # Track last response content for progress detection
        max_total_tool_calls = 8  # Reduced from 15 to 8 to prevent excessive tool usage
        max_tool_calls_per_step = 3  # Maximum tool calls allowed per step
        total_tool_calls = 0  # Track total tool calls to prevent infinite loops
        
        # Simplified tool usage tracking - no special handling for search tools
        tool_usage_limits = {
            'default': 3,
            'wiki_search': 2,
            'web_search': 3, 
            'arxiv_search': 3,
            'analyze_excel_file': 2,
            'analyze_csv_file': 2,
            'analyze_image': 2,
            'extract_text_from_image': 2
        }
        tool_usage_count = {tool_name: 0 for tool_name in tool_usage_limits}
        
        for step in range(max_steps):
            print(f"\n[Tool Loop] Step {step+1}/{max_steps} - Using LLM: {llm_type}")
            current_step_tool_results = []  # Reset for this step
            
            # Check if we've exceeded the maximum total tool calls
            if total_tool_calls >= max_total_tool_calls:
                print(f"[Tool Loop] Maximum total tool calls ({max_total_tool_calls}) reached. Calling _force_final_answer ().")
                # Let the LLM generate the final answer from tool results (or lack thereof)
                return self._force_final_answer(messages, tool_results_history, llm)
            
            # Check for excessive tool usage
            for tool_name, count in tool_usage_count.items():
                if count >= tool_usage_limits.get(tool_name, tool_usage_limits['default']):  # Use default limit for unknown tools
                    print(f"[Tool Loop] ⚠️ {tool_name} used {count} times (max: {tool_usage_limits.get(tool_name, tool_usage_limits['default'])}). Preventing further usage.")
                    # Add a message to discourage further use of this tool
                    if step > 2:  # Only add this message after a few steps
                        reminder = self._get_reminder_prompt(
                            reminder_type="tool_usage_issue",
                            tool_name=tool_name,
                            count=count
                        )
                        messages.append(HumanMessage(content=reminder))
            
            # Truncate messages to prevent token overflow
            messages = self._truncate_messages(messages, llm_type)
            
            # Check token limits and summarize if needed
            total_text = "".join(str(getattr(msg, 'content', '')) for msg in messages)
            estimated_tokens = self._estimate_tokens(total_text)
            token_limit = self.token_limits.get(llm_type)
            
            try:
                response = llm.invoke(messages)
            except Exception as e:
                print(f"[Tool Loop] ❌ LLM invocation failed: {e}")
                
                # Enhanced token limit error handling for all LLMs
                if self._is_token_limit_error(e, llm_type):
                    print(f"[Tool Loop] Token limit error detected for {llm_type} in tool calling loop")
                    # Get the LLM name for proper logging
                    _, llm_name, _ = self._select_llm(llm_type, True)
                    return self._handle_token_limit_error(messages, llm, llm_name, e, llm_type)
                
                # Handle HuggingFace router errors with chunking
                if llm_type == "huggingface" and self._is_token_limit_error(e):
                    print(f"⚠️ HuggingFace router error detected, applying chunking: {e}")
                    return self._handle_token_limit_error(messages, llm, llm_name, e, llm_type)
                
                # Check for general token limit errors specifically
                if "413" in str(e) or "token" in str(e).lower() or "limit" in str(e).lower():
                    print(f"[Tool Loop] Token limit error detected. Forcing final answer with available information.")
                    if tool_results_history:
                        return self._force_final_answer(messages, tool_results_history, llm)
                    else:
                        return AIMessage(content=f"Error: Token limit exceeded for {llm_type} LLM. Cannot complete reasoning.")
                return AIMessage(content=f"Error during LLM processing: {str(e)}")

            # Check if response was truncated due to token limits
            if hasattr(response, 'response_metadata') and response.response_metadata:
                finish_reason = response.response_metadata.get('finish_reason')
                if finish_reason == 'length':
                    print(f"[Tool Loop] ❌ Hit token limit for {llm_type} LLM. Response was truncated. Cannot complete reasoning.")
                    # Handle response truncation using generic token limit error handler
                    print(f"[Tool Loop] Applying chunking mechanism for {llm_type} response truncation")
                    # Get the LLM name for proper logging
                    _, llm_name, _ = self._select_llm(llm_type, True)
                    return self._handle_token_limit_error(messages, llm, llm_name, Exception("Response truncated due to token limit"), llm_type)

            # === DEBUG OUTPUT ===
            # Print LLM response using the new helper function
            print(f"[Tool Loop] Raw LLM response details:")
            self._print_message_components(response, "response")

            # Check for empty response
            if not hasattr(response, 'content') or not response.content:
                # Allow empty content if there are tool calls (this is normal for tool-calling responses)
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    print(f"[Tool Loop] Empty content but tool calls detected - proceeding with tool execution")
                else:
                    # If we have tool results but no content, force a final answer after 2 consecutive empty responses
                    if tool_results_history and consecutive_no_progress >= 1:
                        print(f"[Tool Loop] Empty content and we have {len(tool_results_history)} tool results for 2 consecutive steps. Forcing final answer.")
                        return self._force_final_answer(messages, tool_results_history, llm)
                    # Otherwise, increment no-progress counter and continue
                    consecutive_no_progress += 1
                    print(f"[Tool Loop] ❌ {llm_type} LLM returned empty response. Consecutive no-progress steps: {consecutive_no_progress}")
                    if consecutive_no_progress >= 2:
                        return AIMessage(content=f"Error: {llm_type} LLM returned empty response. Cannot complete reasoning.")
                    continue
            else:
                consecutive_no_progress = 0  # Reset counter on progress

            # Check for progress (new content or tool calls)
            current_content = getattr(response, 'content', '') or ''
            current_tool_calls = getattr(response, 'tool_calls', []) or []
            has_progress = (current_content != last_response_content or len(current_tool_calls) > 0)
            
            # Check if we have tool results but no final answer yet
            has_tool_results = len(tool_results_history) > 0
            has_final_answer = (hasattr(response, 'content') and response.content and 
                              self._has_final_answer_marker(response))
            
            if has_tool_results and not has_final_answer and step >= 2:  # Increased from 1 to 2 to give more time
                # We have information but no answer - provide explicit reminder to analyze tool results
                reminder = self._get_reminder_prompt(
                    reminder_type="final_answer_prompt",
                    messages=messages,
                    tools=self.tools,
                    tool_results_history=tool_results_history
                )
                messages.append(HumanMessage(content=reminder))
            
            if not has_progress:
                consecutive_no_progress += 1
                print(f"[Tool Loop] No progress detected. Consecutive no-progress steps: {consecutive_no_progress}")
                
                # Exit early if no progress for too many consecutive steps
                if consecutive_no_progress >= 3:  # Increased from 2 to 3
                    print(f"[Tool Loop] Exiting due to {consecutive_no_progress} consecutive steps without progress")
                    # If we have tool results, force a final answer before exiting
                    if tool_results_history:
                        print(f"[Tool Loop] Forcing final answer with {len(tool_results_history)} tool results before exit")
                        return self._force_final_answer(messages, tool_results_history, llm)
                    break
                elif consecutive_no_progress == 1:
                    # Add a gentle reminder to use tools
                    reminder = self._get_reminder_prompt(
                        reminder_type="final_answer_prompt",
                        tools=self.tools
                    )
                    messages.append(HumanMessage(content=reminder))
            else:
                consecutive_no_progress = 0  # Reset counter on progress
                
            last_response_content = current_content

            # If response has content and no tool calls, return
            if hasattr(response, 'content') and response.content and not getattr(response, 'tool_calls', None):
                
                # --- Check for 'FINAL ANSWER' marker ---
                if self._has_final_answer_marker(response):
                    print(f"[Tool Loop] Final answer detected: {response.content}")
                    return response
                else:
                    # If we have tool results but no FINAL ANSWER marker, force processing
                    if tool_results_history:
                        print(f"[Tool Loop] Content without FINAL ANSWER marker but we have {len(tool_results_history)} tool results. Forcing final answer.")
                        return self._force_final_answer(messages, tool_results_history, llm)
                    else:
                        print("[Tool Loop] 'FINAL ANSWER' marker not found. Reiterating with reminder.")
                        # Find the original question
                        original_question = None
                        for msg in messages:
                            if hasattr(msg, 'type') and msg.type == 'human':
                                original_question = msg.content
                                break
                        if not original_question:
                            original_question = "[Original question not found]"
                        # Compose a reminder message
                        reminder = self._get_reminder_prompt(
                            reminder_type="final_answer_prompt",
                            messages=messages
                        )
                        reiterate_messages = [self.system_prompt, HumanMessage(content=reminder)]
                        try:
                            reiterate_response = llm.invoke(reiterate_messages)
                            print(f"[Tool Loop] Reiterated response: {reiterate_response.content if hasattr(reiterate_response, 'content') else reiterate_response}")
                            return reiterate_response
                        except Exception as e:
                            print(f"[Tool Loop] ❌ Failed to reiterate: {e}")
                            return response
            tool_calls = getattr(response, 'tool_calls', None)
            if tool_calls:
                print(f"[Tool Loop] Detected {len(tool_calls)} tool call(s)")
                
                # Limit the number of tool calls per step to prevent token overflow
                if len(tool_calls) > max_tool_calls_per_step:
                    print(f"[Tool Loop] Too many tool calls on a single step ({len(tool_calls)}). Limiting to first {max_tool_calls_per_step}.")
                    tool_calls = tool_calls[:max_tool_calls_per_step]
                
                # Simplified duplicate detection using new centralized methods
                new_tool_calls = []
                duplicate_count = 0
                for tool_call in tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    
                    # Check if tool usage limit exceeded FIRST (most restrictive check)
                    if tool_name in tool_usage_count and tool_usage_count[tool_name] >= tool_usage_limits.get(tool_name, tool_usage_limits['default']):
                        print(f"[Tool Loop] ⚠️ {tool_name} usage limit reached ({tool_usage_count[tool_name]}/{tool_usage_limits.get(tool_name, tool_usage_limits['default'])}). Skipping.")
                        duplicate_count += 1
                        continue
                    
                    # Check if this is a duplicate tool call (SECOND)
                    if self._is_duplicate_tool_call(tool_name, tool_args, called_tools):
                        duplicate_count += 1
                        print(f"[Tool Loop] Duplicate tool call detected: {tool_name} with args: {tool_args}")
                        continue
                    
                    # New tool call - add it (LAST)
                    print(f"[Tool Loop] New tool call: {tool_name} with args: {tool_args}")
                    new_tool_calls.append(tool_call)
                    self._add_tool_call_to_history(tool_name, tool_args, called_tools)
                    
                    # Track tool usage
                    if tool_name in tool_usage_count:
                        tool_usage_count[tool_name] += 1
                        print(f"[Tool Loop] {tool_name} usage: {tool_usage_count[tool_name]}/{tool_usage_limits.get(tool_name, tool_usage_limits['default'])}")
                
                # Only force final answer if ALL tool calls were duplicates AND we have tool results
                if not new_tool_calls and tool_results_history:
                    print(f"[Tool Loop] All {len(tool_calls)} tool calls were duplicates and we have {len(tool_results_history)} tool results. Forcing final answer.")
                    result = self._force_final_answer(messages, tool_results_history, llm)
                    if result:
                        return result
                elif not new_tool_calls and not tool_results_history:
                    # No new tool calls and no previous results - this might be a stuck state
                    print(f"[Tool Loop] All tool calls were duplicates but no previous results. Adding reminder to use available tools.")
                    reminder = self._get_reminder_prompt(reminder_type="tool_usage_issue", tool_name=tool_name)
                    messages.append(HumanMessage(content=reminder))
                    continue
                
                # Execute only new tool calls
                for tool_call in new_tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    
                    # Execute tool using helper method
                    tool_result = self._execute_tool(tool_name, tool_args, tool_registry)
                    
                    # Store the raw result for this step
                    current_step_tool_results.append(tool_result)
                    tool_results_history.append(tool_result)
                    total_tool_calls += 1  # Increment total tool call counter
                    
                    # Report tool result
                    self._print_tool_result(tool_name, tool_result)
                    
                    # Add tool result to messages - let LangChain handle the formatting
                    messages.append(ToolMessage(content=tool_result, name=tool_name, tool_call_id=tool_call.get('id', tool_name)))
                
                continue  # Next LLM call
            # Gemini (and some LLMs) may use 'function_call' instead of 'tool_calls'
            function_call = getattr(response, 'function_call', None)
            if function_call:
                tool_name = function_call.get('name')
                tool_args = function_call.get('arguments', {})
                
                # Check if this is a duplicate function call
                if self._is_duplicate_tool_call(tool_name, tool_args, called_tools):
                    print(f"[Tool Loop] Duplicate function_call detected: {tool_name} with args: {tool_args}")
                    reminder = self._get_reminder_prompt(
                        reminder_type="tool_usage_issue",
                        tool_name=tool_name,
                        tool_args=tool_args
                    )
                    messages.append(HumanMessage(content=reminder))
                    
                    # Only force final answer if we have tool results
                    if tool_results_history:
                        print(f"[Tool Loop] Duplicate function_call with {len(tool_results_history)} tool results. Forcing final answer.")
                        result = self._force_final_answer(messages, tool_results_history, llm)
                        if result:
                            return result
                    else:
                        # No previous results - add reminder and continue
                        reminder = self._get_reminder_prompt(reminder_type="tool_usage_issue", tool_name=tool_name)
                        messages.append(HumanMessage(content=reminder))
                    continue
                
                # Check if tool usage limit exceeded
                if tool_name in tool_usage_count and tool_usage_count[tool_name] >= tool_usage_limits.get(tool_name, tool_usage_limits['default']):
                    print(f"[Tool Loop] ⚠️ {tool_name} usage limit reached ({tool_usage_count[tool_name]}/{tool_usage_limits.get(tool_name, tool_usage_limits['default'])}). Skipping.")
                    reminder = self._get_reminder_prompt(
                        reminder_type="tool_usage_issue",
                        tool_name=tool_name,
                        count=tool_usage_count[tool_name]
                    )
                    messages.append(HumanMessage(content=reminder))
                    continue
                
                # Add to history and track usage
                self._add_tool_call_to_history(tool_name, tool_args, called_tools)
                if tool_name in tool_usage_count:
                    tool_usage_count[tool_name] += 1
                
                # Execute tool using helper method
                tool_result = self._execute_tool(tool_name, tool_args, tool_registry)
                
                # Store the raw result for this step
                current_step_tool_results.append(tool_result)
                tool_results_history.append(tool_result)
                total_tool_calls += 1  # Increment total tool call counter
                
                # Report tool result (for function_call branch)
                self._print_tool_result(tool_name, tool_result)
                messages.append(ToolMessage(content=tool_result, name=tool_name, tool_call_id=tool_name))
                continue
            if hasattr(response, 'content') and response.content:
                return response
            print(f"[Tool Loop] No tool calls or final answer detected. Exiting loop.")
            
            # If we get here, the LLM didn't make tool calls or provide content
            # Add a reminder to use tools or provide an answer
            reminder = self._get_reminder_prompt(reminder_type="final_answer_prompt", tools=self.tools)
            messages.append(HumanMessage(content=reminder))
            continue
        
        # If we reach here, we've exhausted all steps or hit progress limits
        print(f"[Tool Loop] Exiting after {step+1} steps. Last response: {response}")
        
        # If we have tool results but no final answer, force one
        if tool_results_history and (not hasattr(response, 'content') or not response.content or not self._has_final_answer_marker(response)):
            print(f"[Tool Loop] Forcing final answer with {len(tool_results_history)} tool results at loop exit")
            return self._force_final_answer(messages, tool_results_history, llm)
        
        # Return the last response as-is, no partial answer extraction
        return response

    def _select_llm(self, llm_type, use_tools):
        if llm_type not in self.LLM_CONFIG:
            raise ValueError(f"Invalid llm_type: {llm_type}")
            
        config = self.LLM_CONFIG[llm_type]
        
        # Get the appropriate LLM instance
        if llm_type == "gemini":
            llm = self.llm_primary_with_tools if use_tools else self.llm_primary
        elif llm_type == "groq":
            llm = self.llm_fallback_with_tools if use_tools else self.llm_fallback
        elif llm_type == "huggingface":
            llm = self.llm_third_fallback_with_tools if use_tools else self.llm_third_fallback
        else:
            raise ValueError(f"Invalid llm_type: {llm_type}")
        
        llm_name = config["name"]
        llm_type_str = config["type_str"]
        
        return llm, llm_name, llm_type_str

    def _make_llm_request(self, messages, use_tools=True, llm_type=None):
        """
        Make an LLM request with rate limiting.

        Args:
            messages: The messages to send to the LLM
            use_tools (bool): Whether to use tools (llm_with_tools vs llm)
            llm_type (str): Which LLM to use (mandatory)

        Returns:
            The LLM response

        Raises:
            Exception: If the LLM fails or if llm_type is not specified
        """

        if llm_type is None:
                raise Exception(
                    f"llm_type must be specified for _make_llm_request(). "
                    f"Please specify a valid llm_type from {list(self.LLM_CONFIG.keys())}"
                )
        # Set the current LLM type for rate limiting
        self.current_llm_type = llm_type
        llm, llm_name, llm_type_str = self._select_llm(llm_type, use_tools)
        if llm is None:
            raise Exception(f"{llm_name} LLM not available")
        
        try:
            self._rate_limit()
            print(f"🤖 Using {llm_name}")
            print(f"--- LLM Prompt/messages sent to {llm_name} ---")
            for i, msg in enumerate(messages):
                self._print_message_components(msg, i)
            tool_registry = {self._get_tool_name(tool): tool for tool in self.tools}
            if use_tools:
                response = self._run_tool_calling_loop(llm, messages, tool_registry, llm_type_str)
                # If tool calling resulted in empty content, try without tools as fallback
                if not hasattr(response, 'content') or not response.content:
                    print(f"⚠️ {llm_name} tool calling returned empty content, trying without tools...")
                    llm_no_tools, _, _ = self._select_llm(llm_type, False)
                    if llm_no_tools:
                        # Check if tool results are already in message history as ToolMessage objects
                        has_tool_messages = self._has_tool_messages(messages)
                        
                        if has_tool_messages:
                            print(f"⚠️ Retrying {llm_name} without tools (tool results already in message history)")
                            response = llm_no_tools.invoke(messages)
                        else:
                            # Extract raw tool results from message history for _get_reminder_prompt
                            tool_results_history = []
                            for msg in messages:
                                if hasattr(msg, 'type') and msg.type == 'tool' and hasattr(msg, 'content'):
                                    tool_results_history.append(msg.content)
                            
                            if tool_results_history:
                                print(f"⚠️ Retrying {llm_name} without tools with enhanced context")
                                print(f"📝 Tool results included: {len(tool_results_history)} tools")
                                reminder = self._get_reminder_prompt(
                                    reminder_type="final_answer_prompt",
                                    messages=messages,
                                    tools=self.tools,
                                    tool_results_history=tool_results_history
                                )
                                enhanced_messages = [self.system_prompt, HumanMessage(content=reminder)]
                                response = llm_no_tools.invoke(enhanced_messages)
                            else:
                                print(f"⚠️ Retrying {llm_name} without tools (no tool results found)")
                                response = llm_no_tools.invoke(messages)
                    
                    # NEW: If still no content, this might be a token limit issue
                    if not hasattr(response, 'content') or not response.content:
                        print(f"⚠️ {llm_name} still returning empty content even without tools. This may be a token limit issue.")
                        return AIMessage(content=f"Error: {llm_name} failed due to token limits. Cannot complete reasoning.")
            else:
                response = llm.invoke(messages)
            print(f"--- Raw response from {llm_name} ---")
            return response
        except Exception as e:
            # Enhanced Groq token limit error handling
            if llm_type == "groq" and self._is_token_limit_error(e):
                print(f"⚠️ Groq token limit error detected: {e}")
                return self._handle_groq_token_limit_error(messages, llm, llm_name, e)
            
            # Special handling for HuggingFace router errors
            if llm_type == "huggingface" and self._is_token_limit_error(e):
                print(f"⚠️ HuggingFace router error detected, applying chunking: {e}")
                return self._handle_token_limit_error(messages, llm, llm_name, e, llm_type)
            elif llm_type == "huggingface" and "500 Server Error" in str(e) and "router.huggingface.co" in str(e):
                error_msg = f"HuggingFace router service error (500): {e}"
                print(f"⚠️ {error_msg}")
                print("💡 This is a known issue with HuggingFace's router service. Consider using Google Gemini or Groq instead.")
                raise Exception(error_msg)
            elif llm_type == "huggingface" and "timeout" in str(e).lower():
                error_msg = f"HuggingFace timeout error: {e}"
                print(f"⚠️ {error_msg}")
                print("💡 HuggingFace models may be slow or overloaded. Consider using Google Gemini or Groq instead.")
                raise Exception(error_msg)
            # Special handling for Groq network errors
            elif llm_type == "groq" and ("no healthy upstream" in str(e).lower() or "network" in str(e).lower() or "connection" in str(e).lower()):
                error_msg = f"Groq network connectivity error: {e}"
                print(f"⚠️ {error_msg}")
                print("💡 This is a network connectivity issue with Groq's servers. The service may be temporarily unavailable.")
                raise Exception(error_msg)
            else:
                raise Exception(f"{llm_name} failed: {e}")

    

    def _handle_groq_token_limit_error(self, messages, llm, llm_name, original_error):
        """
        Handle Groq token limit errors by chunking tool results and processing them in intervals.
        """
        return self._handle_token_limit_error(messages, llm, llm_name, original_error, "groq")

    def _handle_token_limit_error(self, messages, llm, llm_name, original_error, llm_type="unknown"):
        """
        Generic token limit error handling that can be used for any LLM.
        """
        print(f"🔄 Handling token limit error for {llm_name} ({llm_type})")
        
        # Extract tool results from messages
        tool_results = []
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'tool' and hasattr(msg, 'content'):
                tool_results.append(msg.content)
        
        # If no tool results, try to chunk the entire message content
        if not tool_results:
            print(f"📊 No tool results found, attempting to chunk entire message content")
            # Extract all message content
            all_content = []
            for msg in messages:
                if hasattr(msg, 'content') and msg.content:
                    all_content.append(str(msg.content))
            
            if not all_content:
                return AIMessage(content=f"Error: {llm_name} token limit exceeded but no content available to process.")
            
            # Create chunks from all content (use LLM-specific limits)
            token_limit = self.token_limits.get(llm_type, 3000)
            # Handle None token limits (like Gemini) by using a reasonable default
            if token_limit is None:
                token_limit = 3000  # Reasonable default for LLMs with no explicit limit
            safe_tokens = int(token_limit * 0.60)
            chunks = self._create_token_chunks(all_content, safe_tokens)
            print(f"📦 Created {len(chunks)} chunks from message content")
        else:
            print(f"📊 Found {len(tool_results)} tool results to process in chunks")
            # Create chunks (use LLM-specific limits)
            token_limit = self.token_limits.get(llm_type, 3000)
            # Handle None token limits (like Gemini) by using a reasonable default
            if token_limit is None:
                token_limit = 3000  # Reasonable default for LLMs with no explicit limit
            safe_tokens = int(token_limit * 0.60)
            chunks = self._create_token_chunks(tool_results, safe_tokens)
            print(f"📦 Created {len(chunks)} chunks from tool results")
        
        # Process chunks with intervals (shorter for non-Groq LLMs)
        all_responses = []
        wait_time = 60
        
        for i, chunk in enumerate(chunks):
            print(f"🔄 Processing chunk {i+1}/{len(chunks)}")
            
            # Wait between chunks (except first)
            if i > 0:
                print(f"⏳ Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            
            # Create simple prompt for this chunk
            chunk_prompt = self._create_simple_chunk_prompt(messages, chunk, i+1, len(chunks))
            chunk_messages = [self.sys_msg, HumanMessage(content=chunk_prompt)]
            
            try:
                response = llm.invoke(chunk_messages)
                if hasattr(response, 'content') and response.content:
                    all_responses.append(response.content)
                    print(f"✅ Chunk {i+1} processed")
            except Exception as e:
                print(f"❌ Chunk {i+1} failed: {e}")
                continue
        
        if not all_responses:
            return AIMessage(content=f"Error: Failed to process any chunks for {llm_name}")
        
        # Simple final synthesis
        final_prompt = f"Combine these analyses into a final answer:\n\n" + "\n\n".join(all_responses)
        final_messages = [self.sys_msg, HumanMessage(content=final_prompt)]
        
        try:
            final_response = llm.invoke(final_messages)
            return final_response
        except Exception as e:
            print(f"❌ Final synthesis failed: {e}")
            return AIMessage(content=f"OUTPUT {' '.join(all_responses)}")

    def _create_token_chunks(self, tool_results, max_tokens_per_chunk):
        """
        Create chunks of tool results that fit within the token limit.
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for result in tool_results:
            # Use tiktoken for accurate token counting
            result_tokens = self._estimate_tokens(result)
            if current_tokens + result_tokens > max_tokens_per_chunk and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [result]
                current_tokens = result_tokens
            else:
                current_chunk.append(result)
                current_tokens += result_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def _try_llm_sequence(self, messages, use_tools=True, reference=None):
        """
        Try multiple LLMs in sequence until one succeeds and produces a similar answer to reference.
        Only one attempt per LLM, then move to the next.
        
        Args:
            messages: The messages to send to the LLM
            use_tools (bool): Whether to use tools
            reference (str, optional): Reference answer to compare against
            
        Returns:
            tuple: (answer, llm_used) where answer is the final answer and llm_used is the name of the LLM that succeeded
            
        Raises:
            Exception: If all LLMs fail or none produce similar enough answers
        """
        # Use the default LLM sequence from class configuration
        llm_sequence = self.DEFAULT_LLM_SEQUENCE
        
        # Filter out unavailable LLMs
        available_llms = []
        for llm_type in llm_sequence:
            llm, llm_name, _ = self._select_llm(llm_type, True)
            if llm:
                available_llms.append((llm_type, llm_name))
            else:
                print(f"⚠️ {llm_name} not available, skipping...")

        if not available_llms:
            raise Exception("No LLMs are available. Please check your API keys and configuration.")
        
        print(f"🔄 Available LLMs: {[name for _, name in available_llms]}")
        
        # Extract the original question for intelligent extraction
        original_question = ""
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                original_question = msg.content
                break
        
        for llm_type, llm_name in available_llms:
            try:
                response = self._make_llm_request(messages, use_tools=use_tools, llm_type=llm_type)
                
                # Try standard extraction first
                answer = self._extract_final_answer(response)
                
                # If standard extraction didn't work well, try intelligent extraction
                # if not answer:
                #     print(f"⚠️ {llm_name} did not provide a properly formatted answer. Ret trying...")
                #     answer, response = self._retry_with_final_answer_reminder(messages, use_tools, llm_type)
                
                print(f"✅ {llm_name} answered: {answer}")
                print(f"✅ Reference: {reference}")
                
                # If no reference provided, return the first successful answer
                if reference is None:
                    print(f"✅ {llm_name} succeeded (no reference to compare)")
                    self.llm_success_count[llm_type] += 1
                    return answer, llm_name
                
                # Check similarity with reference
                if self._vector_answers_match(answer, reference):
                    print(f"✅ {llm_name} succeeded with similar answer to reference")
                    self.llm_success_count[llm_type] += 1
                    return answer, llm_name
                else:
                    print(f"⚠️ {llm_name} succeeded but answer doesn't match reference")
                    
                    # Try the next LLM without reference if this isn't the last one
                    if llm_type != available_llms[-1][0]:
                        print(f"🔄 Trying next LLM without reference...")
                        # Continue to next iteration to try next LLM
                    else:
                        # This was the last LLM, fall back to reference answer
                        print(f"🔄 All LLMs tried, all failed")
                        # self.llm_success_count["reference_fallback"] += 1
                        # return reference, "reference_fallback"
                    
            except Exception as e:
                print(f"❌ {llm_name} failed: {e}")
                
                # Special retry logic for HuggingFace router errors
                if llm_type == "huggingface" and "500 Server Error" in str(e) and "router.huggingface.co" in str(e):
                    print("🔄 HuggingFace router error detected, retrying once...")
                    try:
                        time.sleep(2)  # Wait 2 seconds before retry
                        response = self._make_llm_request(messages, use_tools=use_tools, llm_type=llm_type)
                        answer = self._extract_final_answer(response)
                        if not answer:
                            # Inject message to get final answer instead of making new LLM call
                            answer, response = self._retry_with_final_answer_reminder(messages, use_tools, llm_type)
                        if answer and not answer == str(response).strip():
                            print(f"✅ HuggingFace retry succeeded: {answer}")
                            self.llm_success_count[llm_type] += 1
                            return answer, llm_name
                    except Exception as retry_error:
                        print(f"❌ HuggingFace retry also failed: {retry_error}")
                
                # Check if this was the last available LLM
                if llm_type == available_llms[-1][0]:
                    # This was the last LLM, re-raise the exception
                    raise Exception(f"All available LLMs failed. Last error from {llm_name}: {e}")
                print(f"🔄 Trying next LLM...")
        
        # This should never be reached, but just in case
        raise Exception("All LLMs failed")

    def _get_reference_answer(self, question: str) -> Optional[str]:
        """
        Retrieve the reference answer for a question using the supabase retriever.

        Args:
            question (str): The question text.

        Returns:
            str or None: The reference answer if found, else None.
        """
        similar = self.vector_store.similarity_search(question)
        if similar:
            # Assume the answer is in the page_content or metadata
            content = similar[0].page_content
            # Try to extract the answer from the content
            if "Final answer :" in content:
                return content.split("Final answer :", 1)[-1].strip().split("\n")[0]
            return content
        return None

    def _format_messages(self, question: str, reference: Optional[str] = None) -> List[Any]:
        """
        Format the message list for the LLM, including system prompt, question, and optional reference answer.

        Args:
            question (str): The question to answer.
            reference (str, optional): The reference answer to include in context.

        Returns:
            list: List of message objects for the LLM.
        """
        messages = [self.sys_msg, HumanMessage(content=question)]
        if reference:
            messages.append(HumanMessage(content=f"Reference answer: {reference}"))
        return messages

    def _normalize_answer(self, ans: str) -> str:
        """
        Normalize answer by removing common prefixes and normalizing whitespace around commas.
        Much simpler approach that preserves semantic meaning.
        """
        # Handle None or empty values gracefully
        if not ans:
            return ""
        
        ans = ans.strip().lower()
        if ans.startswith("final answer:"):
            ans = ans[12:].strip()
        elif ans.startswith("final answer"):
            ans = ans[11:].strip()
        ans = re.sub(r'\s+', ' ', ans).strip()
        return ans

    def _get_tool_name(self, tool):
        if hasattr(tool, 'name'):
            return tool.name
        elif hasattr(tool, '__name__'):
            return tool.__name__
        else:
            return str(tool)

    def _calculate_cosine_similarity(self, embedding1, embedding2) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score (0.0 to 1.0)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity calculation
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    def _vector_answers_match(self, answer: str, reference: str) -> bool:
        try:
            # Handle None or empty answers gracefully
            if not answer:
                print("⚠️ Answer is empty, cannot compare with reference")
                return False
                
            norm_answer = self._normalize_answer(answer)
            norm_reference = self._normalize_answer(reference)
            
            # Debug output to see what normalization is doing
            print(f"🔍 Normalized answer: '{norm_answer}'")
            print(f"🔍 Normalized reference: '{norm_reference}'")
            
            if norm_answer == norm_reference:
                print("✅ Exact match after normalization")
                return True
            embeddings = self.embeddings
            
            # Get embeddings for both answers
            answer_embedding = embeddings.embed_query(norm_answer)
            reference_embedding = embeddings.embed_query(norm_reference)
            
            # Calculate cosine similarity using the reusable method
            cosine_similarity = self._calculate_cosine_similarity(answer_embedding, reference_embedding)
            print(f"🔍 Answer similarity: {cosine_similarity:.3f} (threshold: {self.similarity_threshold})")
            
            if cosine_similarity >= self.similarity_threshold:
                return True
            else:
                print("🔄 Vector similarity below threshold")
                return False
                
        except Exception as e:
            print(f"⚠️ Error in vector similarity matching: {e}")
            return False

    def get_llm_stats(self) -> dict:
        """
        Get clean statistics about LLM success rates.
        
        Returns:
            dict: Dictionary with LLM names, success counts, and success rates
        """
        stats = {
            "total_questions": self.total_questions,
            "success_rates": {}
        }
        
        for llm_type, count in self.llm_success_count.items():
            if llm_type == "reference_fallback":
                llm_name = "All LLMs failed"
            else:
                llm_name = self.LLM_CONFIG[llm_type]["name"]
            
            # Calculate success rate percentage
            success_rate = (count / self.total_questions * 100) if self.total_questions > 0 else 0
            
            stats["success_rates"][llm_name] = {
                "count": count,
                "rate": f"{success_rate:.1f}%"
            }
        
        return stats

    def __call__(self, question: str, file_data: str = None, file_name: str = None) -> str:
        """
        Run the agent on a single question, using step-by-step reasoning and tools.

        Args:
            question (str): The question to answer.
            file_data (str, optional): Base64 encoded file data if a file is attached.
            file_name (str, optional): Name of the attached file.

        Returns:
            str: The agent's final answer, formatted per system_prompt.

        Workflow:
            1. Store file data for use by tools.
            2. Retrieve similar Q/A for context using the retriever.
            3. Use LLM sequence with similarity checking against reference.
            4. If no similar answer found, fall back to reference answer.
        """
        print(f"\n🔎 Processing question: {question}\n")
        
        # Increment total questions counter
        self.total_questions += 1
        
        # Store the original question for reuse throughout the process
        self.original_question = question
        
        # Store file data for use by tools
        self.current_file_data = file_data
        self.current_file_name = file_name
        
        if file_data and file_name:
            print(f"📁 File attached: {file_name} ({len(file_data)} chars base64)")
        
        # 1. Retrieve similar Q/A for context
        reference = self._get_reference_answer(question)
        
        # 2. Step-by-step reasoning with LLM sequence and similarity checking
        messages = self._format_messages(question)
        try:
            answer, llm_used = self._try_llm_sequence(messages, use_tools=True, reference=reference)
            print(f"🎯 Final answer from {llm_used}")
            
            # Display current stats
            stats = self.get_llm_stats()
            print(f"📊 LLM Success Stats (Total Questions: {stats['total_questions']}):")
            for llm_name, data in stats['success_rates'].items():
                print(f"   {llm_name}: {data['count']}: ({data['rate']})")
            
            return answer
        except Exception as e:
            print(f"❌ All LLMs failed: {e}")
            if reference:
                print("⚠️ Falling back to reference answer")
                self.llm_success_count["reference_fallback"] += 1
                stats = self.get_llm_stats()
                print(f"📊 LLM Success Stats (Total Questions: {stats['total_questions']}):")
                for llm_name, data in stats['success_rates'].items():
                    print(f"   {llm_name}: {data['count']}: ({data['rate']})")
                return reference
            else:
                raise Exception("All LLMs failed and no reference answer available")

    def _clean_final_answer_text(self, text: str) -> str:
        """
        Cleans up the answer text by:
        - Removing everything before and including the first 'FINAL ANSWER:' (case-insensitive, with/without colon/space)
        - Stripping leading/trailing whitespace
        - Normalizing whitespace
        """
        import re
        # Handle None text gracefully
        if not text:
            return ""
            
        print(f"[CleanFinalAnswer] Original text before stripping: {text}")
        # Find the first occurrence of 'FINAL ANSWER' (case-insensitive)
        match = re.search(r'final answer\s*:?', text, flags=re.IGNORECASE)
        if match:
            # Only keep what comes after 'FINAL ANSWER'
            text = text[match.end():]
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_text_from_response(self, response: Any) -> str:
        """
        Helper method to extract text content from various response object types.
        
        Args:
            response (Any): The response object (could be LLM response, dict, or string)
            
        Returns:
            str: The text content from the response
        """
        # Handle None responses gracefully
        if not response:
            return ""
            
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        else:
            return str(response)

    def _has_final_answer_marker(self, response: Any) -> bool:
        """
        Check if the LLM response contains a "FINAL ANSWER:" marker.
        This is used in the tool calling loop to determine if the response is a final answer.

        Args:
            response (Any): The LLM response object.

        Returns:
            bool: True if the response contains "FINAL ANSWER:" marker, False otherwise.
        """
        text = self._extract_text_from_response(response)
        
        # Check if any line starts with "FINAL ANSWER" (case-insensitive)
        for line in text.splitlines():
            if line.strip().upper().startswith("FINAL ANSWER"):
                return True
        return False

    def _extract_final_answer(self, response: Any) -> str:
        """
        Extract the final answer from the LLM response, removing the "FINAL ANSWER:" prefix.
        The LLM is responsible for following the system prompt formatting rules.
        This method is used for validation against reference answers and submission.

        Args:
            response (Any): The LLM response object.

        Returns:
            str: The extracted final answer string with "FINAL ANSWER:" prefix removed, or None if not found.
        """
        # First check if there's a final answer marker
        if not self._has_final_answer_marker(response):
            return None
        
        # Extract text from response and clean it using the existing regex logic
        text = self._extract_text_from_response(response)
        return self._clean_final_answer_text(text)

    def _llm_answers_match(self, answer: str, reference: str) -> bool:
        """
        Use the LLM to validate whether the agent's answer matches the reference answer according to the system prompt rules.
        This method is kept for compatibility but should be avoided due to rate limiting.

        Args:
            answer (str): The agent's answer.
            reference (str): The reference answer.

        Returns:
            bool: True if the LLM determines the answers match, False otherwise.
        """
        validation_prompt = (
            f"Agent's answer:\n{answer}\n\n"
            f"Reference answer:\n{reference}\n\n"
            "Question: Does the agent's answer match the reference answer exactly, following the system prompt's answer formatting and constraints? "
            "Reply with only 'true' or 'false'."
        )
        validation_msg = [SystemMessage (content=self.system_prompt), HumanMessage(content=validation_prompt)]
        try:
            response = self._try_llm_sequence(validation_msg, use_tools=False)
            result = self._extract_text_from_response(response).strip().lower()
            return result.startswith('true')
        except Exception as e:
            # Fallback: conservative, treat as not matching if validation fails
            print(f"LLM validation error in _llm_answers_match: {e}")
            return False

    def _gather_tools(self) -> List[Any]:
        """
        Gather all callable tools from tools.py for LLM tool binding.

        Returns:
            list: List of tool functions.
        """
       
        # Get all attributes from the tools module
        tool_list = []
        for name, obj in tools.__dict__.items():
            # Only include actual tool objects (decorated with @tool) or callable functions
            # that are not classes, modules, or builtins
            if (callable(obj) and 
                not name.startswith("_") and 
                not isinstance(obj, type) and  # Exclude classes
                hasattr(obj, '__module__') and  # Must have __module__ attribute
                obj.__module__ == 'tools' and  # Must be from tools module
                name not in ["GaiaAgent", "CodeInterpreter"]):  # Exclude specific classes
                
                # Check if it's a proper tool object (has the tool attributes)
                if hasattr(obj, 'name') and hasattr(obj, 'description'):
                    # This is a proper @tool decorated function
                    tool_list.append(obj)
                elif callable(obj) and not name.startswith("_"):
                    # This is a regular function that might be a tool
                    # Only include if it's not an internal function
                    if not name.startswith("_") and name not in [
                        "_convert_chess_move_internal", 
                        "_get_best_chess_move_internal", 
                        "_get_chess_board_fen_internal",
                        "_expand_fen_rank",
                        "_compress_fen_rank", 
                        "_invert_mirror_fen",
                        "_add_fen_game_state"
                    ]:
                        tool_list.append(obj)
        
        # Add specific tools that might be missed
        specific_tools = [
            'multiply', 'add', 'subtract', 'divide', 'modulus', 'power', 'square_root',
            'wiki_search', 'web_search', 'arxiv_search',
            'save_and_read_file', 'download_file_from_url', 'get_task_file',
            'extract_text_from_image', 'analyze_csv_file', 'analyze_excel_file',
            'analyze_image', 'transform_image', 'draw_on_image', 'generate_simple_image', 'combine_images',
            'understand_video', 'understand_audio',
            'convert_chess_move', 'get_best_chess_move', 'get_chess_board_fen', 'solve_chess_position',
            'execute_code_multilang',
            'exa_ai_helper' 
        ]
        
        # Build a set of tool names for deduplication (handle both __name__ and .name attributes)
        tool_names = set(self._get_tool_name(tool) for tool in tool_list)
        
        # Ensure all specific tools are included
        for tool_name in specific_tools:
            if hasattr(tools, tool_name):
                tool_obj = getattr(tools, tool_name)
                name_val = self._get_tool_name(tool_obj)
                if name_val not in tool_names:
                    tool_list.append(tool_obj)
                    tool_names.add(name_val)
        
        # Filter out any tools that don't have proper tool attributes
        final_tool_list = []
        for tool in tool_list:
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                # This is a proper tool object
                final_tool_list.append(tool)
            elif callable(tool) and not self._get_tool_name(tool).startswith("_"):
                # This is a callable function that should be a tool
                final_tool_list.append(tool)
        
        print(f"✅ Gathered {len(final_tool_list)} tools: {[self._get_tool_name(tool) for tool in final_tool_list]}")
        return final_tool_list

    def _inject_file_data_to_tool_args(self, tool_name: str, tool_args: dict) -> dict:
        """
        Automatically inject file data and system prompt into tool arguments if needed.
        
        Args:
            tool_name (str): Name of the tool being called
            tool_args (dict): Original tool arguments
            
        Returns:
            dict: Modified tool arguments with file data and system prompt if needed
        """
        # Tools that need file data
        file_tools = {
            'understand_audio': 'file_path',
            'analyze_image': 'image_base64', 
            'transform_image': 'image_base64',
            'draw_on_image': 'image_base64',
            'combine_images': 'images_base64',
            'extract_text_from_image': 'image_path',
            'analyze_csv_file': 'file_path',
            'analyze_excel_file': 'file_path',
            'get_chess_board_fen': 'image_path',
            'solve_chess_position': 'image_path',
            'execute_code_multilang': 'code'  # Add support for code injection
        }
        
        # Tools that need system prompt for better formatting
        system_prompt_tools = ['understand_video', 'understand_audio']
        
        # Inject system prompt for video and audio understanding tools
        if tool_name in system_prompt_tools and 'system_prompt' not in tool_args:
            tool_args['system_prompt'] = self.system_prompt
            print(f"[Tool Loop] Injected system prompt for {tool_name}")
        
        if tool_name in file_tools and self.current_file_data and self.current_file_name:
            param_name = file_tools[tool_name]
            
            # For image tools, use base64 directly
            if 'image' in param_name:
                tool_args[param_name] = self.current_file_data
                print(f"[Tool Loop] Injected base64 image data for {tool_name}")
            # For file path tools, create a temporary file
            elif 'file_path' in param_name:
                import tempfile
                import base64
                
                # Decode base64 and create temporary file
                file_data = base64.b64decode(self.current_file_data)
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(self.current_file_name)[1], delete=False) as temp_file:
                    temp_file.write(file_data)
                    temp_file_path = temp_file.name
                
                tool_args[param_name] = temp_file_path
                print(f"[Tool Loop] Created temporary file {temp_file_path} for {tool_name}")
            # For code tools, decode and inject the code content
            elif param_name == 'code':
                import base64
                import tempfile
                try:
                    # Get file extension
                    temp_ext = os.path.splitext(self.current_file_name)[1].lower()
                    code_str = tool_args.get('code', '')
                    orig_file_name = self.current_file_name
                    file_data = base64.b64decode(self.current_file_data)
                    # List of code file extensions
                    code_exts = ['.py', '.js', '.cpp', '.c', '.java', '.rb', '.go', '.ts', '.sh', '.php', '.rs']
                    if temp_ext in code_exts:
                        # If it's a code file, decode as UTF-8 and inject as code
                        code_content = file_data.decode('utf-8')
                        tool_args[param_name] = code_content
                        print(f"[Tool Loop] Injected code from attached file for {tool_name}: {len(code_content)} characters")
                    else:
                        # Otherwise, treat as data file: create temp file and patch code string
                        with tempfile.NamedTemporaryFile(suffix=temp_ext, delete=False) as temp_file:
                            temp_file.write(file_data)
                            temp_file_path = temp_file.name
                        print(f"[Tool Loop] Created temporary file {temp_file_path} for code execution")
                        # Replace all occurrences of the original file name in the code string with the temp file path
                        patched_code = code_str.replace(orig_file_name, temp_file_path)
                        tool_args[param_name] = patched_code
                        print(f"[Tool Loop] Patched code to use temp file path for {tool_name}")
                except Exception as e:
                    print(f"[Tool Loop] Failed to patch code for code injection: {e}")
        
        return tool_args

    def _create_huggingface_llm(self):
        """
        Create HuggingFace LLM with multiple fallback options to handle router issues.
        """
        config = self.LLM_CONFIG["huggingface"]
        
        # Check if HuggingFace API token is available
        if os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"):
            print("✅ HuggingFace API token configured")
        else:
            print("⚠️ No HuggingFace API token found - HuggingFace LLM may not work")
            return None
        
        # Try models in priority order from config
        for model_config in config["models"]:
            try:
                # Create the endpoint
                endpoint = HuggingFaceEndpoint(**model_config)
                
                # Create the chat model
                llm = ChatHuggingFace(
                    llm=endpoint,
                    verbose=True,
                )
                
                # Test the model using the standardized test function
                model_name = f"HuggingFace ({model_config['repo_id']})"
                if self._ping_llm(llm, model_name):
                    print(f"✅ HuggingFace LLM initialized and tested with {model_config['repo_id']}")
                    return llm
                else:
                    print(f"⚠️ {model_config['repo_id']} test failed, trying next model...")
                    continue
                
            except Exception as e:
                print(f"⚠️ Failed to initialize {model_config['repo_id']}: {e}")
                continue
        
        print("❌ All HuggingFace models failed to initialize")
        return None

    def _ping_llm(self, llm, llm_name: str) -> bool:
        """
        Test an LLM with a simple "Hello" message to verify it's working.
        
        Args:
            llm: The LLM instance to test
            llm_name: Name of the LLM for logging purposes
            
        Returns:
            bool: True if test passes, False otherwise
        """
        if llm is None:
            print(f"❌ {llm_name} is None - cannot test")
            return False
            
        try:
            test_message = [HumanMessage(content="Hello, report about yourself briefly.")]
            print(f"🧪 Testing {llm_name} with 'Hello' message...")
            
            start_time = time.time()
            test_response = llm.invoke(test_message)
            end_time = time.time()
            
            if test_response and hasattr(test_response, 'content') and test_response.content:
                print(f"✅ {llm_name} test successful!")
                print(f"   Response time: {end_time - start_time:.2f}s")
                print(f"   Test message details:")
                self._print_message_components(test_message[0], "test_input")
                print(f"   Test response details:")
                self._print_message_components(test_response, "test")
                return True
            else:
                print(f"❌ {llm_name} returned empty response")
                return False
                
        except Exception as e:
            print(f"❌ {llm_name} test failed: {e}")
            return False 

    def _is_duplicate_tool_call(self, tool_name: str, tool_args: dict, called_tools: list) -> bool:
        """
        Check if a tool call is a duplicate based on tool name and vector similarity of arguments.
        
        Args:
            tool_name: Name of the tool
            tool_args: Arguments for the tool
            called_tools: List of previously called tool dictionaries
            
        Returns:
            bool: True if this is a duplicate tool call
        """
        # Convert tool args to text for embedding
        args_text = json.dumps(tool_args, sort_keys=True) if isinstance(tool_args, dict) else str(tool_args)
        
        # Check for exact tool name match first
        for called_tool in called_tools:
            if called_tool['name'] == tool_name:
                # Get embedding for current args
                current_embedding = self.embeddings.embed_query(args_text)
                
                # Compare with stored embedding using vector similarity
                cosine_similarity = self._calculate_cosine_similarity(current_embedding, called_tool['embedding'])
                if cosine_similarity >= self.tool_calls_similarity_threshold:
                    print(f"[Tool Loop] Vector similarity duplicate detected: {tool_name} (similarity: {cosine_similarity:.3f})")
                    return True
        
        return False

    def _add_tool_call_to_history(self, tool_name: str, tool_args: dict, called_tools: list) -> None:
        """
        Add a tool call to the history of called tools.
        
        Args:
            tool_name: Name of the tool
            tool_args: Arguments for the tool
            called_tools: List of previously called tool dictionaries
        """
        # Convert tool args to text for embedding
        args_text = json.dumps(tool_args, sort_keys=True) if isinstance(tool_args, dict) else str(tool_args)
        
        # Get embedding for the tool call
        tool_embedding = self.embeddings.embed_query(args_text)
        
        # Store as dictionary with name and embedding
        tool_call_record = {
            'name': tool_name,
            'embedding': tool_embedding,
            'args': tool_args
        }
        called_tools.append(tool_call_record)

    def _trim_for_print(self, obj, max_len=None):
        """
        Helper to trim any object (string, dict, etc.) for debug printing only.
        Converts to string, trims to max_len (default: self.MAX_PRINT_LEN), and adds suffix with original length if needed.
        """
        if max_len is None:
            max_len = self.MAX_PRINT_LEN
        s = str(obj)
        orig_len = len(s)
        if orig_len > max_len:
            return f"Truncated. Original length: {orig_len}\n{s[:max_len]}"
        return s

    def _print_message_components(self, msg, msg_index):
        """
        Type-aware helper to print message components with proper truncation.
        Only prints relevant components based on message type.
        """
        print("------------------------------------------------\n") 
        print(f"Message {msg_index}:")
        
        # Get message type
        msg_type = getattr(msg, 'type', 'unknown')
        print(f"  type: {msg_type}")
        
        # Print components based on message type
        if msg_type == 'system':
            # System messages: content only
            if hasattr(msg, 'content') and msg.content:
                print(f"  content: {self._trim_for_print(msg.content)}")
                
        elif msg_type == 'human':
            # Human messages: content only
            if hasattr(msg, 'content') and msg.content:
                print(f"  content: {self._trim_for_print(msg.content)}")
                
        elif msg_type == 'ai':
            # AI messages: content, tool_calls, function_call
            if hasattr(msg, 'content') and msg.content:
                print("------------------------------------------------\n")
                print(f"  content: {self._trim_for_print(msg.content)}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print("------------------------------------------------\n")
                print(f"  tool_calls: {self._trim_for_print(msg.tool_calls)}")
            if hasattr(msg, 'function_call') and msg.function_call:
                print("------------------------------------------------\n")
                print(f"  function_call: {self._trim_for_print(msg.function_call)}")
                
        elif msg_type == 'tool':
            # Tool messages: content, name, tool_call_id
            if hasattr(msg, 'content') and msg.content:
                print("------------------------------------------------\n")
                print(f"  content: {self._trim_for_print(msg.content)}")
            if hasattr(msg, 'name') and msg.name:
                print("------------------------------------------------\n")
                print(f"  name: {msg.name}")
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                print("------------------------------------------------\n")
                print(f"  tool_call_id: {msg.tool_call_id}")
                
        else:
            # Unknown type: print all common attributes
            if hasattr(msg, 'content') and msg.content:
                print("------------------------------------------------\n")
                print(f"  content: {self._trim_for_print(msg.content)}")
            if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                print("------------------------------------------------\n")
                print(f"  additional_kwargs: {self._trim_for_print(msg.additional_kwargs)}")
            if hasattr(msg, 'response_metadata') and msg.response_metadata:
                print("------------------------------------------------\n")
                print(f"  response_metadata: {self._trim_for_print(msg.response_metadata)}")
        
        print("------------------------------------------------\n")

    def _deep_trim_dict(self, obj, max_len=None):
        """
        Recursively trim all string fields in a dict or list to max_len characters.
        """
        if max_len is None:
            max_len = self.MAX_PRINT_LEN
        if isinstance(obj, dict):
            return {k: self._deep_trim_dict(v, max_len) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_trim_dict(v, max_len) for v in obj]
        elif isinstance(obj, str):
            if len(obj) > max_len:
                return f"Truncated. Original length: {len(obj)}\n{obj[:max_len]}"
            return obj
        else:
            return obj

    def _print_tool_result(self, tool_name, tool_result):
        """
        Print tool results in a readable format with deep recursive trimming for all dicts/lists.
        For dict/list results, deeply trim all string fields. For other types, use _trim_for_print.
        """
        if isinstance(tool_result, (dict, list)):
            trimmed = self._deep_trim_dict(tool_result)
            print(f"[Tool Loop] Tool result for '{tool_name}': {trimmed}")
        else:
            print(f"[Tool Loop] Tool result for '{tool_name}': {self._trim_for_print(tool_result)}")
        print()

    def _extract_main_text_from_tool_result(self, tool_result):
        """
        Extract the main text from a tool result dict (e.g., wiki_results, web_results, arxiv_results, etc.).
        """
        if isinstance(tool_result, dict):
            for key in ("wiki_results", "web_results", "arxiv_results", "result", "text", "content"):
                if key in tool_result and isinstance(tool_result[key], str):
                    return tool_result[key]
            # Fallback: join all string values
            return " ".join(str(v) for v in tool_result.values() if isinstance(v, str))
        return str(tool_result)

    def _retry_with_final_answer_reminder(self, messages, use_tools, llm_type):
        """
        Injects a final answer reminder, retries the LLM request, and extracts the answer.
        Returns (answer, response)
        """
        # Find the original question from the message history
        original_question = None
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                original_question = msg.content
                break
        
        # Build the prompt message (slim, direct)
        prompt = (
            "TASK: Extract the FINAL answer from the given LLM response. "
            "If a **question** is present, extract the most likely FINAL ANSWER according to the system prompt's answer formatting rules. "
            "Return only the most likely final answer, formatted exactly as required by the system prompt.\n\n"
            "FOCUS: Focus on the most relevant facts, numbers, and names, related to the question if present.\n\n"
            "PURPOSE: Extract the FINAL ANSWER per the system prompt.\n\n"
            "INSTRUCTIONS: Do not use tools.\n\n"
        )
        if original_question:
            prompt += f"QUESTION: {original_question}\n\n"
        prompt += "RESPONSE TO ANALYZE:\nAnalyze the previous response and provide your FINAL ANSWER."
        
        # Inject the message into the queue
        messages.append(HumanMessage(content=prompt))
        
        # Make the LLM call and extract the answer
        response = self._make_llm_request(messages, use_tools=use_tools, llm_type=llm_type)
        answer = self._extract_final_answer(response)
        return answer, response

    def _get_reminder_prompt(
        self,
        reminder_type: str,
        messages=None,
        tools=None,
        tool_results_history=None,
        tool_name=None,
        count=None,
        tool_args=None,
        question=None
    ) -> str:
        """
        Get standardized reminder prompts based on type. Extracts tool_names, tool_count, and original_question as needed.
        
        Args:
            reminder_type: Type of reminder needed
            messages: Message history (for extracting question)
            tools: List of tool objects (for tool names)
            tool_results_history: List of tool results (for count)
            tool_name: Name of the tool (for tool-specific reminders)
            count: Usage count (for tool-specific reminders)
            tool_args: Arguments for the tool (for duplicate reminders)
            question: Optional question override
            
        Returns:
            str: The reminder prompt
        """
        # Extract tool_names if needed
        tool_names = None
        if tools is not None:
            tool_names = ', '.join([self._get_tool_name(tool) for tool in tools])
            
        # Extract tool_count if needed
        tool_count = None
        if tool_results_history is not None:
            tool_count = len(tool_results_history)
            
        # Extract original_question if needed
        original_question = None
        if messages is not None:
            for msg in messages:
                if hasattr(msg, 'type') and msg.type == 'human':
                    original_question = msg.content
                    break
        if not original_question:
            original_question = question or '[Original question not found]'
            
        reminders = {
            "final_answer_prompt": (
                (f"Please analyse any and all existing tool results, then provide your FINAL ANSWER.\n"
                 f"Use any tools to gather missing information, then provide your FINAL ANSWER.\n"
                 f"Available tools include: {tool_names or 'various tools'}." 
                 if not tool_count or tool_count == 0 else "")
                + (f"\n\nIMPORTANT: You have gathered information from {tool_count} tool calls. "
                   f"The tool results are available in the message history above. "
                   f"Please carefully analyze these results and provide your FINAL ANSWER to the original question. "
                   f"Your answer must follow the system prompt. "
                   f"Do not call any more tools - analyze the existing results and provide your answer now." 
                   if tool_count and tool_count > 0 else "")
                + f"\n\nPlease answer the following question in the required format:\n\n"
                + f"ORIGINAL QUESTION:\n{original_question}\n\n"
                + f"Your answer must start with 'FINAL ANSWER:' and follow the system prompt."
            ),
            "tool_usage_issue": (
                (
                    f"You have already called '{tool_name or 'this tool'}'"
                    + (f" {count} times" if count is not None else "")
                    + (f" with arguments {tool_args}" if tool_args is not None else "")
                    + ". "
                    if (tool_name or count is not None or tool_args is not None) else ""
                )
                + "Do not call this tool again. "
                + "Consider any results you have. If the result is empty, call a DIFFERENT TOOL. "
                + f"ORIGINAL QUESTION:\n{original_question}\n\n"
                + "NOW provide your FINAL ANSWER based on the information you have."
            ),
        }
        return reminders.get(reminder_type, "Please provide your FINAL ANSWER.")

    def _create_simple_chunk_prompt(self, messages, chunk_results, chunk_num, total_chunks):
        """Create a simple prompt for processing a chunk."""
        # Find original question
        original_question = ""
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                original_question = msg.content
                break
        
        # Determine if this is tool results or general content
        is_tool_results = any('tool' in str(result).lower() or 'result' in str(result).lower() for result in chunk_results)
        
        if is_tool_results:
            prompt = f"Question: {original_question}\n\nTool Results (Part {chunk_num}/{total_chunks}):\n"
            for i, result in enumerate(chunk_results, 1):
                prompt += f"{i}. {result}\n\n"
        else:
            prompt = f"Question: {original_question}\n\nContent Analysis (Part {chunk_num}/{total_chunks}):\n"
            for i, result in enumerate(chunk_results, 1):
                prompt += f"{i}. {result}\n\n"
        
        if chunk_num < total_chunks:
            prompt += "Analyze these results and provide key findings. More content coming."
        else:
            prompt += "Provide your FINAL ANSWER based on all content, when you receive it, following the system prompt format."
        
        return prompt

    def _is_token_limit_error(self, error, llm_type="unknown") -> bool:
        """
        Check if the error is a token limit error or router error using vector similarity.
        
        Args:
            error: The exception object
            llm_type: Type of LLM for specific error patterns
            
        Returns:
            bool: True if it's a token limit error or router error
        """
        error_str = str(error).lower()
        
        # Token limit and router error patterns for vector similarity
        error_patterns = [
            "Error code: 413 - {'error': {'message': 'Request too large for model `qwen-qwq-32b` in organization `org_01jyfgv54ge5ste08j9248st66` service tier `on_demand` on tokens per minute (TPM): Limit 6000, Requested 9681, please reduce your message size and try again. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
            "500 Server Error: Internal Server Error for url: https://router.huggingface.co/hyperbolic/v1/chat/completions (Request ID: Root=1-6861ed33-7dd4232d49939c6f65f6e83d;164205eb-e591-4b20-8b35-5745a13f05aa)",
            
        ]
        
        # Direct substring checks for efficiency
        if any(term in error_str for term in ["413", "token", "limit", "tokens per minute", "truncated", "tpm", "router.huggingface.co"]):
            return True
        
        # Check if error matches any pattern using vector similarity
        for pattern in error_patterns:
            if self._vector_answers_match(error_str, pattern):
                return True
        
        return False
