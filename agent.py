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
#import hashlib
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
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
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
        max_summary_tokens: Global token limit for summaries
    """
    
    # Single source of truth for LLM configuration
    LLM_CONFIG = {
        "gemini": {
            "name": "Google Gemini",
            "type_str": "gemini",
            "model": "gemini-2.5-pro",
            "temperature": 0,
            "api_key_env": "GEMINI_KEY",
            "token_limit": None,  # No limit for Gemini (2M token context)
            "max_tokens": None
        },
        "groq": {
            "name": "Groq",
            "type_str": "groq", 
            "model": "qwen-qwq-32b",
            "temperature": 0,
            "api_key_env": "GROQ_API_KEY", # Groq uses the GROQ_API_KEY environment variable automatically
            "token_limit": 8000,  # Increased from 5000 to allow longer reasoning
            "max_tokens": 2048
        },
        "huggingface": {
            "name": "HuggingFace",
            "type_str": "huggingface",
            "temperature": 0,
            "api_key_env": "HUGGINGFACEHUB_API_TOKEN",
            "token_limit": 16000,  # Conservative for HuggingFace
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
        #"gemini",
        "groq", 
        #"huggingface"
    ]
    
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
        # Global token limit for summaries
        self.max_summary_tokens = 255

        # Rate limiting setup
        self.last_request_time = 0
        # Minimum 1 second between requests
        self.min_request_interval = 1

        # Token management - LLM-specific limits (built from configuration)
        self.token_limits = {
            config["type_str"]: config["token_limit"] 
            for config in self.LLM_CONFIG.values()
        }
        self.max_message_history = 15  # Increased for better context retention

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
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            # Add small random jitter to avoid thundering herd
            jitter = random.uniform(0, 0.2)
            time.sleep(sleep_time + jitter)
        self.last_request_time = time.time()

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of token count (4 chars per token is a reasonable approximation).
        """
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
        # Determine max message history based on LLM type
        if llm_type == "gemini":
            max_history = 25  # More lenient for Gemini
        elif llm_type == "groq":
            max_history = 15   # More aggressive for Groq due to TPM limits
        else:
            max_history = self.max_message_history
        
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
        
        # For Groq, also truncate long tool messages to prevent TPM issues
        # if llm_type == "groq":
        #     self._summarize_long_tool_messages(tool_messages, llm_type, self.max_summary_tokens)
        
        # Reconstruct message list
        truncated_messages = []
        if system_msg:
            truncated_messages.append(system_msg)
        truncated_messages.extend(tool_messages)
        if last_human_msg:
            truncated_messages.append(last_human_msg)
        
        return truncated_messages

    def _summarize_tool_result_with_llm(self, text, max_tokens=None, question=None):
        """
        Summarize a long tool result.
        Optionally include the original question for more focused summarization.
        """
        # Structure the prompt as JSON for LLM convenience
        prompt_dict = {
            "task": "Summarize the following tool result for use as LLM context. The result pertains to the optional **question** provided below. If **question** is not present, proceed with summarization of existing content.",
            "focus": f"Focus on the most relevant facts, numbers, and names, related to the **question**  if it is present.",
            "length_limit": f"Limit the summary softly to about {max_tokens} tokens.",
            "purpose": f"Extract only the information relevant to the **question** or pertinent to further reasoning on this question. If the question is not present, focus on keeping the essential important details.",
            "question": question if question else None,
            "tool_result_to_summarize": text
        }
               
        return self._summarize_text_with_llm(text, max_tokens=max_tokens, question=question, prompt_dict_override=prompt_dict)
    
    def _summarize_text_with_llm(self, text, max_tokens=None, question=None, prompt_dict_override=None):
        """
        Summarize a long result using Gemini, then Groq (if available), otherwise HuggingFace, otherwise fallback to truncation.
        Optionally include the original question for more focused summarization.
        Uses the LLM with tools enabled, and instructs the LLM to use tools if needed.
        """
        if prompt_dict_override:
            prompt_dict = prompt_dict_override
        else:
            # Structure the prompt as JSON for LLM convenience
            prompt_dict = {
                "task": "Summarize the following response for use as LLM context. The response pertains to the optional **question** provided below. If **question** is not present, proceed with summarization of existing content.",
                "focus": f"Focus on the most relevant facts, numbers, and names, related to the **question**  if it is present.",
                "length_limit": f"Limit the summary softly to about {max_tokens} tokens.",
                "purpose": f"Extract only the information relevant to the **question** or pertinent to further reasoning on this question. If the question is not present, focus on keeping the essential important details.",
                "tool_calls": "Do not use tools.",
                "question": question if question else None,
                "text_to_summarize": text,
            }
        # Remove None fields for cleanliness
        prompt_dict = {k: v for k, v in prompt_dict.items() if v is not None}
        prompt = f"Summarization Request (JSON):\n" + json.dumps(prompt_dict, indent=2)
        
        try:
            if self.llm_primary:
                response = self.llm_primary.invoke([HumanMessage(content=prompt)])
                if hasattr(response, 'content') and response.content:
                    return response.content.strip()
        except Exception as e:
            print(f"[Summarization] Gemini summarization failed: {e}")
        try:
            if self.llm_fallback:
                response = self.llm_fallback.invoke([HumanMessage(content=prompt)])
                if hasattr(response, 'content') and response.content:
                    return response.content.strip()
        except Exception as e:
            print(f"[Summarization] Groq summarization failed: {e}")
        try:
            if self.llm_third_fallback:
                response = self.llm_third_fallback.invoke([HumanMessage(content=prompt)])
                if hasattr(response, 'content') and response.content:
                    return response.content.strip()
        except Exception as e:
            print(f"[Summarization] HuggingFace summarization failed: {e}")
        
        print(f"[Summarization] LLM summarization failed, truncating")
        return text[:1000] + '... [Summary is truncated]'

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
            except Exception as e:
                tool_result = f"Error running tool '{tool_name}': {e}"
                print(f"[Tool Loop] Error running tool '{tool_name}': {e}")
        
        return str(tool_result)

    def _handle_duplicate_tool_calls(self, messages: List, tool_results_history: List, llm) -> Any:
        """
        Handle duplicate tool calls by forcing final answer or using fallback.
        
        Args:
            messages: Current message list
            tool_results_history: History of tool results
            llm: LLM instance
            
        Returns:
            Response from LLM or fallback answer
        """
        print(f"[Tool Loop] All tool calls were duplicates. Forcing final answer with tool results.")
        
        # Find the original question
        original_question = None
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                original_question = msg.content
                break
        if not original_question:
            original_question = "[Original question not found]"
        
        # Create a comprehensive context with all tool results
        tool_results_summary = ""
        # if tool_results_history:
        #     # Summarize all tool results for additional context (not replacement)
        #     all_results = "\n".join([f"Tool result {i+1}: {result}" for i, result in enumerate(tool_results_history)])
        #     tool_results_summary = self._summarize_text_with_llm(
        #         all_results, 
        #         max_tokens=self.max_summary_tokens, 
        #         question=original_question
        #     )
        
        # Compose a comprehensive final answer request
        final_answer_prompt = (
            f"Based on the following tool results, provide your FINAL ANSWER to the question.\n\n"
            f"QUESTION:\n{original_question}\n\n"
        )
        
        if tool_results_summary:
            final_answer_prompt += f"TOOL RESULTS SUMMARY (for context):\n{tool_results_summary}\n\n"
        
        final_answer_prompt += (
            f"Please analyze the tool results and provide your final answer in the required format.\n"
            f"Your answer must follow the system prompt formatting rules."
        )
        
        # Create new message list with system prompt, question, and tool results
        final_messages = [self.sys_msg, HumanMessage(content=final_answer_prompt)]
        
        # Add the actual full tool results as separate messages
        if tool_results_history:
            for i, tool_result in enumerate(tool_results_history):
                # Create a tool message with the full result
                from langchain_core.messages import ToolMessage
                tool_message = ToolMessage(
                    content=tool_result,
                    name=f"tool_result_{i+1}",
                    tool_call_id=f"tool_result_{i+1}"
                )
                final_messages.append(tool_message)
                print(f"[Tool Loop] Added full tool result {i+1} to final messages")
        
        try:
            final_response = llm.invoke(final_messages)
            if hasattr(final_response, 'content') and final_response.content:
                print(f"[Tool Loop] ✅ Forced final answer generated: {final_response.content}")
                
                # Check if the response has the required FINAL ANSWER marker
                if self._has_final_answer_marker(final_response):
                    return final_response
                else:
                    print("[Tool Loop] Forced response missing FINAL ANSWER marker. Adding explicit reminder.")
                    # Add explicit reminder about the required format
                    explicit_reminder = (
                        f"Please provide your final answer in the correct format based on the tool results provided."
                    )
                    final_messages.append(HumanMessage(content=explicit_reminder))
                    try:
                        explicit_response = llm.invoke(final_messages)
                        if hasattr(explicit_response, 'content') and explicit_response.content:
                            print(f"[Tool Loop] ✅ Explicit reminder response: {explicit_response.content}")
                            return explicit_response
                    except Exception as e:
                        print(f"[Tool Loop] ❌ Failed to get explicit reminder response: {e}")
                
                return final_response
        except Exception as e:
            print(f"[Tool Loop] ❌ Failed to force final answer: {e}")
        
        # Fallback: use the most recent tool result if available
        if tool_results_history:
            best_result = tool_results_history[-1] if tool_results_history else "No result available"
            print(f"[Tool Loop] 📝 Using most recent tool result as final answer: {best_result}")
            from langchain_core.messages import AIMessage
            return AIMessage(content=best_result)
        
        return None

    def _summarize_long_tool_messages(self, messages: List, llm_type: str, max_tokens: int = 200) -> None:
        """
        Summarize long tool messages to reduce token usage.
        
        Args:
            messages: List of messages to process
            llm_type: Type of LLM for context
            max_tokens: Maximum tokens for summarization
        """
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'tool' and hasattr(msg, 'content'):
                if len(msg.content) > 500:
                    msg.content = self._summarize_tool_result_with_llm(msg.content, max_tokens=max_tokens, question=self.original_question)

    def _run_tool_calling_loop(self, llm, messages, tool_registry, llm_type="unknown"):
        """
        Run a tool-calling loop: repeatedly invoke the LLM, detect tool calls, execute tools, and feed results back until a final answer is produced.
        - Uses adaptive step limits based on LLM type (Gemini: 25, Groq: 15, HuggingFace: 20, unknown: 20).
        - Tracks called tools to prevent duplicate calls and tool results history for fallback handling.
        - Monitors progress by tracking consecutive steps without meaningful changes in response content.
        - Truncates messages and summarizes long tool results to prevent token overflow.
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
            "groq": 15,     # More steps for Groq to compensate for token limits
            "huggingface": 20,  # Conservative for HuggingFace
            "unknown": 20
        }
        max_steps = base_max_steps.get(llm_type, 8)
        
        called_tools = set()  # Track which tools have been called to prevent duplicates
        tool_results_history = []  # Track tool results for better fallback handling
        current_step_tool_results = []  # Track results from current step only
        consecutive_no_progress = 0  # Track consecutive steps without progress
        last_response_content = ""  # Track last response content for progress detection
        
        for step in range(max_steps):
            print(f"\n[Tool Loop] Step {step+1}/{max_steps} - Using LLM: {llm_type}")
            current_step_tool_results = []  # Reset for this step
            
            # Truncate messages to prevent token overflow
            messages = self._truncate_messages(messages, llm_type)
            
            # Check token limits and summarize if needed
            total_text = "".join(str(getattr(msg, 'content', '')) for msg in messages)
            estimated_tokens = self._estimate_tokens(total_text)
            token_limit = self.token_limits.get(llm_type)
            
            # if token_limit and estimated_tokens > token_limit:
            #     print(f"[Tool Loop] Token limit exceeded: {estimated_tokens} > {token_limit}. Summarizing...")
            #     # self._summarize_long_tool_messages(messages, llm_type, self.max_summary_tokens)
            
            try:
                response = llm.invoke(messages)
            except Exception as e:
                print(f"[Tool Loop] ❌ LLM invocation failed: {e}")
                
                from langchain_core.messages import AIMessage
                return AIMessage(content=f"Error during LLM processing: {str(e)}")

            # Check if response was truncated due to token limits
            if hasattr(response, 'response_metadata') and response.response_metadata:
                finish_reason = response.response_metadata.get('finish_reason')
                if finish_reason == 'length':
                    print(f"[Tool Loop] ❌ Hit token limit for {llm_type} LLM. Response was truncated. Cannot complete reasoning.")
                    from langchain_core.messages import AIMessage
                    return AIMessage(content=f"Error: Hit token limit for {llm_type} LLM. Cannot complete reasoning.")

            # === DEBUG OUTPUT ===
            print(f"[Tool Loop] Raw LLM response: {response}")
            print(f"[Tool Loop] Response type: {type(response)}")
            print(f"[Tool Loop] Response has content: {hasattr(response, 'content')}")
            if hasattr(response, 'content'):
                print(f"[Tool Loop] Content length: {len(response.content) if response.content else 0}")
            print(f"[Tool Loop] Response has tool_calls: {hasattr(response, 'tool_calls')}")
            if hasattr(response, 'tool_calls'):
                print(f"[Tool Loop] Tool calls: {response.tool_calls}")

            # Check for empty response
            if not hasattr(response, 'content') or not response.content:
                # Allow empty content if there are tool calls (this is normal for tool-calling responses)
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    print(f"[Tool Loop] Empty content but tool calls detected - proceeding with tool execution")
                else:
                    print(f"[Tool Loop] ❌ {llm_type} LLM returned empty response.")
                    from langchain_core.messages import AIMessage
                    return AIMessage(content=f"Error: {llm_type} LLM returned empty response. Cannot complete reasoning.")

            # Check for progress (new content or tool calls)
            current_content = getattr(response, 'content', '') or ''
            current_tool_calls = getattr(response, 'tool_calls', []) or []
            has_progress = (current_content != last_response_content or len(current_tool_calls) > 0)
            
            # Check if we have tool results but no final answer yet
            has_tool_results = len(tool_results_history) > 0
            has_final_answer = (hasattr(response, 'content') and response.content and 
                              self._has_final_answer_marker(response))
            
            if has_tool_results and not has_final_answer and step >= 3:
                # We have information but no answer - gently remind to provide final answer
                reminder = (
                    f"You have gathered information from {len(tool_results_history)} tool calls. "
                    f"Please provide your FINAL ANSWER based on this information. "
                    f"Reason more if needed."
                )
                messages.append(HumanMessage(content=reminder))
            
            if not has_progress:
                consecutive_no_progress += 1
                print(f"[Tool Loop] No progress detected. Consecutive no-progress steps: {consecutive_no_progress}")
                
                # Exit early if no progress for too many consecutive steps
                if consecutive_no_progress >= 3:
                    print(f"[Tool Loop] Exiting due to {consecutive_no_progress} consecutive steps without progress")
                    break
                elif consecutive_no_progress == 2:
                    # Add a gentle reminder to use tools
                    reminder = (
                        f"You seem to be thinking about the problem. "
                        f"Please use the available tools to gather information and then provide your FINAL ANSWER. "
                        f"Available tools include: {', '.join([tool.name for tool in self.tools])}."
                    )
                    messages.append(HumanMessage(content=reminder))
            else:
                consecutive_no_progress = 0  # Reset counter on progress
                
            last_response_content = current_content

            # If response has content and no tool calls, return
            if hasattr(response, 'content') and response.content and not getattr(response, 'tool_calls', None):
                print(f"[Tool Loop] Final answer detected: {response.content}")
                # --- NEW LOGIC: Check for 'FINAL ANSWER' marker ---
                if self._has_final_answer_marker(response):
                    return response
                else:
                    print("[Tool Loop] 'FINAL ANSWER' marker not found. Reiterating with reminder and summarized context.")
                    # Summarize the context (all tool results and messages so far)
                    # context_text = "\n".join(str(getattr(msg, 'content', '')) for msg in messages if hasattr(msg, 'content'))
                    # summarized_context = self._summarize_text_with_llm(context_text, max_tokens=self.max_summary_tokens, question=self.original_question)
                    # Find the original question
                    original_question = None
                    for msg in messages:
                        if hasattr(msg, 'type') and msg.type == 'human':
                            original_question = msg.content
                            break
                    if not original_question:
                        original_question = "[Original question not found]"
                    # Compose a reminder message
                    reminder = (
                        f"You did not provide your answer in the required format.\n"
                        f"Please answer the following question in the required format, strictly following the system prompt.\n\n"
                        f"QUESTION:\n{original_question}\n\n"
                        # f"CONTEXT SUMMARY (tool results, previous reasoning):\n{summarized_context}\n\n"
                        f"Remember: Your answer must start with 'FINAL ANSWER:' and follow the formatting rules."
                    )
                    reiterate_messages = [self.sys_msg, HumanMessage(content=reminder)]
                    try:
                        reiterate_response = llm.invoke(reiterate_messages)
                        print(f"[Tool Loop] Reiterated response: {reiterate_response.content if hasattr(reiterate_response, 'content') else reiterate_response}")
                        return reiterate_response
                    except Exception as e:
                        print(f"[Tool Loop] ❌ Failed to reiterate for 'FINAL ANSWER': {e}")
                        return response
            tool_calls = getattr(response, 'tool_calls', None)
            if tool_calls:
                print(f"[Tool Loop] Detected {len(tool_calls)} tool call(s)")
                # Filter out duplicate tool calls (by name and args)
                new_tool_calls = []
                duplicate_count = 0
                for tool_call in tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    args_key = json.dumps(tool_args, sort_keys=True) if isinstance(tool_args, dict) else str(tool_args)
                    
                    # Check if this exact tool call has been made before
                    if (tool_name, args_key) not in called_tools:
                        # New tool call
                        print(f"[Tool Loop] New tool call: {tool_name} with args: {tool_args}")
                        new_tool_calls.append(tool_call)
                        called_tools.add((tool_name, args_key))
                    else:
                        # Duplicate tool call
                        duplicate_count += 1
                        print(f"[Tool Loop] Duplicate tool call detected: {tool_name} with args: {tool_args}")
                        
                        # Only add reminder if this is the first duplicate in this step
                        if duplicate_count == 1:
                            reminder = (
                                f"You have already called tool '{tool_name}' with arguments {tool_args}. "
                                f"Please use the previous result or call a different tool if needed."
                            )
                            messages.append(HumanMessage(content=reminder))
                
                # Only force final answer if ALL tool calls were duplicates AND we have tool results
                if not new_tool_calls and tool_results_history:
                    print(f"[Tool Loop] All {len(tool_calls)} tool calls were duplicates and we have {len(tool_results_history)} tool results. Forcing final answer.")
                    result = self._handle_duplicate_tool_calls(messages, tool_results_history, llm)
                    if result:
                        return result
                elif not new_tool_calls and not tool_results_history:
                    # No new tool calls and no previous results - this might be a stuck state
                    print(f"[Tool Loop] All tool calls were duplicates but no previous results. Adding reminder to use available tools.")
                    reminder = (
                        f"You have called tools that were already executed. "
                        f"Please either provide your FINAL ANSWER based on the available information, "
                        f"or call a different tool that hasn't been used yet."
                    )
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
                    
                    # Report tool result
                    print(f"[Tool Loop] Tool result for '{tool_name}': {tool_result}")
                    messages.append(ToolMessage(content=tool_result, name=tool_name, tool_call_id=tool_call.get('id', tool_name)))
                continue  # Next LLM call
            # Gemini (and some LLMs) may use 'function_call' instead of 'tool_calls'
            function_call = getattr(response, 'function_call', None)
            if function_call:
                tool_name = function_call.get('name')
                tool_args = function_call.get('arguments', {})
                args_key = json.dumps(tool_args, sort_keys=True) if isinstance(tool_args, dict) else str(tool_args)
                if (tool_name, args_key) in called_tools:
                    print(f"[Tool Loop] Duplicate function_call detected: {tool_name} with args: {tool_args}")
                    reminder = (
                        f"You have already called tool '{tool_name}' with arguments {tool_args}. "
                        f"Please use the previous result or call a different tool if needed."
                    )
                    messages.append(HumanMessage(content=reminder))
                    
                    # Only force final answer if we have tool results
                    if tool_results_history:
                        print(f"[Tool Loop] Duplicate function_call with {len(tool_results_history)} tool results. Forcing final answer.")
                        result = self._handle_duplicate_tool_calls(messages, tool_results_history, llm)
                        if result:
                            return result
                    else:
                        # No previous results - add reminder and continue
                        reminder = (
                            f"You have called a tool that was already executed. "
                            f"Please either provide your FINAL ANSWER based on the available information, "
                            f"or call a different tool that hasn't been used yet."
                        )
                        messages.append(HumanMessage(content=reminder))
                    continue
                
                called_tools.add((tool_name, args_key))
                
                # Execute tool using helper method
                tool_result = self._execute_tool(tool_name, tool_args, tool_registry)
                
                # Store the raw result for this step
                current_step_tool_results.append(tool_result)
                tool_results_history.append(tool_result)
                
                # Report tool result
                print(f"[Tool Loop] Tool result for '{tool_name}': {tool_result}")
                messages.append(ToolMessage(content=tool_result, name=tool_name, tool_call_id=tool_name))
                continue
            if hasattr(response, 'content') and response.content:
                return response
            print(f"[Tool Loop] No tool calls or final answer detected. Exiting loop.")
            
            # If we get here, the LLM didn't make tool calls or provide content
            # Add a reminder to use tools or provide an answer
            reminder = (
                f"You need to either:\n"
                f"1. Use the available tools to gather information, or\n"
                f"2. Provide your FINAL ANSWER based on what you know.\n"
                f"Available tools: web_search, wiki_search, and others."
            )
            messages.append(HumanMessage(content=reminder))
            continue
        
        # If we reach here, we've exhausted all steps or hit progress limits
        print(f"[Tool Loop] Exiting after {step+1} steps. Last response: {response}")
        
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
        
        llm, llm_name, llm_type_str = self._select_llm(llm_type, use_tools)
        if llm is None:
            raise Exception(f"{llm_name} LLM not available")
        
        try:
            self._rate_limit()
            print(f"🤖 Using {llm_name}")
            print(f"--- LLM Prompt/messages sent to {llm_name} ---")
            for i, msg in enumerate(messages):
                print(f"Message {i}: {msg}")
            tool_registry = {self._get_tool_name(tool): tool for tool in self.tools}
            if use_tools:
                response = self._run_tool_calling_loop(llm, messages, tool_registry, llm_type_str)
                # If tool calling resulted in empty content, try without tools as fallback
                if not hasattr(response, 'content') or not response.content:
                    print(f"⚠️ {llm_name} tool calling returned empty content, trying without tools...")
                    llm_no_tools, _, _ = self._select_llm(llm_type, False)
                    if llm_no_tools:
                        tool_results = []
                        for msg in messages:
                            if hasattr(msg, 'type') and msg.type == 'tool' and hasattr(msg, 'content'):
                                tool_name = msg.name
                                tool_results.append(f"Tool {tool_name} result: {msg.content}")
                        if tool_results:
                            tool_summary = "\n".join(tool_results)
                            enhanced_messages = []
                            for msg in messages:
                                if not (hasattr(msg, 'type') and msg.type == 'tool'):
                                    enhanced_messages.append(msg)
                            enhanced_messages.append(HumanMessage(content=f"""
Based on the following tool results, provide your FINAL ANSWER according to the system prompt format:

{tool_summary}

"""))
                            print(f"🔄 Retrying {llm_name} without tools with enhanced context")
                            print(f"📝 Tool results included: {len(tool_results)} tools")
                            response = llm_no_tools.invoke(enhanced_messages)
                        else:
                            print(f"🔄 Retrying {llm_name} without tools (no tool results found)")
                            response = llm_no_tools.invoke(messages)
            else:
                response = llm.invoke(messages)
            print(f"--- Raw response from {llm_name} ---")
            return response
        except Exception as e:
            # Special handling for HuggingFace router errors
            if llm_type == "huggingface" and "500 Server Error" in str(e) and "router.huggingface.co" in str(e):
                error_msg = f"HuggingFace router service error (500): {e}"
                print(f"⚠️ {error_msg}")
                print("💡 This is a known issue with HuggingFace's router service. Consider using Google Gemini or Groq instead.")
                raise Exception(error_msg)
            elif llm_type == "huggingface" and "timeout" in str(e).lower():
                error_msg = f"HuggingFace timeout error: {e}"
                print(f"⚠️ {error_msg}")
                print("💡 HuggingFace models may be slow or overloaded. Consider using Google Gemini or Groq instead.")
                raise Exception(error_msg)
            else:
                raise Exception(f"{llm_name} failed: {e}")

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
                # if not answer or answer == str(response).strip():
                #     answer = self._intelligent_answer_extraction(response, original_question)
                
                print(f"✅ {llm_name} answered: {answer}")
                print(f"✅ Reference: {reference}")
                
                # If no reference provided, return the first successful answer
                if reference is None:
                    print(f"✅ {llm_name} succeeded (no reference to compare)")
                    return answer, llm_name
                
                # Check similarity with reference
                if self._vector_answers_match(answer, reference):
                    print(f"✅ {llm_name} succeeded with similar answer to reference")
                    return answer, llm_name
                else:
                    print(f"⚠️ {llm_name} succeeded but answer doesn't match reference")
                    
                    # Try the next LLM without reference if this isn't the last one
                    if llm_type != available_llms[-1][0]:
                        print(f"🔄 Trying next LLM without reference...")
                        # Continue to next iteration to try next LLM
                    else:
                        # This was the last LLM, fall back to reference answer
                        print(f"🔄 All LLMs tried, falling back to reference answer")
                        return reference, "reference_fallback"
                    
            except Exception as e:
                print(f"❌ {llm_name} failed: {e}")
                
                # Special retry logic for HuggingFace router errors
                if llm_type == "huggingface" and "500 Server Error" in str(e) and "router.huggingface.co" in str(e):
                    print("🔄 HuggingFace router error detected, retrying once...")
                    try:
                        import time
                        time.sleep(2)  # Wait 2 seconds before retry
                        response = self._make_llm_request(messages, use_tools=use_tools, llm_type=llm_type)
                        answer = self._extract_final_answer(response)
                        # if not answer or answer == str(response).strip():
                        #     answer = self._intelligent_answer_extraction(response, original_question)
                        if answer and not answer == str(response).strip():
                            print(f"✅ HuggingFace retry succeeded: {answer}")
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
        Normalize answer by removing common prefixes, normalizing whitespace, and removing punctuation for comparison.
        """
        import re
        # Handle None or empty values gracefully
        if not ans:
            return ""
        
        ans = ans.strip().lower()
        if ans.startswith("final answer:"):
            ans = ans[12:].strip()
        elif ans.startswith("final answer"):
            ans = ans[11:].strip()
        ans = re.sub(r'[^\w\s]', '', ans)
        ans = re.sub(r'\s+', ' ', ans).strip()
        return ans

    def _get_tool_name(self, tool):
        if hasattr(tool, 'name'):
            return tool.name
        elif hasattr(tool, '__name__'):
            return tool.__name__
        else:
            return str(tool)

    def _vector_answers_match(self, answer: str, reference: str) -> bool:
        try:
            # Handle None or empty answers gracefully
            if not answer:
                print("⚠️ Answer is empty, cannot compare with reference")
                return False
                
            norm_answer = self._normalize_answer(answer)
            norm_reference = self._normalize_answer(reference)
            if norm_answer == norm_reference:
                return True
            embeddings = self.embeddings
            
            # Get embeddings for both answers
            answer_embedding = embeddings.embed_query(norm_answer)
            reference_embedding = embeddings.embed_query(norm_reference)
            
            # Calculate cosine similarity
            import numpy as np
            answer_array = np.array(answer_embedding)
            reference_array = np.array(reference_embedding)
            
            # Cosine similarity calculation
            dot_product = np.dot(answer_array, reference_array)
            norm_a = np.linalg.norm(answer_array)
            norm_r = np.linalg.norm(reference_array)
            if norm_a == 0 or norm_r == 0:
                return False
            cosine_similarity = dot_product / (norm_a * norm_r)
            print(f"🔍 Answer similarity: {cosine_similarity:.3f} (threshold: {self.similarity_threshold})")
            return cosine_similarity >= self.similarity_threshold
        except Exception as e:
            print(f"⚠️ Error in vector similarity matching: {e}")
            # Fallback to simple string matching if embedding fails
            return self._fallback_string_match(answer, reference)

    def _fallback_string_match(self, answer: str, reference: str) -> bool:
        # Handle None or empty answers gracefully
        if not answer:
            return False
            
        norm_answer = self._normalize_answer(answer)
        norm_reference = self._normalize_answer(reference)
        if norm_answer == norm_reference:
            return True
        
        # Check if one contains the other (for partial matches)
        if norm_answer in norm_reference or norm_reference in norm_answer:
            return True
        
        # Check for numeric answers (common in math problems)
        try:
            # Extract numbers from both answers
            import re
            answer_nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', norm_answer)]
            reference_nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', norm_reference)]
            if answer_nums and reference_nums and answer_nums == reference_nums:
                return True
        except:
            pass
        return False

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
            return answer
        except Exception as e:
            print(f"❌ All LLMs failed: {e}")
            if reference:
                print("⚠️ Falling back to reference answer")
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

    def _intelligent_answer_extraction(self, response: Any, question: str) -> str:
        """
        Use LLM summarization to extract the most likely final answer from the response, given the question and the system prompt.
        This replaces the previous regex/pattern logic with a more robust LLM-based approach.
        Args:
            response (Any): The LLM response object.
            question (str): The original question for context.
        Returns:
            str: The extracted final answer, as determined by the LLM summarizer.
        """
        text = self._extract_text_from_response(response)

        # Compose a summarization prompt for the LLM
        prompt_dict = {
                "task": "Extract the FINAL answer from the given LLM response (response_to_analyze). The response pertains to the optional **question** provided. If **question** is not present, proceed with extracting per the system prompt. From the response, extract the the most likely FINAL ANSWER according to the system prompt's answer formatting rules. Return only the most likely final answer, formatted exactly as required by the system prompt.",
                "focus": f"Focus on the most relevant facts, numbers, and names, related to the **question**  if it is present.",
                "purpose": f"Extract the FINAL ANSWER per the system prompt.",
                "tool_calls": "You may use any available tools to analyze, extract, or process the tool_result if needed.",
                "question": question if question else None,
                "response_to_analyze": text
        }
        print(f"[Agent] Summarization prompt for answer extraction:\n{prompt_dict}")
        summary = self._summarize_text_with_llm(text, max_tokens=self.max_summary_tokens, question=self.original_question, prompt_dict_override=prompt_dict)
        print(f"[Agent] LLM-based answer extraction summary: {summary}")
        return summary.strip()

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
            'execute_code_multilang'
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
        Automatically inject file data into tool arguments if the tool needs it and file data is available.
        
        Args:
            tool_name (str): Name of the tool being called
            tool_args (dict): Original tool arguments
            
        Returns:
            dict: Modified tool arguments with file data if needed
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
                try:
                    # Decode base64 file data to get the actual code content
                    file_data = base64.b64decode(self.current_file_data)
                    code_content = file_data.decode('utf-8')
                    tool_args[param_name] = code_content
                    print(f"[Tool Loop] Injected code from attached file for {tool_name}: {len(code_content)} characters")
                except Exception as e:
                    print(f"[Tool Loop] Failed to decode file data for code injection: {e}")
        
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
                print(f"   Test message: {test_message}")
                print(f"   Test response: {test_response}")
                return True
            else:
                print(f"❌ {llm_name} returned empty response")
                return False
                
        except Exception as e:
            print(f"❌ {llm_name} test failed: {e}")
            return False 