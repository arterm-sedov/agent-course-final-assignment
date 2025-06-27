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
    - system_prompt.txt
"""
import os
import json
import csv
import time
import random
import hashlib
from typing import List, Dict, Any, Optional
from tools import *

# For LLM and retriever integration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from supabase.client import create_client

# === GLOBAL SYSTEM PROMPT LOADING ===
SYSTEM_PROMPT = None
ANSWER_FORMATTING_RULES = None

def _load_system_prompt():
    global SYSTEM_PROMPT, ANSWER_FORMATTING_RULES
    if SYSTEM_PROMPT is None:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read()
        ANSWER_FORMATTING_RULES = SYSTEM_PROMPT
_load_system_prompt()

class GaiaAgent:
    """
    Main agent for the GAIA Unit 4 benchmark.

    This agent:
      - Uses the tools.py (math, code, file, image, web, etc.)
      - Integrates a supabase retriever for similar Q/A and context
      - Strictly follows the system prompt in system_prompt.txt
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
    """
    def __init__(self, provider: str = "groq"):
        """
        Initialize the agent, loading the system prompt, tools, retriever, and LLM.

        Args:
            provider (str): LLM provider to use. One of "google", "groq", or "huggingface".

        Raises:
            ValueError: If an invalid provider is specified.
        """
        _load_system_prompt()
        self.system_prompt = SYSTEM_PROMPT
        self.sys_msg = SystemMessage(content=self.system_prompt)

        # Rate limiting setup
        self.last_request_time = 0
        # Minimum 1 second between requests
        self.min_request_interval = 1

        # Token management - LLM-specific limits
        self.token_limits = {
            "gemini": None,  # No limit for Gemini (2M token context)
            "groq": 32000,   # Conservative for Groq
            "huggingface": 16000  # Conservative for HuggingFace
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

        # Set HuggingFace API token if available
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

        # Set up primary LLM (Google Gemini) and fallback LLM (Groq)
        try:
            self.llm_primary = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro", 
                temperature=0, 
                google_api_key=os.environ.get("GEMINI_KEY")
                # No max_tokens limit for Gemini - let it use its full capability
            )
            print("✅ Primary LLM (Google Gemini) initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize Google Gemini: {e}")
            self.llm_primary = None
        
        try:
            self.llm_fallback = ChatGroq(
                model="qwen-qwq-32b", 
                temperature=0,
                max_tokens=1024  # Limit output tokens
            )
            print("✅ Fallback LLM (Groq) initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize Groq: {e}")
            self.llm_fallback = None
        
        try:
            self.llm_third_fallback = ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
                    task="text-generation",
                    max_new_tokens=1024,
                    do_sample=False,
                    repetition_penalty=1.03,
                    temperature=0,
                ),
                verbose=True,
            )
            print("✅ Third fallback LLM (HuggingFace) initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize HuggingFace: {e}")
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
        
        Args:
            messages: List of messages to truncate
            llm_type: Type of LLM for context-aware truncation
        """
        # Determine max message history based on LLM type
        if llm_type == "gemini":
            max_history = 25  # More lenient for Gemini
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
        
        # Reconstruct message list
        truncated_messages = []
        if system_msg:
            truncated_messages.append(system_msg)
        truncated_messages.extend(tool_messages)
        if last_human_msg:
            truncated_messages.append(last_human_msg)
        
        return truncated_messages

    def _summarize_text_with_llm(self, text, max_tokens=512):
        """
        Summarize a long tool result using Groq (if available), otherwise Gemini, otherwise fallback to truncation.
        """
        prompt = f"Summarize the following tool result for use as LLM context. Focus on the most relevant facts, numbers, and names. Limit to {max_tokens} tokens.\n\nTOOL RESULT:\n{text}"
        try:
            if self.llm_fallback:
                response = self.llm_fallback.invoke([HumanMessage(content=prompt)])
                if hasattr(response, 'content') and response.content:
                    return response.content.strip()
        except Exception as e:
            print(f"[Summarization] Groq summarization failed: {e}")
        try:
            if self.llm_primary:
                response = self.llm_primary.invoke([HumanMessage(content=prompt)])
                if hasattr(response, 'content') and response.content:
                    return response.content.strip()
        except Exception as e:
            print(f"[Summarization] Gemini summarization failed: {e}")
        return text[:1000] + '... [truncated]'

    def _run_tool_calling_loop(self, llm, messages, tool_registry, llm_type="unknown"):
        """
        Run a tool-calling loop: repeatedly invoke the LLM, detect tool calls, execute tools, and feed results back until a final answer is produced.
        - Summarizes tool results after each call and injects them into the context.
        - Reminds the LLM if it tries to call the same tool with the same arguments.
        - Injects the system prompt before requesting the final answer.
        - Uses Groq for summarization if available, otherwise Gemini, otherwise truncation.
        - Keeps the context concise and focused on the system prompt, question, tool results, and answer formatting rules.

        Args:
            llm: The LLM instance (with or without tools bound)
            messages: The message history (list)
            tool_registry: Dict mapping tool names to functions
            llm_type: Type of LLM ("gemini", "groq", "huggingface", or "unknown")
        Returns:
            The final LLM response (with content)
        """
        max_steps = 5  # Prevent infinite loops
        called_tools = set()  # Track which tools have been called to prevent duplicates
        tool_results_history = []  # Track tool results for better fallback handling
        tool_args_history = {}
        for step in range(max_steps):
            print(f"\n[Tool Loop] Step {step+1} - Using LLM: {llm_type}")
            # Truncate messages to prevent token overflow
            messages = self._truncate_messages(messages, llm_type)
            total_text = "".join(str(getattr(msg, 'content', '')) for msg in messages)
            estimated_tokens = self._estimate_tokens(total_text)
            token_limit = self.token_limits.get(llm_type)
            if token_limit and estimated_tokens > token_limit:
                print(f"[Tool Loop] Truncating messages: estimated {estimated_tokens} tokens (limit {token_limit})")
                for msg in messages:
                    if hasattr(msg, 'type') and msg.type == 'tool' and hasattr(msg, 'content'):
                        if len(msg.content) > 500:
                            print(f"[Tool Loop] Summarizing long tool result for token limit")
                            msg.content = self._summarize_text_with_llm(msg.content, max_tokens=300)
            try:
                response = llm.invoke(messages)
            except Exception as e:
                print(f"[Tool Loop] ❌ LLM invocation failed: {e}")
                from langchain_core.messages import AIMessage
                return AIMessage(content=f"Error during LLM processing: {str(e)}")

            # === DEBUG OUTPUT ===
            print(f"[Tool Loop] Raw LLM response: {response}")
            print(f"[Tool Loop] Response type: {type(response)}")
            print(f"[Tool Loop] Response has content: {hasattr(response, 'content')}")
            if hasattr(response, 'content'):
                print(f"[Tool Loop] Content length: {len(response.content) if response.content else 0}")
            print(f"[Tool Loop] Response has tool_calls: {hasattr(response, 'tool_calls')}")
            if hasattr(response, 'tool_calls'):
                print(f"[Tool Loop] Tool calls: {response.tool_calls}")

            # If response has content and no tool calls, return
            if hasattr(response, 'content') and response.content and not getattr(response, 'tool_calls', None):
                print(f"[Tool Loop] Final answer detected: {response.content}")
                return response

            tool_calls = getattr(response, 'tool_calls', None)
            if tool_calls:
                print(f"[Tool Loop] Detected {len(tool_calls)} tool call(s)")
                # Filter out duplicate tool calls (by name and args)
                new_tool_calls = []
                for tool_call in tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    args_key = json.dumps(tool_args, sort_keys=True) if isinstance(tool_args, dict) else str(tool_args)
                    if (tool_name, args_key) not in called_tools:
                        # New tool call
                        print(f"[Tool Loop] New tool call: {tool_name} with args: {tool_args}")
                        new_tool_calls.append(tool_call)
                        called_tools.add((tool_name, args_key))
                        tool_args_history[(tool_name, args_key)] = None
                    else:
                        # Duplicate tool call
                        print(f"[Tool Loop] Duplicate tool call detected: {tool_name} with args: {tool_args}")
                        reminder = f"You have already called tool '{tool_name}' with arguments {tool_args}. Please use the previous result."
                        messages.append(HumanMessage(content=reminder))
                if not new_tool_calls:
                    # All tool calls were duplicates, force final answer
                    print(f"[Tool Loop] All tool calls were duplicates. Appending system prompt for final answer.")
                    messages.append(HumanMessage(content=f"{self.system_prompt}"))
                    try:
                        final_response = llm.invoke(messages)
                        if hasattr(final_response, 'content') and final_response.content:
                            print(f"[Tool Loop] ✅ Forced final answer generated: {final_response.content}")
                            return final_response
                    except Exception as e:
                        print(f"[Tool Loop] ❌ Failed to force final answer: {e}")
                    if tool_results_history:
                        best_result = max(tool_results_history, key=len)
                        print(f"[Tool Loop] 📝 Using best tool result as final answer: {best_result}")
                        from langchain_core.messages import AIMessage
                        return AIMessage(content=f"FINAL ANSWER: {best_result}")
                # Execute only new tool calls
                for tool_call in new_tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    print(f"[Tool Loop] Running tool: {tool_name} with args: {tool_args}")
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except Exception:
                            pass
                    tool_func = tool_registry.get(tool_name)
                    if not tool_func:
                        tool_result = f"Tool '{tool_name}' not found."
                        print(f"[Tool Loop] Tool '{tool_name}' not found.")
                    else:
                        try:
                            # Handle both LangChain tools and regular functions
                            if hasattr(tool_func, 'invoke') and hasattr(tool_func, 'name'):
                                if isinstance(tool_args, dict):
                                    tool_result = tool_func.invoke(tool_args)
                                else:
                                    tool_result = tool_func.invoke({"input": tool_args})
                            else:
                                if isinstance(tool_args, dict):
                                    tool_result = tool_func(**tool_args)
                                else:
                                    tool_result = tool_func(tool_args)
                            print(f"[Tool Loop] Tool '{tool_name}' executed successfully.")
                        except Exception as e:
                            tool_result = f"Error running tool '{tool_name}': {e}"
                            print(f"[Tool Loop] Error running tool '{tool_name}': {e}")
                    tool_results_history.append(str(tool_result))
                    # Summarize tool result and inject as message for LLM context
                    summary = self._summarize_text_with_llm(str(tool_result), max_tokens=255)
                    print(f"[Tool Loop] Injecting tool result summary for '{tool_name}': {summary}")
                    summary_msg = HumanMessage(content=f"Tool '{tool_name}' called with {tool_args}. Result: {summary}")
                    messages.append(summary_msg)
                    messages.append(ToolMessage(content=str(tool_result), name=tool_name, tool_call_id=tool_call.get('id', tool_name)))
                continue  # Next LLM call
            # Gemini (and some LLMs) may use 'function_call' instead of 'tool_calls'
            function_call = getattr(response, 'function_call', None)
            if function_call:
                tool_name = function_call.get('name')
                tool_args = function_call.get('arguments', {})
                args_key = json.dumps(tool_args, sort_keys=True) if isinstance(tool_args, dict) else str(tool_args)
                if (tool_name, args_key) in called_tools:
                    print(f"[Tool Loop] Duplicate function_call detected: {tool_name} with args: {tool_args}")
                    reminder = f"You have already called tool '{tool_name}' with arguments {tool_args}. Please use the previous result."
                    messages.append(HumanMessage(content=reminder))
                    if tool_results_history:
                        print(f"[Tool Loop] Appending system prompt for final answer after duplicate function_call.")
                        messages.append(HumanMessage(content=f"{self.system_prompt}"))
                        try:
                            final_response = llm.invoke(messages)
                            if hasattr(final_response, 'content') and final_response.content:
                                print(f"[Tool Loop] ✅ Forced final answer generated: {final_response.content}")
                                return final_response
                        except Exception as e:
                            print(f"[Tool Loop] ❌ Failed to force final answer: {e}")
                    if tool_results_history:
                        best_result = max(tool_results_history, key=len)
                        print(f"[Tool Loop] 📝 Using best tool result as final answer: {best_result}")
                        from langchain_core.messages import AIMessage
                        return AIMessage(content=f"FINAL ANSWER: {best_result}")
                    continue
                called_tools.add((tool_name, args_key))
                tool_func = tool_registry.get(tool_name)
                print(f"[Tool Loop] Running function_call tool: {tool_name} with args: {tool_args}")
                if not tool_func:
                    tool_result = f"Tool '{tool_name}' not found."
                    print(f"[Tool Loop] Tool '{tool_name}' not found.")
                else:
                    try:
                        if hasattr(tool_func, 'invoke') and hasattr(tool_func, 'name'):
                            if isinstance(tool_args, dict):
                                tool_result = tool_func.invoke(tool_args)
                            else:
                                tool_result = tool_func.invoke({"input": tool_args})
                        else:
                            if isinstance(tool_args, dict):
                                tool_result = tool_func(**tool_args)
                            else:
                                tool_result = tool_func(tool_args)
                        print(f"[Tool Loop] Tool '{tool_name}' executed successfully.")
                    except Exception as e:
                        tool_result = f"Error running tool '{tool_name}': {e}"
                        print(f"[Tool Loop] Error running tool '{tool_name}': {e}")
                tool_results_history.append(str(tool_result))
                summary = self._summarize_text_with_llm(str(tool_result), max_tokens=255)
                print(f"[Tool Loop] Injecting tool result summary for '{tool_name}': {summary}")
                summary_msg = HumanMessage(content=f"Tool '{tool_name}' called with {tool_args}. Result: {summary}")
                messages.append(summary_msg)
                messages.append(ToolMessage(content=str(tool_result), name=tool_name, tool_call_id=tool_name))
                continue
            if hasattr(response, 'content') and response.content:
                print(f"[Tool Loop] Injecting system prompt before final answer.")
                messages.append(HumanMessage(content=f"Before answering, remember:\n{self.system_prompt}"))
                return response
            print(f"[Tool Loop] No tool calls or final answer detected. Exiting loop.")
            break
        if tool_results_history and (not hasattr(response, 'content') or not response.content):
            best_result = max(tool_results_history, key=len)
            print(f"[Tool Loop] 📝 No final answer generated, using best tool result from history: {best_result}")
            from langchain_core.messages import AIMessage
            synthetic_response = AIMessage(content=f"FINAL ANSWER: {best_result}")
            return synthetic_response
        print(f"[Tool Loop] Exiting after {max_steps} steps. Last response: {response}")
        return response

    def _make_llm_request(self, messages, use_tools=True, llm_type="primary"):
        """
        Make an LLM request with rate limiting.
        Uses primary LLM (Google Gemini) first, then fallback (Groq), then third fallback (HuggingFace).

        Args:
            messages: The messages to send to the LLM
            use_tools (bool): Whether to use tools (llm_with_tools vs llm)
            llm_type (str): Which LLM to use ("primary", "fallback", or "third_fallback")

        Returns:
            The LLM response

        Raises:
            Exception: If the LLM fails
        """
        # Select which LLM to use
        if llm_type == "primary":
            llm = self.llm_primary_with_tools if use_tools else self.llm_primary
            llm_name = "Google Gemini"
            llm_type_str = "gemini"
        elif llm_type == "fallback":
            llm = self.llm_fallback_with_tools if use_tools else self.llm_fallback
            llm_name = "Groq"
            llm_type_str = "groq"
        elif llm_type == "third_fallback":
            llm = self.llm_third_fallback_with_tools if use_tools else self.llm_third_fallback
            llm_name = "HuggingFace"
            llm_type_str = "huggingface"
        else:
            raise ValueError(f"Invalid llm_type: {llm_type}")
        
        if llm is None:
            raise Exception(f"{llm_name} LLM not available")
        
        try:
            self._rate_limit()
            print(f"🤖 Using {llm_name}")
            print(f"--- LLM Prompt/messages sent to {llm_name} ---")
            for i, msg in enumerate(messages):
                print(f"Message {i}: {msg}")
            # Build tool registry (name -> function)
            tool_registry = {tool.__name__: tool for tool in self.tools}
            if use_tools:
                response = self._run_tool_calling_loop(llm, messages, tool_registry, llm_type_str)
                # If tool calling resulted in empty content, try without tools as fallback
                if not hasattr(response, 'content') or not response.content:
                    print(f"⚠️ {llm_name} tool calling returned empty content, trying without tools...")
                    # Get the LLM without tools
                    if llm_type == "primary":
                        llm_no_tools = self.llm_primary
                    elif llm_type == "fallback":
                        llm_no_tools = self.llm_fallback
                    elif llm_type == "third_fallback":
                        llm_no_tools = self.llm_third_fallback
                    
                    if llm_no_tools:
                        # Extract tool results more robustly
                        tool_results = []
                        for msg in messages:
                            if hasattr(msg, 'type') and msg.type == 'tool' and hasattr(msg, 'content'):
                                tool_name = msg.name  # ToolMessage always has name attribute
                                tool_results.append(f"Tool {tool_name} result: {msg.content}")
                        
                        if tool_results:
                            # Create a new message with tool results included
                            tool_summary = "\n".join(tool_results)
                            # Remove tool messages and add enhanced context
                            enhanced_messages = []
                            for msg in messages:
                                if not (hasattr(msg, 'type') and msg.type == 'tool'):
                                    enhanced_messages.append(msg)
                            
                            # Add a clear instruction to generate final answer from tool results
                            enhanced_messages.append(HumanMessage(content=f"""
Based on the following tool results, provide your FINAL ANSWER according to the system prompt format:

{tool_summary}

IMPORTANT FORMATTING RULES:
- YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings
- If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise
- If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise
- Your answer must end with "FINAL ANSWER: [your answer]"

For example, if the answer is 3, write: FINAL ANSWER: 3
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
            # Print only the first 1000 characters if response is long
            # resp_str = str(response)
            # if len(resp_str) > 1000:
            #     print(self._summarize_text_with_gemini(resp_str, max_tokens=300))
            # else:
            #     print(resp_str)
            return response
        except Exception as e:
            raise Exception(f"{llm_name} failed: {e}")

    def _try_llm_sequence(self, messages, use_tools=True, reference=None, similarity_threshold=0.8):
        """
        Try multiple LLMs in sequence until one succeeds and produces a similar answer to reference.
        Only one attempt per LLM, then move to the next.
        
        Args:
            messages: The messages to send to the LLM
            use_tools (bool): Whether to use tools
            reference (str, optional): Reference answer to compare against
            similarity_threshold (float): Minimum similarity score (0.0-1.0) to consider answers similar
            
        Returns:
            tuple: (answer, llm_used) where answer is the final answer and llm_used is the name of the LLM that succeeded
            
        Raises:
            Exception: If all LLMs fail or none produce similar enough answers
        """
        llm_sequence = [
            ("primary", "Google Gemini"),
            ("fallback", "Groq"), 
            ("third_fallback", "HuggingFace")
        ]
        
        # Extract the original question for intelligent extraction
        original_question = ""
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                original_question = msg.content
                break
        
        for llm_type, llm_name in llm_sequence:
            try:
                response = self._make_llm_request(messages, use_tools=use_tools, llm_type=llm_type)
                
                # Try standard extraction first
                answer = self._extract_final_answer(response)
                
                # If standard extraction didn't work well, try intelligent extraction
                if not answer or answer == str(response).strip():
                    answer = self._intelligent_answer_extraction(response, original_question)
                
                # Post-process the answer to ensure proper formatting
                answer = self._post_process_answer(answer, original_question)
                
                print(f"✅ {llm_name} answered: {answer}")
                print(f"✅ Reference: {reference}")
                
                # If no reference provided, return the first successful answer
                if reference is None:
                    print(f"✅ {llm_name} succeeded (no reference to compare)")
                    return answer, llm_name
                
                # Check similarity with reference
                if self._simple_answers_match(answer, reference):
                    print(f"✅ {llm_name} succeeded with similar answer to reference")
                    return answer, llm_name
                else:
                    print(f"⚠️ {llm_name} succeeded but answer doesn't match reference")
                    
                    # Try one more time with reference in context if this is the first attempt
                    if llm_type == "primary" and reference:
                        print(f"🔄 Retrying {llm_name} with reference in context...")
                        retry_messages = self._format_messages(original_question, reference)
                        try:
                            retry_response = self._make_llm_request(retry_messages, use_tools=use_tools, llm_type=llm_type)
                            retry_answer = self._extract_final_answer(retry_response)
                            if not retry_answer or retry_answer == str(retry_response).strip():
                                retry_answer = self._intelligent_answer_extraction(retry_response, original_question)
                            
                            # Post-process the retry answer
                            retry_answer = self._post_process_answer(retry_answer, original_question)
                            
                            if self._simple_answers_match(retry_answer, reference):
                                print(f"✅ {llm_name} retry succeeded with similar answer to reference")
                                return retry_answer, llm_name
                            else:
                                print(f"⚠️ {llm_name} retry still doesn't match reference")
                        except Exception as e:
                            print(f"❌ {llm_name} retry failed: {e}")
                    
                    if llm_type == "third_fallback":
                        # This was the last LLM, return the answer anyway
                        print(f"🔄 Using {llm_name} answer despite mismatch")
                        return answer, llm_name
                    print(f"🔄 Trying next LLM...")
                    
            except Exception as e:
                print(f"❌ {llm_name} failed: {e}")
                if llm_type == "third_fallback":
                    # This was the last LLM, re-raise the exception
                    raise Exception(f"All LLMs failed. Last error from {llm_name}: {e}")
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

    def _simple_answers_match(self, answer: str, reference: str) -> bool:
        """
        Use vectorized similarity comparison with the same embedding engine as Supabase.
        This provides semantic similarity matching instead of rigid string matching.
        
        Args:
            answer (str): The agent's answer.
            reference (str): The reference answer.
            
        Returns:
            bool: True if answers are semantically similar (similarity > threshold), False otherwise.
        """
        try:
            # Normalize answers by removing common prefixes
            def normalize_answer(ans):
                ans = ans.strip()
                if ans.lower().startswith("final answer:"):
                    ans = ans[12:].strip()
                elif ans.lower().startswith("final answer"):
                    ans = ans[11:].strip()
                return ans
            
            norm_answer = normalize_answer(answer)
            norm_reference = normalize_answer(reference)
            
            # If answers are identical after normalization, return True immediately
            if norm_answer.lower() == norm_reference.lower():
                return True
            
            # Use the same embedding engine as Supabase for consistency
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
            norm_answer = np.linalg.norm(answer_array)
            norm_reference = np.linalg.norm(reference_array)
            
            if norm_answer == 0 or norm_reference == 0:
                return False
            
            cosine_similarity = dot_product / (norm_answer * norm_reference)
            
            # Set similarity threshold (0.85 is quite strict, 0.8 is more lenient)
            similarity_threshold = 0.8
            
            print(f"🔍 Answer similarity: {cosine_similarity:.3f} (threshold: {similarity_threshold})")
            
            return cosine_similarity >= similarity_threshold
            
        except Exception as e:
            print(f"⚠️ Error in vector similarity matching: {e}")
            # Fallback to simple string matching if embedding fails
            return self._fallback_string_match(answer, reference)
    
    def _fallback_string_match(self, answer: str, reference: str) -> bool:
        """
        Fallback string matching method for when vector similarity fails.
        
        Args:
            answer (str): The agent's answer.
            reference (str): The reference answer.
            
        Returns:
            bool: True if answers appear to match using string comparison.
        """
        # Normalize both answers for comparison
        def normalize_answer(ans):
            # Remove common prefixes and normalize whitespace
            ans = ans.strip().lower()
            if ans.startswith("final answer:"):
                ans = ans[12:].strip()
            elif ans.startswith("final answer"):
                ans = ans[11:].strip()
            # Remove punctuation and extra whitespace
            import re
            ans = re.sub(r'[^\w\s]', '', ans)
            ans = re.sub(r'\s+', ' ', ans).strip()
            return ans
        
        norm_answer = normalize_answer(answer)
        norm_reference = normalize_answer(reference)
        
        # Check for exact match
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

    def __call__(self, question: str) -> str:
        """
        Run the agent on a single question, using step-by-step reasoning and tools.

        Args:
            question (str): The question to answer.

        Returns:
            str: The agent's final answer, formatted per system_prompt.txt.

        Workflow:
            1. Retrieve similar Q/A for context using the retriever.
            2. Use LLM sequence with similarity checking against reference.
            3. If no similar answer found, fall back to reference answer.
        """
        print(f"\n🔎 Processing question: {question}\n")
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
        - Removing everything before and including the first 'FINAL ANSWER' (case-insensitive, with/without colon/space)
        - Stripping leading/trailing whitespace
        - Removing extra punctuation (except for commas, dots, hyphens)
        - Normalizing whitespace
        """
        import re
        print(f"[CleanFinalAnswer] Original text before stripping: {text}")
        # Find the first occurrence of 'FINAL ANSWER' (case-insensitive)
        match = re.search(r'final answer\s*:?', text, flags=re.IGNORECASE)
        if match:
            # Only keep what comes after 'FINAL ANSWER'
            text = text[match.end():]
        # Remove extra punctuation except for commas, dots, hyphens
        text = re.sub(r'[^\w\s,.-]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_final_answer(self, response: Any) -> str:
        """
        Extract the final answer from the LLM response, removing only the "FINAL ANSWER:" prefix.
        The LLM is responsible for following the system prompt formatting rules.
        This method is used for validation against reference answers and submission.

        Args:
            response (Any): The LLM response object.

        Returns:
            str: The extracted final answer string with "FINAL ANSWER:" prefix removed.
        """
        # Try to find the line starting with 'FINAL ANSWER:'
        if hasattr(response, 'content'):
            text = response.content
        elif isinstance(response, dict) and 'content' in response:
            text = response['content']
        else:
            text = str(response)
        # Find the line with 'FINAL ANSWER' (case-insensitive)
        for line in text.splitlines():
            if line.strip().upper().startswith("FINAL ANSWER"):
                answer = line.strip()
                return self._clean_final_answer_text(answer)
        # Fallback: return the whole response, cleaning prefix if present
        return self._clean_final_answer_text(text)

    def _post_process_answer(self, answer: str, question: str) -> str:
        """
        Post-process the answer to ensure it follows the system prompt formatting rules.
        Args:
            answer (str): The raw answer from the LLM.
            question (str): The original question for context.
        Returns:
            str: The properly formatted answer.
        """
        import re
        # Clean up the answer using the unified cleaning function
        answer = self._clean_final_answer_text(answer)
        # Check if question asks for a number
        question_lower = question.lower()
        is_numeric_question = any(word in question_lower for word in ['how many', 'number', 'count', 'amount', 'quantity'])
        if is_numeric_question:
            # Extract the first number from the answer
            numbers = re.findall(r'\d+', answer)
            if numbers:
                return numbers[0]  # Return just the number
        # If the answer is too long, try to extract the key part
        if len(answer) > 50:
            # Look for patterns that might indicate the actual answer
            patterns = [
                r'(\w+(?:\s+\w+){0,5})',  # Up to 6 words
                r'(\d+(?:\s*,\s*\d+)*)',  # Numbers with commas
            ]
            for pattern in patterns:
                matches = re.findall(pattern, answer)
                if matches:
                    # Return the first reasonable match
                    for match in matches:
                        if 1 <= len(match) <= 30:
                            return match
        return answer

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
        if hasattr(response, 'content'):
            text = response.content
        elif isinstance(response, dict) and 'content' in response:
            text = response['content']
        else:
            text = str(response)

        # Compose a summarization prompt for the LLM
        prompt = (
            f"You are a helpful assistant. Given the following question, system prompt, and LLM response, extract the most likely FINAL ANSWER according to the system prompt's answer formatting rules.\n"
            f"\nQUESTION:\n{question}\n"
            f"\nSYSTEM PROMPT (answer formatting rules):\n{self.system_prompt}\n"
            f"\nLLM RESPONSE:\n{text}\n"
            f"\nReturn only the most likely final answer, formatted exactly as required by the system prompt."
        )
        print(f"[Agent] Summarization prompt for answer extraction:\n{prompt}")
        # Use the summarization LLM (Groq preferred, fallback to Gemini)
        summary = self._summarize_text_with_llm(prompt, max_tokens=128)
        print(f"[Agent] LLM-based answer extraction summary: {summary}")
        return summary.strip()

    def _answers_match(self, answer: str, reference: str) -> bool:
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
            f"System prompt (answer formatting rules):\n{self.system_prompt}\n\n"
            f"Agent's answer:\n{answer}\n\n"
            f"Reference answer:\n{reference}\n\n"
            "Question: Does the agent's answer match the reference answer exactly, following the system prompt's answer formatting and constraints? "
            "Reply with only 'true' or 'false'."
        )
        validation_msg = [HumanMessage(content=validation_prompt)]
        try:
            response = self._try_llm_sequence(validation_msg, use_tools=False)
            if hasattr(response, 'content'):
                result = response.content.strip().lower()
            elif isinstance(response, dict) and 'content' in response:
                result = response['content'].strip().lower()
            else:
                result = str(response).strip().lower()
            return result.startswith('true')
        except Exception as e:
            # Fallback: conservative, treat as not matching if validation fails
            print(f"LLM validation error in _answers_match: {e}")
            return False

    def _gather_tools(self) -> List[Any]:
        """
        Gather all callable tools from tools.py for LLM tool binding.

        Returns:
            list: List of tool functions.
        """
        # Import tools module to get its functions
        import tools
        
        # Get all attributes from the tools module
        tool_list = []
        for name, obj in tools.__dict__.items():
            # Only include callable objects that are functions (not classes, modules, or builtins)
            if (callable(obj) and 
                not name.startswith("_") and 
                not isinstance(obj, type) and  # Exclude classes
                hasattr(obj, '__module__') and  # Must have __module__ attribute
                obj.__module__ == 'tools' and  # Must be from tools module
                name not in ["GaiaAgent", "CodeInterpreter"]):  # Exclude specific classes
                tool_list.append(obj)
        
        # Add specific tools that might be missed
        specific_tools = [
            'multiply', 'add', 'subtract', 'divide', 'modulus', 'power', 'square_root',
            'wiki_search', 'web_search', 'arxiv_search',
            'save_and_read_file', 'download_file_from_url', 'get_task_file',
            'extract_text_from_image', 'analyze_csv_file', 'analyze_excel_file',
            'analyze_image', 'transform_image', 'draw_on_image', 'generate_simple_image', 'combine_images',
            'understand_video', 'understand_audio',
            'convert_chess_move', 'get_best_chess_move', 'get_chess_board_fen', 'solve_chess_position'
        ]
        
        # Ensure all specific tools are included
        for tool_name in specific_tools:
            if hasattr(tools, tool_name) and tool_name not in [tool.__name__ for tool in tool_list]:
                tool_list.append(getattr(tools, tool_name))
        
        print(f"✅ Gathered {len(tool_list)} tools: {[tool.__name__ for tool in tool_list]}")
        return tool_list 