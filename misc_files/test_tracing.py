#!/usr/bin/env python3
"""
Test script for the new contextual tracing system.
"""

import os
import sys
import json
from agent import GaiaAgent

def test_tracing():
    """Test the contextual tracing system with a simple question."""
    
    # Initialize the agent
    agent = GaiaAgent(provider="groq")
    
    # Test question
    question = "What is 2 + 2?"
    
    print("🔍 Testing contextual tracing system...")
    print(f"Question: {question}")
    
    # Process the question
    result = agent(question)
    
    print("\n📊 Results:")
    print(f"Answer: {result['answer']}")
    print(f"LLM Used: {result['llm_used']}")
    print(f"Similarity Score: {result['similarity_score']}")
    
    # Get the full trace
    trace = result.get("trace")
    
    print("\n📋 Trace Structure Analysis:")
    if trace:
        print(f"Question: {trace['question']}")
        print(f"Start Time: {trace['start_time']}")
        print(f"End Time: {trace['end_time']}")
        print(f"Total Execution Time: {trace.get('total_execution_time', 'N/A')}s")
        
        # Show LLM traces with contextual logs
        for llm_type, calls in trace['llm_traces'].items():
            print(f"\n🤖 {llm_type.upper()} Calls: {len(calls)}")
            for i, call in enumerate(calls):
                print(f"  Call {i+1}: {call['call_id']}")
                print(f"    Execution Time: {call.get('execution_time', 'N/A')}s")
                print(f"    Tool Executions: {len(call.get('tool_executions', []))}")
                
                # Show LLM call logs
                if call.get('logs'):
                    print(f"    LLM Call Logs: {len(call['logs'])} entries")
                    for log in call['logs'][:2]:  # Show first 2 logs
                        print(f"      [{log['timestamp']}] {log['message']}")
                
                # Show tool execution logs
                for tool_exec in call.get('tool_executions', []):
                    if tool_exec.get('logs'):
                        print(f"    Tool '{tool_exec['tool_name']}' Logs: {len(tool_exec['logs'])} entries")
                        for log in tool_exec['logs'][:1]:  # Show first log
                            print(f"      [{log['timestamp']}] {log['message']}")
                
                if call.get('error'):
                    print(f"    Error: {call['error']['message']}")
        
        # Show tool loop data with logs
        for llm_type, calls in trace['llm_traces'].items():
            for call in calls:
                if call.get('tool_loop_data'):
                    print(f"\n🔄 Tool Loop Data for {call['call_id']}:")
                    for loop_data in call['tool_loop_data']:
                        print(f"  Step {loop_data['step']}: {loop_data['tool_calls_detected']} tool calls")
                        if loop_data.get('logs'):
                            print(f"    Loop Logs: {len(loop_data['logs'])} entries")
                            for log in loop_data['logs'][:1]:  # Show first log
                                print(f"      [{log['timestamp']}] {log['message']}")
        
        # Show final answer enforcement logs
        if trace.get('final_answer_enforcement'):
            print(f"\n🎯 Final Answer Enforcement Logs: {len(trace['final_answer_enforcement'])} entries")
            for log in trace['final_answer_enforcement'][:2]:  # Show first 2 logs
                print(f"  [{log['timestamp']}] {log['message']}")
        
        # Show question-level logs
        if trace.get('logs'):
            print(f"\n📝 Question-Level Logs: {len(trace['logs'])} entries")
            for log in trace['logs'][:2]:  # Show first 2 logs
                print(f"  [{log['timestamp']}] {log['message']}")
        
        # Show final result
        if trace.get('final_result'):
            print(f"\n✅ Final Result: {trace['final_result']['llm_used']}")
        
        # Show complete stdout (last object)
        if trace.get('complete_stdout'):
            print(f"\n📄 Complete Stdout (Length: {len(trace['complete_stdout'])} chars)")
            print("First 200 chars:")
            print(trace['complete_stdout'][:200] + "...")
    else:
        print("❌ No trace available")
    
    # Clear the trace
    agent._trace_clear()
    print("\n🧹 Trace cleared")

def test_contextual_logging():
    """Test the contextual logging structure."""
    
    print("\n🔬 Testing Contextual Logging Structure...")
    
    # Initialize the agent
    agent = GaiaAgent(provider="groq")
    
    # Simple question that will trigger tool usage
    question = "What is the capital of France?"
    
    # Process the question
    result = agent(question)
    trace = result.get("trace")
    
    if trace:
        print("\n📊 Contextual Logging Analysis:")
        
        # Analyze LLM traces
        for llm_type, calls in trace['llm_traces'].items():
            print(f"\n🤖 {llm_type.upper()} Context:")
            for call in calls:
                print(f"  Call ID: {call['call_id']}")
                
                # LLM call logs
                if call.get('logs'):
                    print(f"    LLM Call Logs ({len(call['logs'])}):")
                    for log in call['logs']:
                        print(f"      [{log['function']}] {log['message']}")
                
                # Tool execution logs
                for tool_exec in call.get('tool_executions', []):
                    print(f"    Tool: {tool_exec['tool_name']}")
                    if tool_exec.get('logs'):
                        print(f"      Tool Execution Logs ({len(tool_exec['logs'])}):")
                        for log in tool_exec['logs']:
                            print(f"        [{log['function']}] {log['message']}")
        
        # Final answer enforcement logs
        if trace.get('final_answer_enforcement'):
            print(f"\n🎯 Final Answer Enforcement Context:")
            for log in trace['final_answer_enforcement']:
                print(f"  [{log['function']}] {log['message']}")
    
    agent._trace_clear()

if __name__ == "__main__":
    test_tracing()
    test_contextual_logging() 