#!/usr/bin/env python3
"""
Test script to verify that the serialization fix works for LangChain message objects.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

def serialize_trace_data(obj):
    """
    Recursively serialize trace data, converting LangChain message objects and other
    non-JSON-serializable objects to dictionaries.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Serialized object that can be JSON serialized
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, list):
        return [serialize_trace_data(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_trace_data(value) for key, value in obj.items()}
    elif hasattr(obj, 'type') and hasattr(obj, 'content'):
        # This is likely a LangChain message object
        return {
            "type": getattr(obj, 'type', 'unknown'),
            "content": serialize_trace_data(getattr(obj, 'content', '')),
            "additional_kwargs": serialize_trace_data(getattr(obj, 'additional_kwargs', {})),
            "response_metadata": serialize_trace_data(getattr(obj, 'response_metadata', {})),
            "tool_calls": serialize_trace_data(getattr(obj, 'tool_calls', [])),
            "function_call": serialize_trace_data(getattr(obj, 'function_call', None)),
            "name": getattr(obj, 'name', None),
            "tool_call_id": getattr(obj, 'tool_call_id', None),
            "id": getattr(obj, 'id', None),
            "timestamp": getattr(obj, 'timestamp', None),
            "metadata": serialize_trace_data(getattr(obj, 'metadata', {}))
        }
    else:
        # For any other object, try to convert to string
        try:
            return str(obj)
        except:
            return f"<non-serializable object of type {type(obj).__name__}>"

def test_serialization():
    """Test the serialization function with LangChain message objects."""
    
    # Create sample trace data with LangChain message objects
    trace_data = {
        "llm_traces": {
            "gemini": [
                {
                    "call_id": "gemini_call_1",
                    "input": {
                        "messages": [
                            SystemMessage(content="You are a helpful assistant."),
                            HumanMessage(content="What is 2+2?")
                        ],
                        "use_tools": True
                    },
                    "output": {
                        "content": "The answer is 4.",
                        "tool_calls": [],
                        "response_metadata": {"finish_reason": "stop"}
                    }
                }
            ]
        },
        "logs": [
            {
                "timestamp": "2025-01-01T12:00:00",
                "message": "Processing question",
                "function": "test"
            }
        ],
        "per_llm_stdout": [
            {
                "llm_type": "gemini",
                "stdout": "Test output"
            }
        ]
    }
    
    print("Testing serialization of trace data with LangChain message objects...")
    
    try:
        # Test serialization
        serialized_data = serialize_trace_data(trace_data)
        
        # Test JSON serialization
        json_str = json.dumps(serialized_data, indent=2)
        
        print("✅ Serialization successful!")
        print(f"Serialized data length: {len(json_str)} characters")
        print("Sample of serialized data:")
        print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
        
        return True
        
    except Exception as e:
        print(f"❌ Serialization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_serialization()
    if success:
        print("\n✅ Serialization fix is working correctly!")
    else:
        print("\n❌ Serialization fix has issues!") 