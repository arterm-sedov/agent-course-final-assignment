#!/usr/bin/env python3
"""
Script to convert log files to init JSON files for the dataset.
Extracts timestamp, init_summary, debug_output, and other required fields.
Includes validation, sanitization, and integration with file_helper.
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add parent directory to path to import file_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from file_helper import validate_data_structure, upload_init_summary, get_dataset_features

def sanitize_text(text: str, max_length: int = 100000) -> str:
    """Sanitize text content to prevent issues."""
    if not text:
        return ""
    
    # Remove null bytes and other problematic characters
    text = text.replace('\x00', '')
    text = text.replace('\r', '\n')
    
    # Normalize line endings
    text = text.replace('\r\n', '\n')
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "\n... [TRUNCATED]"
    
    return text

def validate_timestamp(timestamp: str) -> bool:
    """Validate timestamp format."""
    try:
        # Check if it matches expected format YYYYMMDD_HHMMSS
        if not re.match(r'^\d{8}_\d{6}$', timestamp):
            return False
        
        # Try to parse the timestamp
        datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        return True
    except ValueError:
        return False

def extract_timestamp_from_filename(filename: str) -> str:
    """Extract timestamp from filename with validation."""
    # Handle different filename patterns
    patterns = [
        r'(\d{8}_\d{6})_init\.log',  # 20250705_130855_init.log
        r'INIT_(\d{8}_\d{6})\.log',  # INIT_20250703_122618.log
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            timestamp = match.group(1)
            if validate_timestamp(timestamp):
                return timestamp
    
    # Fallback: use current timestamp
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_init_summary(log_content: str) -> str:
    """Extract the initialization summary from log content."""
    # Look for the summary section
    summary_pattern = r'===== LLM Initialization Summary =====\n(.*?)\n======================================================================================================'
    match = re.search(summary_pattern, log_content, re.DOTALL)
    
    if match:
        summary = match.group(1).strip()
        # Clean up the summary
        lines = summary.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip() and not line.startswith('---'):
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    return "No initialization summary found"

def extract_llm_config() -> str:
    """Generate LLM configuration JSON string."""
    config = {
        "default": {
            "type_str": "default",
            "token_limit": 2500,
            "max_history": 15,
            "tool_support": False,
            "force_tools": False,
            "models": []
        },
        "gemini": {
            "name": "Google Gemini",
            "type_str": "gemini",
            "api_key_env": "GEMINI_KEY",
            "max_history": 25,
            "tool_support": True,
            "force_tools": True,
            "models": [
                {
                    "model": "gemini-2.5-pro",
                    "token_limit": 2000000,
                    "max_tokens": 2000000,
                    "temperature": 0
                }
            ]
        },
        "groq": {
            "name": "Groq",
            "type_str": "groq",
            "api_key_env": "GROQ_API_KEY",
            "max_history": 15,
            "tool_support": True,
            "force_tools": True,
            "models": [
                {
                    "model": "qwen-qwq-32b",
                    "token_limit": 3000,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "force_tools": True
                }
            ]
        },
        "huggingface": {
            "name": "HuggingFace",
            "type_str": "huggingface",
            "api_key_env": "HUGGINGFACEHUB_API_TOKEN",
            "max_history": 20,
            "tool_support": False,
            "force_tools": False,
            "models": [
                {
                    "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "task": "text-generation",
                    "token_limit": 1000,
                    "max_new_tokens": 1024,
                    "do_sample": False,
                    "temperature": 0
                },
                {
                    "repo_id": "microsoft/DialoGPT-medium",
                    "task": "text-generation",
                    "token_limit": 1000,
                    "max_new_tokens": 512,
                    "do_sample": False,
                    "temperature": 0
                },
                {
                    "repo_id": "gpt2",
                    "task": "text-generation",
                    "token_limit": 1000,
                    "max_new_tokens": 256,
                    "do_sample": False,
                    "temperature": 0
                }
            ]
        },
        "openrouter": {
            "name": "OpenRouter",
            "type_str": "openrouter",
            "api_key_env": "OPENROUTER_API_KEY",
            "api_base_env": "OPENROUTER_BASE_URL",
            "max_history": 20,
            "tool_support": True,
            "force_tools": False,
            "models": [
                {
                    "model": "deepseek/deepseek-chat-v3-0324:free",
                    "token_limit": 100000,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "mistralai/mistral-small-3.2-24b-instruct:free",
                    "token_limit": 90000,
                    "max_tokens": 2048,
                    "temperature": 0
                }
            ]
        }
    }
    return json.dumps(config)

def extract_available_models() -> str:
    """Generate available models JSON string."""
    models = {
        "gemini": {
            "name": "Google Gemini",
            "models": [
                {
                    "model": "gemini-2.5-pro",
                    "token_limit": 2000000,
                    "max_tokens": 2000000,
                    "temperature": 0
                }
            ],
            "tool_support": True,
            "max_history": 25
        },
        "groq": {
            "name": "Groq",
            "models": [
                {
                    "model": "qwen-qwq-32b",
                    "token_limit": 3000,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "force_tools": True
                }
            ],
            "tool_support": True,
            "max_history": 15
        },
        "huggingface": {
            "name": "HuggingFace",
            "models": [
                {
                    "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "task": "text-generation",
                    "token_limit": 1000,
                    "max_new_tokens": 1024,
                    "do_sample": False,
                    "temperature": 0
                },
                {
                    "repo_id": "microsoft/DialoGPT-medium",
                    "task": "text-generation",
                    "token_limit": 1000,
                    "max_new_tokens": 512,
                    "do_sample": False,
                    "temperature": 0
                },
                {
                    "repo_id": "gpt2",
                    "task": "text-generation",
                    "token_limit": 1000,
                    "max_new_tokens": 256,
                    "do_sample": False,
                    "temperature": 0
                }
            ],
            "tool_support": False,
            "max_history": 20
        },
        "openrouter": {
            "name": "OpenRouter",
            "models": [
                {
                    "model": "deepseek/deepseek-chat-v3-0324:free",
                    "token_limit": 100000,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "mistralai/mistral-small-3.2-24b-instruct:free",
                    "token_limit": 90000,
                    "max_tokens": 2048,
                    "temperature": 0
                }
            ],
            "tool_support": True,
            "max_history": 20
        }
    }
    return json.dumps(models)

def extract_tool_support() -> str:
    """Generate tool support JSON string."""
    tool_support = {
        "gemini": {
            "tool_support": True,
            "force_tools": True
        },
        "groq": {
            "tool_support": True,
            "force_tools": True
        },
        "huggingface": {
            "tool_support": False,
            "force_tools": False
        },
        "openrouter": {
            "tool_support": True,
            "force_tools": False
        }
    }
    return json.dumps(tool_support)

def validate_init_data(data: Dict) -> List[str]:
    """Validate init data and return list of issues."""
    issues = []
    
    # Check required fields
    required_fields = ["timestamp", "init_summary", "debug_output", "llm_config", "available_models", "tool_support"]
    for field in required_fields:
        if field not in data:
            issues.append(f"Missing required field: {field}")
    
    # Validate timestamp
    if "timestamp" in data and not validate_timestamp(data["timestamp"]):
        issues.append(f"Invalid timestamp format: {data['timestamp']}")
    
    # Check data types
    for field in ["init_summary", "debug_output", "llm_config", "available_models", "tool_support"]:
        if field in data and not isinstance(data[field], str):
            issues.append(f"Field {field} must be a string")
    
    # Validate JSON strings
    for field in ["llm_config", "available_models", "tool_support"]:
        if field in data:
            try:
                json.loads(data[field])
            except json.JSONDecodeError:
                issues.append(f"Invalid JSON in field {field}")
    
    return issues

def process_log_file(log_file_path: str, output_dir: str = "dataset", upload_to_hf: bool = False) -> Optional[str]:
    """Process a single log file and create corresponding init JSON file."""
    try:
        print(f"Processing: {os.path.basename(log_file_path)}")
        # Read log file
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        # Sanitize log content
        log_content = sanitize_text(log_content)
        # Extract timestamp from filename
        filename = os.path.basename(log_file_path)
        timestamp = extract_timestamp_from_filename(filename)
        # Extract init summary
        init_summary = extract_init_summary(log_content)
        # Create init JSON object
        init_data = {
            "timestamp": str(timestamp),
            "init_summary": sanitize_text(str(init_summary), max_length=10000),
            "debug_output": str(log_content),
            "llm_config": str(extract_llm_config()),
            "available_models": str(extract_available_models()),
            "tool_support": str(extract_tool_support())
        }
        # Validate data structure
        validation_issues = validate_init_data(init_data)
        if validation_issues:
            print(f"  ⚠️  Validation issues:")
            for issue in validation_issues:
                print(f"    - {issue}")
        # Validate against dataset schema
        if not validate_data_structure(init_data, "init"):
            print(f"  ❌ Data does not match dataset schema, skipping file.")
            return None
        # Create output filename
        output_filename = f"init-{timestamp}.jsonl"
        output_path = os.path.join(output_dir, output_filename)
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Write JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(init_data))
        print(f"  ✅ Created {output_filename} at {os.path.abspath(output_path)}")
        # Upload to HuggingFace if requested
        if upload_to_hf:
            print(f"  📤 Uploading to HuggingFace dataset...")
            if upload_init_summary(init_data):
                print(f"  ✅ Uploaded to HuggingFace dataset")
            else:
                print(f"  ❌ Failed to upload to HuggingFace dataset")
        return output_path
    except Exception as e:
        print(f"  ❌ Error processing {log_file_path}: {e}")
        return None

def main():
    """Main function to process all log files."""
    import argparse
    parser = argparse.ArgumentParser(description="Convert log files to init JSON files")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace dataset")
    parser.add_argument("--output-dir", default="dataset", help="Output directory for JSON files")
    args = parser.parse_args()
    # List of log files to process - try multiple possible paths
    log_files = [
        "logs/20250705_130855_init.log",
        "logs/20250705_131128_init.log",
        "logs/20250705_131406_init.log",
        "logs/20250705_131525_init.log",
        "logs/20250705_132209_init.log",
        "logs/20250705_131702_init.log",
        "logs/20250705_131903_init.log",
        "logs/20250705_132104_init.log",
        "logs/INIT_20250703_122618.log",
        "logs/INIT_20250703_123454.log",
        "logs/INIT_20250703_124712.log",
        "logs/INIT_20250703_153105.log",
        # Try relative to parent directory (if run from misc_files)
        "../logs/20250705_130855_init.log",
        "../logs/20250705_131128_init.log",
        "../logs/20250705_131406_init.log",
        "../logs/20250705_131525_init.log",
        "../logs/20250705_132209_init.log",
        "../logs/20250705_131702_init.log",
        "../logs/20250705_131903_init.log",
        "../logs/20250705_132104_init.log",
        "../logs/INIT_20250703_122618.log",
        "../logs/INIT_20250703_123454.log",
        "../logs/INIT_20250703_124712.log",
        "../logs/INIT_20250703_153105.log"
    ]
    print("Converting log files to init JSON files...")
    if args.upload:
        print("📤 Will upload to HuggingFace dataset")
    print("=" * 60)
    successful_conversions = 0
    processed_files = set()  # Track which files we've already processed
    for log_file in log_files:
        if os.path.exists(log_file) and log_file not in processed_files:
            result = process_log_file(log_file, args.output_dir, args.upload)
            if result:
                successful_conversions += 1
                processed_files.add(log_file)
    if successful_conversions == 0:
        print("❌ No log files found. Please check the following locations:")
        print("   - logs/ (relative to current directory)")
        print("   - ../logs/ (relative to parent directory)")
        print("   - Check if log files exist in the expected locations")
    else:
        print("=" * 60)
        print(f"Conversion complete: {successful_conversions} files processed successfully")
        if successful_conversions > 0:
            print(f"Output directory: {os.path.abspath(args.output_dir)}")
            print("Files created:")
            for file in os.listdir(args.output_dir):
                if file.startswith("init-") and file.endswith(".jsonl"):
                    print(f"  - {file}")

if __name__ == "__main__":
    main() 