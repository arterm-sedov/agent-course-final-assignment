#!/usr/bin/env python3
"""
Script to split an LLM trace log into one dataset entry per question for the 'runs' split.
Each line in the output .jsonl file is a single question with all required fields.
"""
import os
import sys
import re
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import file_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from file_helper import validate_data_structure

LOG_PATH = "../logs/20250704_035108.llm_trace.log"
SCORE_CSV_PATH = "../logs/20250702_202757.Score.csv"

QUESTION_RE = re.compile(r"🔎 Processing question: (.*)")
FILE_RE = re.compile(r"\[File attached: ([^\s]+) - base64 encoded data available\]")
ANSWER_RE = re.compile(r"🎯 First answer above threshold: (.*)")
EXACT_MATCH_RE = re.compile(r"✅ Exact match after normalization = score [0-9]+")
SUBMITTED_ANSWER_RE = re.compile(r"FINAL ANSWER: (.*)")

# Helper to extract the init sequence
def extract_init_sequence(lines):
    init_start = None
    init_end = None
    for i, line in enumerate(lines):
        if "Initializing LLMs based on sequence:" in line:
            init_start = i
        if init_start is not None and line.startswith("Fetching questions from"):
            init_end = i
            break
    if init_start is not None and init_end is not None:
        return "".join(lines[init_start:init_end]), init_end
    return "", 0

# Helper to find all question blocks
def extract_question_blocks(lines, start_idx):
    question_blocks = []
    current_block = None
    for i in range(start_idx, len(lines)):
        match = QUESTION_RE.match(lines[i])
        if match:
            if current_block:
                question_blocks.append(current_block)
            current_block = {
                "question": match.group(1).strip(),
                "start": i,
                "lines": [lines[i]]
            }
        elif current_block:
            current_block["lines"].append(lines[i])
    if current_block:
        question_blocks.append(current_block)
    return question_blocks

def sanitize_text(text, max_length=100000):
    if not text:
        return ""
    text = text.replace('\x00', '')
    text = text.replace('\r', '\n')
    text = text.replace('\r\n', '\n')
    if len(text) > max_length:
        text = text[:max_length] + "\n... [TRUNCATED]"
    return text

def parse_results_df(block_lines, question, idx):
    # Try to extract fields from the trace
    task_id = str(idx+1)
    file_name = ""
    submitted_answer = ""
    for line in block_lines:
        file_match = FILE_RE.search(line)
        if file_match:
            file_name = file_match.group(1)
        answer_match = ANSWER_RE.search(line)
        if answer_match:
            submitted_answer = answer_match.group(1).strip()
        # Fallback: look for FINAL ANSWER
        if not submitted_answer:
            final_match = SUBMITTED_ANSWER_RE.search(line)
            if final_match:
                submitted_answer = final_match.group(1).strip()
    # Fallbacks
    if not submitted_answer:
        submitted_answer = ""
    return [{
        "Task ID": task_id,
        "Question": question,
        "File": file_name,
        "Submitted Answer": submitted_answer
    }]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split LLM trace log into one dataset entry per question for the runs split")
    parser.add_argument("--log", default=LOG_PATH, help="Path to llm trace log file")
    parser.add_argument("--output-dir", default="../dataset", help="Output directory for JSON files")
    parser.add_argument("--username", default="arterm-sedov", help="Username for the run record")
    parser.add_argument("--score-result", default="13 / 20 (65.0%)", help="Score result string")
    parser.add_argument("--final-status", default="NA", help="Final status string")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"❌ Log file not found: {args.log}")
        return

    with open(args.log, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Extract init sequence
    init_seq, after_init_idx = extract_init_sequence(lines)
    if not init_seq:
        print("❌ Could not extract init sequence!")
        return
    print("✅ Extracted init sequence.")

    # Extract question blocks
    question_blocks = extract_question_blocks(lines, after_init_idx)
    print(f"✅ Found {len(question_blocks)} question blocks.")

    # Compose one run record per question
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"runs-{timestamp}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, block in enumerate(question_blocks):
            run_id = f"run_{Path(args.log).stem}_q{idx+1}_{timestamp}"
            question = block["question"]
            trace = sanitize_text("".join(block["lines"]))
            results_log = [{"question": question, "trace": trace}]
            results_df = parse_results_df(block["lines"], question, idx)
            run_data = {
                "run_id": run_id,
                "timestamp": timestamp,
                "questions_count": 1,
                "results_log": json.dumps(results_log, ensure_ascii=False),
                "results_df": json.dumps(results_df, ensure_ascii=False),
                "username": args.username,
                "final_status": args.final_status,
                "score_result": args.score_result
            }
            if not validate_data_structure(run_data, "runs"):
                print(f"❌ Skipping {run_id}: does not match runs schema.")
                continue
            f.write(json.dumps(run_data, ensure_ascii=False) + "\n")
            print(f"  ✅ Wrote entry for question {idx+1}")
    print(f"Done. {len(question_blocks)} entries written to {out_path}")

if __name__ == "__main__":
    main() 