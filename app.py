import os
import gradio as gr
import requests
import inspect
import pandas as pd
import random
import datetime
import subprocess
import json
import re
import base64
from agent import GaiaAgent
from file_helper import TRACES_DIR, upload_run_data

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Main Agent Definition ---
# Instantiate the agent once (choose provider as needed)
AGENT_PROVIDER = os.environ.get("AGENT_PROVIDER", "google")
try:
    agent = GaiaAgent(provider=AGENT_PROVIDER)
except Exception as e:
    agent = None
    print(f"Error initializing GaiaAgent: {e}")

# Helper to save DataFrame as CSV and upload via API
def save_df_to_csv(df, path):
    try:
        # Convert DataFrame to CSV string
        csv_content = df.to_csv(index=False, encoding="utf-8")
        
        # Upload via API
        success = save_and_commit_file(
            file_path=path,
            content=csv_content,
            commit_message=f"Add results CSV {path}"
        )
        if success:
            print(f"✅ Results CSV uploaded successfully: {path}")
        else:
            print(f"⚠️ Results CSV upload failed, saved locally only: {path}")
            # Fallback to local save
            df.to_csv(path, index=False, encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Results CSV upload error: {e}, saving locally only")
        # Fallback to local save
        df.to_csv(path, index=False, encoding="utf-8")
    
    return path

# --- Provide init log for download on app load ---
def get_init_log():
    init_log_path = getattr(agent, "init_log_path", None)
    if init_log_path and os.path.exists(init_log_path):
        return init_log_path
    return None

def generate_run_id(timestamp: str, idx: int) -> str:
    """Generate a unique run ID for a question."""
    return f"{timestamp}_q{idx+1:02d}"

def create_run_data_for_runs_new(
    run_id: str,
    idx: int,
    total_questions: int,
    result: dict,
    llm_stats_json: dict,
    username: str = "N/A",
    total_score: str = "N/A"
) -> dict:
    """
    Create run data for the runs_new split.
    
    Args:
        run_id: Unique identifier for the run
        idx: Index of the question in the batch (0-based)
        total_questions: Total number of questions in the batch
        result: Individual result dictionary
        llm_stats_json: LLM statistics JSON
        username: Username of the person running the agent
        total_score: Overall score for the complete evaluation run
        
    Returns:
        dict: Run data for upload to runs_new split
    """
    # Extract trace data from agent result
    trace = result.get("trace", {})
    
    return {
        "run_id": run_id,
        "questions_count": f"{idx+1}/{total_questions}",
        "input_data": json.dumps([{
            "task_id": result.get("task_id", f"task_{idx+1:03d}"),
            "question": result.get("question", ""),
            "file_name": result.get("file_name", "")
        }]),
        "reference_answer": result.get("reference_answer", "Reference answer not found"),  # Reference answer found by agent
        "final_answer": result.get("submitted_answer", ""),  # Use consistent field name
        "reference_similarity": result.get("similarity_score", 0.0),  # Use similarity score from agent
        "question": result.get("question", ""),  # Question text
        "file_name": result.get("file_name", ""),  # File name
        "file_size": trace.get("file_size"),
        "llm_used": result.get("llm_used", "unknown"),  # LLM used
        "llm_stats_json": json.dumps(llm_stats_json),  # LLM statistics JSON
        "total_score": total_score,  # Overall score for the complete evaluation run
        "start_time": trace.get("start_time"),
        "end_time": trace.get("end_time"),
        "total_execution_time": trace.get("total_execution_time"),
        "tokens_total": trace.get("tokens_total", 0),
        "llm_traces_json": json.dumps(trace.get("llm_traces", {})),
        "logs_json": json.dumps(trace.get("logs", [])),
        "per_llm_stdout_json": json.dumps(trace.get("per_llm_stdout", [])),
        "error": result.get("error", ""),  # Error information
        "username": username.strip() if username else "unknown"
    }

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the GaiaAgent on them, submits all answers,
    and displays the results.
    """
    space_id = os.getenv("SPACE_ID")
    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent (already done globally)
    if agent is None:
        return "Error initializing agent. Check logs for details.", None
    agent_code = f"https://huggingface.co/spaces/{username}/agent-course-final-assignment/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run the Agent
    results_log = []
    answers_payload = []
    print(f"Running GaiaAgent on {len(questions_data)} questions...")
    
    # DEBUG: Select one random task instead of all
    questions_data = random.sample(questions_data, len(questions_data))
    #questions_data = [questions_data[0]]
    
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name", "")  # Extract file_name from question data
        
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        
        # Download file if one is referenced
        file_data = None
        if file_name and file_name.strip():
            try:
                print(f"\U0001F4C1 Downloading file: {file_name} for task {task_id}")
                file_url = f"{api_url}/files/{task_id}"
                file_response = requests.get(file_url, timeout=30)
                file_response.raise_for_status()
                
                # Convert file to base64
                file_data = base64.b64encode(file_response.content).decode('utf-8')
                print(f"✅ Downloaded and encoded file: {file_name} ({len(file_data)} chars)")
            except Exception as e:
                print(f"⚠️ Failed to download file {file_name} for task {task_id}: {e}")
                file_data = None
        
        try:
            # Pass both question text and file data to agent
            if file_data:
                # Create enhanced question with file context
                enhanced_question = f"{question_text}\n\n[File attached: {file_name} - base64 encoded data available]"
                agent_result = agent(enhanced_question, file_data=file_data, file_name=file_name)
            else:
                agent_result = agent(question_text)
            
            # Extract answer and additional info from agent result
            submitted_answer = agent_result.get("submitted_answer", "No answer provided")
            reference_similarity = agent_result.get("similarity_score", 0.0)
            llm_used = agent_result.get("llm_used", "unknown")
            reference_answer = agent_result.get("reference", "Reference answer not found")
            question_text = agent_result.get("question", "")
            file_name = agent_result.get("file_name", "")
            error = agent_result.get("error", "")
            
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "task_id": task_id, 
                "question": question_text, 
                "file_name": file_name, 
                "submitted_answer": submitted_answer,
                "reference_answer": reference_answer,
                "reference_similarity": reference_similarity,
                "llm_used": llm_used,
                "error": error
            })
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({
                "task_id": task_id, 
                "question": question_text, 
                "file_name": file_name, 
                "submitted_answer": f"AGENT ERROR: {e}",
                "reference_answer": reference_answer,
                "reference_similarity": 0.0,
                "llm_used": "none",
                "error": str(e)
            })

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # --- Save results log to logs/ folder with timestamp ---
    #log_path = save_results_log(results_log)  # Re-enabled with API support

    # --- Save results table as CSV for download ---
    results_df = pd.DataFrame(results_log)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Upload each question as a separate run record to runs_new dataset
    successful_uploads = 0
    for idx, result in enumerate(results_log):
        try:
            run_id = generate_run_id(timestamp, idx)
            
            # Get LLM stats JSON for this run
            llm_stats_json = agent._get_llm_stats_json()
            
            # Create run data for runs_new split
            run_data = create_run_data_for_runs_new(
                run_id,
                idx,
                len(results_log),
                result,
                llm_stats_json,
                username,
                "N/A"  # Initial upload - score not available yet
            )
            
            success = upload_run_data(run_data, split="runs_new")
            if success:
                print(f"✅ Question {idx+1} uploaded to runs_new dataset: {run_id}")
                successful_uploads += 1
            else:
                print(f"⚠️ Failed to upload question {idx+1} to runs_new dataset")
                
        except Exception as e:
            print(f"⚠️ Failed to upload question {idx+1}: {e}")
    
    print(f"📊 Uploaded {successful_uploads}/{len(results_log)} questions to runs_new dataset")
    
    # Log complete evaluation run status
    if successful_uploads == len(results_log):
        print(f"✅ Complete evaluation run uploaded to dataset: {timestamp}")
    else:
        print(f"⚠️ Failed to upload complete evaluation run to dataset")

    # 4. Prepare Submission
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        # Extract just the score percentage from the result data
        total_score = f"{result_data.get('score', 'N/A')}% ({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)"
        
        # Update all run records with final status and score result
        for idx, result in enumerate(results_log):
            try:
                run_id = generate_run_id(timestamp, idx)
                
                # Get LLM stats JSON for this run
                llm_stats_json = agent._get_llm_stats_json()
                
                # Create updated run data for this question
                run_data = create_run_data_for_runs_new(
                    run_id,
                    idx,
                    len(results_log),
                    result,
                    llm_stats_json,
                    username,
                    total_score  # Use actual score from submission
                )
                
                success = upload_run_data(run_data, split="runs_new")
                if success:
                    print(f"✅ Updated question {idx+1} with final results: {run_id}")
                else:
                    print(f"⚠️ Failed to update question {idx+1} with final results")
                    
            except Exception as e:
                print(f"⚠️ Failed to update question {idx+1} with final results: {e}")
        
        # Log complete evaluation run update status
        print(f"✅ Complete evaluation run updated with final results: {timestamp}")
            
        return final_status, results_df
    except Exception as e:
        status_message = f"Submission Failed: {e}"
        print(status_message)
        # Set error score result
        total_score = "N/A (Submission Failed)"
        
        # Update all run records with error status and score result
        for idx, result in enumerate(results_log):
            try:
                run_id = generate_run_id(timestamp, idx)
                
                # Get LLM stats JSON for this run
                llm_stats_json = agent._get_llm_stats_json()
                
                # Create updated run data for this question
                run_data = create_run_data_for_runs_new(
                    run_id,
                    idx,
                    len(results_log),
                    result,
                    llm_stats_json,
                    username,
                    total_score  # Use error score result
                )
                
                success = upload_run_data(run_data, split="runs_new")
                if success:
                    print(f"✅ Updated question {idx+1} with error results: {run_id}")
                else:
                    print(f"⚠️ Failed to update question {idx+1} with error results")
                    
            except Exception as upload_e:
                print(f"⚠️ Failed to update question {idx+1} with error results: {upload_e}")
        
        # Log complete evaluation run update status
        print(f"⚠️ Failed to upload complete evaluation run: {e}")
            
        return status_message, results_df

def get_dataset_stats_html():
    """
    Get dataset statistics and return as HTML.
    """
    try:
        from datasets import load_dataset
        
        # Load each config separately
        configs = ['init', 'runs_new']
        stats_html = "<div style='margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 8px;'>"
        stats_html += "<h3>📊 Dataset Statistics</h3>"
        
        for config_name in configs:
            try:
                # Load specific config
                config_data = load_dataset("arterm-sedov/agent-course-final-assignment", config_name)
                
                stats_html += f"<div style='margin: 15px 0; padding: 10px; background: #e9ecef; border-radius: 5px;'>"
                stats_html += f"<h4>🔧 Config: {config_name.upper()}</h4>"
                
                # Get statistics for each split in this config
                for split_name in config_data.keys():
                    split_data = config_data[split_name]
                    stats_html += f"<div style='margin: 8px 0;'>"
                    stats_html += f"<strong>{split_name.upper()} Split:</strong> {len(split_data)} records"
                    stats_html += "</div>"
                
                # Add latest run info for runs_new config
                if config_name == "runs_new" and "default" in config_data:
                    runs_new_data = config_data["default"]
                    if len(runs_new_data) > 0:
                        latest_run = runs_new_data[-1]
                        stats_html += f"<div style='margin: 10px 0; padding: 8px; background: #d4edda; border-radius: 3px;'>"
                        stats_html += f"<strong>Latest Run:</strong> {latest_run.get('run_id', 'N/A')}"
                        stats_html += f"<br><strong>Total Score:</strong> {latest_run.get('total_score', 'N/A')}"
                        stats_html += f"<br><strong>Username:</strong> {latest_run.get('username', 'N/A')}"
                        stats_html += "</div>"
                
                stats_html += "</div>"
                
            except Exception as config_error:
                stats_html += f"<div style='margin: 15px 0; padding: 10px; background: #f8d7da; border-radius: 5px;'>"
                stats_html += f"<h4>❌ Config: {config_name.upper()}</h4>"
                stats_html += f"<div style='margin: 8px 0; color: #721c24;'>Error loading config: {config_error}</div>"
                stats_html += "</div>"
        
        stats_html += "</div>"
        return stats_html
        
    except Exception as e:
        return f"<div style='margin: 20px 0; padding: 15px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;'>⚠️ Could not load dataset statistics: {e}</div>"

def get_logs_html():
    logs_dir = "logs"
    rows = []
    files = []
    
    # Get space ID for repository links
    space_id = os.getenv("SPACE_ID", "arterm-sedov/agent-course-final-assignment")
    repo_base_url = f"https://huggingface.co/spaces/{space_id}/resolve/main"
    
    if os.path.exists(logs_dir):
        for fname in os.listdir(logs_dir):
            fpath = os.path.join(logs_dir, fname)
            if os.path.isfile(fpath):
                timestamp, dt = extract_timestamp_from_filename(fname)
                if not dt:
                    # Fallback to modification time for files without timestamp in filename
                    dt = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
                    timestamp = dt.strftime('%Y-%m-%d %H:%M:%S (mtime)')
                files.append((fname, timestamp, dt, fpath))
        # Sort all files by datetime descending (newest first)
        files.sort(key=lambda x: x[2], reverse=True)
        for fname, timestamp, dt, fpath in files:
            # Create repository download link
            repo_download_url = f"{repo_base_url}/logs/{fname}?download=true"
            download_link = f'<a href="{repo_download_url}" target="_blank" rel="noopener noreferrer">Download from Repo</a>'
            date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            rows.append(f"<tr><td>{fname}</td><td>{date_str}</td><td>{download_link}</td></tr>")
    
    table_html = (
        "<table border='1' style='width:100%;border-collapse:collapse;'>"
        "<thead><tr><th>File Name</th><th>Date/Time</th><th>Download</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )
    return table_html

def extract_timestamp_from_filename(filename):
    """
    Extract timestamp from filename using comprehensive regex patterns for all log formats in @/logs.
    Returns (timestamp_str, datetime_obj) or (None, None) if no timestamp found.
    """
    import re
    
    # Handle multiple extensions by removing all extensions
    name = filename
    while '.' in name:
        name = os.path.splitext(name)[0]
    
    # 1. 14-digit datetime: YYYYMMDDHHMMSS (must be exact 14 digits)
    m = re.match(r'^(\d{14})$', name)
    if m:
        timestamp_str = m.group(1)
        try:
            dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            return timestamp_str, dt
        except ValueError:
            pass
    
    # 2. Leaderboard format: 2025-07-02 090007
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})[ _]+(\d{2})(\d{2})(\d{2})', name)
    if m:
        y, mo, d, h, mi, s = m.groups()
        try:
            dt = datetime.datetime.strptime(f"{y}{mo}{d}{h}{mi}{s}", "%Y%m%d%H%M%S")
            return f"{y}-{mo}-{d} {h}:{mi}:{s}", dt
        except ValueError:
            pass
    
    # 3. LOG prefix with 12-digit timestamp: LOG202506281412
    m = re.match(r'^LOG(\d{12})$', name)
    if m:
        timestamp_str = m.group(1)
        try:
            dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            return f"LOG{timestamp_str}", dt
        except ValueError:
            pass
    
    # 4. LOG prefix with 8-digit date and optional suffix: LOG20250628_2, LOG20250629_1
    m = re.match(r'^LOG(\d{8})(?:_(\d+))?$', name)
    if m:
        date_str, suffix = m.groups()
        try:
            dt = datetime.datetime.strptime(date_str, "%Y%m%d")
            timestamp_str = f"LOG{date_str}"
            if suffix:
                timestamp_str += f"_{suffix}"
            return timestamp_str, dt
        except ValueError:
            pass
    
    # 5. INIT prefix with date and time: INIT_20250704_000343
    m = re.match(r'^INIT_(\d{8})_(\d{6})$', name)
    if m:
        date_str, time_str = m.groups()
        try:
            dt = datetime.datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            return f"INIT_{date_str}_{time_str}", dt
        except ValueError:
            pass
    
    # 6. Date with underscore and time: 20250702_202757, 20250703_135654
    m = re.match(r'^(\d{8})_(\d{6})$', name)
    if m:
        date_str, time_str = m.groups()
        try:
            dt = datetime.datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            return f"{date_str}_{time_str}", dt
        except ValueError:
            pass
    
    # 7. Date only (8 digits): 20250628
    m = re.match(r'^(\d{8})$', name)
    if m:
        date_str = m.group(1)
        try:
            dt = datetime.datetime.strptime(date_str, "%Y%m%d")
            return date_str, dt
        except ValueError:
            pass
    
    # 8. Files with no timestamp pattern (like "Score 60.log")
    # These will return None and fall back to modification time
    
    return None, None

def save_results_log(results_log: list) -> str:
    """
    Save the complete results log to a file and upload via API.
    
    Args:
        results_log (list): List of dictionaries containing task results
        
    Returns:
        str: Path to the saved log file, or None if failed
    """
    try:
        # Create traces directory if it doesn't exist
        os.makedirs(TRACES_DIR, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare log content
        log_content = json.dumps(results_log, indent=2, ensure_ascii=False)
        log_path = f"{TRACES_DIR}/{timestamp}_llm_trace.log"
        
        # Upload via API
        # try:
        #     success = save_and_commit_file(
        #         file_path=log_path,
        #         content=log_content,
        #         commit_message=f"Add LLM trace log {timestamp}"
        #     )
        #     if success:
        #         print(f"✅ LLM trace log uploaded successfully: {log_path}")
        #     else:
        #         print(f"⚠️ LLM trace log upload failed, saved locally only: {log_path}")
        #         # Fallback to local save
        #         with open(log_path, "w", encoding="utf-8") as f:
        #             f.write(log_content)
        # except Exception as e:
        #     print(f"⚠️ LLM trace log upload error: {e}, saving locally only")
        #     # Fallback to local save
        #     with open(log_path, "w", encoding="utf-8") as f:
        #         f.write(log_content)
        
        return log_path
        
    except Exception as e:
        print(f"⚠️ Failed to save results log: {e}")
        return None



# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Unit 4 Agent Evaluation Runner")
    gr.Markdown(
        """
       
        **Instructions:**
        **If you want to test the agent**
        
        1.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.
        
        **If you want to copy the agent**
        
        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit" button, it can take quite some time (this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a separate action or even to answer the questions in async.
        """
    )

    with gr.Tabs():
        with gr.TabItem("Readme"):
            gr.Markdown(open("README.md", "r", encoding="utf-8").read())
        with gr.TabItem("Evaluation"):
            gr.LoginButton()
            run_button = gr.Button("Run Evaluation & Submit All Answers")
            status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
            results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
            # Note: get_init_log() returns a value but demo.load() doesn't expect outputs
            # This is just for initialization, so we ignore the return value
            demo.load(
                fn=lambda: None,  # Use a no-op function instead
                inputs=[]
            )
            run_button.click(
                fn=run_and_submit_all,
                outputs=[status_output, results_table]
            )
        with gr.TabItem("Results dataset"):
            gr.Markdown("## Dataset statistics")
            dataset_stats_output = gr.HTML(get_dataset_stats_html())
            refresh_stats_btn = gr.Button("🔄 Refresh Dataset Statistics")
            refresh_stats_btn.click(fn=get_dataset_stats_html, outputs=dataset_stats_output)
            
            gr.Markdown("## dataset viewer")
            gr.Markdown(
                """
                ### Live Dataset viewer
                View the latest evaluation runs uploaded to the HuggingFace dataset.
                
                **Dataset URL:** [arterm-sedov/agent-course-final-assignment](https://huggingface.co/datasets/arterm-sedov/agent-course-final-assignment)
                
                **Runs New Split:** [View Latest Runs](https://huggingface.co/datasets/arterm-sedov/agent-course-final-assignment/viewer/runs_new)
                
                > **Note:** The dataset viewer may show schema conflicts between different splits (init, runs, runs_new). This is expected as each split has different schemas. The `runs_new` split contains the latest granular evaluation data.
                """
            )
            
            # Embed the dataset viewer
            dataset_viewer_html = """
            <div style="width: 100%; height: 600px; border: 1px solid #ccc; border-radius: 8px; overflow: hidden;">
                <iframe 
                    src="https://huggingface.co/datasets/arterm-sedov/agent-course-final-assignment/viewer/default/runs_new" 
                    width="100%" 
                    height="100%" 
                    frameborder="0"
                    style="border: none;"
                    title="Dataset Viewer">
                </iframe>
            </div>
            """
            gr.HTML(dataset_viewer_html)
        with gr.TabItem("Logs"):
            gr.Markdown("## Logs download links")
            gr.HTML(get_logs_html())

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for GAIA Unit 4 Agent Evaluation...")
    
    demo.launch(debug=True, share=False)