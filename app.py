import os
import gradio as gr
import requests
import inspect
import pandas as pd
import random
from agent import GaiaAgent
import datetime
import yaml
import subprocess
import glob

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

# Helper to save DataFrame as CSV for download
def save_df_to_csv(df, path):
    # Ensure all columns are string type to avoid truncation or encoding issues
    df = df.astype(str)
    df.to_csv(path, index=False, encoding="utf-8")
    # Explicitly flush and close the file to ensure all data is written
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.flush()
    except Exception:
        pass
    return path

# --- Provide init log for download on app load ---
def get_init_log():
    import os
    init_log_path = getattr(agent, "init_log_path", None)
    if init_log_path and os.path.exists(init_log_path):
        return init_log_path
    return None

# --- Provide latest log files for download on app load ---
def get_latest_logs(state=None):
    """
    Returns the latest log, csv, and score files for download links.
    If state is provided and valid, use it; otherwise, discover from disk.
    """
    import glob
    import os
    if state and isinstance(state, list) and any(state):
        # Use state if available and valid
        return state
    log_dir = "logs"
    if not os.path.exists(log_dir):
        return [None, None, None, None]
    # For init log, use the agent's init_log_path if available
    init_log_path = getattr(agent, "init_log_path", None)
    if not init_log_path or not os.path.exists(init_log_path):
        init_log_path = None
    # Find latest log, results.csv, and score.txt
    log_files = sorted(glob.glob(os.path.join(log_dir, "*.log")), reverse=True)
    # Exclude init_log_path from results log if possible
    latest_log = None
    for lf in log_files:
        if lf != init_log_path:
            latest_log = lf
            break
    results_csv_files = sorted(glob.glob(os.path.join(log_dir, "*.results.csv")), reverse=True)
    score_files = sorted(glob.glob(os.path.join(log_dir, "*.score.txt")), reverse=True)
    latest_results_csv = results_csv_files[0] if results_csv_files else None
    latest_score = score_files[0] if score_files else None
    return [init_log_path, latest_log, latest_results_csv, latest_score]

def run_and_submit_all(profile: gr.OAuthProfile | None, state=None):
    """
    Fetches all questions, runs the GaiaAgent on them, submits all answers,
    and displays the results. Also returns new file paths for download links and updates state.
    """
    space_id = os.getenv("SPACE_ID")
    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None, None, None, None, None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent (already done globally)
    if agent is None:
        return "Error initializing agent. Check logs for details.", None, None, None, None, None
    agent_code = f"https://huggingface.co/spaces/arterm-sedov/agent-course-final-assignment/tree/main"
    print(agent_code)

    # --- Provide init log for download ---
    init_log_path = getattr(agent, "init_log_path", None)
    if not init_log_path or not os.path.exists(init_log_path):
        init_log_path = None

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None, init_log_path, None, None, None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None, init_log_path, None, None, None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None, init_log_path, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None, init_log_path, None, None, None

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
                import base64
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
                submitted_answer = agent(enhanced_question, file_data=file_data, file_name=file_name)
            else:
                submitted_answer = agent(question_text)
            # Ensure submitted_answer is always a string (never None)
            if submitted_answer is None:
                submitted_answer = ""
            else:
                submitted_answer = str(submitted_answer)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            # Also ensure all values in results_log are strings for robust CSV output
            results_log.append({
                "Task ID": str(task_id) if task_id is not None else "",
                "Question": str(question_text) if question_text is not None else "",
                "File": str(file_name) if file_name is not None else "",
                "Submitted Answer": submitted_answer
            })
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({
                "Task ID": str(task_id) if task_id is not None else "",
                "Question": str(question_text) if question_text is not None else "",
                "File": str(file_name) if file_name is not None else "",
                "Submitted Answer": f"AGENT ERROR: {e}"
            })

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log), init_log_path, None, None, None

    # --- Save log to logs/ folder with timestamp ---
    try:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"logs/{timestamp}.log"
        with open(log_path, "w", encoding="utf-8") as f:
            yaml.dump(results_log, f, allow_unicode=True)
        print(f"✅ Results log saved to: {log_path}")
    except Exception as e:
        print(f"⚠️ Failed to save results log: {e}")
        log_path = None

    # --- Save results table as CSV for download ---
    results_df = pd.DataFrame(results_log)
    csv_path = f"logs/{timestamp}.results.csv"
    save_df_to_csv(results_df, csv_path)

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
        # Save final status to a text file for download
        score_path = f"logs/{timestamp}.score.txt"
        with open(score_path, "w", encoding="utf-8") as f:
            f.write(final_status)
        # Return new file paths and update state
        new_state = [init_log_path, log_path, csv_path, score_path]
        return final_status, results_df, init_log_path, log_path, csv_path, score_path, new_state
    except Exception as e:
        status_message = f"Submission Failed: {e}"
        print(status_message)
        # Save error status to a text file for download
        score_path = f"logs/{timestamp}.score.txt"
        with open(score_path, "w", encoding="utf-8") as f:
            f.write(status_message)
        # Return new file paths and update state
        new_state = [init_log_path, log_path, csv_path, score_path]
        return status_message, results_df, init_log_path, log_path, csv_path, score_path, new_state


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Unit 4 Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit" button, it can take quite some time (this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a separate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    # Remove direct instantiation of gr.OAuthProfile (not needed in recent Gradio)
    # profile = gr.OAuthProfile()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
    init_log_file = gr.File(label="Download LLM Initialization Log")
    results_log_file = gr.File(label="Download Full Results Log")
    results_csv_file = gr.File(label="Download Results Table (CSV)")
    score_file = gr.File(label="Download Final Score/Status")
    file_state = gr.State([None, None, None, None])  # [init_log, results_log, csv, score]

    # On app load, show the latest logs (if available), using state if present
    demo.load(
        fn=get_latest_logs,
        inputs=[file_state],
        outputs=[init_log_file, results_log_file, results_csv_file, score_file, file_state],
    )

    # Use gr.OAuthProfile as an input type for run_and_submit_all, but do not instantiate it directly
    run_button.click(
        fn=run_and_submit_all,
        inputs=[gr.OAuthProfile(), file_state],
        outputs=[status_output, results_table, init_log_file, results_log_file, results_csv_file, score_file, file_state]
    )

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