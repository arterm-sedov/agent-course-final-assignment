import os
import gradio as gr
import requests
import pandas as pd
import random
from agent import GaiaAgent
import datetime
import glob

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Main Agent Definition ---
AGENT_PROVIDER = os.environ.get("AGENT_PROVIDER", "google")
try:
    agent = GaiaAgent(provider=AGENT_PROVIDER)
except Exception as e:
    agent = None
    print(f"Error initializing GaiaAgent: {e}")

# --- Agent Evaluation Logic (close to reference) ---
def run_and_submit_all(profile: gr.OAuthProfile | None):
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
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
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

    # 3. Run the Agent (random sampling as in your version)
    results_log = []
    answers_payload = []
    print(f"Running GaiaAgent on {len(questions_data)} questions...")
    questions_data = random.sample(questions_data, len(questions_data))
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name", "")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        file_data = None
        if file_name and file_name.strip():
            try:
                print(f"\U0001F4C1 Downloading file: {file_name} for task {task_id}")
                file_url = f"{api_url}/files/{task_id}"
                file_response = requests.get(file_url, timeout=30)
                file_response.raise_for_status()
                import base64
                file_data = base64.b64encode(file_response.content).decode('utf-8')
                print(f"✅ Downloaded and encoded file: {file_name} ({len(file_data)} chars)")
            except Exception as e:
                print(f"⚠️ Failed to download file {file_name} for task {task_id}: {e}")
                file_data = None
        try:
            if file_data:
                enhanced_question = f"{question_text}\n\n[File attached: {file_name} - base64 encoded data available]"
                submitted_answer = agent(enhanced_question, file_data=file_data, file_name=file_name)
            else:
                submitted_answer = agent(question_text)
            if submitted_answer is None:
                submitted_answer = ""
            else:
                submitted_answer = str(submitted_answer)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
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
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

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
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except Exception as e:
        status_message = f"Submission Failed: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df

# --- Logs Tab Logic ---
def list_logs():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        return pd.DataFrame(columns=["File Name", "Download"])
    files = sorted(glob.glob(os.path.join(log_dir, "*")), reverse=True)
    data = []
    for f in files:
        fname = os.path.basename(f)
        data.append({"File Name": fname, "Download": f})
    return pd.DataFrame(data)

def download_log(file_path):
    return file_path

# --- Build Gradio Interface with Tabs ---
with gr.Blocks() as demo:
    with gr.Tab("Agent Evaluation"):
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
        run_button = gr.Button("Run Evaluation & Submit All Answers")
        status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
        results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
        run_button.click(
            fn=run_and_submit_all,
            inputs=[gr.OAuthProfile()],
            outputs=[status_output, results_table]
        )
    with gr.Tab("Logs & Downloads"):
        gr.Markdown("# Logs & Downloads")
        logs_df = gr.DataFrame(
            value=list_logs(),
            label="Log Files (click to download)",
            interactive=False
        )
        log_file = gr.File(label="Download Selected Log File")
        def on_select(evt: gr.SelectData):
            # evt.value is the row index
            df = list_logs()
            if evt.value is not None and int(evt.value) < len(df):
                return df.iloc[int(evt.value)]["Download"]
            return None
        logs_df.select(on_select, outputs=log_file)

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