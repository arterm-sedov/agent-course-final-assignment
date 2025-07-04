import os
import subprocess
import datetime

def save_and_commit_file(
    file_path: str,
    content: str,
    commit_message: str = None,
    user_name: str = None,
    user_email: str = None,
    hf_token_env: str = "HF_TOKEN"
):
    """
    Save a file, commit, and push it to the HuggingFace Space repo for persistence.

    Args:
        file_path (str): Path to save the file (e.g., 'logs/mylog.txt')
        content (str): File content to write
        commit_message (str): Commit message (optional, will use timestamp and file name if not provided)
        user_name (str): Git user.name (optional, from env or fallback)
        user_email (str): Git user.email (optional, from env or fallback)
        hf_token_env (str): Name of the env var holding the HF token
    """
    # 1. Write the file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    # 2. Get user info
    user_name = user_name or os.environ.get("GIT_USER_NAME", "HF Space Bot")
    user_email = user_email or os.environ.get("GIT_USER_EMAIL", "hfspacebot@users.noreply.huggingface.co")

    # 3. Configure git user
    subprocess.run(['git', 'config', '--global', 'user.name', user_name], check=True)
    subprocess.run(['git', 'config', '--global', 'user.email', user_email], check=True)

    # 4. Get repo info from env
    space_id = os.environ.get("SPACE_ID")
    hf_token = os.environ.get(hf_token_env)
    if not space_id or not hf_token:
        raise RuntimeError("SPACE_ID or HF_TOKEN not set in environment variables/secrets.")

    repo_url = f"https://{hf_token}@huggingface.co/spaces/{space_id}.git"
    subprocess.run(['git', 'remote', 'set-url', 'origin', repo_url], check=True)

    # Debug prints for troubleshooting authentication issues
    print("HF_TOKEN present:", bool(hf_token))
    print("Remote URL:", repo_url[:30] + '...' + repo_url[-20:])  # Mask token in output

    # 5. Add, commit, and push
    subprocess.run(['git', 'add', file_path], check=True)
    if not commit_message:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Add {file_path} at {timestamp}"
    subprocess.run(['git', 'commit', '-m', commit_message], check=True)
    # Use GIT_ASKPASS=echo to prevent password prompt
    env = os.environ.copy()
    env["GIT_ASKPASS"] = "echo"
    subprocess.run(['git', 'push', 'origin', 'main'], check=True, env=env) 