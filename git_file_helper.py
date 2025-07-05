import os
import datetime
from typing import Optional, Union, Dict, Any
from pathlib import Path

# Global constants
TRACES_DIR = "traces"  # Directory for uploading trace files (won't trigger Space restarts)

# Import huggingface_hub components for API-based file operations
try:
    from huggingface_hub import HfApi, CommitOperationAdd
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")

def get_hf_api_client(token: Optional[str] = None) -> Optional[HfApi]:
    """
    Create and configure an HfApi client for repository operations.
    
    Args:
        token (str, optional): HuggingFace token. If None, uses environment variable.
        
    Returns:
        HfApi: Configured API client or None if not available
    """
    if not HF_HUB_AVAILABLE:
        return None
        
    try:
        # Get token from parameter or environment
        hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            print("Warning: No HuggingFace token found. API operations will fail.")
            return None
            
        # Create API client
        api = HfApi(token=hf_token)
        return api
    except Exception as e:
        print(f"Error creating HfApi client: {e}")
        return None

def get_repo_info() -> tuple[Optional[str], Optional[str]]:
    """
    Get repository information from environment variables.
    
    Returns:
        tuple: (space_id, repo_type) or (None, None) if not found
    """
    space_id = os.environ.get("SPACE_ID")
    repo_type = os.environ.get("REPO_TYPE", "space")  # Default to space type
    
    return space_id, repo_type

def upload_file_via_api(
    file_path: str,
    content: Union[str, bytes],
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
    repo_id: Optional[str] = None,
    repo_type: str = "space"
) -> bool:
    """
    Upload a file to HuggingFace repository using the API (CommitOperationAdd).
    
    Args:
        file_path (str): Path in the repository where to save the file
        content (Union[str, bytes]): File content to upload
        commit_message (str, optional): Commit message
        token (str, optional): HuggingFace token
        repo_id (str, optional): Repository ID. If None, uses SPACE_ID from env
        repo_type (str): Repository type (space, model, dataset)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not HF_HUB_AVAILABLE:
        print("Error: huggingface_hub not available for API operations")
        return False
        
    try:
        # Get API client
        api = get_hf_api_client(token)
        if not api:
            return False
            
        # Get repository info
        if not repo_id:
            repo_id, repo_type = get_repo_info()
            if not repo_id:
                print("Error: No repository ID found in environment variables")
                return False
        
        # Prepare content
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content
            
        # Create commit operation
        operation = CommitOperationAdd(
            path_in_repo=file_path,
            path_or_fileobj=content_bytes
        )
        
        # Generate commit message if not provided
        if not commit_message:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f"Add {file_path} at {timestamp}"
        
        # Commit the operation
        commit_info = api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=[operation],
            commit_message=commit_message
        )
        
        print(f"✅ File uploaded successfully via API: {file_path}")
        print(f"   Commit: {commit_info.commit_url}")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading file via API: {e}")
        return False

def save_and_commit_file(
    file_path: str,
    content: str,
    commit_message: str = None,
    token: Optional[str] = None,
    repo_id: Optional[str] = None,
    repo_type: str = "space"
) -> bool:
    """
    Save a file and commit it to the HuggingFace repository using the API.
    
    This function uses CommitOperationAdd for efficient file uploads.
    Used primarily for saving log files.
    
    Args:
        file_path (str): Path to save the file (e.g., 'logs/mylog.txt')
        content (str): File content to write
        commit_message (str, optional): Commit message
        token (str, optional): HuggingFace token
        repo_id (str, optional): Repository ID
        repo_type (str): Repository type
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not HF_HUB_AVAILABLE:
        print("Error: huggingface_hub not available. Install with: pip install huggingface_hub")
        return False
        
    try:
        # Upload file via API
        success = upload_file_via_api(
            file_path=file_path,
            content=content,
            commit_message=commit_message,
            token=token,
            repo_id=repo_id,
            repo_type=repo_type
        )
        
        if success:
            print(f"✅ File saved and committed successfully: {file_path}")
        else:
            print(f"❌ Failed to save and commit file: {file_path}")
            
        return success
        
    except Exception as e:
        print(f"❌ Error in save_and_commit_file: {e}")
        return False

def batch_upload_files(
    files_data: Dict[str, Union[str, bytes]],
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
    repo_id: Optional[str] = None,
    repo_type: str = "space"
) -> Dict[str, bool]:
    """
    Upload multiple files in a single commit using the API.
    
    Useful for uploading multiple log files at once.
    
    Args:
        files_data (Dict[str, Union[str, bytes]]): Dictionary mapping file paths to content
        commit_message (str, optional): Commit message
        token (str, optional): HuggingFace token
        repo_id (str, optional): Repository ID
        repo_type (str): Repository type
        
    Returns:
        Dict[str, bool]: Dictionary mapping file paths to success status
    """
    if not HF_HUB_AVAILABLE:
        print("Error: huggingface_hub not available for batch operations")
        return {path: False for path in files_data.keys()}
        
    try:
        # Get API client
        api = get_hf_api_client(token)
        if not api:
            return {path: False for path in files_data.keys()}
            
        # Get repository info
        if not repo_id:
            repo_id, repo_type = get_repo_info()
            if not repo_id:
                print("Error: No repository ID found in environment variables")
                return {path: False for path in files_data.keys()}
        
        # Create operations for all files
        operations = []
        for file_path, content in files_data.items():
            # Prepare content
            if isinstance(content, str):
                content_bytes = content.encode('utf-8')
            else:
                content_bytes = content
                
            operation = CommitOperationAdd(
                path_in_repo=file_path,
                path_or_fileobj=content_bytes
            )
            operations.append(operation)
        
        # Generate commit message if not provided
        if not commit_message:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_count = len(files_data)
            commit_message = f"Batch upload {file_count} files at {timestamp}"
        
        # Commit all operations
        commit_info = api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=commit_message
        )
        
        print(f"✅ Batch upload successful: {len(files_data)} files")
        print(f"   Commit: {commit_info.commit_url}")
        return {path: True for path in files_data.keys()}
        
    except Exception as e:
        print(f"❌ Error in batch upload: {e}")
        return {path: False for path in files_data.keys()} 