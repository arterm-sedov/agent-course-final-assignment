# HuggingFace Hub API File Upload Guide

This guide explains how to use the enhanced file upload functionality that integrates `CommitOperationAdd` from the HuggingFace Hub API.

## Overview

The enhanced `git_file_helper.py` provides API-based file operations for uploading files to HuggingFace repositories:

1. **API-based**: Uses `CommitOperationAdd` from `huggingface_hub`
2. **Clean and focused**: No git-based fallback logic

## Features

### ✅ API-Based Operations
- **Single file upload** with `CommitOperationAdd`
- **Batch file upload** with multiple files in single commit
- **Log file management** for saving agent logs and results
- **Clean and focused** - no unnecessary operations

### ✅ Benefits of API Approach
- **Faster**: No need to clone/pull/push git repository
- **More reliable**: Direct API calls with better error handling
- **Atomic operations**: Multiple files in single commit
- **Better logging**: Detailed success/failure information
- **Token-based auth**: Uses HuggingFace tokens directly
- **Clean code**: No complex git fallback logic

## Installation & Setup

### 1. Dependencies
The `huggingface_hub` package is already included in `requirements.txt`:

```bash
pip install huggingface_hub
```

### 2. Environment Variables
Ensure these environment variables are set:

```bash
# Required
HF_TOKEN=your_huggingface_token
SPACE_ID=your_space_id

# Optional
HUGGINGFACEHUB_API_TOKEN=your_token  # Alternative token name
REPO_TYPE=space  # Default: space
```

### 3. Token Setup
Get your HuggingFace token from: https://huggingface.co/settings/tokens

## Usage Examples

### Basic File Upload

```python
from git_file_helper import upload_file_via_api

# Upload a text file
success = upload_file_via_api(
    file_path="logs/my_log.txt",
    content="This is my log content",
    commit_message="Add log file"
)

if success:
    print("✅ File uploaded successfully!")
else:
    print("❌ Upload failed")
```

### Upload Binary Files

```python
from git_file_helper import upload_file_via_api

# Upload binary data
with open("image.png", "rb") as f:
    image_data = f.read()

success = upload_file_via_api(
    file_path="images/test.png",
    content=image_data,  # bytes object
    commit_message="Add test image"
)
```

### Batch Upload Multiple Files

```python
from git_file_helper import batch_upload_files

# Prepare multiple files
files_data = {
    "logs/error.log": "Error log content",
    "logs/info.log": "Info log content", 
    "data/results.json": '{"result": "success"}',
    "images/screenshot.png": image_bytes  # binary data
}

# Upload all files in one commit
results = batch_upload_files(
    files_data=files_data,
    commit_message="Batch upload: logs and data"
)

# Check results
for file_path, success in results.items():
    status = "✅" if success else "❌"
    print(f"{status} {file_path}")
```

### Log File Upload

```python
from git_file_helper import upload_file_via_api

# Upload log file
log_content = f"""Log Entry
Timestamp: {datetime.datetime.now()}
Level: INFO
Message: Agent evaluation completed
Status: Success
"""

success = upload_file_via_api(
    file_path="logs/evaluation_log.txt",
    content=log_content,
    commit_message="Add evaluation log"
)
```

### API-based save_and_commit_file

The `save_and_commit_file` function now uses API-based upload:

```python
from git_file_helper import save_and_commit_file

# Use API-based upload
success = save_and_commit_file(
    file_path="logs/api_test.txt",
    content="Test content"
)

if success:
    print("✅ File uploaded successfully!")
else:
    print("❌ Upload failed")
```

## API Reference

### `upload_file_via_api()`

Upload a single file using `CommitOperationAdd`.

**Parameters:**
- `file_path` (str): Path in repository where to save file
- `content` (Union[str, bytes]): File content
- `commit_message` (str, optional): Commit message
- `token` (str, optional): HuggingFace token
- `repo_id` (str, optional): Repository ID
- `repo_type` (str): Repository type ("space", "model", "dataset")

**Returns:** `bool` - Success status

### `batch_upload_files()`

Upload multiple files in a single commit.

**Parameters:**
- `files_data` (Dict[str, Union[str, bytes]]): File paths to content mapping
- `commit_message` (str, optional): Commit message
- `token` (str, optional): HuggingFace token
- `repo_id` (str, optional): Repository ID
- `repo_type` (str): Repository type

**Returns:** `Dict[str, bool]` - Success status for each file



## Error Handling

The API functions include comprehensive error handling:

```python
try:
    success = upload_file_via_api("test.txt", "content")
    if success:
        print("✅ Upload successful")
    else:
        print("❌ Upload failed - check logs")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
```

## Testing

Run the test script to verify functionality:

```bash
python test_hf_api_upload.py
```

This will test:
- ✅ API availability
- ✅ Client creation
- ✅ Single file upload
- ✅ Batch file upload
- ✅ File operations (upload/copy/delete)
- ✅ API vs Git comparison

## Integration with Existing Code

The API-based functions are designed to be easy to use and integrate with existing code.

### Migration Guide

**Before (if you had git-based code):**
```python
from git_file_helper import save_and_commit_file

save_and_commit_file("logs/test.txt", "content")
```

**After (API-based):**
```python
from git_file_helper import save_and_commit_file

success = save_and_commit_file("logs/test.txt", "content")
if success:
    print("✅ Upload successful!")
```

**Direct API approach:**
```python
from git_file_helper import upload_file_via_api

success = upload_file_via_api("logs/test.txt", "content")
```

## Performance Benefits

| Operation | API Method |
|-----------|------------|
| Single file | ~1-2s |
| Batch files | ~2-3s |
| Error handling | Detailed |
| Network usage | Minimal |
| Code complexity | Low |

## Troubleshooting

### Common Issues

1. **"huggingface_hub not available"**
   ```bash
   pip install huggingface_hub
   ```

2. **"No HuggingFace token found"**
   - Set `HF_TOKEN` environment variable
   - Or set `HUGGINGFACEHUB_API_TOKEN`

3. **"No repository ID found"**
   - Set `SPACE_ID` environment variable
   - Or pass `repo_id` parameter explicitly

4. **Authentication errors**
   - Verify token is valid
   - Check token permissions
   - Ensure repository access

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# API operations will show detailed logs
upload_file_via_api("test.txt", "content")
```

## Best Practices

1. **Use API functions**: All functions are API-based
2. **Batch operations**: Use `batch_upload_files()` for multiple files
3. **Error handling**: Always check return values
4. **Token security**: Store tokens in environment variables
5. **Clean code**: No complex fallback logic needed

## Advanced Usage

### Custom Repository

```python
upload_file_via_api(
    file_path="my_file.txt",
    content="content",
    repo_id="username/repo-name",
    repo_type="model"  # or "dataset"
)
```

### Custom Token

```python
upload_file_via_api(
    file_path="my_file.txt", 
    content="content",
    token="hf_your_custom_token"
)
```

### Large Files

For large files, consider chunking:

```python
def upload_large_file(file_path: str, local_path: str):
    with open(local_path, 'rb') as f:
        content = f.read()
    
    return upload_file_via_api(file_path, content)
```

## Conclusion

The enhanced file upload functionality provides a robust, efficient way to manage files in HuggingFace repositories. The API-based approach offers better performance and reliability while maintaining backward compatibility with existing code.

For more information, see the [HuggingFace Hub documentation](https://huggingface.co/docs/huggingface_hub/v0.32.3/en/package_reference/hf_api#huggingface_hub.CommitOperationAdd). 