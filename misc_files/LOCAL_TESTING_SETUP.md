# Local Testing Setup Guide

This guide helps you set up environment variables for local testing of the HuggingFace Hub API functionality.

## 🔧 **Step 1: Get Your HuggingFace Token**

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "Local Testing")
4. Select "Write" permissions
5. Copy the token (starts with `hf_`)

## 🔧 **Step 2: Get Your Repository ID**

### **Option A: Use an Existing Repository**
If you have an existing HuggingFace Space, Model, or Dataset:
```bash
SPACE_ID=your-username/your-repo-name
```

### **Option B: Create a Test Repository**
1. Go to [HuggingFace](https://huggingface.co)
2. Click "New" → "Space" (recommended for testing)
3. Give it a name (e.g., "test-file-uploads")
4. Use the repository ID: `your-username/test-file-uploads`

## 🔧 **Step 3: Set Environment Variables**

### **Method 1: Create .env File (Recommended)**

Create a `.env` file in your project root:

```bash
# .env file
HF_TOKEN=hf_your_token_here
SPACE_ID=your-username/your-repo-name
REPO_TYPE=space
```

### **Method 2: Set Environment Variables Directly**

#### **For Windows (PowerShell):**
```powershell
$env:HF_TOKEN="hf_your_token_here"
$env:SPACE_ID="your-username/your-repo-name"
$env:REPO_TYPE="space"
```

#### **For Windows (Command Prompt):**
```cmd
set HF_TOKEN=hf_your_token_here
set SPACE_ID=your-username/your-repo-name
set REPO_TYPE=space
```

#### **For Linux/macOS:**
```bash
export HF_TOKEN="hf_your_token_here"
export SPACE_ID="your-username/your-repo-name"
export REPO_TYPE="space"
```

## 🔧 **Step 4: Test Your Setup**

Run the test script to verify your configuration:

```bash
python test_hf_api_upload.py
```

Or run the example script:

```bash
python example_api_usage.py
```

## 🔧 **Step 5: Verify Environment Variables**

You can verify your environment variables are set correctly:

```python
import os
print(f"HF_TOKEN: {'✅ Set' if os.getenv('HF_TOKEN') else '❌ Not set'}")
print(f"SPACE_ID: {'✅ Set' if os.getenv('SPACE_ID') else '❌ Not set'}")
print(f"REPO_TYPE: {os.getenv('REPO_TYPE', 'space (default)')}")
```

## 🔧 **Troubleshooting**

### **"No repository ID found" Error**
- Make sure `SPACE_ID` is set correctly
- Format should be: `username/repository-name`
- No leading/trailing spaces

### **"No HuggingFace token found" Error**
- Make sure `HF_TOKEN` is set correctly
- Token should start with `hf_`
- Check token permissions (needs "Write" access)

### **Authentication Errors**
- Verify your token is valid
- Check that you have write access to the repository
- Ensure the repository exists

## 🔧 **Example Configuration**

Here's a complete example:

```bash
# .env file
HF_TOKEN=hf_abc123def456ghi789jkl012mno345pqr678stu901vwx234yz567
SPACE_ID=myusername/test-file-uploads
REPO_TYPE=space
```

## 🔧 **Security Notes**

- Never commit your `.env` file to version control
- Add `.env` to your `.gitignore` file
- Use different tokens for different environments
- Regularly rotate your tokens

## 🔧 **Next Steps**

Once your environment variables are set up:

1. Test the API functionality: `python test_hf_api_upload.py`
2. Try the examples: `python example_api_usage.py`
3. Run your agent: `python app.py`

The API-based file upload should now work for local testing! 🚀 