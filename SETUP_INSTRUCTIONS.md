# GAIA Unit 4 - Vector Store Setup Instructions

This guide will help you set up the vector store for your GAIA Unit 4 agent using your Supabase and Hugging Face credentials.

## 🐍 Python Virtual Environment Setup

### Quick Setup (Automated)

**For a one-command setup, use the automated script:**
```bash
python setup_venv.py
```

This script will automatically:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Verify installation
- ✅ Provide next steps

### Manual Setup

If you prefer to set up manually or the automated script doesn't work:

### Step 0: Create and Activate Virtual Environment

**For Windows:**
```bash
# Create virtual environment (try these commands in order)
py -m venv venv
# OR if py doesn't work:
python -m venv venv
# OR if python doesn't work:
python3 -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (should show venv path)
where python
```

**For macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show venv path)
which python
```

**For Hugging Face Spaces:**
```bash
# HF Spaces automatically creates a virtual environment
# Just install requirements
pip install -r requirements.txt
```

### Step 0.1: Verify Python Version

Make sure you have Python 3.8+ installed:

```bash
# Windows
py --version
# OR
python --version

# macOS/Linux
python3 --version
# Should show Python 3.8.x or higher
```

### Step 0.2: Upgrade pip (Recommended)

```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip
```

### Step 0.3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 0.4: Verify Installation

```bash
# Test that key packages are installed
python -c "import langchain, supabase, gradio; print('✅ All packages installed successfully!')"
```

### Virtual Environment Management

**To deactivate the virtual environment:**
```bash
deactivate
```

**To reactivate later:**
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**To delete and recreate virtual environment:**
```bash
# Deactivate first
deactivate

# Delete old environment
rm -rf venv  # macOS/Linux
# OR
rmdir /s venv  # Windows

# Create new environment (repeat Step 0)
```

### Windows-Specific Troubleshooting

**If you get "python is not recognized":**
1. Make sure Python is installed and added to PATH
2. Try using `py` instead of `python`
3. Try using the full path to Python

**If you get "venv is not recognized":**
1. Make sure you're using Python 3.3+ (which includes venv)
2. Try: `py -m venv venv` or `python -m venv venv`

**If activation fails:**
1. Make sure you're in the correct directory
2. Try: `venv\Scripts\activate.bat` (Windows)
3. Check if the venv folder was created properly

**If pip install fails:**
1. Try upgrading pip first: `python -m pip install --upgrade pip`
2. Check your internet connection
3. Try installing packages one by one to identify the problematic one

**Alternative Windows Setup:**
```bash
# If the automated script fails, try this manual approach:
py -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Prerequisites

1. **Python 3.8+**: Make sure you have Python 3.8 or higher installed
2. **Supabase Account**: You need a Supabase project with pgvector extension enabled
3. **Hugging Face Account**: For embeddings and API access
4. **Virtual Environment**: Use the setup above to create an isolated Python environment

## Step 1: Set Up Environment Variables

Create a `.env` file in the `arterm-sedov` directory with your credentials:

```bash
# REQUIRED: Supabase credentials (for vector store)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your_service_role_key_here

# REQUIRED: Google Gemini credentials (for LLM - default provider)
GEMINI_KEY=your_gemini_api_key_here

# OPTIONAL: Hugging Face credentials (for embeddings - uses free models by default)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# OPTIONAL: Alternative LLM providers (only needed if you want to use these instead of Gemini)
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### How to get Supabase credentials:

1. Go to [supabase.com](https://supabase.com) and create a project
2. In your project dashboard, go to Settings → API
3. Copy the "Project URL" (this is your `SUPABASE_URL`)
4. Copy the "service_role" key (this is your `SUPABASE_KEY`)

### How to get Google Gemini API key:

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. Use this key as your `GEMINI_KEY`

### How to get Hugging Face API key (optional):

1. Go to [huggingface.co](https://huggingface.co) and create an account
2. Go to Settings → Access Tokens
3. Create a new token with "read" permissions
4. Use this token as your `HUGGINGFACE_API_KEY`
5. **Note**: This is optional - the embeddings model works without an API key for basic usage

### How to get Groq API key (optional):

1. Go to [console.groq.com](https://console.groq.com/)
2. Sign up or log in to your Groq account
3. Navigate to the API Keys section
4. Create a new API key
5. Use this key as your `GROQ_API_KEY`
6. **Note**: This is optional - only needed if you want to use Groq instead of Gemini

### How to get Tavily API key (optional):

1. Go to [tavily.com](https://tavily.com/)
2. Sign up for an account
3. Get your API key from the dashboard
4. Use this key as your `TAVILY_API_KEY`
5. **Note**: This is optional - only needed if you want to use web search tools

**Tavily Implementation Details:**
- The `web_search()` function uses Tavily's search API to find real-time web results
- Returns up to 3 search results with source URLs and content snippets
- Useful for finding current information, recent events, and up-to-date data
- Automatically handles API key validation and error handling
- Returns formatted results that can be easily parsed by the agent

**Example Usage:**
```python
# In your agent, the web_search tool can be called like:
result = web_search("latest SpaceX launch date")
# Returns formatted web search results about recent SpaceX launches
```

## Step 2: Set Up Supabase Database

### 2.1 Enable pgvector Extension

In your Supabase SQL editor, run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2.2 Create the Table

```sql
CREATE TABLE agent_course_reference (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(768)
);
```

### 2.3 Create the Similarity Search Function

```sql
CREATE OR REPLACE FUNCTION match_agent_course_reference_langchain(
    query_embedding vector(768),
    match_count integer DEFAULT 5,
    filter jsonb DEFAULT '{}'
)
RETURNS TABLE (
    id bigint,
    content text,
    metadata jsonb,
    embedding vector(768),
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        agent_course_reference.id,
        agent_course_reference.content,
        agent_course_reference.metadata,
        agent_course_reference.embedding,
        1 - (agent_course_reference.embedding <=> query_embedding) AS similarity
    FROM agent_course_reference
    WHERE agent_course_reference.metadata @> filter
    ORDER BY agent_course_reference.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

### 2.4 Create Table Truncate Function (Optional)

For more reliable table clearing during setup:

```sql
CREATE OR REPLACE FUNCTION truncate_agent_course_reference()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    TRUNCATE TABLE agent_course_reference RESTART IDENTITY;
END;
$$;
```

## Step 3: Copy Required Data Files

Make sure to have the metadata file:

```bash
metadata.jsonl .
```

## Step 4: Install Required Packages

Make sure you have all required packages installed:

```bash
pip install -r requirements.txt
```

If you don't have a requirements.txt, install these packages:

```bash
pip install langchain langchain-community langchain-core langchain-google-genai langchain-huggingface langchain-groq supabase python-dotenv pandas numpy pillow pytesseract requests langchain-tavily sentence-transformers
```

## Step 5: Run the Setup Script

### Option A: Run the Python Script

```bash
python setup_vector_store.py
```

### Option B: Run the Jupyter Notebook

```bash
jupyter notebook explore_metadata.ipynb
```

## Step 6: Verify the Setup

The setup script will:

1. ✅ Load metadata.jsonl data
2. ✅ Connect to Supabase
3. ✅ Populate the vector store with Q&A data
4. ✅ Test similarity search functionality
5. ✅ Analyze tools used in the dataset
6. ✅ Test GaiaAgent integration

You should see output like:

```
🚀 GAIA Unit 4 - Vector Store Setup
==================================================
📁 Loading metadata.jsonl...
✅ Loaded 1000 questions from metadata.jsonl

🔍 Exploring sample data...
==================================================
Task ID: d1af70ea-a9a4-421a-b9cc-94b5e02f1788
Question: As of the 2020 census, what was the population difference...
...

🔗 Setting up Supabase connection...
✅ Supabase URL: https://your-project.supabase.co
✅ Supabase Key: eyJhbGciOi...
✅ Supabase connection established

📊 Populating vector store...
✅ Prepared 1000 documents for insertion
✅ Cleared existing data from agent_course_reference table
✅ Successfully inserted 1000 documents into agent_course_reference table
✅ Saved documents to supabase_docs.csv as backup

🧪 Testing vector store...
✅ Vector store initialized
✅ Found 1 similar documents
✅ Top match: Content: Question : On June 6, 2023...

🛠️  Analyzing tools used in dataset...
Total number of unique tools: 83
Top 20 most used tools:
  ├── web browser: 107
  ├── search engine: 101
  ├── calculator: 34
  ...

🤖 Testing GaiaAgent integration...
✅ GaiaAgent initialized
✅ Reference answer found: 80GSFC21M0002

==================================================
📋 SETUP SUMMARY
==================================================
✅ Metadata loaded: 1000 questions
✅ Supabase connection: Success
✅ Vector store population: Success
✅ Vector store testing: Success
✅ Agent integration: Success

🎉 Vector store setup completed successfully!
GaiaAgent is ready to use with the vector store.
```

## Troubleshooting

### Common Issues:

1. **"metadata.jsonl not found"**
   - Make sure you copied the file from fisherman611 folder
   - Run: `cp ../fisherman611/metadata.jsonl .`

2. **"Missing Supabase credentials"**
   - Check that the `.env` file exists and has correct credentials
   - Make sure you're using the service_role key, not the anon key

3. **"Error inserting data into Supabase"**
   - Check if the table exists and has the correct schema
   - Verify pgvector extension is enabled
   - Check your Supabase permissions

4. **"Error in similarity search"**
   - Verify the function `match_agent_course_reference_langchain` exists
   - Check if data was properly inserted into the table

5. **"Error testing GaiaAgent integration"**
   - Make sure you have `GEMINI_KEY` in your `.env` file
   - Check if all required packages are installed

6. **"ModuleNotFoundError: No module named 'sentence-transformers'"**
   - Install the missing package: `pip install sentence-transformers`
   - This package is required for HuggingFace embeddings
   - Re-run the setup script after installation

7. **"ImportError: Could not import sentence_transformers"**
   - Make sure you're in the virtual environment
   - Run: `pip install sentence-transformers`
   - If that doesn't work, try: `pip install --upgrade sentence-transformers`

### Getting Help:

- Check the Supabase logs in your project dashboard
- Verify your table structure matches the expected schema
- Test the similarity function directly in Supabase SQL editor

## Next Steps

Once the setup is complete:

1. The vector store is populated with reference Q&A data
2. The GaiaAgent can use similarity search to find relevant answers
3. You can run the full evaluation with `python app.py`
4. The agent will automatically use the vector store for reference answers

## Files Created/Modified:

- `explore_metadata.ipynb` - Jupyter notebook for exploration
- `setup_vector_store.py` - Python script for setup
- `