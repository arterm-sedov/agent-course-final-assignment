# Windows Installation Guide

## Problem
The original requirements.txt had issues with pandas 2.2.2 trying to build from source on Windows, which requires Visual Studio build tools.

## Solution
We've fixed the requirements.txt and created installation scripts that install dependencies in the correct order.

## Installation Options

### Option 1: Use PowerShell Script (Recommended)
```powershell
# Run PowerShell as Administrator and execute:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install_windows.ps1
```

### Option 2: Use Batch Script
```cmd
# Run Command Prompt and execute:
install_windows.bat
```

### Option 3: Manual Installation
If the scripts don't work, follow these steps manually:

1. **Create and activate virtual environment:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Upgrade pip and install build tools:**
   ```cmd
   python -m pip install --upgrade pip
   pip install wheel setuptools
   ```

3. **Install numpy first:**
   ```cmd
   pip install "numpy>=1.24.0"
   ```

4. **Install pandas (avoiding 2.2.2):**
   ```cmd
   pip install "pandas>=2.0.0,<2.2.0"
   ```

5. **Install remaining dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

## Key Details

**Changed pandas version** from `==2.2.2` to `>=2.0.0,<2.2.0` to avoid Windows build issues
**Added proper version constraints** to prevent conflicts
**Created installation scripts** that install in the correct order

## Troubleshooting

### If you still get build errors:
1. Install Visual Studio Build Tools 2019 or later
2. Or use conda instead of pip:
   ```cmd
   conda install pandas numpy
   pip install -r requirements.txt
   ```

### If you get permission errors:
1. Run PowerShell/Command Prompt as Administrator
2. Or use `--user` flag:
   ```cmd
   pip install --user -r requirements.txt
   ```

## Verification
After installation, test that everything works:
```python
import pandas as pd
import numpy as np
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")
``` 