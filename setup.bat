@echo off
echo [INFO] Setting up development environment...

:: Ensure Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed. Please install it first.
    exit /b
)

:: Ensure pip is installed
python -m ensurepip --default-pip

:: Ensure virtual environment exists
if not exist env (
    echo [INFO] Creating virtual environment...
    python -m venv env
) else (
    echo [INFO] Virtual environment already exists.
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call env\Scripts\activate

:: Upgrade pip
echo [INFO] Upgrading pip...
pip install --upgrade pip

:: Install dependencies
echo [INFO] Installing dependencies...
pip install -r requirements.txt

:: GPU Setup
echo [INFO] Checking for GPU support...
python -c "import torch; print(torch.cuda.is_available())" | findstr /C:"True" >nul
if %errorlevel%==0 (
    echo [INFO] GPU detected! Installing GPU-accelerated PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo [WARNING] No GPU detected. Installing CPU-based PyTorch...
    pip install torch torchvision torchaudio
)

echo [✅] Setup complete. Run 'env\Scripts\activate' to activate the environment.
