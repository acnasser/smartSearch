#!/bin/bash

echo "[INFO] Setting up development environment..."

# Ensure Python 3 and virtualenv are installed
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 is not installed. Please install it first."
    exit 1
fi

if ! python3 -m pip &>/dev/null; then
    echo "[ERROR] Pip is not installed. Installing now..."
    python3 -m ensurepip --default-pip
fi

if ! python3 -m venv --help &>/dev/null; then
    echo "[ERROR] Virtualenv is not installed. Installing now..."
    python3 -m pip install --user virtualenv
fi

# Create virtual environment
if [ ! -d "env" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv env
else
    echo "[INFO] Virtual environment already exists."
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source env/bin/activate

# Upgrade pip
echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "[INFO] Installing dependencies..."
pip install -r requirements.txt

# GPU Setup
echo "[INFO] Checking for GPU..."
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "[INFO] GPU detected! Installing GPU-accelerated PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "[WARNING] No GPU detected. Installing CPU-based PyTorch..."
    pip install torch torchvision torchaudio
fi

echo "[✅] Setup complete. Run 'source env/bin/activate' to activate the environment."
