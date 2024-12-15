#!/bin/bash

# Default Python version
PYTHON_VERSION="11"
VENV_NAME="rl-homework-venv"

# Check if argument is provided
if [ $# -eq 1 ]; then
    if [[ "$1" =~ ^(10|11|12)$ ]]; then
        PYTHON_VERSION="$1"
    else
        echo "Error: Python version must be 10, 11, or 12"
        exit 1
    fi
fi

# Create virtual environment
python3."$PYTHON_VERSION" -m venv "$VENV_NAME"

# Activate virtual environment
source "${VENV_NAME}/bin/activate"

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found"
fi

echo "Virtual environment $VENV_NAME created and activated with Python 3.$PYTHON_VERSION"
