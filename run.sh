#!/bin/bash

# エンコーディング設定
export PYTHONIOENCODING=utf-8

# Windowsの場合、コマンドプロンプトのコードページをUTF-8に設定
if [ "$OSTYPE" = "msys" ] || [ "$OSTYPE" = "win32" ] || [[ "$OSTYPE" == "cygwin"* ]]; then
  chcp.com 65001 > /dev/null 2>&1
fi

# 明示的にAnacondaのPythonを指定
CONDA_PATH="/c/Users/borik/Anaconda3"
PYTHON="$CONDA_PATH/envs/remdis/python.exe"

# Get the base directory (where this script is located)
BASE_DIR=$(cd $(dirname $0); pwd)
MOD_DIR=$BASE_DIR/modules

# Change to modules directory
cd $MOD_DIR

# Kill all background processes when script exits
trap 'echo "Terminating processes..."; kill $(jobs -p) 2>/dev/null || true' INT TERM EXIT

echo "Starting Remdis system..."
echo "Using Python: $PYTHON"
echo "Python version: $($PYTHON --version 2>&1)"
echo "Current directory: $(pwd)"

# Start input.py with logging
echo "Starting input.py..."
$PYTHON input.py > input.log 2>&1 &
PID_INPUT=$!

# Start text_vap.py with logging
echo "Starting text_vap.py..."
$PYTHON text_vap.py > text_vap.log 2>&1 &
PID_TEXT_VAP=$!

# Start asr.py with output displayed in terminal
echo "Starting asr.py..."
$PYTHON asr.py &
PID_ASR=$!

# Start dialogue.py with output displayed in terminal
echo "Starting dialogue.py..."
$PYTHON dialogue.py &
PID_DIALOGUE=$!

# Start tts.py with output displayed in terminal
echo "Starting tts.py..."
$PYTHON tts.py &
PID_TTS=$!

# Start output.py with logging
echo "Starting output.py..."
$PYTHON output.py > output.log 2>&1 &
PID_OUTPUT=$!

echo "All processes started with PIDs:"
echo "input.py: $PID_INPUT"
echo "text_vap.py: $PID_TEXT_VAP"
echo "asr.py: $PID_ASR"
echo "dialogue.py: $PID_DIALOGUE"
echo "tts.py: $PID_TTS"
echo "output.py: $PID_OUTPUT"

echo "Press Ctrl+C to terminate all processes"
# Wait for all processes to complete (or until interrupted)
wait

echo "All processes terminated. Check log files for details."