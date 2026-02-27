#!/bin/bash
# V5 Training Launcher: auto TensorBoard + browser open + training
# Usage: ./run_train.sh [config_path] [port]

CONFIG=${1:-configs/train_trafficplanner.cfg}
PORT=${2:-6006}

# Extract output dir from config
OUT_DIR=$(grep "^out:" "$CONFIG" | awk '{print $2}')
TB_DIR="${OUT_DIR}/tb_logs"

echo "=== V5 Training Launcher ==="
echo "Config:      $CONFIG"
echo "Output:      $OUT_DIR"
echo "TB logs:     $TB_DIR"
echo "TB port:     $PORT"
echo ""

# Kill existing TensorBoard on same port
pkill -f "tensorboard.*--port $PORT" 2>/dev/null
sleep 1

# Start TensorBoard in background
mkdir -p "$TB_DIR"
source ~/venvs/strive_env/bin/activate
tensorboard --logdir "$TB_DIR" --port "$PORT" --bind_all &
TB_PID=$!
echo "TensorBoard started (PID: $TB_PID)"

# Wait for TB server, then auto-open browser
sleep 2
xdg-open "http://localhost:$PORT" 2>/dev/null \
  || google-chrome "http://localhost:$PORT" 2>/dev/null \
  || firefox "http://localhost:$PORT" 2>/dev/null \
  || echo "Auto browser open failed -> http://localhost:$PORT"
echo ""

# Run training
python3 src/train_trafficplanner.py --config "$CONFIG"

# Cleanup info
echo ""
echo "Training finished. TensorBoard still running (PID: $TB_PID)"
echo "Stop with: kill $TB_PID"
