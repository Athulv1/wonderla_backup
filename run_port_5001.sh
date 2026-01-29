#!/bin/bash

# Manual run script for RTSP Dashboard on Port 5001
# Uses separate database: head_counter_5001.db

WORK_DIR="/home/user/Documents/THINKNEURALAI/wonderla_backup"
cd "$WORK_DIR"

echo "========================================"
echo "🚀 Starting RTSP Dashboard - Port 5001"
echo "========================================"
echo "📊 Port: 5001"
echo "💾 Database: head_counter_5001.db"
echo "🌐 URL: http://localhost:5001"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Activate virtual environment if it exists
if [ -d "venv/bin" ]; then
    source venv/bin/activate
fi

# Run the application
python rtsp_dashboard.py --port 5001 --db head_counter_5001.db
