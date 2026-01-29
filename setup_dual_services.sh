#!/bin/bash

# Setup script for RTSP Dashboard
# Auto-run: Port 5000 with database head_counter_5000.db
# Manual: Port 5001 with database head_counter_5001.db (use run_port_5001.sh)

echo "========================================"
echo "RTSP Dashboard Setup (Port 5000)"
echo "========================================"
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root or with sudo"
    echo "Usage: sudo ./setup_dual_services.sh"
    exit 1
fi

WORK_DIR="/home/user/Documents/THINKNEURALAI/wonderla_backup"

# Stop existing services if running
echo "🛑 Stopping existing services..."
systemctl stop rtsp-dashboard.service 2>/dev/null
systemctl stop rtsp-dashboard-5000.service 2>/dev/null

# Copy service file to systemd directory
echo "📋 Installing service file for port 5000..."
cp "$WORK_DIR/rtsp-dashboard-5000.service" /etc/systemd/system/

# Reload systemd
echo "🔄 Reloading systemd..."
systemctl daemon-reload

# Enable service to start on boot
echo "✅ Enabling service..."
systemctl enable rtsp-dashboard-5000.service

# Start service
echo "🚀 Starting service..."
systemctl start rtsp-dashboard-5000.service

echo ""
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "📊 Service Status:"
echo "----------------------------------------"
systemctl status rtsp-dashboard-5000.service --no-pager -l | head -n 10
echo ""
echo "========================================"
echo "🌐 Access Your Dashboard:"
echo "----------------------------------------"
echo "Port 5000 (Auto-run): http://localhost:5000"
echo "Database: head_counter_5000.db"
echo ""
echo "Port 5001 (Manual): Use ./run_port_5001.sh"
echo "Database: head_counter_5001.db"
echo "========================================"
echo ""
echo "📝 Useful Commands:"
echo "  Check status: sudo systemctl status rtsp-dashboard-5000.service"
echo "  View logs:    sudo journalctl -u rtsp-dashboard-5000.service -f"
echo "  Restart:      sudo systemctl restart rtsp-dashboard-5000.service"
echo "  Stop:         sudo systemctl stop rtsp-dashboard-5000.service"
echo ""
