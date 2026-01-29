#!/bin/bash
# Setup script for RTSP Dashboard service

echo "Setting up RTSP Dashboard as a system service..."

# Copy service file to systemd directory
sudo cp rtsp-dashboard.service /etc/systemd/system/

# Reload systemd to recognize new service
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable rtsp-dashboard.service

# Start the service
sudo systemctl start rtsp-dashboard.service

# Check status
sudo systemctl status rtsp-dashboard.service

echo ""
echo "✓ Service installed successfully!"
echo ""
echo "Useful commands:"
echo "  Start:   sudo systemctl start rtsp-dashboard"
echo "  Stop:    sudo systemctl stop rtsp-dashboard"
echo "  Restart: sudo systemctl restart rtsp-dashboard"
echo "  Status:  sudo systemctl status rtsp-dashboard"
echo "  Logs:    sudo journalctl -u rtsp-dashboard -f"
echo "  Disable: sudo systemctl disable rtsp-dashboard"
