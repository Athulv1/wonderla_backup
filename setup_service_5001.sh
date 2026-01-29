#!/bin/bash

# Installation script for Port 5001 service (best.pt model)
# This does NOT affect the existing Port 5000 service

echo "=============================================="
echo "Installing RTSP Dashboard Service (Port 5001)"
echo "Model: best.pt (Custom Head Detection)"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if port 5000 service is running
echo -e "\n${YELLOW}Checking existing services...${NC}"
if systemctl is-active --quiet rtsp-dashboard.service; then
    echo -e "${GREEN}✓ Port 5000 service is running${NC}"
else
    echo -e "${YELLOW}⚠ Port 5000 service not found (this is OK)${NC}"
fi

# Stop port 5001 service if already running
echo -e "\n${YELLOW}Checking Port 5001 service...${NC}"
if systemctl is-active --quiet rtsp-dashboard-5001.service; then
    echo "Stopping existing Port 5001 service..."
    sudo systemctl stop rtsp-dashboard-5001.service
fi

# Copy service file
echo -e "\n${YELLOW}Installing service file...${NC}"
sudo cp rtsp-dashboard-5001.service /etc/systemd/system/

# Reload systemd
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable service
echo "Enabling Port 5001 service..."
sudo systemctl enable rtsp-dashboard-5001.service

# Start service
echo "Starting Port 5001 service..."
sudo systemctl start rtsp-dashboard-5001.service

# Wait a moment
sleep 3

# Check status
echo -e "\n=============================================="
echo "Service Status Check"
echo "=============================================="

echo -e "\n${YELLOW}Port 5000 (yolo11x.pt):${NC}"
if systemctl is-active --quiet rtsp-dashboard.service; then
    echo -e "${GREEN}✓ Running${NC}"
    sudo systemctl status rtsp-dashboard.service --no-pager -n 3
else
    echo -e "${YELLOW}Not running${NC}"
fi

echo -e "\n${YELLOW}Port 5001 (best.pt):${NC}"
if systemctl is-active --quiet rtsp-dashboard-5001.service; then
    echo -e "${GREEN}✓ Running${NC}"
    sudo systemctl status rtsp-dashboard-5001.service --no-pager -n 3
else
    echo -e "${RED}✗ Failed to start${NC}"
    echo "Check logs with: sudo journalctl -u rtsp-dashboard-5001.service -n 50"
fi

# Show URLs
echo -e "\n=============================================="
echo "Dashboard URLs"
echo "=============================================="
echo -e "${GREEN}Port 5000 (yolo11x.pt):${NC} http://localhost:5000"
echo -e "${GREEN}Port 5001 (best.pt):${NC}    http://localhost:5001"

echo -e "\n=============================================="
echo "Useful Commands"
echo "=============================================="
echo "View Port 5001 logs:"
echo "  sudo journalctl -u rtsp-dashboard-5001.service -f"
echo ""
echo "Check service status:"
echo "  sudo systemctl status rtsp-dashboard-5001.service"
echo ""
echo "Restart service:"
echo "  sudo systemctl restart rtsp-dashboard-5001.service"
echo ""
echo "Stop service:"
echo "  sudo systemctl stop rtsp-dashboard-5001.service"
echo ""
echo "View log files:"
echo "  tail -f logs/service_5001.log"
echo "=============================================="
