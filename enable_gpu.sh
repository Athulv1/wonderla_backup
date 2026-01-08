#!/bin/bash
# Script to enable GPU after disabling Secure Boot

echo "==============================================="
echo "GPU Setup Verification Script"
echo "==============================================="
echo ""

# Check Secure Boot status
echo "1. Checking Secure Boot status..."
SECUREBOOT=$(mokutil --sb-state 2>/dev/null)
echo "   $SECUREBOOT"
echo ""

if [[ $SECUREBOOT == *"enabled"* ]]; then
    echo "⚠️  SECURE BOOT IS ENABLED"
    echo ""
    echo "To use NVIDIA GPU, you need to:"
    echo "1. Restart computer and enter BIOS/UEFI (usually press F10, F2, or Del during startup)"
    echo "2. Find 'Security' or 'Boot' section"
    echo "3. Disable 'Secure Boot'"
    echo "4. Save and Exit"
    echo "5. Run this script again"
    echo ""
    exit 1
fi

# Check if NVIDIA driver is loaded
echo "2. Checking NVIDIA driver..."
if lsmod | grep -q nvidia; then
    echo "   ✓ NVIDIA driver loaded"
else
    echo "   Loading NVIDIA driver..."
    sudo modprobe nvidia
    if [ $? -eq 0 ]; then
        echo "   ✓ NVIDIA driver loaded successfully"
    else
        echo "   ✗ Failed to load NVIDIA driver"
        exit 1
    fi
fi
echo ""

# Run nvidia-smi
echo "3. Checking GPU status..."
nvidia-smi
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ GPU is working correctly!"
    echo ""
else
    echo "✗ GPU check failed"
    exit 1
fi

# Check PyTorch CUDA
echo "4. Checking PyTorch CUDA..."
cd /home/user/Documents/THINKNEURALAI/wonderla_backup
source venv/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

echo "==============================================="
echo "✓ All checks passed! GPU is ready to use."
echo "==============================================="
