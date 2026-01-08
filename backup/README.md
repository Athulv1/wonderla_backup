# Quick Start - Training YOLOv11 on RunPod

## 🎯 What You'll Get
- Trained YOLOv11 model for detecting MOBILE phones
- Ability to count OUT objects
- Ready-to-use inference script

## 📦 Files Created

1. **train_yolov11.py** - Main training script (optimized for RTX 5090)
2. **inference.py** - Detection and counting script
3. **requirements.txt** - All dependencies
4. **setup_runpod.sh** - Automated setup for RunPod
5. **RUNPOD_TRAINING_GUIDE.md** - Complete step-by-step guide

## ⚡ Quick Start (5 Steps)

### 1. Prepare for Upload
Create a zip of your dataset:
```bash
cd "/home/rasheeque/VS CODE FOLDER/MARUTI"
zip -r dataset.zip OUT_MOBILE.v1i.yolov11/
```

### 2. Rent RTX 5090 on RunPod
- Go to https://www.runpod.io
- Deploy → GPU Instances → Select RTX 5090
- Choose PyTorch template
- Deploy!

### 3. Upload Files
Upload to `/workspace/` on RunPod:
- dataset.zip (or entire OUT_MOBILE.v1i.yolov11 folder)
- train_yolov11.py
- requirements.txt
- setup_runpod.sh
- inference.py

### 4. Setup & Train
In RunPod terminal:
```bash
cd /workspace
unzip dataset.zip  # if you zipped it
bash setup_runpod.sh
python train_yolov11.py
```

### 5. Download Trained Model
After training (2-4 hours), download:
```
/workspace/mobile_out_detection/train/weights/best.pt
```

## 📊 What to Expect

**Training Stats:**
- Time: 2-4 hours
- GPU Usage: ~90%
- Cost: $2-10 (depending on spot/on-demand)
- Output: best.pt model file (~6-50 MB depending on model size)

**Performance Targets:**
- mAP50: >0.70 (higher is better)
- Can detect mobile phones with high accuracy
- Can count OUT objects in images/videos

## 🔥 Test Your Model

After training, test it:

```bash
# Test on single image
python inference.py \
    --model mobile_out_detection/train/weights/best.pt \
    --source test_image.jpg \
    --output result.jpg

# Test on video
python inference.py \
    --model mobile_out_detection/train/weights/best.pt \
    --source video.mp4 \
    --output output_video.mp4

# Batch process images
python inference.py \
    --model mobile_out_detection/train/weights/best.pt \
    --source images_folder/ \
    --output results/
```

## 📖 Full Documentation

See **RUNPOD_TRAINING_GUIDE.md** for:
- Detailed explanations
- Troubleshooting
- Advanced configurations
- Cost optimization tips

## 🎛️ Configuration Options

Edit `train_yolov11.py` before training to customize:

```python
# Model size (accuracy vs speed tradeoff)
MODEL_SIZE = "yolo11n.pt"  # Options: n, s, m, l, x

# Training duration
EPOCHS = 300  # More epochs = better model (but longer)

# Batch size (adjust based on GPU memory)
BATCH_SIZE = 64  # Reduce if you get memory errors

# Image size
IMG_SIZE = 512  # Matches your dataset
```

## ⚠️ Important Notes

1. **Don't stop the pod mid-training** - You'll lose progress
2. **Download best.pt immediately** after training
3. **Use spot instances** to save 50-70% on costs
4. **Start with yolo11n** (fastest) and upgrade if needed

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce BATCH_SIZE to 32 or 16 |
| Can't find dataset | Check data.yaml paths |
| CUDA not available | Run `nvidia-smi` to verify GPU |
| Training too slow | Verify GPU usage with `nvidia-smi` |

## 📞 Support

For detailed help, check the full guide:
- **RUNPOD_TRAINING_GUIDE.md** - Complete instructions
- RunPod Discord - Community support
- Ultralytics Docs - YOLOv11 documentation

---

**Ready to train? Follow the steps above and you'll have a working model in 3-4 hours! 🚀**
