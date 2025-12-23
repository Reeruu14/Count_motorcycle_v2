# Model Evaluation Configuration & Usage Guide

## Quick Start

### 1. Prerequisites
```bash
# Install required packages
pip install ultralytics torch opencv-python numpy pandas scikit-learn matplotlib seaborn tqdm
```

### 2. Prepare Test Data
```
project_root/
├── streamlit_deployment/
│   ├── model_evaluation.py
│   ├── motorcycle_detector_best.pt
│   ├── test_images/          ← Create this folder
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   └── ... (more test images)
│   └── test_video.mp4        ← Optional
```

### 3. Run Evaluation
```bash
# Navigate to project directory
cd streamlit_deployment

# Run evaluation script
python model_evaluation.py
```

---

## Detailed Usage

### Option 1: Full Evaluation

```python
from model_evaluation import MotorcycleModelEvaluator

# Initialize
evaluator = MotorcycleModelEvaluator(
    model_path="motorcycle_detector_best.pt",
    confidence_threshold=0.5,
    iou_threshold=0.5
)

# Get model info
model_info = evaluator.get_model_info()
print(model_info)

# Evaluate on images
img_results = evaluator.evaluate_directory("test_images")

# Evaluate on video (optional)
video_results = evaluator.evaluate_video("test_video.mp4", sample_rate=5)

# Benchmark performance
benchmark_results = evaluator.benchmark_performance(
    image_sizes=[320, 416, 512, 640, 768],
    num_iterations=50
)

# Generate report
evaluator.generate_report('evaluation_report.json')
```

### Option 2: Single Image Evaluation

```python
from model_evaluation import MotorcycleModelEvaluator

evaluator = MotorcycleModelEvaluator("motorcycle_detector_best.pt")

# Evaluate single image
result = evaluator.evaluate_image("path/to/image.jpg")

print(f"Detections: {result['num_detections']}")
print(f"Inference Time: {result['inference_time_ms']:.2f} ms")
print(f"FPS: {result['fps']:.2f}")
```

### Option 3: Video Evaluation

```python
from model_evaluation import MotorcycleModelEvaluator

evaluator = MotorcycleModelEvaluator("motorcycle_detector_best.pt")

# Evaluate video
video_results = evaluator.evaluate_video(
    "path/to/video.mp4",
    sample_rate=5  # Process every 5th frame
)

print(f"Total Frames: {video_results['total_frames']}")
print(f"Avg FPS: {video_results['fps']['mean']:.2f}")
print(f"Avg Detections/Frame: {video_results['avg_detections_per_frame']:.2f}")
```

### Option 4: Benchmark Only

```python
from model_evaluation import MotorcycleModelEvaluator

evaluator = MotorcycleModelEvaluator("motorcycle_detector_best.pt")

# Run benchmark at different resolutions
benchmark = evaluator.benchmark_performance(
    image_sizes=[320, 416, 512, 640],
    num_iterations=100
)

for size, metrics in benchmark.items():
    print(f"{size}x{size}: {metrics['fps']['mean']:.2f} FPS")
```

---

## Configuration Parameters

### Confidence & IOU Thresholds

| Threshold Type | Range | Default | Description |
|---|---|---|---|
| **Confidence** | 0.0 - 1.0 | 0.5 | Detection confidence level |
| **IOU** | 0.0 - 1.0 | 0.5 | Non-Maximum Suppression threshold |

**Recommendations:**
- **Conservative (High Accuracy)**: confidence=0.7, iou=0.5
- **Balanced**: confidence=0.5, iou=0.5
- **Aggressive (Speed)**: confidence=0.3, iou=0.4

### Benchmark Resolutions

```python
# Standard resolutions
image_sizes = [320, 416, 512, 640, 768]

# For different devices
edge_devices = [320, 416]
standard = [512, 640]
high_accuracy = [640, 768]
```

---

## Interpreting Results

### Metrics Explanation

**Precision**
```
What: Percentage of predictions that are correct
Formula: TP / (TP + FP)
Good Value: >95%
Meaning: Few false alarms
```

**Recall**
```
What: Percentage of actual objects found
Formula: TP / (TP + FN)
Good Value: >95%
Meaning: Few missed detections
```

**F1 Score**
```
What: Harmonic mean of precision and recall
Formula: 2 × (P × R) / (P + R)
Good Value: >0.95
Meaning: Balanced performance
```

**mAP (mean Average Precision)**
```
What: Average precision across different IoU thresholds
mAP50: IoU threshold at 0.50
mAP50-95: IoU from 0.50 to 0.95 (strict)
Good Value: mAP50 >90%, mAP50-95 >80%
```

---

## Output Files

### JSON Report
```json
{
  "metadata": {
    "timestamp": "2025-12-23T10:30:00",
    "model_path": "motorcycle_detector_best.pt",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.5
  },
  "model_info": {
    "device": "GPU (CUDA)",
    "total_parameters": 25900000,
    "model_size_mb": 86
  },
  "evaluation_results": {
    "image_evaluation": { ... },
    "video_evaluation": { ... },
    "benchmark": { ... }
  }
}
```

---

## Troubleshooting

### Issue: Out of Memory Error

**Solution:**
```python
# Reduce batch processing or use quantized model
# OR reduce image size for evaluation
evaluator.benchmark_performance(image_sizes=[320, 512])
```

### Issue: Slow Evaluation

**Solution:**
```python
# Skip video evaluation or reduce sample rate
video_results = evaluator.evaluate_video(
    "video.mp4",
    sample_rate=10  # Process every 10th frame
)

# Limit image evaluation
import os
images = os.listdir("test_images")[:50]  # First 50 images only
```

### Issue: No Test Images Found

**Solution:**
```bash
# Create test_images folder
mkdir test_images

# Add sample images
# Download from: https://roboflow.com/datasets
# Or use images from your own dataset
```

### Issue: GPU Not Detected

**Solution:**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Performance Optimization Tips

### For Faster Evaluation

```python
# 1. Reduce resolution
evaluator.benchmark_performance(image_sizes=[320, 512])

# 2. Reduce iterations
benchmark = evaluator.benchmark_performance(num_iterations=10)

# 3. Skip video evaluation
# Only run image and benchmark

# 4. Use sample_rate for video
video_results = evaluator.evaluate_video("video.mp4", sample_rate=10)
```

### For Accuracy Evaluation

```python
# 1. Use full resolution
image_sizes = [640, 768]

# 2. Use strict IOU threshold
evaluator = MotorcycleModelEvaluator(
    "model.pt",
    iou_threshold=0.75
)

# 3. Evaluate all frames in video
video_results = evaluator.evaluate_video("video.mp4", sample_rate=1)

# 4. Use more test images
# Ensure test_images contains diverse scenarios
```

---

## Batch Processing

```python
from model_evaluation import MotorcycleModelEvaluator
import os
from pathlib import Path

# Evaluate multiple models
models = [
    "motorcycle_detector_best.pt",
    "yolov8m.pt",
    "yolov8s.pt"
]

results = {}
for model_path in models:
    print(f"Evaluating {model_path}...")
    evaluator = MotorcycleModelEvaluator(model_path)
    
    img_results = evaluator.evaluate_directory("test_images")
    benchmark = evaluator.benchmark_performance(image_sizes=[320, 512, 640])
    
    results[model_path] = {
        'images': img_results,
        'benchmark': benchmark
    }
    
    # Save report
    report_name = f"report_{Path(model_path).stem}.json"
    evaluator.generate_report(report_name)

print("All evaluations complete!")
```

---

## Integration with Streamlit

```python
# In streamlit_app.py
import streamlit as st
from model_evaluation import MotorcycleModelEvaluator

if st.sidebar.checkbox("Run Model Evaluation"):
    st.write("## Model Evaluation")
    
    evaluator = MotorcycleModelEvaluator("motorcycle_detector_best.pt")
    model_info = evaluator.get_model_info()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Device", model_info['device'])
    with col2:
        st.metric("Model Size (MB)", f"{model_info['model_size_mb']:.2f}")
    with col3:
        st.metric("Parameters (M)", f"{model_info['total_parameters']/1e6:.1f}")
    
    # Run benchmarks
    if st.button("Run Benchmark"):
        with st.spinner("Running benchmark..."):
            benchmark = evaluator.benchmark_performance(
                image_sizes=[320, 512, 640],
                num_iterations=30
            )
            
            # Display results
            for size, metrics in benchmark.items():
                st.write(f"**{size}x{size}**: {metrics['fps']['mean']:.2f} FPS")
```

---

## Command Line Usage

```bash
# Basic evaluation
python model_evaluation.py

# With custom parameters (modify the script)
# Or create wrapper script:
```

```python
# evaluate_cli.py
import sys
import argparse
from model_evaluation import MotorcycleModelEvaluator

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='motorcycle_detector_best.pt')
parser.add_argument('--images', default='test_images')
parser.add_argument('--video', default='test_video.mp4')
parser.add_argument('--conf', type=float, default=0.5)
parser.add_argument('--iou', type=float, default=0.5)
parser.add_argument('--output', default='eval_report.json')

args = parser.parse_args()

evaluator = MotorcycleModelEvaluator(args.model, args.conf, args.iou)
img_results = evaluator.evaluate_directory(args.images)
benchmark = evaluator.benchmark_performance()
evaluator.generate_report(args.output)
```

```bash
# Run with custom parameters
python evaluate_cli.py \
  --model motorcycle_detector_best.pt \
  --images test_images \
  --conf 0.6 \
  --output my_report.json
```

---

## Support & Questions

For issues or questions:
1. Check MODEL_EVALUATION_REPORT.md for detailed metrics explanation
2. Review logs in evaluation report JSON
3. Consult troubleshooting section above

---

**Version**: 1.0  
**Last Updated**: December 23, 2025  
**Maintained by**: TBES Research Team
