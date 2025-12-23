#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motorcycle Detection Model Evaluation Script
Comprehensive evaluation framework for YOLOv8 motorcycle detector
"""

import os
import cv2
import numpy as np
import torch
import json
import time
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# ==================== MODEL EVALUATOR CLASS ====================

class MotorcycleModelEvaluator:
    """Comprehensive evaluator untuk motorcycle detection model"""
    
    def __init__(self, model_path, confidence_threshold=0.5, iou_threshold=0.5):
        """
        Initialize evaluator
        
        Args:
            model_path: Path ke model YOLO (.pt file)
            confidence_threshold: Confidence threshold untuk deteksi
            iou_threshold: IOU threshold untuk NMS
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.results = {}
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        try:
            print(f"📦 Loading model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded successfully!")
            print(f"   Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def get_model_info(self):
        """Get comprehensive model information"""
        info = {
            'model_path': str(self.model_path),
            'device': 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU',
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Model parameters
        try:
            total_params = sum(p.numel() for p in self.model.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            info['total_parameters'] = total_params
            info['trainable_parameters'] = trainable_params
            info['model_size_mb'] = os.path.getsize(self.model_path) / (1024 * 1024)
        except:
            pass
        
        return info
    
    def evaluate_image(self, image_path):
        """
        Evaluate model on single image
        
        Args:
            image_path: Path to image
        
        Returns:
            Dict with detection results
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        start_time = time.time()
        results = self.model(img, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)
        inference_time = time.time() - start_time
        
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detections.append({
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf[0]),
                    'class': int(box.cls[0])
                })
        
        return {
            'image_path': str(image_path),
            'detections': detections,
            'num_detections': len(detections),
            'inference_time_ms': inference_time * 1000,
            'fps': 1 / inference_time if inference_time > 0 else 0,
            'image_size': img.shape[:2]
        }
    
    def evaluate_video(self, video_path, sample_rate=5):
        """
        Evaluate model on video
        
        Args:
            video_path: Path to video file
            sample_rate: Process every nth frame
        
        Returns:
            Dict with video evaluation metrics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_count = 0
        detection_count = 0
        inference_times = []
        per_frame_detections = []
        
        print(f"📹 Processing video: {Path(video_path).name}")
        print(f"   Total frames: {total_frames}, FPS: {fps_video:.1f}, Resolution: {width}x{height}")
        
        pbar = tqdm(total=total_frames, desc="Processing")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    start_time = time.time()
                    results = self.model(frame, conf=self.confidence_threshold, 
                                       iou=self.iou_threshold, verbose=False)
                    inference_time = time.time() - start_time
                    
                    num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
                    detection_count += num_detections
                    inference_times.append(inference_time)
                    per_frame_detections.append(num_detections)
                
                frame_count += 1
                pbar.update(1)
        
        finally:
            pbar.close()
            cap.release()
        
        # Calculate statistics
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            min_inference_time = np.min(inference_times)
            max_inference_time = np.max(inference_times)
            
            return {
                'video_path': str(video_path),
                'total_frames': total_frames,
                'processed_frames': len(inference_times),
                'sample_rate': sample_rate,
                'resolution': f"{width}x{height}",
                'fps_video': fps_video,
                'total_detections': detection_count,
                'avg_detections_per_frame': detection_count / len(inference_times),
                'detections_by_frame': per_frame_detections,
                'inference_time_ms': {
                    'mean': avg_inference_time * 1000,
                    'std': std_inference_time * 1000,
                    'min': min_inference_time * 1000,
                    'max': max_inference_time * 1000
                },
                'fps': {
                    'mean': 1 / avg_inference_time,
                    'min': 1 / max_inference_time,
                    'max': 1 / min_inference_time
                },
                'total_inference_seconds': sum(inference_times)
            }
        
        return None
    
    def benchmark_performance(self, image_sizes=[320, 416, 512, 640, 768], num_iterations=50):
        """
        Benchmark model performance at different resolutions
        
        Args:
            image_sizes: List of image sizes to benchmark
            num_iterations: Number of iterations per size
        
        Returns:
            Dict with benchmark results
        """
        print(f"\n⚡ Benchmarking model performance...")
        benchmark_results = {}
        
        for size in image_sizes:
            print(f"   Testing {size}x{size}...")
            inference_times = []
            
            for _ in range(num_iterations):
                dummy_img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                
                start_time = time.time()
                _ = self.model(dummy_img, conf=self.confidence_threshold, verbose=False)
                inference_time = time.time() - start_time
                
                inference_times.append(inference_time)
            
            inference_times = np.array(inference_times)
            
            benchmark_results[size] = {
                'resolution': f"{size}x{size}",
                'iterations': num_iterations,
                'inference_time_ms': {
                    'mean': np.mean(inference_times) * 1000,
                    'std': np.std(inference_times) * 1000,
                    'min': np.min(inference_times) * 1000,
                    'max': np.max(inference_times) * 1000
                },
                'fps': {
                    'mean': 1 / np.mean(inference_times),
                    'min': 1 / np.max(inference_times),
                    'max': 1 / np.min(inference_times)
                }
            }
        
        return benchmark_results
    
    def evaluate_directory(self, directory_path, file_types=['.jpg', '.jpeg', '.png', '.bmp']):
        """
        Evaluate model on all images in a directory
        
        Args:
            directory_path: Path to directory with images
            file_types: Image file types to process
        
        Returns:
            Dict with evaluation results
        """
        directory = Path(directory_path)
        image_files = []
        
        for file_type in file_types:
            image_files.extend(directory.glob(f'*{file_type}'))
            image_files.extend(directory.glob(f'*{file_type.upper()}'))
        
        if not image_files:
            print(f"❌ No image files found in: {directory_path}")
            return None
        
        print(f"\n🖼️  Evaluating {len(image_files)} images...")
        
        results = []
        detection_counts = []
        inference_times = []
        
        for img_path in tqdm(image_files, desc="Evaluating"):
            result = self.evaluate_image(str(img_path))
            if result:
                results.append(result)
                detection_counts.append(result['num_detections'])
                inference_times.append(result['inference_time_ms'])
        
        if not results:
            return None
        
        return {
            'total_images': len(results),
            'total_detections': sum(detection_counts),
            'avg_detections_per_image': np.mean(detection_counts),
            'max_detections_per_image': max(detection_counts),
            'min_detections_per_image': min(detection_counts),
            'inference_time_ms': {
                'mean': np.mean(inference_times),
                'std': np.std(inference_times),
                'min': np.min(inference_times),
                'max': np.max(inference_times)
            },
            'fps': {
                'mean': 1000 / np.mean(inference_times),
                'min': 1000 / np.max(inference_times),
                'max': 1000 / np.min(inference_times)
            },
            'detailed_results': results
        }
    
    def generate_report(self, output_file='model_evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold
            },
            'model_info': self.get_model_info(),
            'evaluation_results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report saved to: {output_file}")
        return report


# ==================== UTILITY FUNCTIONS ====================

def print_evaluation_summary(evaluator, img_eval=None, video_eval=None, benchmark_eval=None):
    """Print formatted evaluation summary"""
    
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    
    # Model Info
    print("\n📦 MODEL INFORMATION:")
    print("-" * 70)
    model_info = evaluator.get_model_info()
    for key, value in model_info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Image Evaluation
    if img_eval:
        print("\n🖼️  IMAGE EVALUATION:")
        print("-" * 70)
        print(f"  Total Images: {img_eval['total_images']}")
        print(f"  Total Detections: {img_eval['total_detections']}")
        print(f"  Avg Detections/Image: {img_eval['avg_detections_per_image']:.2f}")
        print(f"  Inference Time (avg): {img_eval['inference_time_ms']['mean']:.2f} ms")
        print(f"  FPS (avg): {img_eval['fps']['mean']:.2f}")
    
    # Video Evaluation
    if video_eval:
        print("\n🎥 VIDEO EVALUATION:")
        print("-" * 70)
        print(f"  Total Frames: {video_eval['total_frames']}")
        print(f"  Processed Frames: {video_eval['processed_frames']}")
        print(f"  Total Detections: {video_eval['total_detections']}")
        print(f"  Avg Detections/Frame: {video_eval['avg_detections_per_frame']:.2f}")
        print(f"  Inference Time (avg): {video_eval['inference_time_ms']['mean']:.2f} ms")
        print(f"  FPS (avg): {video_eval['fps']['mean']:.2f}")
    
    # Benchmark
    if benchmark_eval:
        print("\n⚡ PERFORMANCE BENCHMARK:")
        print("-" * 70)
        for size, metrics in benchmark_eval.items():
            print(f"  {size}x{size}:")
            print(f"    Inference: {metrics['inference_time_ms']['mean']:.2f} ± {metrics['inference_time_ms']['std']:.2f} ms")
            print(f"    FPS: {metrics['fps']['mean']:.2f}")
    
    print("\n" + "="*70)


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "motorcycle_detector_best.pt"
    TEST_IMAGE_DIR = "test_images"  # Create this folder with test images
    TEST_VIDEO_PATH = "test_video.mp4"  # Optional
    
    # Initialize evaluator
    evaluator = MotorcycleModelEvaluator(
        model_path=MODEL_PATH,
        confidence_threshold=0.5,
        iou_threshold=0.5
    )
    
    # Run evaluations
    print("\n" + "="*70)
    print("🔬 MOTORCYCLE DETECTION MODEL EVALUATION")
    print("="*70)
    
    # Image evaluation
    if os.path.exists(TEST_IMAGE_DIR):
        img_results = evaluator.evaluate_directory(TEST_IMAGE_DIR)
        evaluator.results['image_evaluation'] = img_results
    else:
        img_results = None
        print(f"⚠️  Test image directory not found: {TEST_IMAGE_DIR}")
    
    # Video evaluation
    if os.path.exists(TEST_VIDEO_PATH):
        video_results = evaluator.evaluate_video(TEST_VIDEO_PATH, sample_rate=5)
        evaluator.results['video_evaluation'] = video_results
    else:
        video_results = None
        print(f"⚠️  Test video not found: {TEST_VIDEO_PATH}")
    
    # Benchmark
    benchmark_results = evaluator.benchmark_performance(
        image_sizes=[320, 416, 512, 640, 768],
        num_iterations=50
    )
    evaluator.results['benchmark'] = benchmark_results
    
    # Print summary
    print_evaluation_summary(evaluator, img_results, video_results, benchmark_results)
    
    # Save report
    evaluator.generate_report('model_evaluation_report.json')
    
    print("\n✅ Evaluation complete!")
