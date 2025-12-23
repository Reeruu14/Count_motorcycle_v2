import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import os
from collections import defaultdict
from scipy.spatial import distance
import time
import imageio
import sys

# Try to import cv2
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# WebRTC for cloud webcam support
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# ==================== ENVIRONMENT DETECTION ====================
def is_cloud_environment():
    """Detect if running on Streamlit Cloud or local"""
    return "STREAMLIT_CLOUD" in os.environ or "KUBERNETES_SERVICE_HOST" in os.environ

IS_CLOUD = is_cloud_environment()

# ==================== MOTORCYCLE TRACKER ====================
class MotorcycleTracker:
    """Track motorcycles dan hitung yang melewati garis counting"""
    
    def __init__(self, frame_height, line_position=0.5):
        """
        Initialize tracker
        
        Args:
            frame_height: Tinggi frame video
            line_position: Posisi garis counting (0.0-1.0, default 0.5 = tengah)
        """
        self.frame_height = frame_height
        self.line_position = int(frame_height * line_position)
        self.tracks = {}  # {track_id: {'centroid': (x, y), 'counted': bool}}
        self.next_track_id = 0
        self.motorcycle_count = 0
        self.max_distance = 50  # Max jarak untuk match track dengan detection
        
    def update(self, detections):
        """
        Update tracks dengan detections baru
        
        Args:
            detections: List of detection boxes [[x1, y1, x2, y2], ...]
        
        Returns:
            Dict dengan track info dan motorcycle_count
        """
        # Calculate centroids dari detections
        new_centroids = []
        for box in detections:
            try:
                # Convert to numpy if needed
                if hasattr(box, 'cpu'):
                    box = box.cpu().numpy()
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                new_centroids.append((cx, cy))
            except:
                continue
        
        # Match detections dengan existing tracks
        if len(self.tracks) == 0:
            # Jika belum ada track, buat baru
            for centroid in new_centroids:
                self.tracks[self.next_track_id] = {
                    'centroid': centroid,
                    'counted': False,
                    'frames_since_seen': 0
                }
                self.next_track_id += 1
        else:
            # Match dengan existing tracks
            used_detections = set()
            used_tracks = set()
            
            # Calculate distances
            for track_id, track_data in list(self.tracks.items()):
                if len(new_centroids) == 0:
                    track_data['frames_since_seen'] += 1
                    if track_data['frames_since_seen'] > 30:  # Remove jika 30 frames tidak terdeteksi
                        del self.tracks[track_id]
                    continue
                
                distances = [
                    distance.euclidean(track_data['centroid'], centroid)
                    for centroid in new_centroids
                ]
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                
                if min_distance < self.max_distance and min_distance_idx not in used_detections:
                    # Match found
                    old_cy = track_data['centroid'][1]
                    new_cy = new_centroids[min_distance_idx][1]
                    
                    # Cek apakah melewati counting line
                    if old_cy < self.line_position <= new_cy or old_cy > self.line_position >= new_cy:
                        if not track_data['counted']:
                            self.motorcycle_count += 1
                            track_data['counted'] = True
                    
                    # Update centroid
                    track_data['centroid'] = new_centroids[min_distance_idx]
                    track_data['frames_since_seen'] = 0
                    used_detections.add(min_distance_idx)
                    used_tracks.add(track_id)
                else:
                    track_data['frames_since_seen'] += 1
                    if track_data['frames_since_seen'] > 30:
                        del self.tracks[track_id]
            
            # Create new tracks untuk detections yang tidak match
            for i, centroid in enumerate(new_centroids):
                if i not in used_detections:
                    self.tracks[self.next_track_id] = {
                        'centroid': centroid,
                        'counted': False,
                        'frames_since_seen': 0
                    }
                    self.next_track_id += 1
        
        return {
            'count': self.motorcycle_count,
            'current_detections': len(new_centroids),
            'active_tracks': len(self.tracks)
        }
    
    def draw_line(self, frame):
        """Draw counting line di frame using PIL"""
        # Convert numpy array to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(frame)
        else:
            frame_pil = frame
        
        draw = ImageDraw.Draw(frame_pil)
        h, w = frame_pil.size[1], frame_pil.size[0]
        
        color = (0, 255, 255)  # Cyan (RGB)
        thickness = 2
        
        # Draw line
        draw.line(
            [(0, self.line_position), (w, self.line_position)],
            fill=color,
            width=thickness
        )
        
        # Draw text
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text(
            (10, self.line_position - 10),
            "Counting Line",
            fill=color,
            font=font
        )
        
        return np.array(frame_pil)

# ==================== PAGE CONFIGURATION ====================

# Page configuration
st.set_page_config(
    page_title="Motorcycle Detection & Counting",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #FF6B35;
        text-align: center;
        padding: 1rem 0;
    }
    h2 {
        color: #4ECDC4;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .big-number {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏍️ Motorcycle Detection & Counting System")

# Initialize session state
if "motorcycle_count" not in st.session_state:
    st.session_state.motorcycle_count = 0
if "model" not in st.session_state:
    st.session_state.model = None
if "running" not in st.session_state:
    st.session_state.running = False
if "tracker" not in st.session_state:
    st.session_state.tracker = None

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_options = {
        "Best Model (Recommended)": "motorcycle_detector_best.pt",
        "YOLOv8 Nano": "yolov8n.pt",
        "YOLOv8 Small": "yolov8s.pt",
        "YOLOv8 Medium": "yolov8m.pt"
    }
    
    # Note: Model files must be in the repository for Streamlit Cloud deployment
    
    selected_model = st.selectbox(
        "Select Model:",
        list(model_options.keys()),
        index=0,
        help="Pilih model yang ingin digunakan untuk deteksi"
    )
    model_path = model_options[selected_model]
    
    # Confidence threshold
    conf_threshold = st.slider(
        "Confidence Threshold:",
        min_value=0.1,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Semakin tinggi = semakin strict dalam deteksi"
    )
    
    # IOU threshold
    iou_threshold = st.slider(
        "IOU Threshold:",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Threshold untuk NMS (Non-Maximum Suppression)"
    )
    
    # Counting line position
    st.markdown("---")
    st.subheader("🎯 Counting Settings")
    line_position = st.slider(
        "Counting Line Position:",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Posisi garis counting (0.1 = atas, 0.9 = bawah)"
    )
    
    # Detection mode
    st.markdown("---")
    
    # Adjust available modes based on environment
    if IS_CLOUD:
        st.subheader("📋 Detection Mode")
        if WEBRTC_AVAILABLE:
            st.info("☁️ Running on Streamlit Cloud - WebRTC Webcam Enabled")
            detection_mode = st.radio(
                "Choose Mode:",
                ["📹 Webcam (WebRTC)", "🖼️ Upload Image", "🎥 Upload Video"],
                help="Pilih sumber input untuk deteksi"
            )
        else:
            st.warning("⚠️ WebRTC tidak tersedia - gunakan Upload Image/Video")
            detection_mode = st.radio(
                "Choose Mode:",
                ["🖼️ Upload Image", "🎥 Upload Video"],
                help="Pilih sumber input untuk deteksi"
            )
    else:
        detection_mode = st.radio(
            "Detection Mode:",
            ["📹 Webcam (Local)", "📹 Webcam (WebRTC)", "🖼️ Upload Image", "🎥 Upload Video"],
            help="Pilih sumber input untuk deteksi"
        )
    
    # Display info
    st.markdown("---")
    st.subheader("📊 Info")
    if st.session_state.model:
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        st.info(f"✅ Model loaded\n\n📍 Device: {device_info}")
    else:
        st.warning("⚠️ Model not loaded yet")


@st.cache_resource
def load_model(model_path):
    """Load YOLO model"""
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None
        
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


def process_frame_with_tracking(frame, model, tracker, conf, iou, line_position):
    """Process frame with detection and tracking"""
    if frame is None:
        return None, None, {}
    
    # Process frame for detection
    annotated_frame, detections = process_frame(frame, model, conf, iou)
    
    # Update tracker
    if detections:
        track_info = tracker.update(detections)
    else:
        track_info = tracker.update([])
    
    # Draw counting line
    annotated_frame = tracker.draw_line(annotated_frame)
    
    return annotated_frame, detections, track_info


def process_frame(frame, model, conf, iou):
    """Process single frame for detection"""
    if frame is None:
        return None, []
    
    # Run inference
    results = model(frame, conf=conf, iou=iou, verbose=False)
    
    # Draw results
    annotated_frame = results[0].plot()
    
    # Extract detections as simple box coordinates
    detections = []
    if results[0].boxes:
        for box in results[0].boxes:
            try:
                # Extract coordinates as list
                coords = box.xyxy[0].cpu().numpy()
                detections.append(coords)
            except:
                pass
    
    return annotated_frame, detections


def main():
    # Load model
    st.session_state.model = load_model(model_path)
    
    if st.session_state.model is None:
        st.error("❌ Gagal memuat model. Pastikan file model ada di folder yang benar.")
        return
    
    # Main content based on selected mode
    
    # WebRTC Webcam (Cloud & Local)
    if detection_mode == "📹 Webcam (WebRTC)":
        st.subheader("🎥 Webcam Real-Time Detection (WebRTC)")
        
        if not WEBRTC_AVAILABLE:
            st.error("❌ streamlit-webrtc tidak tersedia. Install dengan: pip install streamlit-webrtc aiortc av")
            return
        
        st.info("✅ WebRTC Webcam aktif - Bekerja di cloud dan lokal!")
        
        try:
            from av import VideoFrame
        except ImportError:
            st.error("❌ av library tidak tersedia")
            return
        
        # Create a video processor callback
        class MotorcycleProcessor:
            def __init__(self, model, conf, iou, frame_height, line_position):
                self.model = model
                self.conf = conf
                self.iou = iou
                self.frame_height = frame_height
                self.line_position = int(frame_height * line_position)
                self.tracker = MotorcycleTracker(frame_height, line_position)
                self.frame_count = 0
                self.prev_time = time.time()
                self.fps = 0
                
            def recv(self, frame):
                try:
                    # Get frame as BGR (OpenCV format)
                    img = frame.to_ndarray(format="bgr24")
                    h, w = img.shape[:2]
                    
                    # Resize if needed (keep consistent ratio)
                    target_width = 640
                    if w != target_width:
                        scale = target_width / w
                        new_w = target_width
                        new_h = int(h * scale)
                        # Convert BGR to RGB for PIL
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else img
                        img_pil = Image.fromarray(img_rgb)
                        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        img = np.array(img_pil)
                        # Convert back to BGR for YOLO
                        if not isinstance(img, np.ndarray) or len(img.shape) != 3:
                            img = np.array(img)
                        h, w = img.shape[:2]
                    
                    # Process frame (YOLO expects BGR)
                    try:
                        annotated_frame, detections = process_frame(img, self.model, self.conf, self.iou)
                    except Exception as e:
                        print(f"Detection error: {str(e)}")
                        annotated_frame = img.copy()
                        detections = []
                    
                    # Update tracker dan hitung motorcycles
                    if detections:
                        track_info = self.tracker.update(detections)
                    else:
                        track_info = self.tracker.update([])
                    
                    # Draw counting line
                    try:
                        annotated_frame = self.tracker.draw_line(annotated_frame)
                    except Exception as e:
                        print(f"Draw line error: {str(e)}")
                    
                    # Calculate FPS
                    self.frame_count += 1
                    current_time = time.time()
                    if (current_time - self.prev_time) >= 1.0:
                        self.fps = self.frame_count / (current_time - self.prev_time)
                        self.frame_count = 0
                        self.prev_time = current_time
                    
                    # Add info text using OpenCV (more reliable)
                    if CV2_AVAILABLE and isinstance(annotated_frame, np.ndarray):
                        try:
                            cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"Current: {track_info['current_detections']}", (10, 70),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(annotated_frame, f"Total: {track_info['count']}", (10, 110),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        except Exception as e:
                            print(f"Text overlay error: {str(e)}")
                    else:
                        # Fallback: use PIL
                        try:
                            if isinstance(annotated_frame, np.ndarray):
                                frame_pil = Image.fromarray(annotated_frame.astype('uint8'))
                            else:
                                frame_pil = annotated_frame
                            
                            draw = ImageDraw.Draw(frame_pil)
                            try:
                                font = ImageFont.load_default()
                            except:
                                font = None
                            
                            draw.text((10, 10), f"FPS: {self.fps:.1f}", fill=(0, 255, 0), font=font)
                            draw.text((10, 30), f"Current: {track_info['current_detections']}", fill=(255, 0, 0), font=font)
                            draw.text((10, 50), f"Total: {track_info['count']}", fill=(0, 165, 255), font=font)
                            
                            annotated_frame = np.array(frame_pil)
                        except Exception as e:
                            print(f"PIL text error: {str(e)}")
                    
                    # Ensure frame is uint8 and BGR for output
                    if isinstance(annotated_frame, np.ndarray):
                        annotated_frame = annotated_frame.astype(np.uint8)
                    
                    # Store info di session state untuk ditampilkan di sidebar
                    st.session_state.webrtc_fps = self.fps
                    st.session_state.webrtc_current_det = track_info['current_detections']
                    st.session_state.webrtc_total_count = track_info['count']
                    st.session_state.webrtc_active_tracks = track_info['active_tracks']
                    
                    # Return as VideoFrame
                    return VideoFrame.from_ndarray(annotated_frame, format="bgr24")
                
                except Exception as e:
                    print(f"WebRTC recv error: {str(e)}")
                    # Return original frame on error
                    return frame
        
        # WebRTC configuration
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Initialize session state untuk WebRTC stats
        if "webrtc_fps" not in st.session_state:
            st.session_state.webrtc_fps = 0
        if "webrtc_current_det" not in st.session_state:
            st.session_state.webrtc_current_det = 0
        if "webrtc_total_count" not in st.session_state:
            st.session_state.webrtc_total_count = 0
        if "webrtc_active_tracks" not in st.session_state:
            st.session_state.webrtc_active_tracks = 0
        
        # Get frame height untuk tracker (default 480)
        frame_height = 480
        
        # Extract model to local variable (avoid session_state access in thread)
        model = st.session_state.model
        conf = conf_threshold
        iou = iou_threshold
        line_pos = line_position
        
        # Reset Count button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Reset Count", key="webrtc_reset"):
                st.session_state.webrtc_total_count = 0
                st.session_state.webrtc_current_det = 0
        
        st.markdown("---")
        
        # Create streamer - Auto start
        webrtc_ctx = webrtc_streamer(
            key="motorcycle-detection-webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
            video_processor_factory=lambda: MotorcycleProcessor(model, conf, iou, frame_height, line_pos),
            desired_playing_state=True  # Auto-start, user can click STOP button
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 📊 Status")
            if webrtc_ctx.state.playing:
                st.success("✅ Webcam aktif - Detection & Counting sedang berjalan")
            else:
                st.info("⏸️ Webcam dihentikan - Klik START untuk melanjutkan")
        
        with col2:
            st.markdown("### 📈 Statistics")
            st.metric("🏆 Total Count", st.session_state.webrtc_total_count)
            st.metric("📍 Current", st.session_state.webrtc_current_det)
            st.metric("⚡ FPS", f"{st.session_state.webrtc_fps:.1f}")
            st.metric("🔍 Tracking", st.session_state.webrtc_active_tracks)
    
    # Local Webcam (Lokal saja)
    elif detection_mode == "📹 Webcam (Local)":
        st.subheader("🎥 Webcam Real-Time Detection (Local)")
        
        # Warning about webcam limitations
        st.warning(
            "⚠️ **Mode ini hanya bekerja di lokal (localhost)!**\n\n"
            "Gunakan WebRTC untuk cloud atau fitur:\n"
            "- 📷 **Upload Image** untuk mendeteksi gambar statis\n"
            "- 🎥 **Upload Video** untuk mendeteksi dari file video"
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Placeholder untuk video
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
        
        with col2:
            st.markdown("### 📈 Statistics")
            total_count_placeholder = st.empty()
            current_det_placeholder = st.empty()
            fps_placeholder = st.empty()
            conf_display = st.empty()
        
        # Start/Stop button
        col1, col2, col3 = st.columns(3)
        with col1:
            start_btn = st.button("🟢 Start Camera", key="start_btn")
        with col2:
            stop_btn = st.button("🔴 Stop Camera", key="stop_btn")
        with col3:
            reset_btn = st.button("🔄 Reset Count", key="reset_btn")
        
        if start_btn:
            st.session_state.running = True
            st.session_state.tracker = None
        
        if stop_btn:
            st.session_state.running = False
        
        if reset_btn:
            st.session_state.motorcycle_count = 0
            st.session_state.tracker = None
        
        if st.session_state.running:
            try:
                try:
                    import cv2
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame
                    
                    if not cap.isOpened():
                        st.error("❌ Webcam tidak tersedia. \n\nCatatan: Webcam hanya bekerja di lokal, tidak tersedia di Streamlit Cloud.")
                        st.info("💡 Gunakan fitur 'Upload Video' atau 'Upload Image' untuk testing di Streamlit Cloud")
                        st.session_state.running = False
                        return
                    
                    # Set camera properties untuk mencegah auto-zoom
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable auto-focus
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Disable auto-exposure
                    cap.set(cv2.CAP_PROP_EXPOSURE, -8)  # Manual exposure (lebih terang)
                    cap.set(cv2.CAP_PROP_ZOOM, 0)  # No zoom (reset ke default)
                except ImportError:
                    st.error("❌ OpenCV (cv2) tidak tersedia. Instalasi dengan: pip install opencv-python")
                    st.info("💡 Gunakan fitur 'Upload Video' atau 'Upload Image' untuk alternative")
                    st.session_state.running = False
                    return
            except Exception as e:
                st.error(f"❌ Error saat mengakses webcam: {str(e)}")
                st.info("💡 Gunakan fitur 'Upload Video' atau 'Upload Image' untuk alternative")
                st.session_state.running = False
                return
            
            # Get frame dimensions untuk tracker (dengan timeout)
            frame_height = 480
            for _ in range(5):  # Try 5 times
                ret, test_frame = cap.read()
                if ret:
                    frame_height = test_frame.shape[0]
                    break
            
            # Initialize tracker jika belum
            if st.session_state.tracker is None:
                st.session_state.tracker = MotorcycleTracker(frame_height, line_position)
            
            frame_count = 0
            prev_time = time.time()
            fps = 0
            max_frames = 500  # Process max 500 frames before Streamlit reruns
            
            try:
                status_placeholder.success("✅ Webcam aktif - Loading frames...")
                
                for frame_idx in range(max_frames):
                    if not st.session_state.running:
                        break
                    
                    ret, frame = cap.read()
                    
                    if not ret:
                        status_placeholder.error("❌ Gagal membaca frame dari webcam")
                        break
                    
                    # Process frame
                    annotated_frame, detections = process_frame(
                        frame, 
                        st.session_state.model, 
                        conf_threshold, 
                        iou_threshold
                    )
                    
                    # Update tracker
                    if detections:
                        track_info = st.session_state.tracker.update(detections)
                        total_motorcycles = track_info['count']
                        current_detections = track_info['current_detections']
                    else:
                        track_info = st.session_state.tracker.update([])
                        total_motorcycles = track_info['count']
                        current_detections = 0
                    
                    # Draw counting line
                    annotated_frame = st.session_state.tracker.draw_line(annotated_frame)
                    
                    # Calculate FPS
                    frame_count += 1
                    current_time = time.time()
                    if (current_time - prev_time) >= 1.0:
                        fps = frame_count / (current_time - prev_time)
                        frame_count = 0
                        prev_time = current_time
                    
                    # Add info to frame using PIL
                    frame_pil = Image.fromarray(annotated_frame.astype('uint8')) if isinstance(annotated_frame, np.ndarray) else annotated_frame
                    draw = ImageDraw.Draw(frame_pil)
                    try:
                        font = ImageFont.load_default()
                    except:
                        font = None
                    
                    draw.text((10, 30), f"FPS: {fps:.1f}", fill=(0, 255, 0), font=font)
                    draw.text((10, 70), f"Current: {current_detections}", fill=(255, 0, 0), font=font)
                    draw.text((10, 110), f"Total Passed: {total_motorcycles}", fill=(0, 165, 255), font=font)
                    annotated_frame = np.array(frame_pil)
                    
                    # Convert frame to RGB using PIL if needed
                    if isinstance(annotated_frame, np.ndarray):
                        annotated_frame_pil = Image.fromarray(annotated_frame.astype('uint8'))
                        annotated_frame_rgb = np.array(annotated_frame_pil)
                    else:
                        annotated_frame_rgb = annotated_frame
                    
                    # Display frame
                    frame_placeholder.image(annotated_frame_rgb, use_column_width=True)
                    
                    # Update stats
                    status_placeholder.success(f"✅ Running... | Tracking {track_info['active_tracks']} motorcycles")
                    total_count_placeholder.metric("🏆 Total Motorcycles Passed", total_motorcycles, delta=current_detections if current_detections > 0 else None)
                    current_det_placeholder.metric("📍 Current in Frame", current_detections)
                    fps_placeholder.metric("⚡ FPS", f"{fps:.1f}")
                    conf_display.metric("🎯 Confidence", f"{conf_threshold:.2f}")
                    
                    # Allow Streamlit to process button clicks
                    time.sleep(0.01)
                
                cap.release()
            
            except Exception as e:
                st.error(f"❌ Error saat menggunakan webcam: {str(e)}")
                st.info("💡 Gunakan fitur 'Upload Video' atau 'Upload Image' untuk alternative")
                st.session_state.running = False
                if 'cap' in locals():
                    cap.release()
        else:
            frame_placeholder.info("👆 Klik tombol '🟢 Start Camera' untuk memulai deteksi dan counting")


    elif detection_mode == "🖼️ Upload Image":
        st.subheader("🖼️ Image Detection")
        
        uploaded_image = st.file_uploader(
            "Upload image:",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload gambar untuk deteksi"
        )
        
        if uploaded_image is not None:
            # Convert to numpy array
            image = Image.open(uploaded_image)
            image_array = np.array(image)
            
            # Process image
            image_bgr = image_array
            
            # Process image
            annotated_frame, detections = process_frame(
                image_bgr,
                st.session_state.model,
                conf_threshold,
                iou_threshold
            )
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.markdown("### Detection Result")
                st.image(annotated_frame, use_column_width=True)
            
            # Show statistics
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🏍️ Total Motorcycles Detected", len(detections))
            
            with col2:
                if detections:
                    avg_conf = np.mean([d['conf'] for d in detections])
                    st.metric("Average Confidence", f"{avg_conf:.2%}")
            
            with col3:
                if detections:
                    max_conf = max([d['conf'] for d in detections])
                    st.metric("Max Confidence", f"{max_conf:.2%}")
            
            # Show detailed detections
            if detections:
                st.markdown("### 📋 Detection Details")
                for i, det in enumerate(detections, 1):
                    st.write(f"**Detection {i}**: Confidence = {det['conf']:.2%}")

    elif detection_mode == "🎥 Upload Video":
        st.subheader("🎥 Video Detection")
        
        uploaded_video = st.file_uploader(
            "Upload video:",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload video untuk deteksi"
        )
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            temp_video_path = "temp_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            
            # Open video with imageio
            try:
                video_data = imageio.get_reader(temp_video_path)
                fps = int(video_data.get_meta_data().get('fps', 30))
            except Exception as e:
                st.error(f"❌ Gagal membuka video: {str(e)}")
                return
            
            # Get video properties
            frames = list(video_data)
            total_frames = len(frames)
            
            st.info(f"📊 Total Frames: {total_frames} | FPS: {fps}")
            
            # Frame slider
            frame_number = st.slider("Select frame:", 0, total_frames - 1, 0)
            
            if 0 <= frame_number < len(frames):
                frame = frames[frame_number]
                
                # Ensure frame is numpy array
                if not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                # Process frame
                annotated_frame, detections = process_frame(
                    frame,
                    st.session_state.model,
                    conf_threshold,
                    iou_threshold
                )
                
                # Convert to PIL Image for display
                if isinstance(annotated_frame, np.ndarray):
                    annotated_rgb = Image.fromarray(annotated_frame.astype('uint8'))
                else:
                    annotated_rgb = annotated_frame
                
                st.image(annotated_rgb, use_column_width=True)
                
                # Show metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🏍️ Motorcycles in Frame", len(detections))
                with col2:
                    st.metric("Frame Number", f"{frame_number} / {total_frames}")
            
            # Clean up temp file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)


if __name__ == "__main__":
    main()
