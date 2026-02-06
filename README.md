# 🏍️ Motorcycle Detection & Counting System

Sitem deteksi dan hitung sepeda motor menggunakan YOLOv8 dengan real-time tracking.
https://countmotorcyclev2-5aapp6euqkkn2p7acselvhm.streamlit.app/

## ✨ Features

- 🎯 **Real-time Detection**: Deteksi sepeda motor menggunakan YOLOv8
- 📊 **Counting System**: Hitung sepeda motor yang melewati counting line
- 🎬 **Multiple Input Modes**:
  - 📹 Webcam (lokal)
  - 🖼️ Upload Image
  - 🎥 Upload Video
- ⚙️ **Customizable**:
  - Model selection (Nano, Small, Medium, Best)
  - Confidence threshold adjustment
  - Counting line position control

## 🚀 Quick Start

### Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

Akses di: `http://localhost:8501`

### Deploy ke Streamlit Cloud
1. Push ke GitHub
2. Buka https://share.streamlit.io/
3. Connect repository dan deploy
4. Streamlit Cloud akan auto-install dependencies

## 📋 File Structure

```
count_motorcycle/
├── streamlit_app.py         # Main application
├── requirements.txt         # Python dependencies (sudah updated untuk Python 3.13)
├── models/
│   └── motorcycle_detector_best.pt  # Model (optional)
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── .gitignore
└── README.md
```

## 🔧 Model Selection

Pilih dari 4 model tersedia:
- **Best Model** (Recommended): `motorcycle_detector_best.pt` - Akurasi terbaik
- **YOLOv8 Nano**: Paling ringan, cepat
- **YOLOv8 Small**: Balance antara akurasi dan speed
- **YOLOv8 Medium**: Akurasi lebih baik

## ⚠️ Catatan Penting

### Webcam (Hanya Lokal)
- ✅ Bekerja saat run lokal dengan `streamlit run`
- ❌ Tidak tersedia di Streamlit Cloud (headless environment)
- 💡 Gunakan Upload Image/Video mode untuk cloud

### Requirements
- Python 3.10 atau lebih baru
- Dependencies sudah dioptimasi untuk Python 3.13

### Large Files
- Model files (.pt) tidak disarankan di upload ke GitHub (gunakan `.gitignore`)
- Alternative: Model akan auto-download saat pertama kali (jika pakai YOLO pretrained)

## 🔗 Links

- Streamlit Cloud: https://share.streamlit.io/
- YOLOv8 Documentation: https://docs.ultralytics.com/
- GitHub: [Ganti dengan URL repo Anda]

## 📝 Usage

1. **Buka aplikasi**
2. **Select Mode**:
   - Webcam: Klik "Start Camera" (hanya lokal)
   - Image: Upload file gambar
   - Video: Upload file video
3. **Adjust Settings**:
   - Confidence Threshold (lebih tinggi = lebih strict)
   - IOU Threshold (untuk NMS)
   - Counting Line Position
4. **View Results**: Lihat detection dengan bounding boxes dan count

## 🐛 Troubleshooting

### Import Error
```
ModuleNotFoundError: No module named 'cv2'
```
**Solution**: `pip install -r requirements.txt`

### Webcam Error di Streamlit Cloud
```
❌ Webcam tidak tersedia
```
**Expected**: Webcam tidak bisa di Streamlit Cloud (headless). Gunakan Upload mode.

### Model Not Found
- Pastikan model file ada di folder `models/`
- Atau gunakan pretrained YOLO (auto-download)

### Memory Issues
- Gunakan model lebih kecil (Nano)
- Reduce input resolution
- Close other applications

## 📊 Output Metrics

- **Total Motorcycles Passed**: Jumlah sepeda motor yang melewati garis
- **Current in Frame**: Deteksi aktif dalam frame
- **FPS**: Frame rate saat ini
- **Confidence**: Threshold confidence yang digunakan

### Latest Updates:
- ✅ Fixed `numpy` compatibility dengan Python 3.13 (1.26.0+)
- ✅ Changed `opencv-python` → `opencv-python-headless` untuk Streamlit Cloud
- ✅ Fixed `pyyaml` dependency issue
- ✅ Added better error handling untuk webcam
- ✅ Compatible dengan Streamlit Cloud deployment

GITHUB LINK : https://github.com/Reeruu14/Count_motorcycle_v2.git
STREAMLIT LINK : https://countmotorcyclev2-5aapp6euqkkn2p7acselvhm.streamlit.app/
