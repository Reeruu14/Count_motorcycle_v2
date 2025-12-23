# 🎥 WebRTC Webcam Support untuk Cloud

## 📌 PENTING: Cara Setup Yang Benar

### Step 1️⃣: Install Dependencies
```bash
pip install -r requirements.txt
```

Pastikan file ini include:
- `streamlit-webrtc>=0.47.0`
- `aiortc>=1.5.0`
- `av>=10.0.0`
- `opencv-python-headless>=4.8.0`

### Step 2️⃣: Test Dulu
```bash
# Test WebRTC setup
streamlit run test_webrtc.py
```

Pastikan semua dependencies ✅ dan WebRTC bisa stream

### Step 3️⃣: Jalankan Aplikasi
```bash
streamlit run streamlit_app.py
```

---

## 🎯 Mode Webcam Yang Tersedia

### 📹 **Webcam (Local)**
- ✅ Paling cepat & responsif
- ✅ Tracking yang akurat
- ❌ Hanya untuk lokal (localhost)
- **Use case:** Development, local testing

### 📹 **Webcam (WebRTC)** ⭐ BARU
- ✅ Bekerja di cloud & lokal
- ✅ Browser-based streaming
- ⚠️ Sedikit lebih lambat
- **Use case:** Cloud deployment, Streamlit Cloud

---

## 🚀 Deployment ke Cloud

### Streamlit Cloud
1. Push code ke GitHub
2. Buka https://streamlit.io/cloud
3. Deploy dengan repository Anda
4. Pilih **📹 Webcam (WebRTC)** saat di cloud

**Auto detect:** App otomatis pilih mode yang sesuai

---

## ⚙️ Troubleshooting

### ❌ "WebRTC tidak bekerja"

**Check 1: Kamera permissions**
- Browser minta izin akses kamera → Allow
- Pastikan HTTPS (cloud) atau localhost (lokal)

**Check 2: Dependencies**
```bash
pip install streamlit-webrtc aiortc av --upgrade
```

**Check 3: Network**
- Check internet connection
- STUN server accessible? 
  - Default: `stun.l.google.com:19302`
  - Alternative: `stun.stunprotocol.org:3478`

**Check 4: Browser**
- Chrome/Firefox/Safari (modern versions)
- Clear cache & cookies
- Try incognito mode

### ❌ "Frame tidak terlihat"

**Solution 1: Simplify model**
```python
# Gunakan YOLOv8 Nano (lebih ringan)
model_path = "yolov8n.pt"
```

**Solution 2: Lower resolution**
- Update di line ~420 dalam streamlit_app.py
- Set `max_width = 480` (default: 640)

**Solution 3: Reduce detection frequency**
```python
# Process setiap 2 frame
if frame_count % 2 == 0:
    process_frame(...)
```

### ❌ "Memory error"

**Solution:**
- Gunakan model yang lebih kecil (Nano)
- Reduce frame resolution
- Upgrade to paid Streamlit Cloud plan

---

## 📊 Performance Tips

| Environment | Recommended Mode | Notes |
|-----------|-----------------|-------|
| **Lokal** | Webcam (Local) | Paling cepat |
| **Cloud** | Webcam (WebRTC) | Sedikit latency |
| **Testing** | Upload Video | Paling stabil |

---

## 🔧 Advanced Config

### Custom STUN Server
Edit dalam `streamlit_app.py` line ~430:
```python
rtc_configuration = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.stunprotocol.org:3478"]},
        {"urls": ["stun:stun1.stunprotocol.org:3478"]},
    ]}
)
```

### Disable Tracking (lebih cepat)
Hapus atau comment `tracker.update()` untuk performa lebih baik

---

## ✅ Checklist Before Deploy

- [ ] `pip install -r requirements.txt` 
- [ ] `streamlit run test_webrtc.py` → Semua ✅
- [ ] Model file ada di folder `models/`
- [ ] Push ke GitHub
- [ ] Deploy di Streamlit Cloud
- [ ] Test 📹 Webcam (WebRTC) di cloud

---

## 📚 Links

- 🎬 streamlit-webrtc: https://github.com/whitphx/streamlit-webrtc
- 📖 Docs: https://github.com/whitphx/streamlit-webrtc/blob/main/README.md
- 🆘 Issues: https://github.com/whitphx/streamlit-webrtc/issues

**🎉 Sekarang webcam bekerja di mana saja!**
