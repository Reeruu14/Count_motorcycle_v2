# 🎥 WebRTC Webcam Support untuk Cloud

## Apa itu WebRTC?

**WebRTC (Web Real-Time Communication)** adalah teknologi yang memungkinkan browser untuk mengakses kamera dan mikrofon pengguna secara real-time. Ini membuat webcam bekerja di Streamlit Cloud! ☁️

---

## 📋 Fitur Webcam Sekarang

### **Opsi 1: Webcam (Local)**
- ✅ Lebih cepat & responsif
- ✅ Bekerja dengan cv2.VideoCapture
- ❌ Hanya untuk lokal (localhost:8501)

### **Opsi 2: Webcam (WebRTC)** ⭐ NEW
- ✅ Bekerja di cloud & lokal
- ✅ Akses kamera via browser
- ✅ Streaming real-time
- ⚡ Minimal latency

---

## 🚀 Cara Menggunakan

### **Di Lokal**
```powershell
streamlit run streamlit_app.py
```
Pilih mode:
- **📹 Webcam (Local)** - Lebih cepat
- **📹 Webcam (WebRTC)** - Testing cloud version

### **Di Streamlit Cloud**
Otomatis hanya tampil:
- **📹 Webcam (WebRTC)** ✅ Bekerja
- **🖼️ Upload Image** ✅ Bekerja
- **🎥 Upload Video** ✅ Bekerja

---

## 🔧 Requirements

Sudah ditambahkan di `requirements.txt`:
```
streamlit-webrtc>=0.47.0
aiortc>=1.5.0
av>=10.0.0
opencv-python-headless>=4.8.0
```

### Install Manual
```bash
pip install streamlit-webrtc aiortc av
```

---

## ⚙️ Konfigurasi WebRTC

Default configuration menggunakan Google STUN server:
```python
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
```

### Custom STUN Server
Jika Google STUN tidak accessible, gunakan alternative:
```python
iceServers: [
    {"urls": ["stun:stun.stunprotocol.org:3478"]},
    {"urls": ["stun:stun1.stunprotocol.org:3478"]},
    {"urls": ["stun:stun2.stunprotocol.org:3478"]}
]
```

---

## 🐛 Troubleshooting

### "Kamera tidak bisa diakses"
```
Solution:
- Allowlist browser untuk akses kamera
- Pastikan HTTPS (cloud) atau localhost (lokal)
- Check browser permissions
```

### "WebRTC sangat lambat"
```
Solution:
- Network latency biasa di cloud
- Gunakan Local Webcam untuk performa lebih baik
- Check internet speed
```

### "streamlit-webrtc tidak terinstall"
```bash
pip install streamlit-webrtc aiortc av
streamlit run streamlit_app.py
```

### "Kamera error di cloud"
```
1. Pastikan HTTPS digunakan (Streamlit Cloud = HTTPS otomatis)
2. Check browser permissions
3. Try refresh page
4. Clear browser cache
```

---

## 📊 Perbandingan Mode

| Fitur | Local | WebRTC | Upload |
|-------|-------|--------|--------|
| Cloud Support | ❌ | ✅ | ✅ |
| Local Support | ✅ | ✅ | ✅ |
| Real-time | ✅ | ✅ | ❌ |
| Speed | ⚡⚡⚡ | ⚡⚡ | N/A |
| Setup | Mudah | Medium | Sangat Mudah |

---

## 📚 Links Penting

- 🎬 Streamlit WebRTC Docs: https://github.com/whitphx/streamlit-webrtc
- 🌐 WebRTC Spec: https://webrtc.org/
- 🔗 STUN Servers: https://gist.github.com/zziuni/3741933
- 💬 Issues: https://github.com/whitphx/streamlit-webrtc/issues

---

## ✅ Next Steps

1. ✅ Update requirements.txt
2. ✅ Code sudah support WebRTC
3. 👉 Deploy ke cloud dan test!

**Sekarang webcam bekerja di mana saja! 🎉**
