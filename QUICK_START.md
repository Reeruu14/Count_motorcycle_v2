# 🚀 QUICK START GUIDE - WebRTC Cloud Deployment

## 📌 Langkah 1: Instalasi Dependencies

```bash
# Masuk ke folder project
cd c:\document\visikom\tbes\streamlit_deployment

# Install semua dependencies
pip install -r requirements.txt
```

**⏱️ Waktu:** ~2-5 menit

---

## 📌 Langkah 2: Test Setup

```bash
# Test apakah semua dependencies installed
python check_setup.py

# Test WebRTC (optional)
streamlit run test_webrtc.py
```

Klik **START** → Pastikan kamera bisa diakses ✅

---

## 📌 Langkah 3: Jalankan Aplikasi Lokal

```bash
streamlit run streamlit_app.py
```

Browser akan buka di `http://localhost:8501`

**Pilih mode:**
- **📹 Webcam (Local)** - Tercepat, untuk development
- **📹 Webcam (WebRTC)** - Test cloud version

---

## 📌 Langkah 4: Deploy ke Streamlit Cloud

### A. Prepare untuk Cloud

1. Push code ke GitHub
```bash
git add .
git commit -m "Add WebRTC support for cloud"
git push
```

2. Edit `requirements.txt` - pastikan sudah lengkap:
```
streamlit>=1.28.0
numpy
torch
ultralytics
streamlit-webrtc>=0.47.0
aiortc>=1.5.0
av>=10.0.0
```

### B. Deploy

1. Buka https://streamlit.io/cloud
2. Klik **"New app"**
3. Pilih:
   - Repository: `your-repo/streamlit_deployment`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. Klik **Deploy**

⏱️ Tunggu 2-5 menit sampai deploy selesai

---

## 📌 Langkah 5: Test di Cloud

1. Buka deployed app URL
2. Sidebar akan otomatis tampil:
   - ☁️ "Running on Streamlit Cloud"
   - Hanya mode: **📹 Webcam (WebRTC)** + Image + Video

3. Klik **"Webcam (WebRTC)"** 
4. Klik tombol **START**
5. Allow camera access → Harus di-allow! ✅

---

## ✅ Checklist

- [ ] `pip install -r requirements.txt` berhasil
- [ ] `python check_setup.py` semua ✅
- [ ] `streamlit run test_webrtc.py` bisa stream
- [ ] `streamlit run streamlit_app.py` jalan di lokal
- [ ] GitHub repo siap push
- [ ] Deploy di Streamlit Cloud berhasil
- [ ] WebRTC streaming di cloud berfungsi

---

## 🐛 Troubleshooting

### ❌ "Kamera tidak bekerja di cloud"

**Step 1: Check browser permissions**
- Browser harus izin akses kamera
- Buka browser settings → Privacy
- Pastikan site diizinkan akses camera

**Step 2: Reload page**
```
- Hard refresh: Ctrl+Shift+R (Windows) atau Cmd+Shift+R (Mac)
- Clear cache & cookies
- Try incognito mode
```

**Step 3: Check HTTPS**
- Streamlit Cloud sudah HTTPS ✅
- Lokal pastikan `localhost` atau `127.0.0.1`

**Step 4: Check network**
```bash
# Test STUN server
ping stun.l.google.com
```

### ❌ "Error: 'av' module not found"

```bash
pip install av aiortc
pip install -r requirements.txt --upgrade
```

### ❌ "WebRTC memory error"

**Solution:**
- Gunakan YOLOv8 Nano (lebih ringan)
- Reduce frame resolution
- Upgrade ke plan berbayar Streamlit Cloud

---

## 🎯 Performance Tips

**Di Lokal:**
- Gunakan **Webcam (Local)** → Tercepat
- FPS normal: 20-30 fps

**Di Cloud:**
- Gunakan **Webcam (WebRTC)** → Sedikit latency
- FPS normal: 10-15 fps
- Network latency: 200-500ms normal

---

## 📚 Additional Resources

📖 Full Guide: `WEBRTC_GUIDE.md`
📖 Deployment: `CLOUD_DEPLOYMENT.md`
🧪 Test file: `test_webrtc.py`

---

## 🆘 Masih Bermasalah?

1. Check `test_webrtc.py` untuk diagnostik
2. Read `WEBRTC_GUIDE.md` section Troubleshooting
3. Check Streamlit logs: Bottom right → Manage app → View logs
4. Check browser console: F12 → Console tab

---

**Good luck! 🚀 Semoga lancar!**
