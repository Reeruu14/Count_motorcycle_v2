#!/usr/bin/env python3
"""
Simple WebRTC test untuk debug
"""
import streamlit as st

st.set_page_config(page_title="WebRTC Test", layout="wide")

st.title("🧪 WebRTC Webcam Test")

# Check dependencies
st.subheader("📋 Dependency Check")

deps = {
    "streamlit": False,
    "streamlit-webrtc": False,
    "aiortc": False,
    "av": False,
    "torch": False,
    "ultralytics": False,
}

try:
    import streamlit
    deps["streamlit"] = True
except:
    pass

try:
    from streamlit_webrtc import webrtc_streamer
    deps["streamlit-webrtc"] = True
except:
    pass

try:
    import aiortc
    deps["aiortc"] = True
except:
    pass

try:
    import av
    deps["av"] = True
except:
    pass

try:
    import torch
    deps["torch"] = True
except:
    pass

try:
    from ultralytics import YOLO
    deps["ultralytics"] = True
except:
    pass

# Display status
col1, col2 = st.columns([2, 2])
for i, (name, status) in enumerate(deps.items()):
    if i < len(deps) // 2:
        with col1:
            if status:
                st.success(f"✅ {name}")
            else:
                st.error(f"❌ {name}")
    else:
        with col2:
            if status:
                st.success(f"✅ {name}")
            else:
                st.error(f"❌ {name}")

# Show install command
if not all(deps.values()):
    st.error("⚠️ Ada dependencies yang belum terinstall")
    st.code("pip install -r requirements.txt", language="bash")
else:
    st.success("✅ Semua dependencies tersedia!")
    
    # Test WebRTC
    st.subheader("🎥 WebRTC Test")
    
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
        
        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        webrtc_ctx = webrtc_streamer(
            key="test",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
        )
        
        if webrtc_ctx.state.playing:
            st.success("✅ WebRTC Streaming Active!")
            st.info("Kamera Anda sedang aktif. Jika tidak bisa akses kamera, periksa browser permissions.")
        else:
            st.info("👆 Klik START untuk test WebRTC")
            
    except Exception as e:
        st.error(f"❌ WebRTC Error: {str(e)}")
        st.code(str(e), language="text")

st.markdown("---")
st.subheader("📝 Troubleshooting")
st.markdown("""
### Jika WebRTC tidak bekerja:

1. **Install dependencies:**
   ```bash
   pip install streamlit-webrtc aiortc av
   ```

2. **Browser permissions:**
   - Allow akses kamera saat diminta browser
   - Pastikan menggunakan HTTPS (atau localhost)

3. **Network:**
   - Check internet connection
   - Pastikan firewall tidak block WebRTC

4. **Alternative:**
   - Gunakan Webcam (Local) untuk performance lebih baik
   - Atau Upload Video/Image untuk alternative

### Environment Variables (opsional):
```bash
STREAMLIT_LOGGER_LEVEL=debug
```

### Untuk debug lebih detail:
```bash
streamlit run streamlit_app.py --logger.level=debug
```
""")
