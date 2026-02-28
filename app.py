import streamlit as st
import cv2
import numpy as np
from PIL import Image

# นำเข้า Logic
from riceimg_logic import process_rice_logic
from ricerealtime_logic import process_rice_logic as process_realtime

# --- UI Configuration ---
st.set_page_config(page_title="Rice Quality Inspection AI", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid #dee2e6 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }
    div[data-testid="stMetricValue"] { color: #2ecc71 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("🌾 Rice Grading System")
    app_mode = st.selectbox("Select Mode", ["📤 Upload Image", "📷 Real-time Camera"])
    
    st.divider()
    
    # --- เพิ่มส่วนเลือกกล้อง (Camera Selection) ---
    st.subheader("📷 Camera Settings")
    # สร้างรายชื่อกล้องให้เลือก (0, 1, 2, 3) 
    # ปกติ Iriun มักจะไปอยู่ที่ 1 หรือ 2 ส่วน EGA อาจจะเป็น 1 หรือ 2 เช่นกัน
    cam_index = st.selectbox("Select Camera Source", [0, 1, 2, 3], index=0, 
                             help="0: กล้องหลัก, 1-3: กล้องแยก/Iriun")
    
    st.divider()
    st.subheader("⚙️ Analysis Settings")
    dist_threshold = st.slider("Separation Sensitivity", 0.1, 0.9, 0.4)
    yellow_threshold = st.slider("Spoiled Sensitivity", 0.05, 0.5, 0.12)

# --- Main Dashboard ---
st.title("Rice Quality Dashboard")
col_main, col_stats = st.columns([3, 1])

if app_mode == "📤 Upload Image":
    with col_stats:
        st.subheader("Input")
        uploaded_file = st.file_uploader("Choose a rice image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        res_img, stats = process_rice_logic(img_bgr, dist_threshold, yellow_threshold)
        
        with col_main:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            is_success, buffer = cv2.imencode(".jpg", res_img)
            st.download_button("💾 Download Results", buffer.tobytes(), "rice_analysis.jpg", "image/jpeg")
            
        with col_stats:
            st.subheader("Results")
            total = sum(stats.values())
            st.metric("Total Count", total)
            for k, v in stats.items():
                icon = "✅" if k in ["Good", "Pass"] else "❌"
                st.write(f"{icon} **{k}:** {v}")

else: # Mode: Real-time Camera
    with col_stats:
        st.subheader("Camera Control")
        run_camera = st.toggle("Start Camera", value=False)
        st_total = st.empty()
        st_details = st.empty()

    img_placeholder = col_main.empty()

    if run_camera:
        # ใช้ cam_index ที่เลือกจาก Sidebar
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            st.error(f"❌ ไม่สามารถเปิดกล้อง Index {cam_index} ได้ กรุณาลองเลือกตัวเลขอื่นใน Sidebar")
            run_camera = False # สั่งหยุดถ้าเปิดไม่ได้
        else:
            # ตั้งค่าความละเอียด
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ ไม่ได้รับสัญญาณภาพ (กรุณาเช็คว่าเลือก Index ถูกต้อง)")
                    break
                
                res_img, stats = process_realtime(frame, dist_threshold, yellow_threshold)
                
                img_placeholder.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                total = sum(stats.values())
                st_total.metric("Live Count", total)
                st_details.write(stats)
                
            cap.release()
    else:
        img_placeholder.info(f"ขณะนี้เลือกใช้งานกล้องที่ Index: {cam_index} (กรุณาเปิดสวิตช์เพื่อเริ่ม)")