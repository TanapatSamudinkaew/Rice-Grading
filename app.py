import streamlit as st
import cv2
import numpy as np
from PIL import Image

# นำเข้า Logic แยกตามโหมดที่ต้องการ
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
        run_camera = st.toggle("Start EGA Camera", value=False)
        st_total = st.empty()
        st_details = st.empty()

    img_placeholder = col_main.empty()

    if run_camera:
        # --- ฟังก์ชันค้นหากล้อง EGA ---
        cap = None
        # พยายามเปิด Index 1 (มักเป็นกล้อง USB) ก่อน ถ้าไม่ได้ค่อยไป 0 หรือ 2
        for idx in [1, 0, 2]:
            temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if temp_cap.isOpened():
                cap = temp_cap
                break
            temp_cap.release()

        if cap is None or not cap.isOpened():
            st.error("❌ ไม่พบกล้อง EGA! กรุณาเช็คสาย USB หรือปิดแอปอื่นที่ใช้กล้องอยู่")
        else:
            # ตั้งค่าความละเอียดสำหรับกล้อง EGA (ช่วยให้ภาพชัดขึ้น)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ การรับภาพขัดข้อง (Frame Empty)")
                    break
                
                # ประมวลผล
                res_img, stats = process_realtime(frame, dist_threshold, yellow_threshold)
                
                # แสดงผล
                img_placeholder.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                total = sum(stats.values())
                st_total.metric("Live Count", total)
                st_details.write(stats)
                
            cap.release()
    else:
        img_placeholder.info("สวิตช์กล้องปิดอยู่ กรุณาเปิด 'Start EGA Camera'")