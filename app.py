import streamlit as st
import cv2
import numpy as np
from PIL import Image

# นำเข้า Logic แยกตามโหมดที่ต้องการ
from riceimg_logic import process_rice_logic        # สำหรับภาพนิ่ง
from ricerealtime_logic import process_rice_logic as process_realtime # สำหรับกล้อง (เปลี่ยนชื่อเพื่อไม่ให้ซ้ำ)

# --- UI Configuration ---
st.set_page_config(page_title="Rice Quality Inspection AI", layout="wide")

# Custom CSS เพื่อความสวยงามและอ่านง่าย
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid #dee2e6 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }
    div[data-testid="stMetricValue"] { color: #2ecc71 !important; } /* สีเขียวสำหรับตัวเลขหลัก */
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar (Control Panel) ---
with st.sidebar:
    st.title("🌾 Rice Grading System")
    app_mode = st.selectbox("Select Mode", ["📤 Upload Image", "📷 Real-time Camera"])
    
    st.divider()
    st.subheader("⚙️ Analysis Settings")
    # พารามิเตอร์สำหรับแบ่งส่วนของเมล็ดข้าว (Conceptual: Watershed)
    dist_threshold = st.slider("Separation Sensitivity", 0.1, 0.9, 0.4, 
                               help="ปรับระดับการแยกเมล็ดข้าวที่วางชิดกัน")
    
    # พารามิเตอร์สำหรับตรวจจับข้าวเสีย (Conceptual: HSV Thresholding)
    yellow_threshold = st.slider("Spoiled Sensitivity", 0.05, 0.5, 0.12, 
                                 help="ปรับระดับการตรวจจับสีเหลือง/น้ำตาลของข้าวเสีย")

# --- Main Dashboard ---
st.title("Rice Quality Dashboard")
col_main, col_stats = st.columns([3, 1])

# --- Mode 1: Upload Image (เน้นประมวลผลละเอียด) ---
if app_mode == "📤 Upload Image":
    with col_stats:
        st.subheader("Input")
        uploaded_file = st.file_uploader("Choose a rice image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        # อ่านไฟล์ภาพ
        img = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # แปลงสี BGR
        
        # เรียกใช้ Logic สำหรับภาพนิ่ง (riceimg_logic.py)
        res_img, stats = process_rice_logic(img_bgr, dist_threshold, yellow_threshold)
        
        with col_main:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            # ปุ่มดาวน์โหลดผลลัพธ์
            is_success, buffer = cv2.imencode(".jpg", res_img)
            st.download_button("💾 Download Results", buffer.tobytes(), "rice_analysis.jpg", "image/jpeg")
            
        with col_stats:
            st.subheader("Results")
            total = sum(stats.values())
            st.metric("Total Count", total)
            
            # แสดงรายละเอียด Pass/Fail หรือแยกประเภท
            for k, v in stats.items():
                icon = "✅" if k == "Good" or k == "Pass" else "❌"
                st.write(f"{icon} **{k}:** {v}")

# --- Mode 2: Real-time Camera (เน้นความเร็ว) ---
else:
    with col_stats:
        st.subheader("Camera Control")
        run_camera = st.toggle("Power Camera On/Off", value=False)
        st_total = st.empty()
        st_details = st.empty()

    img_placeholder = col_main.empty()

    if run_camera:
        # เปิดกล้องด้วย CAP_DSHOW เพื่อความเร็วใน Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            st.error("Cannot access camera. Please check your system permissions.")
        
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to receive frame from camera.")
                break
            
            # เรียกใช้ Logic สำหรับ Real-time (ricerealtime_logic.py)
            # ซึ่งอาจมีการลด Pre-processing เพื่อความลื่นไหล
            res_img, stats = process_realtime(frame, dist_threshold, yellow_threshold)
            
            # แสดงภาพสด
            img_placeholder.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # อัปเดต Stats แบบสดๆ
            total = sum(stats.values())
            st_total.metric("Live Count", total)
            st_details.write(stats) # แสดงสถิติแบบ Real-time
            
        cap.release()
    else:
        img_placeholder.info("Waiting for camera... Please toggle the switch to start.")