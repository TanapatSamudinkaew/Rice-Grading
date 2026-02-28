import streamlit as st
import cv2
import numpy as np
from PIL import Image

# นำเข้า Logic
from riceimg_logic import process_rice_logic
from ricerealtime_logic import process_rice_logic as process_realtime

# --- UI Configuration ---
st.set_page_config(page_title="Rice Quality AI", layout="wide")

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
    app_mode = st.selectbox("เลือกโหมดการทำงาน", ["📤 อัปโหลดรูปภาพ", "📷 กล้องสด (Real-time)"])
    
    st.divider()
    st.subheader("📷 ตั้งค่ากล้อง")
    cam_index = st.selectbox("เลือกแหล่งที่มาของกล้อง", [0, 1, 2, 3], index=0, 
                             help="0: กล้องหลัก, 1-3: กล้องแยก/Iriun")
    
    st.divider()
    st.subheader("⚙️ ตั้งค่าการวิเคราะห์")
    dist_threshold = st.slider("ความละเอียดในการแยกเมล็ด", 0.1, 0.9, 0.4)
    yellow_threshold = st.slider("ความไวการตรวจเมล็ดเสีย", 0.05, 0.5, 0.12)

# --- Main Dashboard ---
st.title("Rice Quality Dashboard")
col_main, col_stats = st.columns([3, 1])

# ฟังก์ชันสำหรับกรองเอาเฉพาะ Pass/Fail มาโชว์
def display_filtered_stats(stats_dict):
    # รวมกลุ่มถ้ามีคีย์อื่นหลุดมา หรือแสดงเฉพาะที่มี
    pass_count = stats_dict.get("Pass", stats_dict.get("Good", 0))
    fail_count = stats_dict.get("Fail", 0)
    
    # ถ้าใน logic มี Broken/Spoiled ให้รวมเข้า Fail ไปเลยเพื่อความง่าย
    if "Broken" in stats_dict: fail_count += stats_dict["Broken"]
    if "Spoiled" in stats_dict: fail_count += stats_dict["Spoiled"]
    if "Foreign" in stats_dict: fail_count += stats_dict["Foreign"]

    st.metric("จำนวนเมล็ดทั้งหมด", pass_count + fail_count)
    st.write(f"✅ **ผ่านเกณฑ์ (Pass):** {pass_count}")
    st.write(f"❌ **ไม่ผ่านเกณฑ์ (Fail):** {fail_count}")

# --- โหมดอัปโหลดรูปภาพ ---
if app_mode == "📤 อัปโหลดรูปภาพ":
    with col_stats:
        st.subheader("นำเข้าข้อมูล")
        uploaded_file = st.file_uploader("เลือกรูปภาพข้าว...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        res_img, stats = process_rice_logic(img_bgr, dist_threshold, yellow_threshold)
        
        with col_main:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            is_success, buffer = cv2.imencode(".jpg", res_img)
            st.download_button("💾 ดาวน์โหลดผลลัพธ์", buffer.tobytes(), "rice_analysis.jpg", "image/jpeg")
            
        with col_stats:
            st.subheader("ผลการวิเคราะห์")
            display_filtered_stats(stats)

# --- โหมดกล้องสด ---
else:
    with col_stats:
        st.subheader("ควบคุมกล้อง")
        run_camera = st.toggle("เปิดการใช้งานกล้อง", value=False)
        st_metrics = st.empty()

    img_placeholder = col_main.empty()

    if run_camera:
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error(f"❌ ไม่สามารถเปิดกล้อง Index {cam_index} ได้")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            while run_camera:
                ret, frame = cap.read()
                if not ret: break
                
                res_img, stats = process_realtime(frame, dist_threshold, yellow_threshold)
                img_placeholder.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with st_metrics.container():
                    display_filtered_stats(stats)
                
            cap.release()
    else:
        img_placeholder.info(f"เลือกกล้องที่ Index: {cam_index} และกดเปิดใช้งาน")