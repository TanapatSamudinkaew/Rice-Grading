import streamlit as st
import cv2
import numpy as np
from PIL import Image

# นำเข้า Logic
from rice_logic import process_rice_image
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
    st.title(" Rice Detect System")
    app_mode = st.selectbox("เลือกโหมดการทำงาน", ["📤 อัปโหลดรูปภาพ", "📷 กล้องสด (Real-time)"])
    
    st.divider()
    st.subheader("📷 ตั้งค่ากล้อง")
    cam_index = st.selectbox("เลือกแหล่งที่มาของกล้อง", [0, 1, 2, 3], index=0)
    
    st.divider()
    st.subheader("🎨 ปรับจูนแสงและสี (HSV)")
    h_range = st.slider("ช่วงเฉดสี (Hue)", 0, 180, (20, 40))
    s_range = st.slider("ความสดของสี (Saturation)", 0, 255, (40, 255))
    v_range = st.slider("ความสว่าง (Value)", 0, 255, (150, 255))
    
    st.divider()
    st.subheader("⚙️ ตั้งค่าการวิเคราะห์")
    dist_threshold = st.slider("ความละเอียดในการแยกเมล็ด", 0.1, 0.9, 0.4)

# --- Main Dashboard ---
st.title("Rice Detection Dashboard")
col_main, col_stats = st.columns([3, 1])

def display_filtered_stats(stats_dict):
    pass_count = stats_dict.get("Pass", stats_dict.get("Good", 0))
    fail_count = stats_dict.get("Fail", 0)
    
    additional_fails = ["Broken", "Spoiled", "Foreign"]
    for key in additional_fails:
        fail_count += stats_dict.get(key, 0)

    st.metric("จำนวนเมล็ดทั้งหมด", pass_count + fail_count)
    st.write(f"✅ **ผ่านเกณฑ์ (Pass):** {pass_count}")
    ##st.write(f"❌ **ไม่ผ่านเกณฑ์ (Fail):** {fail_count}")

# --- โหมดอัปโหลดรูปภาพ ---
if app_mode == "📤 อัปโหลดรูปภาพ":
    with col_stats:
        st.subheader("นำเข้าข้อมูล")
        uploaded_file = st.file_uploader("เลือกรูปภาพข้าว...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        res_img, stats = process_rice_image(img_bgr, dist_threshold)
        
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
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            st.error(f"❌ ไม่สามารถเปิดกล้อง Index {cam_index} ได้")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # ✅ เรียกแบบใหม่ (ไม่มี yellow_sensitivity)
                res_img, stats = process_realtime(
                    frame,
                    h_range,
                    s_range,
                    v_range
                )

                img_placeholder.image(
                    cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
                
                with st_metrics.container():
                    display_filtered_stats(stats)
                
            cap.release()
    else:
        img_placeholder.info("จูนค่าสีทางซ้ายมือ และกดเปิดกล้องเพื่อเริ่มวิเคราะห์")