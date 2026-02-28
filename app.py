import streamlit as st
import cv2
import numpy as np
from PIL import Image
from riceimg_logic import process_rice_logic

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Rice Grading AI", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 2px solid #3498db !important;
        padding: 15px !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    div[data-testid="stMetric"] label, 
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1a1a1a !important;
    }
    .stSidebar { background-color: #262730; }
    .stSidebar .stMarkdown { color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1147/1147805.png", width=100)
    st.title("Control Panel")
    app_mode = st.selectbox("Select Mode", ["📷 Real-time Camera", "📤 Upload Image"])
    
    st.divider()
    st.subheader("⚙️ Analysis Settings")
    dist_threshold = st.slider("Grain Separation", 0.1, 0.9, 0.4, help="เลื่อนเพื่อปรับการแยกเมล็ดที่ติดกัน")
    yellow_threshold = st.slider("Spoiled Sensitivity", 0.05, 0.5, 0.12, help="ปรับความไวในการตรวจเมล็ดสีเหลือง/เสีย")

# --- Main Content ---
st.title("🌾 Rice Quality Inspection AI")
col_left, col_right = st.columns([2.5, 1])

if app_mode == "📤 Upload Image":
    with col_right:
        st.subheader("Data Input")
        uploaded_file = st.file_uploader("Upload rice image...", type=["jpg", "png", "jpeg"])
        
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        res_img, stats = process_rice_logic(img_bgr, dist_threshold, yellow_threshold)
        
        with col_left:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col_right:
            total = sum(stats.values())
            st.metric("Total Count", f"{total} grains")
            st.write(f"✅ Good: {stats['Good']}")
            st.write(f"❌ Broken: {stats['Broken']}")
            st.write(f"⚠️ Spoiled: {stats['Spoiled']}")
            st.write(f"🌑 Foreign: {stats['Foreign']}")
            
            is_success, buffer = cv2.imencode(".jpg", res_img)
            st.download_button(label="💾 Save Result", data=buffer.tobytes(), file_name="analyzed_rice.jpg", mime="image/jpeg")

else: # Mode: Real-time Camera
    with col_right:
        st.subheader("Camera Control")
        # ใช้ปุ่มกดแทน checkbox เพื่อความเสถียรในการเคลียร์ cache กล้อง
        run_camera = st.toggle("Start Camera", value=False)
        st_count = st.empty()
        st_metrics = st.empty()

    img_placeholder = col_left.empty()

    if run_camera:
        # ลองเปิดกล้องด้วย CAP_DSHOW (สำหรับ Windows) เพื่อความเร็ว
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # ตรวจสอบว่าเปิดกล้องได้จริงไหม
        if not cap.isOpened():
            st.error("Cannot access camera. Please check your connection or Privacy Settings.")
        else:
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame.")
                    break
                
                # ประมวลผลภาพ
                res_img, stats = process_rice_logic(frame, dist_threshold, yellow_threshold)
                
                # แสดงผลภาพแบบ Real-time
                img_placeholder.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # อัปเดต Dashboard
                total = sum(stats.values())
                st_count.metric("Total Grains", f"{total}")
                st_metrics.write(f"Good: {stats['Good']} | Broken: {stats['Broken']} | Spoiled: {stats['Spoiled']}")
                
                # เช็คอีกรอบเผื่อ user กดปิดปุ่ม toggle ระหว่าง loop
                # (Streamlit จะทำการ rerun หน้าใหม่เมื่อกด toggle)
            
            cap.release()
    else:
        img_placeholder.info("Camera is currently OFF. Please turn on 'Start Camera' in the control panel.")