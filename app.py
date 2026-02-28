import streamlit as st
import cv2
import numpy as np
from PIL import Image

# นำเข้าฟังก์ชัน process_rice_logic จากไฟล์ rice_logic.py ที่เราเพิ่งสร้าง
from riceimg_logic import process_rice_logic

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Rice Grading AI", layout="wide")

# แก้ไข CSS เพื่อให้ตัวอักษรเห็นชัดเจน
st.markdown("""
    <style>
    /* พื้นหลังของแอป */
    .main { background-color: #f0f2f6; }
    
    /* กล่อง Metric (Dashboard) */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 2px solid #3498db !important;
        padding: 15px !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    
    /* บังคับสีตัวอักษรใน Metric ให้เป็นสีเข้ม */
    div[data-testid="stMetric"] label, 
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1a1a1a !important;
    }

    /* ปรับแต่ง sidebar */
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
        
        # เรียกใช้งานฟังก์ชันที่ Import มา
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
            
            # ปุ่มดาวน์โหลด
            is_success, buffer = cv2.imencode(".jpg", res_img)
            st.download_button(label="💾 Save Result", data=buffer.tobytes(), file_name="analyzed_rice.jpg", mime="image/jpeg")

else:
    with col_right:
        st.subheader("Camera Control")
        run_camera = st.checkbox("Toggle Camera On/Off", value=False)
        st_count = st.empty()
        st_metrics = st.empty()

    if run_camera:
        cap = cv2.VideoCapture(0)
        img_placeholder = col_left.empty()
        
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot access camera.")
                break
            
            # เรียกใช้งานฟังก์ชันที่ Import มา
            res_img, stats = process_rice_logic(frame, dist_threshold, yellow_threshold)
            img_placeholder.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            total = sum(stats.values())
            st_count.metric("Total Grains", total)
            st_metrics.json(stats) # แสดงผลแบบ JSON ให้ดูทันสมัย
            
        cap.release()
    else:
        col_left.info("Waiting for camera to start... Please check 'Toggle Camera' in the sidebar.")