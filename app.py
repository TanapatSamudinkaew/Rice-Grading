import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- ฟังก์ชันหลักในการประมวลผล (Core Logic) ---
def process_rice_logic(img, dist_val=0.4, yellow_sens=0.12):
    original = img.copy()
    
    # 1. White Balance
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_f, a_f, b_f = l.astype(np.float32), a.astype(np.float32), b.astype(np.float32)
    avg_a, avg_b = np.mean(a_f), np.mean(b_f)
    a_f -= ((avg_a - 128) * (l_f / 255.0) * 1.1)
    b_f -= ((avg_b - 128) * (l_f / 255.0) * 1.1)
    lab = cv2.merge([np.clip(l_f,0,255).astype(np.uint8), 
                     np.clip(a_f,0,255).astype(np.uint8), 
                     np.clip(b_f,0,255).astype(np.uint8)])
    img_wb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Pre-processing & Watershed
    gray = cv2.cvtColor(img_wb, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)
    
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 2)
    sure_bg = cv2.dilate(opening, kernel, 3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, dist_val * dist.max(), 255, 0)
    
    unknown = cv2.subtract(sure_bg, np.uint8(sure_fg))
    _, markers = cv2.connectedComponents(np.uint8(sure_fg))
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(original, markers)

    # 3. Color & Shape Analysis
    hsv = cv2.cvtColor(img_wb, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([18, 40, 120]), np.array([40, 255, 255]))
    
    stats = {"Good": 0, "Broken": 0, "Spoiled": 0, "Foreign": 0}
    
    for label in np.unique(markers):
        if label <= 1: continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area < 120 or area > 8000: continue

            x, y, w, h = cv2.boundingRect(c)
            peri = cv2.arcLength(c, True)
            circularity = 4 * np.pi * area / (peri * peri + 1e-5)
            yellow_ratio = cv2.countNonZero(yellow_mask[y:y+h, x:x+w]) / (w * h + 1e-5)

            if area < 350:
                text, color, key = "Broken", (0, 0, 255), "Broken"
            elif yellow_ratio > yellow_sens:
                text, color, key = "Spoiled", (0, 165, 255), "Spoiled"
            elif circularity > 0.7:
                text, color, key = "Foreign", (255, 0, 0), "Foreign"
            else:
                text, color, key = "Good", (0, 255, 0), "Good"
            
            stats[key] += 1
            cv2.rectangle(original, (x, y), (x+w, y+h), color, 2)
            cv2.putText(original, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
    return original, stats

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
            
            res_img, stats = process_rice_logic(frame, dist_threshold, yellow_threshold)
            img_placeholder.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            total = sum(stats.values())
            st_count.metric("Total Grains", total)
            st_metrics.json(stats) # แสดงผลแบบ JSON ให้ดูทันสมัย
            
        cap.release()
    else:
        col_left.info("Waiting for camera to start... Please check 'Toggle Camera' in the sidebar.")