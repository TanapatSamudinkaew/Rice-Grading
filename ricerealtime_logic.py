import cv2
import numpy as np

def process_rice_logic(img, h_range, s_range, v_range, yellow_sensitivity):
    # --- 1. เตรียมภาพ (Preprocessing) ---
    # ใช้ Bilateral Filter เพื่อลด Noise แต่ยังคงความคมของขอบเมล็ดข้าวไว้
    blurred = cv2.bilateralFilter(img, 9, 75, 75)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # สร้าง Mask สำหรับตรวจจับสีเสีย (ตามค่าที่จูนจาก Slider)
    lower_fail = np.array([h_range[0], s_range[0], v_range[0]])
    upper_fail = np.array([h_range[1], s_range[1], v_range[1]])
    fail_mask = cv2.inRange(hsv, lower_fail, upper_fail)

    # --- 2. การแยกพื้นหลัง (Segmentation) ---
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ลบจุดเล็กๆ และเติมเต็มรูในเมล็ด
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # --- 3. ระยะห่าง (Distance Transform) เพื่อแยกเมล็ดที่ติดกัน ---
    # คำนวณระยะห่างจากพิกเซลสีขาวไปยังพื้นหลังสีดำ
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # หา "ยอด" ของระยะห่าง (จุดที่อยู่กลางเมล็ดที่สุด)
    # ปรับ 0.4-0.6 ตามความเบียดของเมล็ด
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # --- 4. การนับและวิเคราะห์ทีละเมล็ด ---
    # ใช้ Connected Components เพื่อให้แต่ละเมล็ดมี ID ของตัวเอง
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    display_img = img.copy()
    stats = {"Pass": 0, "Fail": 0}

    for c in contours:
        # ขยายขอบเขตจากจุดศูนย์กลาง (sure_fg) กลับมาเป็นขนาดเมล็ดจริง
        # โดยการใช้ Bounding Box ครอบจุดกึ่งกลางแล้วขยายเล็กน้อย
        x, y, w, h = cv2.boundingRect(c)
        
        # ค้นหาขอบเขตเมล็ดข้าวที่แท้จริงรอบๆ จุดนี้ในภาพต้นฉบับ
        # (ใช้พื้นที่เดิมจาก thresh เพื่อความแม่นยำ)
        grain_roi_thresh = opening[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
        grain_roi_fail = fail_mask[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
        
        total_pixels = cv2.countNonZero(grain_roi_thresh)
        if total_pixels < 100: continue # กรองฝุ่น
        
        # หาพิกเซลสีเสียที่ทับซ้อนกับตัวเมล็ด
        fail_pixels = cv2.countNonZero(cv2.bitwise_and(grain_roi_thresh, grain_roi_fail))
        fail_ratio = fail_pixels / (total_pixels + 1e-5)

        # --- 5. ตัดสินผลและวาดกรอบ ---
        if fail_ratio > yellow_sensitivity:
            label, color = "Fail", (0, 0, 255) # แดง
            stats["Fail"] += 1
        else:
            label, color = "Pass", (0, 255, 0) # เขียว
            stats["Pass"] += 1

        # ตีกรอบสี่เหลี่ยมรอบเมล็ดที่แยกได้
        cv2.rectangle(display_img, (x-2, y-2), (x+w+2, y+h+2), color, 2)
        cv2.putText(display_img, label, (x, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return display_img, stats