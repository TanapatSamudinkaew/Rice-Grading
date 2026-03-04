import cv2
import numpy as np

def process_rice_logic(img, h_range, s_range, v_range):
    # --- Configuration ---
    FAIL_THRESHOLD = 0.15      # สัดส่วนสีเสียในเมล็ด (15%)
    MIN_GRAIN_AREA = 200       # ขนาดพื้นที่ขั้นต่ำของเมล็ดข้าว
    SOLIDITY_THRESHOLD = 0.94  # ความ "ตัน" ของเมล็ด (ปรับให้สูงขึ้นตามคำแนะนำก่อนหน้า)
    MIN_GRAIN_LENGTH = 50      # ความยาวขั้นต่ำของเมล็ด (ต้องปรับค่านี้ตามหน้างานจริง)

    # --- 1. Preprocessing ---
    blurred = cv2.bilateralFilter(img, 9, 75, 75)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    lower_fail = np.array([h_range[0], s_range[0], v_range[0]])
    upper_fail = np.array([h_range[1], s_range[1], v_range[1]])
    fail_mask = cv2.inRange(hsv, lower_fail, upper_fail)

    # --- 2. Segmentation ---
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # --- 3. Contour Detection ---
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    display_img = img.copy()
    stats = {"Pass": 0, "Fail": 0}

    for c in contours:
        # กรองพื้นที่เล็กเกินไป (Noise)
        area = cv2.contourArea(c)
        if area < MIN_GRAIN_AREA:
            continue

        # --- กฎข้อที่ 1: ตรวจสอบรูปทรง (Solidity) ---
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        is_solidity_fail = solidity < SOLIDITY_THRESHOLD

        # --- กฎข้อที่ 2: ตรวจสอบความยาว (minAreaRect) ---
        # ใช้ minAreaRect เพื่อหาขนาดเมล็ดที่หมุนตามองศาจริงของเม็ดข้าว
        rect = cv2.minAreaRect(c)
        # rect[1] จะได้ (width, height) เรานำมาเรียงใหม่ให้ด้านที่ยาวที่สุดเป็น length เสมอ
        width, length = sorted(rect[1]) 
        
        is_length_fail = length < MIN_GRAIN_LENGTH # ตรวจสอบเมล็ดหัก/สั้น

        # --- กฎข้อที่ 3: ตรวจสอบสีเสีย (Color Analysis) ---
        mask_single_grain = np.zeros_like(opening)
        cv2.drawContours(mask_single_grain, [c], -1, 255, -1)

        total_pixels = cv2.countNonZero(mask_single_grain)
        grain_fail_pixels = cv2.bitwise_and(mask_single_grain, fail_mask)
        fail_pixels_count = cv2.countNonZero(grain_fail_pixels)
        
        fail_ratio = fail_pixels_count / (total_pixels + 1e-5)
        is_color_fail = fail_ratio > FAIL_THRESHOLD

        # --- สรุปผลการตรวจสอบ ---
        x, y, w, h = cv2.boundingRect(c)
        
        # ถ้านิยามของ Fail คือ หัก OR สีเสีย OR ทรงเบี้ยว
        if is_color_fail or is_solidity_fail or is_length_fail:
            # ลำดับการแสดงเหตุผล
            if is_color_fail:
                reason = "Color"
            elif is_length_fail:
                reason = "Short/Broken"
            else:
                reason = "Shape"
                
            label = f"Fail ({reason})"
            color = (0, 0, 255) # สีแดง
            stats["Fail"] += 1
        else:
            label = "Pass"
            color = (0, 255, 0) # สีเขียว
            stats["Pass"] += 1

        # วาดเส้นรอบรูปและแสดงสถานะ
        cv2.drawContours(display_img, [c], -1, color, 2)
        # แสดงความยาวจริง (Optional - สำหรับ Debug)
        cv2.putText(display_img, f"{label} L:{int(length)}", (x, y-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return display_img, stats