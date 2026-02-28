import cv2
import numpy as np

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
            
            # --- [ส่วนที่แก้ไข] เพิ่มเงื่อนไขการกรองวัตถุที่เข้มงวดขึ้น ---
            
            # 1. กรองขนาด: ลดขนาดสูงสุดลงเพื่อตัดกำแพง/เฟอร์นิเจอร์ (จาก 8000 เหลือ 3000)
            if area < 120 or area > 3000: continue
            
            # 2. กรองความทึบ (Solidity): ตัดรูปทรงแปลกๆ แหว่งๆ ที่ไม่ใช่เมล็ดข้าว
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area
            if solidity < 0.75: continue # เมล็ดข้าวควรมีความทึบสูงกว่า 75%
            
            # ----------------------------------------------------

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