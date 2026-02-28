import cv2
import numpy as np

def process_rice_logic(img, dist_threshold=0.4, yellow_sensitivity=0.12):
    # 1. เตรียมภาพ
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # เพิ่มความคมชัดและลด Noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. ทำ Threshold เพื่อแยกเมล็ดออกจากพื้นหลัง
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 51, 2)
    
    # 3. แยกเมล็ดที่ติดกันด้วย Watershed
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, dist_threshold * dist_transform.max(), 255, 0)
    
    # 4. ตรวจจับสีเสียด้วย HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([18, 40, 120])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 5. วิเคราะห์แต่ละเมล็ด
    stats = {"Pass": 0, "Fail": 0}
    contours, _ = cv2.findContours(np.uint8(sure_fg), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 120: continue # กรองจุดเล็กๆ ทิ้ง

        x, y, w, h = cv2.boundingRect(c)
        
        # ตรวจสอบสีเสียในขอบเขตเมล็ด
        roi_yellow = yellow_mask[y:y+h, x:x+w]
        yellow_ratio = cv2.countNonZero(roi_yellow) / (w * h + 1e-5)

        # ตัดสิน Pass/Fail
        if area < 350 or yellow_ratio > yellow_sensitivity:
            status = "Fail"
            color = (0, 0, 255) # สีแดง
            stats["Fail"] += 1
        else:
            status = "Pass"
            color = (0, 255, 0) # สีเขียว
            stats["Pass"] += 1

        cv2.rectangle(original, (x, y), (x + w, y + h), color, 2)
        cv2.putText(original, status, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return original, stats