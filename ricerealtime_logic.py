import cv2
import numpy as np

def process_rice_logic(img, dist_threshold=0.4, yellow_sensitivity=0.12):
    # 1. ลดความละเอียดภาพ (Downscaling) เพื่อเพิ่ม FPS ให้ลื่นขึ้น
    # ถ้าภาพจากกล้องใหญ่เกินไป จะทำให้ประมวลผลไม่ทัน
    h, w = img.shape[:2]
    scale = 640 / w if w > 640 else 1.0
    if scale < 1.0:
        img_proc = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    else:
        img_proc = img.copy()

    original_display = img_proc.copy()
    
    # 2. Pre-processing แบบเร็ว (ใช้ Median Blur แทน Bilateral เพื่อความไว)
    gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5) 
    
    # 3. Thresholding แบบรวดเร็ว
    # ใช้ OTSU ร่วมกับ Adaptive เพื่อให้ทนต่อแสงที่เปลี่ยนไปตอนขยับกล้อง
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # กำจัดจุดรบกวนเล็กๆ (Noise Removal)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 4. ตรวจจับสีเสียด้วย HSV (ประมวลผลเฉพาะในหน้ากากเมล็ดข้าว)
    hsv = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 50, 100]) # ขยายช่วงสีให้กว้างขึ้นเล็กน้อยเพื่อ Real-time
    upper_yellow = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 5. วิเคราะห์ด้วย Contours (ไม่ต้องใช้ Watershed ในโหมด Real-time ถ้าไม่จำเป็น เพื่อความลื่น)
    stats = {"Pass": 0, "Fail": 0}
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        # ปรับค่า Area ตามสเกลภาพ
        min_area = 150 * scale
        if area < min_area: continue 

        x, y, w_box, h_box = cv2.boundingRect(c)
        
        # ตรวจสอบสีเสีย
        roi_yellow = yellow_mask[y:y+h_box, x:x+w_box]
        # นับเฉพาะจุดสีเหลืองที่อยู่ในตัวเมล็ดจริงๆ
        yellow_pixels = cv2.countNonZero(roi_yellow)
        yellow_ratio = yellow_pixels / (w_box * h_box + 1e-5)

        # คำนวณความยาวเมล็ด (Aspect Ratio) เพื่อดูว่าเมล็ดหักไหม
        aspect_ratio = float(w_box)/h_box if w_box > h_box else float(h_box)/w_box

        # เงื่อนไขคัดแยก (ปรับจูนตามหน้างาน)
        # Fail ถ้า: เมล็ดเล็กไป (หัก) OR สีเหลืองเยอะไป OR ทรงกลมเกินไป (ไม่ใช่ข้าว)
        if area < (300 * scale) or yellow_ratio > yellow_sensitivity or aspect_ratio < 1.5:
            status = "Fail"
            color = (0, 0, 255) # แดง
            stats["Fail"] += 1
        else:
            status = "Pass"
            color = (0, 255, 0) # เขียว
            stats["Pass"] += 1

        # วาดเฉพาะ Pass หรือ Fail ตามที่เราต้องการเน้น
        cv2.rectangle(original_display, (x, y), (x + w_box, y + h_box), color, 2)
        cv2.putText(original_display, status, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return original_display, stats