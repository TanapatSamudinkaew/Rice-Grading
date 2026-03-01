import cv2
import numpy as np

# --- พารามิเตอร์ช่วงสีสำหรับการคัดแยก (ปรับแต่งได้) ---
# 1. ช่วงสีแดงสำหรับจับพื้นที่ถาด (HSV แยกเป็น 2 ช่วงเพราะสีแดงอยู่ตรงขอบสเกล)
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

# 2. ช่วงสีขาวสำหรับจับเมล็ดข้าว (ความอิ่มตัวสีต่ำ, ความสว่างสูง)
lower_white = np.array([0, 0, 150])
upper_white = np.array([180, 60, 255])

# ==========================================
# โหมด 1: สำหรับภาพนิ่ง (Upload Image)
# ==========================================
def process_rice_image(img, dist_val=0.4):
    # ปรับขนาดภาพให้กว้าง 800px เสมอ
    h, w = img.shape[:2]
    new_w = 800
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_w, new_h))
    original = img.copy()

    # 1. สร้างหน้ากาก (Mask) ค้นหาถาดสีแดงเพื่อตัดพื้นหลังโต๊ะออก
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
                              cv2.inRange(hsv, lower_red2, upper_red2))
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_open)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_tray = np.zeros(img.shape[:2], dtype="uint8")

    if contours_red:
        cnt_tray = max(contours_red, key=cv2.contourArea)
        hull_tray = cv2.convexHull(cnt_tray)
        cv2.drawContours(mask_tray, [hull_tray], -1, 255, -1)

    # เก็บเฉพาะภาพที่อยู่ข้างในพื้นที่ถาดสีแดง
    roi = cv2.bitwise_and(img, img, mask=mask_tray) if contours_red else img

    # 2. ค้นหาเมล็ดข้าวด้วยสีขาวและ Otsu Threshold
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    _, thresh_gray = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # รวมความแม่นยำจากสีขาวและความสว่างเข้าด้วยกัน
    mask_rice = cv2.bitwise_and(thresh_gray, mask_white)
    
    # ลบ Noise เล็กๆ และแยกเมล็ดที่อาจจะแตะกันอยู่
    kernel_ellip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_rice = cv2.morphologyEx(mask_rice, cv2.MORPH_OPEN, kernel_ellip, iterations=1)

    # 3. วิเคราะห์และคัดกรองรูปร่าง
    contours_rice, _ = cv2.findContours(mask_rice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats = {"Good": 0}

    for c in contours_rice:
        area = cv2.contourArea(c)
        # กรอง 1: ขนาด (อิงภาพ 800px)
        if area < 300 or area > 4000: 
            continue

        # กรอง 2: ความทึบ ตัดคีมคีบและข้าวที่แหว่งมากๆ
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        if solidity < 0.75: continue 

        # กรอง 3: สัดส่วน ตัดวัตถุเส้นยาวๆ 
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / float(h)
        if aspect_ratio < 0.25 or aspect_ratio > 4.0: continue

        stats["Good"] += 1
        cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(original, "Good", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return original, stats