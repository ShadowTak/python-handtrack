import cv2
import mediapipe as mp
import time
import math
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Fix scipy/numpy compatibility for Python 3.14
if not hasattr(np, 'long'):
    np.long = np.int64
if not hasattr(np, 'ulong'):
    np.ulong = np.uint64

# ==================== INSIGHTFACE (AGE + FACE DETECTION) ====================
from insightface.app import FaceAnalysis

print("Loading InsightFace age model...")
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace model loaded!")

# ==================== MEDIAPIPE SETUP (Hand Landmarker) ====================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = HandLandmarker.create_from_options(options)

# ==================== SEGMENTER ====================
segmenter = None
try:
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    seg_options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path='selfie_segmenter.tflite'),
        running_mode=VisionRunningMode.VIDEO,
        output_category_mask=True
    )
    segmenter = ImageSegmenter.create_from_options(seg_options)
except Exception as e:
    print("Segmenter tidak tersedia:", e)


# ==================== AGE HELPERS ====================
def get_age_label(age):
    """Convert age number to display label."""
    if age <= 3:
        return "(0-3)"
    elif age <= 7:
        return "(4-7)"
    elif age <= 12:
        return "(8-12)"
    elif age <= 17:
        return "(13-17)"
    elif age <= 25:
        return "(18-25)"
    elif age <= 35:
        return "(26-35)"
    elif age <= 45:
        return "(36-45)"
    elif age <= 55:
        return "(46-55)"
    elif age <= 65:
        return "(56-65)"
    else:
        return "(66+)"


# ==================== GALAXY BACKGROUND ====================
galaxy_bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
galaxy_bg[:] = (30, 10, 40)
for _ in range(800):
    sx = np.random.randint(0, 1920)
    sy = np.random.randint(0, 1080)
    galaxy_bg[sy, sx] = (255, 255, 255)
for _ in range(100):
    sx = np.random.randint(0, 1920)
    sy = np.random.randint(0, 1080)
    cv2.circle(galaxy_bg, (sx, sy), np.random.randint(2, 6),
               (np.random.randint(150, 255), np.random.randint(100, 255), 255), -1)

# ==================== CAMERA ====================
cap = cv2.VideoCapture(0)
os.makedirs("captures", exist_ok=True)

print("=" * 60)
print("  RETROLENS - Hand Tracking Filter & Portal")
print("  + InsightFace Age Detection (Deep Learning) + Photo Capture")
print("=" * 60)
print("Gestures:")
print("  Portal:        Thumb + Index of both hands (4 tips)")
print("  Change Filter: Touch thumb+pinky, or index tips together")
print("  Photo:         Hold open palm (5 fingers) 3 sec -> countdown 3 sec")
print("  Manual shot:   Press 's'")
print("  Quit:          Press 'q'")
print("=" * 60)

filters = ["MONO", "DUAL-TONE", "PIXELATE", "INVERT", "SEPIA",
           "BLUR", "THERMAL", "SKETCH", "GLITCH", "NEON", "GALAXY", "CYBER"]
current_filter = 0
gesture_triggered = False

# ==================== PHOTO CAPTURE STATE ====================
flat_hand_start = None
countdown_start = None
countdown_active = False
HOLD_THRESHOLD = 3.0
COUNTDOWN_SECONDS = 3
photo_captured = False
capture_msg = ""
capture_msg_time = 0


def is_flat_hand_open(hand_landmarks, w, h):
    """Check if all 5 fingers are fully extended (open palm)."""
    lm = hand_landmarks
    fingers = 0
    if lm[8].y < lm[6].y:
        fingers += 1
    if lm[12].y < lm[10].y:
        fingers += 1
    if lm[16].y < lm[14].y:
        fingers += 1
    if lm[20].y < lm[18].y:
        fingers += 1
    thumb_tip_x = lm[4].x * w
    palm_cx = lm[9].x * w
    if abs(thumb_tip_x - palm_cx) > 40:
        fingers += 1
    return fingers >= 5


def apply_cyber_filter(roi):
    """Apply cyberpunk-style filter with neon colors, scanlines, and grid."""
    h_r, w_r = roi.shape[:2]
    filtered = roi.copy()

    # Color shift to cyberpunk palette (cyan + magenta)
    b, g, r = cv2.split(filtered)
    b = cv2.add(b, 30)
    r = cv2.add(r, 20)
    g = cv2.subtract(g, 10)
    filtered = cv2.merge([b, g, r])

    # Add cyan/magenta tint
    overlay = filtered.copy()
    overlay[:, :w_r // 2, 0] = cv2.add(overlay[:, :w_r // 2, 0], 40)
    overlay[:, :w_r // 2, 1] = cv2.add(overlay[:, :w_r // 2, 1], 20)
    overlay[:, w_r // 2:, 0] = cv2.add(overlay[:, w_r // 2:, 0], 30)
    overlay[:, w_r // 2:, 2] = cv2.add(overlay[:, w_r // 2:, 2], 40)
    filtered = cv2.addWeighted(filtered, 0.7, overlay, 0.3, 0)

    # Scanlines
    scanline_img = filtered.copy()
    for y in range(0, h_r, 3):
        scanline_img[y, :] = cv2.add(scanline_img[y, :], 15)
    filtered = cv2.addWeighted(filtered, 0.85, scanline_img, 0.15, 0)

    # Grid overlay
    grid_color = (255, 0, 128)
    for y in range(0, h_r, 40):
        cv2.line(filtered, (0, y), (w_r, y), grid_color, 1)
    for x in range(0, w_r, 40):
        cv2.line(filtered, (x, 0), (x, h_r), grid_color, 1)

    # Chromatic aberration
    shift = 2
    if w_r > shift:
        filtered[:, shift:, 0] = filtered[:, :-shift, 0]
        filtered[:, :-shift, 2] = filtered[:, shift:, 2]

    # Vignette
    rows, cols = filtered.shape[:2]
    X = cv2.getGaussianKernel(cols, cols / 2)
    Y = cv2.getGaussianKernel(rows, rows / 2)
    mask = Y * X.T
    mask = mask / mask.max()
    mask = np.stack([mask] * 3, axis=-1)
    filtered = (filtered * mask).astype(np.uint8)

    return filtered


def apply_filter(roi, filter_name, x=0, y=0, mask_person=None, frame_galaxy=None):
    if filter_name == "MONO":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif filter_name == "INVERT":
        return cv2.bitwise_not(roi)
    elif filter_name == "BLUR":
        return cv2.GaussianBlur(roi, (25, 25), 0)
    elif filter_name == "SEPIA":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        filtered = cv2.transform(roi, kernel)
        return np.clip(filtered, 0, 255).astype(np.uint8)
    elif filter_name == "DUAL-TONE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_c = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        filtered = np.zeros_like(roi)
        filtered[mask_c == 255] = [0, 165, 255]
        filtered[mask_c == 0] = [147, 20, 255]
        return filtered
    elif filter_name == "PIXELATE":
        h_r, w_r = roi.shape[:2]
        if h_r > 10 and w_r > 10:
            small = cv2.resize(roi, (w_r // 10, h_r // 10), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w_r, h_r), interpolation=cv2.INTER_NEAREST)
    elif filter_name == "THERMAL":
        return cv2.applyColorMap(roi, cv2.COLORMAP_JET)
    elif filter_name == "SKETCH":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif filter_name == "GLITCH":
        h_r, w_r = roi.shape[:2]
        shift = max(5, w_r // 20)
        glitch_roi = roi.copy()
        if w_r > shift:
            glitch_roi[:, :-shift, 2] = roi[:, shift:, 2]
            glitch_roi[:, shift:, 0] = roi[:, :-shift, 0]
        return glitch_roi
    elif filter_name == "NEON":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_bgr[np.where((edges_bgr == [255, 255, 255]).all(axis=2))] = [255, 255, 0]
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(edges_bgr, kernel, iterations=1)
    elif filter_name == "GALAXY" and mask_person is not None and frame_galaxy is not None:
        bh, bw = roi.shape[:2]
        roi_mask = mask_person[y:y + bh, x:x + bw]
        roi_galaxy = frame_galaxy[y:y + bh, x:x + bw]
        bg_condition = (roi_mask == 0)
        filtered = roi.copy()
        filtered[bg_condition] = roi_galaxy[bg_condition]
        return filtered
    elif filter_name == "CYBER":
        return apply_cyber_filter(roi)
    return roi


def apply_filter_to_full_frame(frame, filter_name, mask_person=None, frame_galaxy=None):
    """Apply filter to entire frame for photo capture."""
    h, w = frame.shape[:2]
    return apply_filter(frame, filter_name, 0, 0, mask_person, frame_galaxy)


# ==================== MAIN LOOP ====================
face_frame_counter = 0
cached_faces = []      # list of (x1, y1, x2, y2)
cached_age_labels = []  # list of "Age: 25 (18-25)"
face_age_history = {}   # {face_id: [age1, age2, ...]} for smoothing
face_id_counter = 0
EMA_ALPHA = 0.3  # smoothing factor (lower = more stable, 0.1-0.3 recommended)

while True:
    success, img = cap.read()
    if not success:
        print("Gagal membaca kamera!")
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    frame_galaxy = galaxy_bg[:h, :w]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    timestamp_ms = time.time_ns() // 1_000_000

    # Hand detection
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    # ==================== INSIGHTFACE AGE DETECTION ====================
    # Run every 8 frames for stability (less frequent = more stable age)
    face_frame_counter += 1
    if face_frame_counter % 8 == 0 and not countdown_active:
        try:
            faces = face_analyzer.get(img)
            new_faces = []
            new_age_labels = []
            matched_ids = set()

            for face in faces:
                bbox = face.bbox.astype(int)
                x1 = max(0, bbox[0])
                y1 = max(0, bbox[1])
                x2 = min(w, bbox[2])
                y2 = min(h, bbox[3])
                if x2 <= x1 or y2 <= y1:
                    continue

                # Try to match with existing face by IoU
                best_id = None
                best_iou = 0.3  # minimum IoU threshold
                for fid, fdata in face_age_history.items():
                    if fid in matched_ids:
                        continue
                    fx1, fy1, fx2, fy2 = fdata['bbox']
                    # Calculate IoU
                    ix1 = max(x1, fx1)
                    iy1 = max(y1, fy1)
                    ix2 = min(x2, fx2)
                    iy2 = min(y2, fy2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (fx2 - fx1) * (fy2 - fy1)
                    union = area1 + area2 - inter
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_id = fid

                age_raw = int(face.age)
                gender = "M" if face.gender == 1 else "F"

                if best_id is not None:
                    # Matched: apply EMA smoothing
                    matched_ids.add(best_id)
                    prev_age = face_age_history[best_id]['smoothed_age']
                    smoothed = EMA_ALPHA * age_raw + (1 - EMA_ALPHA) * prev_age
                    face_age_history[best_id]['smoothed_age'] = smoothed
                    face_age_history[best_id]['bbox'] = (x1, y1, x2, y2)
                    face_age_history[best_id]['gender'] = gender
                    age_final = int(round(smoothed))
                else:
                    # New face: create entry
                    face_id_counter += 1
                    best_id = face_id_counter
                    face_age_history[best_id] = {
                        'smoothed_age': float(age_raw),
                        'bbox': (x1, y1, x2, y2),
                        'gender': gender
                    }
                    matched_ids.add(best_id)
                    age_final = age_raw

                new_faces.append((x1, y1, x2, y2))
                new_age_labels.append(f"Age: {age_final} {get_age_label(age_final)} ({gender})")

            # Remove unmatched faces (face disappeared)
            for fid in list(face_age_history.keys()):
                if fid not in matched_ids:
                    del face_age_history[fid]

            cached_faces = new_faces
            cached_age_labels = new_age_labels
        except Exception as e:
            pass

    # Draw face boxes + age (InsightFace)
    face_colors = [(0, 255, 200), (255, 100, 255), (255, 200, 0),
                   (0, 200, 255), (100, 255, 100)]
    for idx, (x1, y1, x2, y2) in enumerate(cached_faces):
        color = face_colors[idx % len(face_colors)]

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        cl = min(25, (x2 - x1) // 4, (y2 - y1) // 4)
        cv2.line(img, (x1, y1), (x1 + cl, y1), color, 3)
        cv2.line(img, (x1, y1), (x1, y1 + cl), color, 3)
        cv2.line(img, (x2, y1), (x2 - cl, y1), color, 3)
        cv2.line(img, (x2, y1), (x2, y1 + cl), color, 3)
        cv2.line(img, (x1, y2), (x1 + cl, y2), color, 3)
        cv2.line(img, (x1, y2), (x1, y2 - cl), color, 3)
        cv2.line(img, (x2, y2), (x2 - cl, y2), color, 3)
        cv2.line(img, (x2, y2), (x2, y2 - cl), color, 3)

        # Age label
        age_text = cached_age_labels[idx] if idx < len(cached_age_labels) else "Age: ?"
        (tw, th), _ = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 14), (x1 + tw + 10, y1), color, -1)
        cv2.putText(img, age_text, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    filter_name = filters[current_filter]

    # Galaxy segmenter
    mask_person = None
    if filter_name == "GALAXY" and segmenter is not None:
        seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
        if seg_result.category_mask is not None:
            mask_person = seg_result.category_mask.numpy_view()
            if mask_person.shape != (h, w):
                mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)

    pts_portal = []
    change_filter = False

    # ==================== FLAT HAND PHOTO DETECTION ====================
    flat_hand_detected = False
    if results.hand_landmarks:
        for hand_lms in results.hand_landmarks:
            if is_flat_hand_open(hand_lms, w, h):
                flat_hand_detected = True
                break

    now = time.time()

    if flat_hand_detected and not countdown_active:
        if flat_hand_start is None:
            flat_hand_start = now
        if (now - flat_hand_start) >= HOLD_THRESHOLD:
            countdown_active = True
            countdown_start = now
            flat_hand_start = None
    elif not flat_hand_detected:
        flat_hand_start = None

    # ==================== COUNTDOWN (CENTERED, LARGE) ====================
    if countdown_active:
        elapsed = now - countdown_start
        remaining = COUNTDOWN_SECONDS - int(elapsed)

        if remaining > 0:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

            text = str(remaining)
            font_scale = 10.0
            thickness = 20
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            tx = (w - tw) // 2
            ty = (h + th) // 2

            # Glow effect
            cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 80, 180), thickness + 10)
            cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 150, 255), thickness + 4)
            cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

            # Sub text
            sub_text = "Hold still..."
            (stw, sth), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(img, sub_text, ((w - stw) // 2, ty + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            # Circular progress
            center = (w // 2, h // 2 - 150)
            radius = 80
            progress = (elapsed % 1.0)
            start_angle = -90
            end_angle = int(start_angle + progress * 360)
            cv2.ellipse(img, center, (radius, radius), 0, 0, 360, (80, 80, 80), 6)
            cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, (0, 255, 255), 8)

        else:
            # ==================== CAPTURE PHOTO WITH FILTER ====================
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"captures/photo_{timestamp_str}.jpg"

            save_img = apply_filter_to_full_frame(img, filter_name, mask_person, frame_galaxy)
            save_img = cv2.flip(save_img, 1)
            cv2.imwrite(filename, save_img)
            print(f"Foto tersimpan: {filename}")
            capture_msg = f"Saved: {filename}"
            capture_msg_time = now
            photo_captured = True
            countdown_active = False
            countdown_start = None

    # Flash effect
    if photo_captured and (now - capture_msg_time) < 1.5:
        flash_alpha = max(0, 1.0 - (now - capture_msg_time) / 1.5)
        flash = np.ones_like(img) * 255
        img = cv2.addWeighted(flash, flash_alpha * 0.6, img, 1.0, 0)
    else:
        photo_captured = False

    # ==================== HAND GESTURES ====================
    if results.hand_landmarks:
        if len(results.hand_landmarks) >= 2:
            idx0 = results.hand_landmarks[0][8]
            idx1 = results.hand_landmarks[1][8]
            pt0 = (int(idx0.x * w), int(idx0.y * h))
            pt1 = (int(idx1.x * w), int(idx1.y * h))
            if math.hypot(pt0[0] - pt1[0], pt0[1] - pt1[1]) < 40:
                change_filter = True

        for hand_lms in results.hand_landmarks:
            tx, ty = int(hand_lms[4].x * w), int(hand_lms[4].y * h)
            px, py = int(hand_lms[20].x * w), int(hand_lms[20].y * h)
            if math.hypot(tx - px, ty - py) < 40:
                change_filter = True

        if change_filter:
            if not gesture_triggered:
                current_filter = (current_filter + 1) % len(filters)
                gesture_triggered = True
        else:
            gesture_triggered = False

        for hand_lms in results.hand_landmarks:
            for id, lm in enumerate(hand_lms):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id in [4, 8]:
                    pts_portal.append([cx, cy])
                    cv2.circle(img, (cx, cy), 8, (255, 255, 0), cv2.FILLED)

        if len(pts_portal) == 4:
            pts_portal.sort(key=lambda p: p[1])
            top_pts = pts_portal[:2]
            bottom_pts = pts_portal[2:]
            top_pts.sort(key=lambda p: p[0])
            bottom_pts.sort(key=lambda p: p[0])

            poly_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.int32)

            x, y, bw, bh = cv2.boundingRect(poly_pts)
            x, y = max(0, x), max(0, y)
            bw, bh = min(w - x, bw), min(h - y, bh)

            if bw > 0 and bh > 0:
                roi = img[y:y + bh, x:x + bw].copy()
                filtered_roi = apply_filter(roi, filter_name, x, y, mask_person, frame_galaxy)

                mask = np.zeros((bh, bw), dtype=np.uint8)
                poly_roi = poly_pts - [x, y]
                cv2.fillPoly(mask, [poly_roi], 255)
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                img[y:y + bh, x:x + bw] = np.where(mask_3ch == 255, filtered_roi, roi)

                cv2.polylines(img, [poly_pts], True, (255, 255, 255), 2)

                for i in range(4):
                    pt1 = poly_pts[i]
                    pt2 = poly_pts[(i + 1) % 4]
                    for _ in range(5):
                        alpha = np.random.random()
                        ppx = int(pt1[0] * alpha + pt2[0] * (1 - alpha)) + np.random.randint(-15, 15)
                        ppy = int(pt1[1] * alpha + pt2[1] * (1 - alpha)) + np.random.randint(-15, 15)
                        cv2.circle(img, (ppx, ppy), np.random.randint(1, 4), (0, 255, 255), -1)

                cv2.putText(img, f"PORTAL: {filter_name}",
                            (top_pts[0][0], top_pts[0][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    # ==================== HUD ====================
    cv2.putText(img, "Ganti Filter: Thumb+Pinky / Index tips", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, f"Filter: {filters[current_filter]}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(img, "Photo: Open palm 3 sec | 's' = manual shot", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Hold progress bar
    if flat_hand_start is not None and not countdown_active:
        elapsed = now - flat_hand_start
        pct = min(elapsed / HOLD_THRESHOLD, 1.0)
        bx, by, bw_bar, bh_bar = 10, 95, 200, 15
        cv2.rectangle(img, (bx, by), (bx + bw_bar, by + bh_bar), (80, 80, 80), -1)
        cv2.rectangle(img, (bx, by), (bx + int(bw_bar * pct), by + bh_bar), (0, 255, 0), -1)
        cv2.rectangle(img, (bx, by), (bx + bw_bar, by + bh_bar), (255, 255, 255), 1)
        cv2.putText(img, f"Hold: {elapsed:.1f}s / {HOLD_THRESHOLD}s", (bx, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Capture message
    if capture_msg and (now - capture_msg_time) < 3.0:
        (tw, th), _ = cv2.getTextSize(capture_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (w // 2 - tw // 2 - 10, h - 50),
                      (w // 2 + tw // 2 + 10, h - 20), (0, 0, 0), -1)
        cv2.putText(img, capture_msg, (w // 2 - tw // 2, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Face count
    if cached_faces:
        cv2.putText(img, f"Faces: {len(cached_faces)}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('RETROLENS - InsightFace Age + Portal', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_img = apply_filter_to_full_frame(img, filter_name, mask_person, frame_galaxy)
        save_img = cv2.flip(save_img, 1)
        cv2.imwrite(f"captures/manual_{ts}.jpg", save_img)
        print(f"Manual screenshot: captures/manual_{ts}.jpg")

cap.release()
cv2.destroyAllWindows()
