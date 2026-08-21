# 🎭 Retrolens - แอป Filter กล้องด้วย Hand Tracking

แอปพลิเคชันกล้องถ่ายภาพแบบ Real-time ที่ใช้ **MediaPipe Hand Tracking** ในการควบคุมฟิลเตอร์ และ **InsightFace Deep Learning** ในการตรวจจับอายุใบหน้า

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.35-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-orange)
![InsightFace](https://img.shields.io/badge/InsightFace-1.0-red)

## ✨ ฟีเจอร์หลัก

### 🖐️ Hand Tracking Portal
- ใช้นิ้วหัวแม่มือ + นิ้วชี้ทั้งสองมือสร้าง "Portal" สี่เหลี่ยม
- ฟิลเตอร์จะแสดงผลภายใน Portal ที่สร้างขึ้น

### 🎨 12 ฟิลเตอร์
| ฟิลเตอร์ | คำอธิบาย |
|----------|----------|
| MONO | ภาพขาวดำ |
| DUAL-TONE | สองโทนสี (ส้ม + ชมพู) |
| PIXELATE | ภาพแบบ Pixel Art |
| INVERT | กลับสี |
| SEPIA | โทนสีเก่าแบบหนังย้อนยุค |
| BLUR | ภาพเบลอ |
| THERMAL | ภาพแบบกล้องอินฟราเรด |
| SKETCH | ภาพแบบดินสอวาด |
| GLITCH | ภาพแบบสัญญาณรบกวน |
| NEON | ภาพขอบเรืองแสง |
| GALAXY | แทนที่พื้นหลังเป็นอวกาศ |
| CYBER | ธีม Cyberpunk (Neon + Scanlines + Grid) |

### 🧠 ระบบจับอายุ (InsightFace)
- ใช้ **Deep Learning Model** 5 ตัว (buffalo_l)
- ตรวจจับใบหน้าได้ **สูงสุด 5 คน** พร้อมกัน
- บอก **อายุ + ช่วงอายุ + เพศ (M/F)**
- แสดงกรอบสี่เหลี่ยมพร้อม Corner Accents สีสวย

### 📸 ระบบถ่ายภาพ
- **แบมือ 5 นิ้วค้าง 3 วินาที** → นับถอยหลัง 3-2-1 → ถ่ายภาพ
- **กด `s`** = ถ่ายภาพ Manual
- ภาพที่ถ่ายจะเก็บ **ฟิลเตอร์ที่ใช้งานอยู่** ด้วย
- บันทึกไฟล์ในโฟลเดอร์ `captures/`

## 📋 ต้องการ

- Python 3.8 ขึ้นไป
- Webcam
- ระบบปฏิบัติการ: macOS / Linux / Windows

## 🚀 วิธีติดตั้ง

### 1. Clone Repository
```bash
git clone https://github.com/ShadowTak/python-handtrack.git
cd python-handtrack
```

### 2. สร้าง Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 4. ดาวน์โหลด Model อัตโนมัติ
เมื่อรันครั้งแรก InsightFace จะดาวน์โหลดโมเดลไปที่ `~/.insightface/models/buffalo_l/` โดยอัตโนมัติ

### 5. รันแอป
```bash
python main.py
```

## 🎮 วิธีใช้งาน

### การเปลี่ยนฟิลเตอร์
| ท่าทาง | วิธีทำ |
|--------|--------|
| 👌 Thumb + Pinky | แตะนิ้วหัวแม่มือกับนิ้วก้อย |
| ☝️ Index Tips | แตะปลายนิ้วชี้ทั้งสองมือเข้าด้วยกัน |

### การถ่ายภาพ
| ท่าทาง | วิธีทำ |
|--------|--------|
| 🖐️ Open Palm | แบมือ 5 นิ้วค้าง 3 วินาที |
| ⌨️ แป้นพิมพ์ | กด `s` = ถ่ายภาพ Manual |

### การปิดแอป
- กด `q` บนแป้นพิมพ์

## 📁 โครงสร้างโปรเจค

```
python-handtrack/
├── main.py                      # โค้ดหลัก
├── requirements.txt             # Dependencies
├── hand_landmarker.task         # MediaPipe Hand Model
├── selfie_segmenter.tflite      # MediaPipe Segmenter Model
├── face_landmarker.task         # MediaPipe Face Model
├── captures/                    # โฟลเดอร์เก็บภาพถ่าย
└── README.md
```

## 🧠 เทคโนโลยีที่ใช้

| เทคโนโลยี | ใช้ทำอะไร |
|-----------|----------|
| **MediaPipe** | Hand Tracking, Face Landmarks, Segmentation |
| **OpenCV** | ประมวลผลภาพ, ฟิลเตอร์, การแสดงผล |
| **InsightFace** | ตรวจจับอายุ + เพศใบหน้า (Deep Learning) |
| **NumPy** | ประมวลผล Array |

## ⚠️ หมายเหตุ

- **การให้สิทธิ์กล้อง**: macOS ต้องอนุญาตให้ Terminal เข้าถึงกล้องที่ **System Settings → Privacy & Security → Camera**
- **แสง**: ใช้ในที่มีแสงเพียงพอเพื่อผลลัพธ์ที่ดีที่สุด
- **InsightFace Model**: ดาวน์โหลดอัตโนมัติครั้งแรก (~280MB)

## 📄 License

MIT License

---

<p align="center">Made with ❤️ by <a href="https://github.com/ShadowTak">ShadowTak</a></p>
