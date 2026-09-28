import os

# กำหนดนามสกุลไฟล์ที่ต้องการเปลี่ยน เช่น '.jpg', '.png', '.txt' หรือปล่อยว่าง '' เพื่อเปลี่ยนทุกไฟล์
target_extension = '.jpg' 

# ดึงรายการไฟล์ทั้งหมดในโฟลเดอร์ปัจจุบัน
files = os.listdir('.')

# กรองเฉพาะไฟล์ (ไม่เอาโฟลเดอร์) และเรียงลำดับชื่อเดิม
files = sorted([f for f in files if os.path.isfile(f) and f != 'rename.py'])

# วนลูปเปลี่ยนชื่อ
for index, filename in enumerate(files, start=1):
    # ตรวจสอบนามสกุลไฟล์ (ถ้ามีการกำหนด)
    ext = os.path.splitext(filename)[1] if target_extension == '' else target_extension
    
    # จัดรูปแบบเลขลำดับให้เป็น 3 หลัก (001, 002, ...)
    new_name = f"BlackThorn_{index:03d}{ext}"
    
    # เปลี่ยนชื่อไฟล์
    os.rename(filename, new_name)
    print(f"เปลี่ยน {filename} -> {new_name}")

print("เปลี่ยนชื่อไฟล์เสร็จเรียบร้อยแล้ว!")