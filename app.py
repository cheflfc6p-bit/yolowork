import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. ส่วนหัวของเว็บ: กำหนดชื่อเรื่องและข้อความต้อนรับ
st.title("ระบบตรวจจับ (หมี, กระต่าย, แกะ) ด้วย YOLO")
st.write("อัปโหลดรูปภาพเพื่อทดสอบการทำงานของโมเดล")

# 2. ส่วนโหลดโมเดล AI: 
# ใส่ @st.cache_resource ไว้ข้างบน เพื่อให้เว็บจำโมเดลไว้ในหน่วยความจำ
# เวลามีคนกดเข้ามาใหม่ เว็บจะได้ไม่ต้องเสียเวลาโหลดไฟล์ best.pt ซ้ำทุกครั้ง (ช่วยให้เว็บเร็วขึ้น)
@st.cache_resource
def load_model():
    return YOLO("best.pt") # ดึงไฟล์น้ำหนัก (Weight) ที่เราเทรนเสร็จแล้วมาใช้งาน

# สั่งให้ฟังก์ชันด้านบนทำงาน เพื่อเตรียมโมเดลให้พร้อม
model = load_model()

# 3. ส่วนรับไฟล์ภาพจากผู้ใช้งาน: สร้างปุ่มให้กดอัปโหลดไฟล์ (รองรับ jpg, jpeg, png)
uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพที่ต้องการตรวจจับ...", type=["jpg", "jpeg", "png"])

# 4. ส่วนแสดงผล: ถ้าผู้ใช้งานอัปโหลดไฟล์เข้ามาแล้ว (ไม่เท่ากับ None) ให้ทำงานในบล็อกนี้
if uploaded_file is not None:
    
    # เปิดไฟล์ภาพที่อัปโหลดมา และแสดงบนหน้าเว็บให้ผู้ใช้ดู
    image = Image.open(uploaded_file)
    st.image(image, caption="ภาพต้นฉบับ", use_container_width=True)

    # 5. ส่วนวิเคราะห์ภาพ: สร้างปุ่ม "กดเพื่อวิเคราะห์รูปภาพ"
    if st.button("กดเพื่อวิเคราะห์รูปภาพ"):
        
        # แสดงข้อความหมุนๆ "กำลังประมวลผล..." ระหว่างที่รอ AI คิด
        with st.spinner("กำลังประมวลผล..."):
            
            # ส่งรูปภาพเข้าไปให้โมเดล YOLO ทำการทำนาย (Detect/Segment)
            results = model.predict(image)
            
            # นำผลลัพธ์ที่ได้ (กรอบสี่เหลี่ยม หรือ พื้นที่สี) มาวาดทับลงบนรูปภาพเดิม
            res_plotted = results[0].plot()
            
            # 6. ส่วนโชว์ผลลัพธ์: แจ้งเตือนสีเขียวว่าเสร็จแล้ว และแสดงภาพที่ AI วาดเสร็จแล้วบนเว็บ
            st.success("ประมวลผลเสร็จสิ้น!")
            st.image(res_plotted, caption="ผลลัพธ์จากการตรวจจับ", channels="BGR", use_container_width=True)