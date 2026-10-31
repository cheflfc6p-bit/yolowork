import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. ตั้งชื่อหัวเว็บ
st.title("ระบบตรวจจับ(หมี, กระต่าย, แกะ)ด้วย YOLO")
st.write("อัปโหลดรูปภาพเพื่อทดสอบ")

# 2. ฟังก์ชันโหลดโมเดล (อ่านไฟล์ best.pt)
# ใช้ @st.cache_resource เพื่อให้โหลดโมเดลแค่ครั้งแรกครั้งเดียว จะได้เว็บไม่หน่วง
@st.cache_resource
def load_model():
    return YOLO("best.pt")

# เรียกใช้ฟังก์ชันโหลดโมเดล
model = load_model()

# 3. สร้างกล่องสำหรับอัปโหลดไฟล์ภาพ
uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ...", type=["jpg", "jpeg", "png"])

# ถ้ามีการอัปโหลดไฟล์เข้ามา
if uploaded_file is not None:
    # 4. แสดงภาพต้นฉบับที่อัปโหลด
    image = Image.open(uploaded_file)
    st.image(image, caption="ภาพต้นฉบับ", use_container_width=True)

    # 5. สร้างปุ่มสำหรับกดเพื่อประมวลผล
    if st.button("กดเพื่อวิเคราะห์รูปภาพ"):
        with st.spinner("กำลังประมวลผล..."):
            
            # สั่งให้โมเดลวิเคราะห์ภาพ
            results = model.predict(image)
            
            # ดึงภาพผลลัพธ์ที่ AI วาดกรอบหรือระบายสีแล้วออกมา
            res_plotted = results[0].plot()
            
            # 6. แสดงภาพผลลัพธ์บนเว็บ
            st.success("ประมวลผลเสร็จสิ้น!")
            st.image(res_plotted, caption="ผลลัพธ์", channels="BGR", use_container_width=True)
