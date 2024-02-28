<h2 align="center">Inson xususiyatlarini tanib olish</h2>
Inson xususiyatlarini tanib olish (Human Attribute Recognition[HAR]) - Computer Vision va Pattern Recognitionda keng qo'llaniladigan tadqiqot sohasi hisoblanadi. HAR da insonning tashqi ko'rinishi xususiyatlaridan kelib chiqqan holda turli xil yondoshuvlar ishlab chiqiladi. Ushbu loyiha quyidagi yondoshuvlarga asoslangan:

<p align="center">
    <img width="600" src="https://github.com/MisterFoziljon/Paddle-Human-Attribute-Recognition/blob/main/src/main.png" alt="Material Bread logo">
</p>

### Loyiha 3 qismdan iborat:
1. Odam aniqlash (Person detection) - tasvirdan odam sinfiga tegishli obyektlar aniqlanadi. PaddlePaddle texnologiyasining [PersonDetection](https://drive.google.com/drive/folders/1zXrG1WNqC6ugG-xQMSCtsJnI0Bw0dRzu) modelidan foydalanildi;
2. Odam xususiyatlarini aniqlash (Human attribute recognition) - yuqoridagi tasvirda keltirilgan xususiyatlarni tanib olish uchun qo'llaniladi. PaddlePaddle texnologiyasining [Human Attribute Recognition](https://drive.google.com/drive/folders/1CLh4D-ep2RI8ux4jlTATB0bWWEScLTO5) modelidan foydalanildi.
3. Deploy - Modellarni Paddle freymvorki va CPU yordamida foydalanish uchun tayyor holatga keltirish.

### Modellar:
1. High Precision.
2. Balanced.
3. Fast.

### Dasturni ishga tushirish:
```cmd
python deploy.py
```

### Demo:
1. High Precision Model ishlatilgandagi natija:
<p align="center">
    <img width="600" src="https://github.com/MisterFoziljon/Paddle-Human-Attribute-Recognition/blob/main/src/high_precision.jpg" alt="Material Bread logo">
</p>

2. Balanced Model ishlatilgandagi natija:
<p align="center">
    <img width="600" src="https://github.com/MisterFoziljon/Paddle-Human-Attribute-Recognition/blob/main/src/balanced.jpg" alt="Material Bread logo">
</p>

3. Fast Model ishlatilgandagi natija:
<p align="center">
    <img width="600" src="https://github.com/MisterFoziljon/Paddle-Human-Attribute-Recognition/blob/main/src/fast.jpg" alt="Material Bread logo">
</p>
