# منصة المساعدين الأذكياء (Flask)

## التشغيل محليًا
1. `pip install -r requirements.txt`
2. انسخ `.env.example` إلى `.env` وضع مفاتيحك:
   - GEMINI_API_KEY من https://aistudio.google.com
   - GROQ_API_KEY من https://console.groq.com
   - HF_API_KEY من https://huggingface.co/settings/tokens (اختياري)
3. `python app.py` ثم افتح http://127.0.0.1:5000

## النشر المجاني (Render)
ارفع المجلد إلى GitHub، أنشئ Web Service في Render، أمر البناء `pip install -r requirements.txt`،
أمر التشغيل `gunicorn app:app --timeout 180`، وأضف المفاتيح في Environment Variables (لا ترفع ملف .env).

## ملاحظات
- أسماء النماذج في `.env` (GEMINI_MODEL / GROQ_MODEL) لأن المزوّدين يغيّرون النماذج المجانية من وقت لآخر.
- توليد الصور: دالة `make_image` في app.py؛ بدّل الرابط فيها إن غيّر Pollinations واجهته.
