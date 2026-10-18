import os, re, json, base64, random, urllib.parse
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"\'')
GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip().strip('"\'')
HF_KEY = os.getenv("HF_API_KEY", "").strip().strip('"\'')
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

FMT = (" نسّق إجابتك بـ Markdown منظّم: ابدأ بفقرة واحدة قصيرة تلخّص الفكرة الرئيسية، ثم عناوين فرعية (##) وقوائم نقطية قصيرة،"
       " واجعل الكلمات المفتاحية **عريضة**، وضع الأكواد داخل كتل ``` مع اسم اللغة، وتجنّب الإطالة والحشو.")

MATH = (r" اكتب كل المعادلات والرموز العلمية بصيغة LaTeX صحيحة: داخل الجملة بين علامتي $ مفردتين مثل $CO_2$ و $x^2$،"
        r" والمعادلة المستقلة بين $$ و $$ في سطر منفصل، ولا تستعمل $$ داخل الجملة، واكتب النص العادي داخل المعادلة بـ \text{}"
        r" والأسهم بـ \rightarrow، ولا تترك أي أمر LaTeX (مثل \rightarrow أو \frac) خارج علامات $.")
FMT += MATH

SYSTEM = {
    "fast": "اسمك زيب، المهندس السريع في البرمجة والحسابات وصياغة النصوص. أجب بإيجاز ودقة وبلغة المستخدم." + FMT,
    "academic": "اسمك بروف، الأستاذ الأكاديمي: اشرح خطوة بخطوة، حلّ المسائل بوضوح، لخّص بدقة، ولا تخترع مصادر." + FMT,
    "vision": "اسمك ماكس، محلل الصور: حلّل الصور، استخرج النصوص منها (OCR)، وترجمها عند الطلب، وأجب بلغة المستخدم." + FMT,
}

ROUTER = """أنت موجّه نوايا. أعد JSON فقط بلا أي شرح بهذا الشكل:
{"agent":"vision|creative|fast|academic|action","action":null}
- vision: تحليل صورة مرفقة أو قراءة نص منها أو ترجمتها.
- creative: طلب توليد أو رسم صورة.
- academic: شرح علمي، حل مسائل، تلخيص مقالات.
- fast: برمجة، حسابات، صياغة، أسئلة عامة.
- action: أمر لتغيير الموقع نفسه. ضع action مثل {"type":"theme","value":"dark"} أو {"type":"theme","value":"light"} أو {"type":"open","value":"vision|creative|fast|academic"}."""


class ProviderError(RuntimeError):
    pass


def _detail(r, name):
    try:
        msg = r.json()["error"]["message"][:300]
    except Exception:
        msg = r.text[:200]
    hint = {401: "المفتاح غير صحيح", 403: "المفتاح مرفوض أو غير مفعّل لهذه الخدمة",
            429: "تجاوزت الحد المجاني، انتظر قليلاً"}.get(r.status_code, "")
    return ProviderError(f"{name} ({r.status_code}) {hint}: {msg}")


_models = {}


def _gemini_candidates():
    """نماذج Gemini المتاحة لمفتاحك: المستقرة أولاً (الأحدث فالأقدم)."""
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                     headers={"x-goog-api-key": GEMINI_KEY}, timeout=30)
    if not r.ok:
        raise _detail(r, "Gemini")
    names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    bad = ("lite", "image", "tts", "live", "audio", "exp", "embedding", "robotics",
           "computer", "thinking", "omni", "preview", "customtools")
    stable = [n for n in names if re.fullmatch(r"gemini-\d+(\.\d+)?-flash(-\d+)?", n)]
    stable.sort(key=lambda n: float(re.search(r"gemini-(\d+(?:\.\d+)?)", n).group(1)), reverse=True)
    others = [n for n in names if "flash" in n and not any(x in n for x in bad) and n not in stable]
    pool = stable + others + [n for n in names if n not in stable + others and "pro" in n and "exp" not in n]
    if not pool:
        raise ProviderError("هذا المفتاح لا يملك أي نموذج Gemini متاح.")
    return pool


def _pick_groq():
    r = requests.get("https://api.groq.com/openai/v1/models",
                     headers={"Authorization": f"Bearer {GROQ_KEY}"}, timeout=30)
    if not r.ok:
        raise _detail(r, "Groq")
    bad = ("whisper", "guard", "tts", "orpheus", "playai", "compound", "safeguard")
    ids = [m["id"] for m in r.json().get("data", []) if not any(x in m["id"] for x in bad)]
    pref = [i for i in ids if "llama" in i and ("versatile" in i or "70b" in i)]
    if not (pref or ids):
        raise ProviderError("هذا المفتاح لا يملك أي نموذج Groq متاح.")
    return sorted(pref or ids, reverse=True)[0]


def gemini(text, history=(), image=None, system=""):
    if not GEMINI_KEY:
        raise RuntimeError("مفتاح GEMINI_API_KEY غير موجود في ملف .env")
    contents = [{"role": "model" if h["role"] == "assistant" else "user",
                 "parts": [{"text": h["content"]}]} for h in history]
    parts = [{"text": text}]
    if image:
        parts.append({"inline_data": {"mime_type": image[0],
                                      "data": base64.b64encode(image[1]).decode()}})
    contents.append({"role": "user", "parts": parts})
    body = {"contents": contents}
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    model, tried, cands = _models.get("g") or GEMINI_MODEL, set(), None
    for _ in range(8):
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": GEMINI_KEY}, json=body, timeout=90)
        no_quota = r.status_code == 429 and "limit: 0" in r.text   # لا حصة مجانية لهذا النموذج
        bad_model = r.status_code in (400, 404) and "model" in r.text.lower()
        if no_quota or bad_model:
            tried.add(model)
            cands = cands or _gemini_candidates()
            rest = [c for c in cands if c not in tried]
            if not rest:
                raise _detail(r, "Gemini")
            model = rest[0]
            continue
        if not r.ok:
            raise _detail(r, "Gemini")
        _models["g"] = model
        try:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            raise ProviderError("Gemini لم يُرجع نصًا (ربما حُجب الطلب لأسباب الأمان).")
    raise ProviderError("لم يُوجد نموذج Gemini يعمل بمفتاحك. جرّب مفتاحًا جديدًا من aistudio.google.com")


def groq(text, history=(), system=""):
    msgs = ([{"role": "system", "content": system}] if system else []) + list(history)
    msgs.append({"role": "user", "content": text})
    for attempt in range(2):
        model = _models.get("q") or GROQ_MODEL
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {GROQ_KEY}"},
                          json={"model": model, "messages": msgs}, timeout=60)
        if r.status_code in (400, 404) and attempt == 0 and "model" in r.text.lower():
            _models["q"] = _pick_groq()
            continue
        if not r.ok:
            raise _detail(r, "Groq")
        return r.json()["choices"][0]["message"]["content"]


def fast_llm(text, history=(), system=""):
    """Groq إن توفر مفتاحه، وإلا Gemini."""
    return groq(text, history, system) if GROQ_KEY else gemini(text, history, system=system)


def route(message, has_image):
    if has_image:
        return {"agent": "vision", "action": None}
    try:
        out = fast_llm(message, system=ROUTER)
        data = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
        if data.get("agent") in ("vision", "creative", "fast", "academic", "action"):
            return data
    except Exception:
        pass
    return {"agent": "fast", "action": None}


def to_image_prompt(idea):
    return fast_llm(idea, system="حوّل الفكرة إلى وصف إنجليزي مفصّل لصورة (سطر واحد، أسلوب، إضاءة، تكوين). أعد الوصف فقط.").strip()


def make_image(prompt, provider="pollinations"):
    """يعيد رابطًا (Pollinations) أو data URL (Hugging Face)."""
    if provider == "hf" and HF_KEY:
        r = requests.post(
            "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
            headers={"Authorization": f"Bearer {HF_KEY}"}, json={"inputs": prompt}, timeout=120)
        r.raise_for_status()
        mime = r.headers.get("content-type", "image/png")
        return f"data:{mime};base64," + base64.b64encode(r.content).decode()
    return ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt)
            + f"?width=1024&height=1024&nologo=true&seed={random.randint(1, 10**6)}")


def history_from(form):
    try:
        h = json.loads(form.get("history", "[]"))
        return [x for x in h if x.get("role") in ("user", "assistant")][-10:]
    except Exception:
        return []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        msg = request.form.get("message", "").strip()
        agent = request.form.get("agent", "auto")
        provider = request.form.get("provider", "pollinations")
        hist = history_from(request.form)
        f = request.files.get("image")
        image = (f.mimetype, f.read()) if f else None
        if not msg and not image:
            return jsonify(error="اكتب طلبك أولًا."), 400

        action = None
        if agent == "auto":
            r = route(msg, bool(image))
            agent, action = r["agent"], r.get("action")

        if agent == "action":
            if not action:
                return jsonify(agent="fast", type="text", text=fast_llm(msg, hist, SYSTEM["fast"]))
            return jsonify(agent=agent, type="action", action=action, text="تم تنفيذ الأمر.")
        if agent == "creative":
            prompt = to_image_prompt(msg)
            return jsonify(agent=agent, type="image", text=prompt, image=make_image(prompt, provider))
        if agent == "vision":
            if not image:
                return jsonify(error="ارفع صورة لهذا المساعد."), 400
            text = gemini(msg or "حلّل هذه الصورة واستخرج أي نص فيها.", hist, image, SYSTEM["vision"])
        elif agent == "academic":
            text = gemini(msg, hist, system=SYSTEM["academic"])
        else:
            agent = "fast"
            text = fast_llm(msg, hist, SYSTEM["fast"])
        return jsonify(agent=agent, type="text", text=text)
    except ProviderError as e:
        return jsonify(error=str(e)), 502
    except RuntimeError as e:
        return jsonify(error=str(e)), 400
    except requests.HTTPError as e:
        return jsonify(error=f"رفض المزوّد الطلب ({e.response.status_code}): {e.response.text[:200]}"), 502
    except Exception as e:
        return jsonify(error=f"خطأ غير متوقع: {e}"), 500


@app.route("/api/diagnose")
def diagnose():
    """افتح http://127.0.0.1:5000/api/diagnose لمعرفة سبب أي فشل."""
    out = {}
    for name, fn in (("gemini", lambda: gemini("قل: تم")), ("groq", lambda: groq("قل: تم"))):
        try:
            out[name] = "OK: " + fn()[:40]
        except Exception as e:
            out[name] = f"FAIL: {e}"
    out["models_used"] = {"gemini": _models.get("g") or GEMINI_MODEL, "groq": _models.get("q") or GROQ_MODEL}
    return jsonify(out)


@app.route("/api/workflow", methods=["POST"])
def workflow():
    """فكرة -> وصف صورة -> صورة -> تحليل شامل للصورة."""
    try:
        idea = request.form.get("message", "").strip()
        if not idea:
            return jsonify(error="اكتب فكرتك أولًا."), 400
        prompt = to_image_prompt(idea)
        url = make_image(prompt, request.form.get("provider", "pollinations"))
        if url.startswith("data:"):
            mime, b64 = url[5:].split(";base64,")
            img = (mime, base64.b64decode(b64))
        else:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            img = (r.headers.get("content-type", "image/jpeg").split(";")[0], r.content)
        analysis = gemini("حلّل هذه الصورة تحليلًا شاملًا: العناصر، الألوان، الأسلوب، والانطباع.", image=img, system=SYSTEM["vision"])
        return jsonify(prompt=prompt, image=url, analysis=analysis)
    except RuntimeError as e:
        return jsonify(error=str(e)), 502 if isinstance(e, ProviderError) else 400
    except Exception as e:
        return jsonify(error=f"تعذّر إكمال المهمة المركبة: {e}"), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
