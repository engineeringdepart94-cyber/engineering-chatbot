# 🏗️ Engineering Standard AI Chatbot (Streamlit)

PDF (engineering standard) se sawalon ke jawab dene wala chatbot — type ya voice se
sawal poocha ja sakta hai, jawab text aur AI awaaz (voice) dono mein milta hai.

---

## Zaroori: Pehle Colab mein data taiyar karein

Yeh app har baar PDF se OCR/processing nahi karta (bohat slow hoga). Is liye pehle
apni Colab notebook mein (jahan aapne PDF process kiya tha — chunks, embeddings,
FAISS index waghera bana chuke hain), yeh naya cell chalayein taake do files ban
sakein:

```python
import json
import faiss

# chunks ko JSON file mein save karein
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False)

# FAISS index ko file mein save karein
faiss.write_index(index, "faiss_index.bin")

from google.colab import files
files.download("chunks.json")
files.download("faiss_index.bin")
```

Yeh 2 files (`chunks.json` aur `faiss_index.bin`) download ho jayengi. Inhein
is project ke `data/` folder mein daal dein (`data/README.md` wali placeholder
file ko hata kar).

---

## Step-by-Step: GitHub par upload karein

1. https://github.com par jaake login karein (account nahi hai to free bana lein)
2. **New repository** banayein — naam dein jaise `engineering-chatbot`
   - **Public** rakhein (Streamlit Cloud free tier public repos ke sath aasan hai)
   - ⚠️ **Add a README** ka checkbox mat lagayein (hum apni README already de rahe hain)
3. Repository ban jane ke baad, **"uploading an existing file"** link par click karein
4. Is poore folder ki tamam files upload kar dein:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `.gitignore`
   - `README.md`
   - `data/chunks.json` ⬅️ (Colab se banai hui)
   - `data/faiss_index.bin` ⬅️ (Colab se banai hui)
   - `.streamlit/secrets.toml.example` (yeh sirf reference ke liye hai)
5. **Commit changes** dabayein

⚠️ `secrets.toml` (asal key wali file) **kabhi GitHub par upload na karein** — API
key sirf Streamlit Cloud ke dashboard mein dalen (neechay Step wali baat).

---

## Step-by-Step: Streamlit Cloud par deploy karein (FREE)

1. https://share.streamlit.io par jayein aur GitHub account se sign in karein
2. **"Create app"** > **"Deploy a public app from GitHub"** par click karein
3. Apna repository select karein (`engineering-chatbot`)
4. **Main file path** mein likhein: `app.py`
5. **"Advanced settings"** > **"Secrets"** mein yeh dalen:
   ```toml
   GROQ_API_KEY = "apni_asal_groq_key_yahan"
   ```
   (Free key: https://console.groq.com/keys)
6. **Deploy** par click karein

2-5 minute mein app ban kar taiyar ho jayega, aur ek public link milega jaise:
`https://your-app-name.streamlit.app` — yeh link kisi ko bhi bhej sakte hain,
wo seedha browser mein khol kar chat kar sakega.

---

## Local computer par chalane ka tareeqa (agar chahein)

```bash
pip install -r requirements.txt
```

`.streamlit/secrets.toml.example` ko copy kar ke `.streamlit/secrets.toml` bana lein
aur usmein apni asal Groq key dal dein, phir:

```bash
streamlit run app.py
```

---

## Kaise kaam karta hai

1. User sawal likhta hai ya mic se bolta hai
2. Agar voice hai to Whisper (faster-whisper) usay text mein badalta hai
3. Sawal se related PDF ka hissa (chunks) FAISS search se dhoonda jata hai
4. Groq API (Llama 3.3 model) us context ke sath jawab banata hai — usi zaban mein
   jis mein sawal poocha gaya (Urdu/Roman Urdu/English)
5. Jawab ko gTTS awaaz mein badal kar user ko sunaya jata hai

## Notes

- Free Groq API key: https://console.groq.com/keys
- Agar app pehli baar slow load ho to normal hai — models download ho rahe hote hain
- Naya PDF process karna ho to phir Colab mein wahi steps dohrayein aur nayi
  `chunks.json` / `faiss_index.bin` GitHub par update kar dein
