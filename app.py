import streamlit as st
import os
import json
import tempfile
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from gtts import gTTS
from faster_whisper import WhisperModel
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="Engineering Standard AI Chatbot", page_icon="🏗️", layout="centered")

# ---------------------------------------------------------------
# Cached resource loaders (yeh sirf ek dafa load hote hain, har
# naye message par dobara load nahi hote)
# ---------------------------------------------------------------

@st.cache_resource(show_spinner="Embedding model load ho raha hai...")
def load_embedder():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource(show_spinner="PDF data load ho raha hai...")
def load_chunks_and_index():
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    index = faiss.read_index("data/faiss_index.bin")
    return chunks, index


@st.cache_resource(show_spinner="AI se connect ho raha hai...")
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        st.error("GROQ_API_KEY nahi mili. Streamlit Cloud ke Settings > Secrets mein add karein.")
        st.stop()
    return Groq(api_key=api_key)


@st.cache_resource(show_spinner="Voice recognition model load ho raha hai...")
def load_whisper_model():
    return WhisperModel("small", device="cpu", compute_type="int8")


embedder = load_embedder()
chunks, index = load_chunks_and_index()
client = load_groq_client()
stt_model = load_whisper_model()

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------

def get_relevant_chunks(query, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]


def ask_ai(user_question, retrieval_query=None):
    query_for_search = retrieval_query if retrieval_query else user_question
    context_chunks = get_relevant_chunks(query_for_search)
    context = "\n\n".join(context_chunks)

    system_prompt = (
        "Tum ek engineering assistant ho. Sirf diye gaye context se jawab do. "
        "Jis zaban mein user sawal karay (Urdu, Roman Urdu, ya English), usi zaban mein jawab do. "
        "Agar context mein jawab na mile to bolo 'Yeh information PDF mein maujood nahi'."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nSawal: {user_question}"},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def translate_to_english(text):
    """PDF English mein hai, is liye voice se aaye sawal ko English mein translate
    karte hain taake FAISS search behtar (zyada accurate) chunks dhoond sake.
    Chota/tez model use karta hai taake asal jawab wale model ka quota na khaye."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Translate the given text to English. Reply with ONLY the English translation, nothing else."},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def detect_response_lang(text):
    for ch in text:
        if "\u0600" <= ch <= "\u06FF":
            return "ur"
    return "en"


def generate_voice(text):
    lang = detect_response_lang(text)
    tts = gTTS(text=text, lang=lang, slow=False)
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(path)
    return path


def transcribe_audio_bytes(audio_bytes):
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    segments, info = stt_model.transcribe(tmp_path, language="ur")
    text = " ".join([seg.text for seg in segments])
    return text.strip()


def handle_question(user_question, tag="", retrieval_query=None):
    st.session_state.messages.append({"role": "user", "content": f"{tag}{user_question}"})
    with st.spinner("Jawab tayar kiya ja raha hai..."):
        answer = ask_ai(user_question, retrieval_query=retrieval_query)
        audio_path = generate_voice(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "audio": audio_path})


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------

st.title("🏗️ Engineering Standard AI Chatbot")
st.caption("Type karein ya mic se bolein — Urdu, Roman Urdu ya English mein sawal poochein")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")

st.divider()
st.write("🎤 Bol kar poochein:")
audio = mic_recorder(start_prompt="Recording shuru karein", stop_prompt="Recording rokein", key="mic")

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

if audio and audio.get("id") != st.session_state.last_audio_id:
    st.session_state.last_audio_id = audio.get("id")
    with st.spinner("Aapki awaaz samjhi ja rahi hai..."):
        transcribed = transcribe_audio_bytes(audio["bytes"])
    if transcribed:
        with st.spinner("Behtar jawab ke liye tayari ho rahi hai..."):
            translated_query = translate_to_english(transcribed)
        handle_question(transcribed, tag="🎤 ", retrieval_query=translated_query)
        st.rerun()
    else:
        st.warning("Kuch samajh nahi aaya, dobara koshish karein.")

user_text = st.chat_input("Apna sawal type karein...")
if user_text:
    handle_question(user_text)
    st.rerun()
