import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import spacy
import numpy as np
import librosa
import soundfile as sf
import tempfile
from pathlib import Path

# Optional: Speech recognition
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

# Optional: Audio recorder
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

# --- CONFIG ---
st.set_page_config(
    page_title="ABSA Voice Emotion & Sentiment Analyzer",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# (Your CSS styling block unchanged)...
# --- CUSTOM UI STYLING ---
st.markdown("""
<style>
/* --- MAIN BACKGROUND (Orange Gradient) --- */
.stApp {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
}

/* --- MAIN CARD --- */
.main .block-container {
    background-color: #ffffff;
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 8px 25px rgba(255,140,0,0.2);
}

/* --- HEADER --- */
.main-header {
    font-size: 2.7rem;
    text-align: center;
    color: #e65100;
    font-weight: bold;
}

/* --- TEXT AREA --- */
.stTextArea textarea {
    background: #ffffff !important;
    color: #000 !important;
    border: 2px solid #ff9800 !important;
    border-radius: 10px !important;
    padding: 15px !important;
}
.stTextArea textarea:focus {
    border-color: #fb8c00 !important;
    box-shadow: 0 0 8px rgba(251,140,0,0.35);
}

/* --- BUTTON --- */
.stButton > button {
    background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%) !important;
    border: 2px solid #e65100 !important;
    color: white !important;
    padding: 0.5rem 2rem !important;
    border-radius: 10px !important;
    font-weight: bold;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(255,152,0,0.4);
}

/* --- TABS --- */
.stTabs [data-baseweb="tab"] {
    background: #fff3e0;
    border-radius: 10px 10px 0 0;
    padding: 8px 15px;
    border: 2px solid #ffb74d;
    color: #e65100;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%) !important;
    color: white !important;
    border-bottom: none;
}

/* --- RESULT CARDS --- */
.result-card {
    background: #fff8e1;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #ffb74d;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.emotion-card { border-left-color: #fb8c00; }
.transcript-card { border-left-color: #f57c00; }
.aspect-card { border-left-color: #ffa726; }

.final-card {
    border-left-color: #e65100;
    background: #e65100;
    color: white;
}

/* --- ASPECT TAG --- */
.aspect-tag {
    background: #ff9800;
    color: white;
    padding: 7px 12px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: bold;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)


# --- FOOD KEYWORDS (new) ---
FOOD_KEYWORDS = [
    "food", "taste", "flavor", "meal", "dish", "rice", "biryani", "pizza",
    "burger", "pasta", "noodles", "chicken", "beef", "mutton", "spices",
    "salt", "sweetness", "freshness", "quality", "portion", "service", "menu"
]

# --- MODEL PATHS ---
ROBERTA_MODEL_PATH = "./Model/RoBERTa_ABSA_Final"
EMOTION_MODEL_PATH = "./Model/wav2vec2-ser-ravdess"
FALLBACK_MODEL = "roberta-base"

EMOTION_LABELS = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    tokenizer = model = nlp = emotion_processor = emotion_model = None

    # RoBERTa
    try:
        model_path = Path(ROBERTA_MODEL_PATH)
        if model_path.exists() and any(model_path.glob("pytorch_model*.bin")):
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)
    except:
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)

    # Emotion model
    try:
        emotion_path = Path(EMOTION_MODEL_PATH)
        if emotion_path.exists():
            emotion_processor = Wav2Vec2Processor.from_pretrained(str(emotion_path))
            emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(str(emotion_path))
    except:
        pass

    # spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    return tokenizer, model, nlp, emotion_processor, emotion_model

tokenizer, model, nlp, emotion_processor, emotion_model = load_models()

# --- HELPER FUNCTIONS ---
def predict_sentiment(text):
    if not text.strip():
        return "neutral", 0.5
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        conf = torch.max(probs).item()
        label = model.config.id2label.get(logits.argmax().item(), "neutral")
        # optional normalization:
        SENTIMENT_MAP = {"LABEL_0": "negative", "LABEL_1": "positive", "LABEL_2": "neutral"}
        label = SENTIMENT_MAP.get(label, label)
        return label, conf
    except:
        return "neutral", 0.5

def extract_aspects(text):
    if not nlp or not text.strip():
        return []
    doc = nlp(text)
    aspects = []
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        for keyword in FOOD_KEYWORDS:
            if keyword in chunk_text:
                aspects.append(chunk.text.strip())
                break
    if not aspects:
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):
                tok = token.text.lower()
                for keyword in FOOD_KEYWORDS:
                    if keyword in tok:
                        aspects.append(token.text)
                        break
    return list(dict.fromkeys([a for a in aspects if len(a)>1]))

def detect_emotion(audio, sr):
    if not emotion_processor or not emotion_model:
        return "neutral"
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    inputs = emotion_processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    pred = logits.argmax().item()
    return EMOTION_LABELS.get(pred, "neutral")

def transcribe_audio(audio, sr):
    if not SPEECH_AVAILABLE:
        return "Speech recognition unavailable."
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio, sr)
            tmp_path = tmp.name
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio_rec = r.record(source)
            text = r.recognize_google(audio_rec)
        return text
    except:
        return "Could not transcribe audio."

def process_uploaded_file(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.read())
            path = tmp.name
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        return audio, sr
    except Exception as e:
        st.error(f"Audio processing failed: {e}")
        return None, None

# --- APP STATE ---
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "audio_sr" not in st.session_state:
    st.session_state.audio_sr = None
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "emotion" not in st.session_state:
    st.session_state.emotion = None

# --- UI ---
st.markdown('<h1 style="text-align:center;">🎤 ABSA Voice Emotion & Sentiment Analyzer</h1>', unsafe_allow_html=True)

tabs = st.tabs(["📝 Text Input", "🎤 Live Voice Recording", "📁 Upload Audio File"])
input_text = ""

# --- TEXT INPUT TAB ---
with tabs[0]:
    text = st.text_area("Enter your review or feedback:", placeholder="Example: Your food is disgusting!")
    if st.button("🔍 Analyze Text"):
        input_text = text.strip()
        # set session values for text path
        st.session_state.transcript = input_text
        st.session_state.audio_data = None
        st.session_state.audio_sr = None
        st.session_state.emotion = None  # no emotion from text

# --- LIVE VOICE RECORDING TAB ---
with tabs[1]:
    if AUDIO_RECORDER_AVAILABLE:
        audio_bytes = audio_recorder()
        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            audio, sr = sf.read(tmp_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            st.session_state.audio_data = audio     
            st.session_state.audio_sr = sr
            st.session_state.emotion = detect_emotion(audio, sr)
            st.session_state.transcript = transcribe_audio(audio, sr)
            input_text = st.session_state.transcript
            st.success(f"🎯 Transcribed: {input_text}")
            # DO NOT show emotion here (removed per request)
    else:
        st.warning("Audio recorder not installed. Install `audio_recorder_streamlit` package.")

# --- UPLOAD FILE TAB ---
with tabs[2]:
    uploaded_file = st.file_uploader("Upload audio file (wav, mp3, m4a, flac):", type=["wav","mp3","m4a","flac"])
    if uploaded_file:
        st.session_state.audio_data, st.session_state.audio_sr = process_uploaded_file(uploaded_file)
        if st.session_state.audio_data is not None:
            st.session_state.emotion = detect_emotion(st.session_state.audio_data, st.session_state.audio_sr)
            st.session_state.transcript = transcribe_audio(st.session_state.audio_data, st.session_state.audio_sr)
            input_text = st.session_state.transcript
            st.success(f"🎯 Transcribed: {input_text}")
            # DO NOT show emotion here (removed per request)
            st.audio(uploaded_file)

# --- ANALYSIS SECTION ---
if input_text:
    st.divider()
    st.markdown("### 📊 Analysis Results")
    aspects = extract_aspects(input_text)
    sentiment, conf = predict_sentiment(input_text)  # compute but do not display raw model label
    # show emotion once in analysis
    if st.session_state.emotion:
        st.info(f"Detected Emotion: {st.session_state.emotion}")
    # DO NOT show predicted sentiment label line (removed)
    if aspects:
        st.info(f"Detected Aspects: {', '.join(aspects)}")
    # build final output with food focus
    customer_state = st.session_state.emotion.lower() if st.session_state.emotion else sentiment.lower()
    if not aspects:
        if any(k in input_text.lower() for k in FOOD_KEYWORDS):
            aspects = ["food"]
    final_out = f"Customer is {customer_state}"
    if aspects:
        final_out += f" about: {', '.join(aspects)}"
    st.success(f"🚀 FINAL OUTPUT: {final_out}")
else:
    st.info("👆 Enter text, record live audio, or upload audio to start analysis.")
