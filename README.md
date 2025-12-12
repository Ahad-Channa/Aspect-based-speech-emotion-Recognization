**Overview**
- **Project**: Aspect-based Speech Emotion Recognition (ABSER).
- **Short description**: Combines Aspect-Based Sentiment Analysis (ABSA) on text with a Wav2Vec2-based speech emotion recognizer to produce a final, human-friendly output describing customer sentiment/emotion and the aspect (e.g., food) being discussed.

**Key Files**
- **`ABSA.ipynb`**: Notebook implementing and fine-tuning the RoBERTa/transformers model for aspect-based sentiment analysis (text).
- **`wav2vec2.ipynb`**: Notebook for training/fine-tuning a Wav2Vec2-based speech emotion recognition model.
- **`ABSER.ipynb`**: Notebook that integrates the two models; demonstrates full audio → transcript → ABSA + emotion pipeline.
- **`app.py`**: Streamlit web app that exposes the combined system: accepts text, live recordings, or uploaded audio and returns detected aspects, sentiment/emotion, and a concise final output.
- **`FinalReport.pdf`**: Project report with methods, experiments, results, and analysis.

**Project Structure (important paths)**
- **`Model/`**: Expected location for saved model artifacts used by `app.py`:
  - `./Model/RoBERTa_ABSA_Final` — fine-tuned RoBERTa model for ABSA (tokenizer + model files).
  - `./Model/wav2vec2-ser-ravdess` — Wav2Vec2-based speech emotion model.
- Notebooks produce and save trained models here (see notebooks for training & export steps).

**Dependencies (high-level)**
- Core: `python` (3.8+ recommended), `numpy`, `pandas`.
- Deep learning / NLP / Speech: `torch`, `transformers`, `torchaudio` (optional), `librosa`, `soundfile`.
- Web UI: `streamlit`.
- NLP helper: `spacy` + model `en_core_web_sm`.
- Optional (features shown in `app.py`): `speech_recognition`, `audio_recorder_streamlit`.

Suggested minimal `pip` install (adjust `torch` install per your CUDA/CPU setup):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install streamlit transformers librosa soundfile numpy pandas scikit-learn spacy
# Install PyTorch following instructions at https://pytorch.org/ (choose the correct CUDA/CPU wheel)
# Optional features
pip install SpeechRecognition audio_recorder_streamlit
python -m spacy download en_core_web_sm
```

Note: Installing `torch` should follow the official instructions on https://pytorch.org/ (CPU vs GPU builds). If you prefer a single-line quick install for CPU-only, run:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**How to run the Streamlit app (local)**
- From the project root (`d:\7th Semester\DL Project`):

```powershell
# Activate virtual env if created
.venv\Scripts\Activate.ps1
# (Optional) Ensure spaCy model is present
python -m spacy download en_core_web_sm
# Start app
streamlit run app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`) in a browser.

**App usage (UI flow)**
- **Text Input**: Paste or type a review; the app extracts aspects, predicts sentiment (via RoBERTa ABSA model) and returns a final, human-readable statement such as: "Customer is happy about: food".
- **Live Voice Recording**: (Requires `audio_recorder_streamlit`) record audio in the browser; the app transcribes audio (if `speech_recognition` available), detects emotion using the Wav2Vec2 model, then runs ABSA on the transcript.
- **Upload Audio File**: Upload `.wav`, `.mp3`, `.m4a`, `.flac`. The app preprocesses audio, optionally resamples to 16 kHz, runs emotion recognition and (if available) transcription.

**Notes about models**
- `app.py` will attempt to load models from `./Model/RoBERTa_ABSA_Final` and `./Model/wav2vec2-ser-ravdess`. If those directories are missing or incomplete, `app.py` falls back to Hugging Face defaults for tokenizers/models (for text) and disables emotion detection if no Wav2Vec2 model is available.
- To reproduce exact results, run the training/fine-tuning cells in `ABSA.ipynb` and `wav2vec2.ipynb` and: save the trained model folders to the paths above.

**Repro & Development tips**
- Notebooks include training, evaluation, and saving steps. Inspect `ABSA.ipynb` for the ABSA training loop and `wav2vec2.ipynb` for speech emotion fine-tuning.
- Use GPU for faster training; set up CUDA-enabled `torch` accordingly.
- For accurate speech transcription, either run `ABSER.ipynb` examples with Hugging Face models or enable `speech_recognition` (which uses Google Web Speech API by default for `recognize_google`, requiring internet).

**Troubleshooting**
- If `streamlit run app.py` fails due to missing spaCy model, run `python -m spacy download en_core_web_sm`.
- If `app.py` does not detect emotions, ensure `./Model/wav2vec2-ser-ravdess` exists and contains the model files. If not, either fine-tune per `wav2vec2.ipynb` or download a pre-trained speech emotion model.
- Audio upload errors: ensure uploaded audio is readable by `soundfile`/`librosa` and not corrupt; supported types include WAV/MP3/FLAC/M4A.

**Credits & References**
- Wav2Vec2 and Transformers: Hugging Face `transformers` library.
- Audio processing: `librosa`, `soundfile`.
- Streamlit for the interactive UI.

**Contact / Next steps**
- If you want, I can:
  - generate a `requirements.txt` from discovered imports,
  - add a small `run_app.ps1` helper script for Windows,
  - or create a minimal demo dataset and instructions to quickly reproduce results.

---
README generated automatically. File path: `d:\7th Semester\DL Project\README.md`.
