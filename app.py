# app.py
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
# ---- CONFIG ----
MODEL_NAME = "ai4bharat/indictrans2-indic-indic-dist-320M"

LANGUAGE_TAGS = {
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
    "Tamil": "tam_Taml",
    "Kannada": "kan_Knda",
    "Punjabi": "pan_Guru",
    "Gujarati": "guj_Gujr",
    "Marathi": "mar_Deva",
    "Bengali": "ben_Beng",
}

# ---- UI ----
st.set_page_config(page_title="IndicTrans2 Demo", layout="centered")
st.title("IndicTrans2 demo")
st.write("Select source and target languages, paste a short paragraph/sentences and press Translate.")

col1, col2 = st.columns(2)
with col1:
    src_lang = st.selectbox("From", list(LANGUAGE_TAGS.keys()), index=list(LANGUAGE_TAGS.keys()).index("Hindi"))
with col2:
    tgt_lang = st.selectbox("To", list(LANGUAGE_TAGS.keys()), index=list(LANGUAGE_TAGS.keys()).index("Telugu"))

src_tag = LANGUAGE_TAGS[src_lang]
tgt_tag = LANGUAGE_TAGS[tgt_lang]

text_input = st.text_area("Enter text (short paragraphs are best)", height=220, placeholder="आज सुबह मौसम बहुत अच्छा था...")
warmup = st.button("Warm up (load model)")

# ---- Model loader ----
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    # Prefer st.secrets (Spaces) then env var
    token = None
    try:
        token = st.secrets.get("HF_TOKEN")
    except Exception:
        token = None
    if not token:
        token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build kwargs for from_pretrained. Only include 'token' (not use_auth_token).
    load_kwargs = {"trust_remote_code": True}
    if token:
        load_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


# Try warmup if pressed
if warmup:
    try:
        with st.spinner("Loading model (warm up)..."):
            tokenizer, model, device = load_model(MODEL_NAME)
        st.success(f"Model loaded on device: {device}")
    except Exception as e:
        st.error("Model load failed. See details below.")
        st.exception(e)

# Safe attempt to get tokenizer/model (but don't crash the UI)
tokenizer = model = device = None
try:
    tokenizer, model, device = load_model(MODEL_NAME)
except Exception as e:
    # Keep tokenizer/model None but show message when Translate is pressed
    pass

# Caching translations: ignore tokenizer/model by using leading underscores
@st.cache_data(show_spinner=False)
def translate_cached(_tokenizer, _model, _device, src_tag, tgt_tag, text):
    return translate_once(_tokenizer, _model, _device, src_tag, tgt_tag, text)

def translate_once(tokenizer, model, device, src_tag, tgt_tag, text, max_length=512, num_beams=4):
    if tokenizer is None or model is None:
        raise RuntimeError("Model or tokenizer not loaded.")
    # Prepare prompt the same way as in Kaggle notebook
    prompt = f"{src_tag} {tgt_tag} {text.strip()}"
    # Tokenize — single-shot (no custom splitting)
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    # Ensure tensors and move to device
    inputs = {k: v.to(device) for k, v in tokenized.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    return decoded

# ---- Translate button ----
if st.button("Translate"):
    if not text_input.strip():
        st.warning("Enter some text first.")
        st.stop()

    if tokenizer is None or model is None:
        st.error("Model/tokenizer not available. Common fixes:\n"
                 "1) Install sentencepiece (pip install sentencepiece)\n"
                 "2) If model is gated, set HF token in env or Space secrets (see README instructions)\n"
                 "3) Use a public model (change MODEL_NAME)")
        st.stop()

    try:
        # Use cache to speed up repeated identical inputs
        print("hello app")
        ip = IndicProcessor(inference=True)
        translation = translate_cached(tokenizer, model, device, src_tag, tgt_tag, text_input)
        st.subheader("Translation")
        print("translation ",translation)
        translated = ip.postprocess_batch(translation, lang=tgt_lang)
        print("translated ",translated)
        st.write(translated[0])
    except Exception as e:
        st.error("Inference failed. See details below.")
        st.exception(e)

st.markdown("---")
st.markdown("**Notes:**\n- Short paragraphs (<= ~800 chars) work best without chunking. "
            "\n- If model is gated, provide a HF token in `st.secrets['HF_TOKEN']` on Spaces or set `HUGGINGFACEHUB_API_TOKEN` locally.")



