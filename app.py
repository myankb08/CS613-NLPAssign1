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
    src_lang_name = st.selectbox("From", list(LANGUAGE_TAGS.keys()), index=list(LANGUAGE_TAGS.keys()).index("Hindi"))
with col2:
    tgt_lang_name = st.selectbox("To", list(LANGUAGE_TAGS.keys()), index=list(LANGUAGE_TAGS.keys()).index("Telugu"))

# Use tags internally (hf-style tags) — these go to IndicProcessor and forced token
src_tag = LANGUAGE_TAGS[src_lang_name]
tgt_tag = LANGUAGE_TAGS[tgt_lang_name]

text_input = st.text_area("Enter text (short paragraphs are best)", height=220, placeholder="आज सुबह मौसम बहुत अच्छा था...")
warmup = st.button("Warm up (load model)")

# ---- Model loader ----
@st.cache_resource(show_spinner=False)
def load_model_and_processor(model_name: str):
    # Prefer st.secrets then env var
    token = None
    try:
        token = st.secrets.get("HF_TOKEN")
    except Exception:
        token = None
    if not token:
        token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_kwargs = {"trust_remote_code": True}
    if token:
        load_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
    model = model.to(device)
    model.eval()

    # Create IndicProcessor once and return it
    ip = IndicProcessor(inference=True)

    return tokenizer, model, ip, device

# Try warmup if pressed
if warmup:
    try:
        with st.spinner("Loading model (warm up)..."):
            tokenizer, model, ip, device = load_model_and_processor(MODEL_NAME)
        st.success(f"Model + processor loaded on device: {device}")
    except Exception as e:
        st.error("Model load failed. See details below.")
        st.exception(e)

# Safe attempt to get tokenizer/model/ip (but don't crash the UI)
tokenizer = model = ip = device = None
try:
    tokenizer, model, ip, device = load_model_and_processor(MODEL_NAME)
except Exception:
    # Keep tokenizer/model/ip None but show message when Translate is pressed
    pass

# ---- Translation logic (uses your approach) ----
def translate_docs(docs, src_tag, tgt_tag, tokenizer, model, ip, device, num_beams=5, max_length=2048):
    """
    docs: list of raw document strings
    src_tag, tgt_tag: HF-style language tags like 'hin_Deva', 'tel_Telu'
    tokenizer, model, ip: loaded objects
    device: 'cuda' or 'cpu'
    returns: list of translated strings (one per doc)
    """
    translations = []
    for idx, doc in enumerate(docs):
        if not doc or not doc.strip():
            translations.append("")
            continue

        try:
            # preprocess_batch expects a list of strings and src/tgt tags
            preprocessed = ip.preprocess_batch([doc], src_lang=src_tag, tgt_lang=tgt_tag)
            if not preprocessed:
                # Preprocessing can return empty list -> skip
                translations.append("")
                continue
        except Exception as e:
            # keep robust in UI: append empty string on preprocess error
            st.warning(f"Preprocessing error for index {idx}: {e}")
            translations.append("")
            continue

        # Tokenize; result is dict of tensors -> move to device
        tokenized = tokenizer(preprocessed, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in tokenized.items()}

        # Build forced bos token id from target tag (the model uses tokens like '<2tel_Telu>')
        forced_bos_token_id = None
        try:
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(f"<2{tgt_tag}>")
            # convert_tokens_to_ids returns 0 for unknown token sometimes; check if token exists in vocab
            if forced_bos_token_id == tokenizer.unk_token_id:
                forced_bos_token_id = None
        except Exception:
            forced_bos_token_id = None

        gen_kwargs = {
            "num_beams": num_beams,
            "max_length": max_length,
        }
        if forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

        try:
            with torch.no_grad():
                generated = model.generate(**inputs, **gen_kwargs)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        except Exception as e:
            st.warning(f"Generation error for index {idx}: {e}")
            translations.append("")
            continue

        try:
            postprocessed = ip.postprocess_batch(decoded, lang=tgt_tag)
            # postprocessed is a list corresponding to decoded inputs -> take the 1st entry
            if postprocessed and len(postprocessed) > 0:
                translations.append(postprocessed[0])
            else:
                translations.append(decoded[0] if decoded else "")
        except Exception as e:
            # fall back to decoded raw output
            st.warning(f"Postprocessing error for index {idx}: {e}")
            translations.append(decoded[0] if decoded else "")

    return translations

# Use caching for repeated identical inputs. We prefix the model-like args with underscores so
# st.cache_data won't try to hash the model objects (they're ignored in the cache key).
@st.cache_data(show_spinner=False)
def translate_cached(_tokenizer, _model, _ip, _device, src_tag, tgt_tag, text):
    return translate_docs([text], src_tag, tgt_tag, _tokenizer, _model, _ip, _device)[0]

# ---- Translate button ----
if st.button("Translate"):
    if not text_input.strip():
        st.warning("Enter some text first.")
        st.stop()

    if tokenizer is None or model is None or ip is None:
        st.error(
            "Model/tokenizer/processor not available. Common fixes:\n"
            "1) Install sentencepiece (pip install sentencepiece)\n"
            "2) If model is gated, set HF token in env or Space secrets (see README instructions)\n"
            "3) Use a public model (change MODEL_NAME)"
        )
        st.stop()

    try:
        with st.spinner("Translating..."):
            # Use the cached wrapper which ignores the non-hashable model/tokenizer objects
            translated_text = translate_cached(tokenizer, model, ip, device, src_tag, tgt_tag, text_input)
        st.subheader("Translation")
        st.write(translated_text)
    except Exception as e:
        st.error("Inference failed. See details below.")
        st.exception(e)

st.markdown("---")
st.markdown(
    "**Notes:**\n- Short paragraphs (<= ~800 chars) work best without chunking. "
    "\n- If model is gated, provide a HF token in `st.secrets['HF_TOKEN']` on Spaces or set `HUGGINGFACEHUB_API_TOKEN` locally."
)
