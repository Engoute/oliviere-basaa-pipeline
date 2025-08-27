# FILE: app/pipeline_core.py
"""
Central pipeline logic:
- Input language selection controls which ASR to use
- Mode (chat vs translate) controls how we route text through Qwen + M2M + Orpheus
- Returns labeled text and audio (both streamable chunks and a final WAV)
"""

import os
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    AutoModelForCausalLM, pipeline
)

from .utils_audio import decode_audio_to_16k_float_mono, wav_bytes_from_float32
from .tts_orpheus import Orpheus


Lang = Literal["fr", "en", "lg"]         # lg = Basaa
Mode = Literal["chat", "translate"]


# --------- model paths from env (populated by your bootstrap) ----------
MODELS_DIR = os.environ.get("MODELS_DIR", "/data/models")
PATH_WHISPER_GENERAL = os.environ.get("PATH_WHISPER_GENERAL", os.path.join(MODELS_DIR, "whisper_general"))
PATH_WHISPER_BASAA   = os.environ.get("PATH_WHISPER_BASAA",   os.path.join(MODELS_DIR, "whisper_hf"))
PATH_M2M             = os.environ.get("PATH_M2M",             os.path.join(MODELS_DIR, "m2m100_1p2B"))
PATH_QWEN            = os.environ.get("PATH_QWEN",            os.path.join(MODELS_DIR, "qwen2_5_instruct_7b"))
PATH_ORPHEUS         = os.environ.get("PATH_ORPHEUS",         os.path.join(MODELS_DIR, "orpheus_bundle"))

# M2M language codes (adjust if your bundle uses different tags)
M2M_CODE_FR = os.environ.get("M2M_CODE_FR", "fr")
M2M_CODE_EN = os.environ.get("M2M_CODE_EN", "en")
M2M_CODE_LG = os.environ.get("M2M_CODE_LG", "bas")  # Basaa code in M2M100; change if your bundle differs


@dataclass
class Models:
    asr_general: any
    asr_basaa: any
    m2m_tok: AutoTokenizer
    m2m: AutoModelForSeq2SeqLM
    qwen_tok: AutoTokenizer
    qwen: AutoModelForCausalLM
    orpheus: Orpheus


def load_models(device: str = None) -> Models:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ASR
    asr_general = pipeline(
        "automatic-speech-recognition",
        model=PATH_WHISPER_GENERAL,
        device=0 if dev == "cuda" else -1,
    )
    asr_basaa = pipeline(
        "automatic-speech-recognition",
        model=PATH_WHISPER_BASAA,
        device=0 if dev == "cuda" else -1,
    )

    # M2M100
    m2m_tok = AutoTokenizer.from_pretrained(PATH_M2M, local_files_only=True)
    m2m = AutoModelForSeq2SeqLM.from_pretrained(PATH_M2M, local_files_only=True).to(dev).eval()

    # Qwen (French replies)
    qwen_tok = AutoTokenizer.from_pretrained(PATH_QWEN, local_files_only=True, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(PATH_QWEN, local_files_only=True, trust_remote_code=True).to(dev).eval()

    # Orpheus TTS
    orpheus = Orpheus(PATH_ORPHEUS)

    return Models(asr_general, asr_basaa, m2m_tok, m2m, qwen_tok, qwen, orpheus)


# -------------- helpers --------------

def m2m_translate(models: Models, text: str, src: Lang, tgt: Lang) -> str:
    tok = models.m2m_tok
    mdl = models.m2m
    code_map = {"fr": M2M_CODE_FR, "en": M2M_CODE_EN, "lg": M2M_CODE_LG}
    src_code = code_map[src]
    tgt_code = code_map[tgt]

    # For M2M100: set the target language by forced_bos_token_id
    tok.src_lang = src_code
    encoded = tok(text, return_tensors="pt").to(mdl.device)
    gen = mdl.generate(
        **encoded,
        forced_bos_token_id=tok.get_lang_id(tgt_code),
        max_length=512,
        num_beams=4,
    )
    out = tok.batch_decode(gen, skip_special_tokens=True)[0]
    return out.strip()


def qwen_chat_fr(models: Models, user_fr: str) -> str:
    """
    Ask Qwen to respond in *French* (polite, concise).
    """
    tok, mdl = models.qwen_tok, models.qwen
    system = "Vous êtes un assistant utile. Répondez en français, de façon claire et concise."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_fr},
    ]
    input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
    gen = mdl.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    out = tok.decode(gen[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return out.strip()


def run_pipeline(
    models: Models,
    mode: Mode,
    input_lang: Lang,
    raw_audio_bytes: bytes,
) -> Tuple[str, str, str, np.ndarray, int]:
    """
    Returns (asr_text, fr_text, basaa_text, wav_float32, sr_out)
    wav_float32 is the audio to be streamed/encoded; sr_out is its sampling rate.
    """
    # 1) decode user audio to 16k mono float
    x16k = decode_audio_to_16k_float_mono(raw_audio_bytes)

    # 2) transcribe with the correct ASR
    if input_lang in ("fr", "en"):
        asr = models.asr_general(x16k, sampling_rate=16000)
    else:
        asr = models.asr_basaa(x16k, sampling_rate=16000)

    asr_text = (asr["text"] if isinstance(asr, dict) else str(asr)).strip()

    # 3) route by mode
    if mode == "chat":
        if input_lang in ("fr", "en"):
            # user FR/EN -> Qwen (FR) -> M2M FR->LG -> Basaa TTS
            user_fr = asr_text if input_lang == "fr" else m2m_translate(models, asr_text, "en", "fr")
            fr_text = qwen_chat_fr(models, user_fr)
            basaa_text = m2m_translate(models, fr_text, "fr", "lg")
        else:
            # user LG -> FR via M2M -> Qwen -> FR->LG for TTS
            fr_in = m2m_translate(models, asr_text, "lg", "fr")
            fr_text = qwen_chat_fr(models, fr_in)
            basaa_text = m2m_translate(models, fr_text, "fr", "lg")
    else:  # translate mode
        if input_lang in ("fr", "en"):
            # FR/EN -> LG, TTS Basaa
            pivot_fr = asr_text if input_lang == "fr" else m2m_translate(models, asr_text, "en", "fr")
            fr_text = pivot_fr
            basaa_text = m2m_translate(models, pivot_fr, "fr", "lg")
        else:
            # LG -> FR (text). We still speak Basaa so user hears the LG line.
            fr_text = m2m_translate(models, asr_text, "lg", "fr")
            basaa_text = asr_text

    # 4) synthesize TTS (Basaa)
    wav = models.orpheus.tts(basaa_text)
    sr_out = models.orpheus.sr_out
    return asr_text, fr_text, basaa_text, wav, sr_out


def labeled_block(asr: str, fr: str, lg: str) -> str:
    """
    Matches your ListeningPage parser:
      "Vous:\nFR:\nBasaa:"
    """
    parts = []
    if asr.strip():
        parts.append(f"Vous: {asr.strip()}")
    if fr.strip():
        parts.append(f"FR: {fr.strip()}")
    if lg.strip():
        parts.append(f"Basaa: {lg.strip()}")
    return "\n".join(parts)
