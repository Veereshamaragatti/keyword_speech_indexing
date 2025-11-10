import os
from pathlib import Path
from typing import List, Dict, Any
import json
import importlib

# ===============================================================
# 🔧 STEP 1: Force-set absolute FFmpeg path for Whisper
# ===============================================================
FFMPEG_BIN = Path(r"E:\Bindu\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe")
if FFMPEG_BIN.exists():
    ffmpeg_dir = str(FFMPEG_BIN.parent)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
    os.environ["FFMPEG_BINARY"] = str(FFMPEG_BIN)
    print("✅ FFmpeg found at:", FFMPEG_BIN)
else:
    print("❌ FFmpeg not found at:", FFMPEG_BIN)
    raise FileNotFoundError(f"ffmpeg.exe not found at {FFMPEG_BIN}")
# ===============================================================


# ===============================================================
# STEP 2: Import external dependencies (whisper, deep-translator)
# ===============================================================
try:
    import whisper
except Exception:
    whisper = None

try:
    module = importlib.import_module("deep_translator")
    GoogleTranslator = getattr(module, "GoogleTranslator", None)
except Exception:
    GoogleTranslator = None
# ===============================================================


# ===============================================================
# STEP 3: Supported Indian languages
# ===============================================================
SUPPORTED_LANGS = [
    "en", "hi", "kn", "te", "ta", "ml", "mr", "gu", "bn", "pa", "or", "ur"
]
# ===============================================================


# ===============================================================
# STEP 4: Utility functions
# ===============================================================
def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm for VTT."""
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_vtt(segments: List[Dict[str, Any]], out_path: Path):
    """Write list of {start, end, text} segments into .vtt format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip().replace("-->", "→")
            f.write(f"{start} --> {end}\n{text}\n\n")
# ===============================================================


# ===============================================================
# STEP 5: Translation helper using deep-translator
# ===============================================================
def translate_segments(segments: List[Dict[str, Any]], target_lang: str) -> List[Dict[str, Any]]:
    """Translate subtitle segments into target language using Google Translate."""
    if target_lang == "en":
        return segments

    if GoogleTranslator is None:
        raise RuntimeError("Translation requires 'deep-translator'. Install via: pip install deep-translator")

    print(f"🔁 Translating subtitles to {target_lang}...")
    translator = GoogleTranslator(source='auto', target=target_lang)
    out = []
    delim = " ||| "
    joined = delim.join([s["text"] for s in segments])

    try:
        translated = translator.translate(joined)
        parts = [p.strip() for p in translated.split("|||")]
        if len(parts) != len(segments):
            raise ValueError("Batch translation mismatch — using fallback")
    except Exception as e:
        print(f"⚠️ Batch translation failed for {target_lang}: {e}")
        parts = []
        for s in segments:
            try:
                parts.append(translator.translate(s["text"]))
            except Exception as ex:
                print(f"⚠️ Line translation failed ({target_lang}): {ex}")
                parts.append(s["text"])

    for s, t in zip(segments, parts):
        out.append({"start": s["start"], "end": s["end"], "text": t})

    print(f"✅ Translated {len(out)} segments to {target_lang}")
    return out
# ===============================================================


# ===============================================================
# STEP 6: Whisper transcription
# ===============================================================
def transcribe_core(audio_path: str) -> List[Dict[str, Any]]:
    """Transcribe a single audio/video file using Whisper."""
    if whisper is None:
        raise RuntimeError("Whisper not installed. Run: pip install -U openai-whisper")

    print(f"🎙️ Transcribing: {audio_path}")
    model_name = os.environ.get("WHISPER_MODEL", "small")
    model = whisper.load_model(model_name)

    # Transcribe using ffmpeg under the hood
    result = model.transcribe(audio_path, task="transcribe")
    segments = [
        {"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"]}
        for seg in result.get("segments", [])
    ]

    if not segments and result.get("text"):
        segments = [{"start": 0.0, "end": 0.01, "text": result["text"]}]

    print(f"✅ Transcription complete — {len(segments)} segments.")
    return segments
# ===============================================================


# ===============================================================
# STEP 7: Main transcribe + translate + save
# ===============================================================
def transcribe_to_vtt_many(media_path: str, vtt_dir: Path, langs: List[str], video_id: str = None) -> Dict[str, str]:
    """
    Transcribes and translates media into multiple languages.
    Returns dict: {lang_code: vtt_path_str}
    """
    p = Path(media_path)
    if video_id is None:
        video_id = p.stem

    print(f"🎬 Starting transcription for: {p.name}")
    base_segments = transcribe_core(media_path)

    out = {}
    for code in langs:
        if code not in SUPPORTED_LANGS:
            print(f"⚠️ Skipping unsupported language: {code}")
            continue

        try:
            if code == "en":
                segs = base_segments
            else:
                segs = translate_segments(base_segments, code)
        except Exception as e:
            print(f"⚠️ Translation failed for {code}: {e}")
            segs = base_segments

        vtt_path = vtt_dir / f"{video_id}.{code}.vtt"
        write_vtt(segs, vtt_path)
        out[code] = str(vtt_path)
        print(f"💾 Saved {code} subtitles → {vtt_path.name}")

    # Save manifest for indexing
    manifest = {
        "video_id": video_id,
        "media_file": os.path.basename(media_path),
        "langs": list(out.keys())
    }
    with (vtt_dir / f"{video_id}.manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"📜 Manifest saved for {video_id}")
    return out
# ===============================================================
