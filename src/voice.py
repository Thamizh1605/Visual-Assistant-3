# src/voice.py
"""
voice.py
- TTS via pyttsx3 (offline)
- Recording via sounddevice
- STT via Vosk (offline)
"""

import queue
import sounddevice as sd
import wavio
import os
from pathlib import Path
import json
from typing import Optional
from .utils import model_path, timestamped_filename

# TTS
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 160)
except Exception as e:
    print("[voice] pyttsx3 init failed:", e)
    _tts_engine = None

def speak(text: str):
    if _tts_engine is not None:
        try:
            _tts_engine.say(text)
            _tts_engine.runAndWait()
            return
        except Exception as e:
            print("[voice] pyttsx3 failed:", e)
    # fallback to print
    print("[TTS]", text)

# Recording helpers
_SAMPLE_RATE = 16000
_channels = 1

_recording_queue = None
_recording_stream = None
_recording_file = None

def start_recording(output_dir="recordings"):
    global _recording_queue, _recording_stream, _recording_file
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    fname = timestamped_filename("question", "wav")
    _recording_file = Path(output_dir) / fname
    _recording_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print("[voice] status:", status)
        _recording_queue.put(indata.copy())

    _recording_stream = sd.InputStream(samplerate=_SAMPLE_RATE, channels=_channels, dtype='int16', callback=callback)
    _recording_stream.start()
    print("[voice] Recording started ->", _recording_file)
    return str(_recording_file)

def stop_recording_and_save():
    global _recording_queue, _recording_stream, _recording_file
    if _recording_stream is None:
        return None
    _recording_stream.stop()
    _recording_stream.close()
    # drain queue into array
    frames = []
    while not _recording_queue.empty():
        frames.append(_recording_queue.get())
    if frames:
        import numpy as np
        data = np.concatenate(frames, axis=0)
        # wavio expects float or int16
        wavio.write(str(_recording_file), data, _SAMPLE_RATE, sampwidth=2)
        print("[voice] Saved recording:", _recording_file)
        # reset
        _recording_queue = None
        _recording_stream = None
        fpath = str(_recording_file)
        _recording_file = None
        return fpath
    else:
        print("[voice] No audio data captured.")
        _recording_queue = None
        _recording_stream = None
        _recording_file = None
        return None

# Vosk transcription
try:
    from vosk import Model, KaldiRecognizer
    _has_vosk = True
except Exception:
    _has_vosk = False

def transcribe_vosk(wav_path: str, vosk_model_dir: Optional[str] = None) -> Optional[str]:
    if not _has_vosk:
        print("[voice] Vosk not installed.")
        return None
    vosk_model_dir = vosk_model_dir or str(model_path("/Users/thamizharasan/Desktop/Visual-Assistant-3/models/vosk-model/vosk-model-en-us-0.22"))
    if not os.path.exists(vosk_model_dir):
        print("[voice] Vosk model not found at", vosk_model_dir)
        return None
    try:
        wf = None
        import wave
        wf = wave.open(wav_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != _SAMPLE_RATE:
            print("[voice] WAV file not in required format, trying to convert.")
            # naive conversion via ffmpeg would be ideal; here we try numpy + wavio but keep simple
        model = Model(vosk_model_dir)
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        result_text = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                result_text.append(res.get("text", ""))
        res = json.loads(rec.FinalResult())
        result_text.append(res.get("text", ""))
        return " ".join([t for t in result_text if t])
    except Exception as e:
        print("[voice] Vosk transcription failed:", e)
        return None
