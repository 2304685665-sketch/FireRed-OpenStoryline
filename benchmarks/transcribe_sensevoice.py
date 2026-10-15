"""
Speech transcription of footage with SenseVoice (sherpa-onnx) + Silero VAD.

Produces one record per speech segment: file, start/end seconds, detected language,
emotion and text. Used as a speech channel for retrieval and to locate funny lines.

Model files (GitHub release assets of k2-fsa/sherpa-onnx, tag asr-models):
  sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/{model.int8.onnx,tokens.txt}
  silero_vad.onnx

    pip install sherpa-onnx numpy
    for f in footage/*.mp4; do ffmpeg -i "$f" -vn -ac 1 -ar 16000 "wav/$(basename "${f%.mp4}").wav"; done
    python benchmarks/transcribe_sensevoice.py --wav-dir wav --model-dir <sensevoice dir> \
        --vad silero_vad.onnx --out runs/speech/transcript.json
"""

from __future__ import annotations

import argparse
import json
import re
import wave
from pathlib import Path

import numpy as np


def main() -> None:
    import sherpa_onnx

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--vad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    m = Path(args.model_dir)
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(m / "model.int8.onnx"), tokens=str(m / "tokens.txt"), num_threads=args.threads,
        use_itn=True, language="auto")
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = args.vad
    cfg.silero_vad.min_silence_duration = 0.4
    cfg.silero_vad.max_speech_duration = 20
    cfg.sample_rate = 16000

    def key(p: Path):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]

    out = []
    for f in sorted(Path(args.wav_dir).glob("*.wav"), key=key):
        w = wave.open(str(f))
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
        for i in range(0, len(a), 512):
            vad.accept_waveform(a[i:i + 512])
        vad.flush()
        while not vad.empty():
            seg = vad.front
            s = rec.create_stream()
            s.accept_waveform(16000, seg.samples)
            rec.decode_stream(s)
            text = s.result.text.strip()
            if text:
                out.append({"clip": f.stem, "start": round(seg.start / 16000, 1),
                            "end": round((seg.start + len(seg.samples)) / 16000, 1),
                            "lang": s.result.lang, "emotion": s.result.emotion, "event": s.result.event,
                            "text": text})
            vad.pop()
        print(f.name, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(out)} speech segments -> {args.out}")


if __name__ == "__main__":
    main()
