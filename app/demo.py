"""Speak-a-question demo: record or upload Hindi/Hinglish audio, get the transcript, optionally an LLM answer.

    python app/demo.py            # http://127.0.0.1:7860
    python app/demo.py --base     # zero-shot whisper-small, for comparison

Works without any API key (transcript only). Set ANTHROPIC_API_KEY or
OPENAI_API_KEY (see src/hindi_asr/llm.py) to also get an answer.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr  # noqa: E402

from hindi_asr.inference import transcribe  # noqa: E402
from hindi_asr.llm import get_client  # noqa: E402


def build(adapter) -> gr.Blocks:
    client = get_client()

    def run(audio_path, ask):
        if not audio_path:
            return "", "", "Record or upload some audio first."
        t0 = time.perf_counter()
        text = transcribe(audio_path, adapter=adapter)
        status = f"Transcribed in {time.perf_counter() - t0:.2f}s"
        answer = ""
        if ask:
            if client is None:
                status += " - no LLM key configured, transcript only"
            else:
                try:
                    answer = client.answer(text)
                except Exception as e:  # noqa: BLE001 - show provider errors in the UI
                    answer, status = "", f"{status} - LLM error: {e}"
        return text, answer, status

    label = "zero-shot whisper-small" if adapter is None else "whisper-small + Hindi LoRA"
    with gr.Blocks(title="Hindi voice ASR") as demo:
        gr.Markdown(f"## Hindi / Hinglish speech to text\nModel: **{label}**. Speak a question or upload a clip (up to a minute or so).")
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
        ask = gr.Checkbox(value=client is not None, label=f"Ask an LLM ({client.name if client else 'not configured'})")
        btn = gr.Button("Transcribe", variant="primary")
        text = gr.Textbox(label="Transcript", lines=3)
        answer = gr.Textbox(label="Answer", lines=5)
        status = gr.Markdown()
        btn.click(run, [audio, ask], [text, answer, status])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="no adapter (zero-shot base model)")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = ap.parse_args()
    adapter = None if args.base else (args.adapter or os.environ.get("HINDI_ASR_ADAPTER") or str(Path(__file__).resolve().parents[1] / "artifacts" / "adapter"))
    build(adapter).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
