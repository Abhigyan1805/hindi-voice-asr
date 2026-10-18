"""Command line: ``python -m hindi_asr.cli audio.wav [--ask] [--base]``."""

from __future__ import annotations

import argparse
import time

from .inference import transcribe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Transcribe Hindi/Hinglish speech with Whisper-small + LoRA.")
    ap.add_argument("audio", nargs="+", help="audio file(s): wav, flac, mp3, ...")
    ap.add_argument("--adapter", help="LoRA adapter dir (default: artifacts/adapter or $HINDI_ASR_ADAPTER)")
    ap.add_argument("--base", action="store_true", help="use the zero-shot base model, no adapter")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--ask", action="store_true", help="also send the transcript to the configured LLM")
    args = ap.parse_args(argv)

    kwargs = {"num_beams": args.beams}
    if args.base:
        kwargs["adapter"] = None
    elif args.adapter:
        kwargs["adapter"] = args.adapter
    client = None
    if args.ask:
        from .llm import get_client

        client = get_client()
        if client is None:
            print("(no LLM configured: set ANTHROPIC_API_KEY or OPENAI_API_KEY; transcribing only)")
    for path in args.audio:
        t0 = time.perf_counter()
        text = transcribe(path, **kwargs)
        print(f"{path}\t{text}\t[{time.perf_counter() - t0:.2f}s]")
        if client is not None:
            print(f"  -> {client.answer(text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
