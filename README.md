# hindi-voice-asr

Hindi / Hinglish speech-to-text: **`openai/whisper-small` fine-tuned with LoRA on Google FLEURS Hindi**, evaluated honestly against the zero-shot base model (and whisper-medium), and packaged as:

- a Python API, `transcribe("clip.wav") -> "हिंदी टेक्स्ट"`, with the adapter merged into the base model so decoding costs exactly what the base model costs;
- a CLI (`hindi-asr clip.wav`);
- a small **speak-a-question demo** (Gradio): record or upload audio, read the transcript, and optionally send it to an LLM for an answer. It works without any API key and then only transcribes.

Everything is reproducible: one Kaggle T4 script kernel trains and evaluates every model in the tables below, with fixed seeds and the exact code bundled from this repo.

## Results (FLEURS Hindi test, 418 utterances, Kaggle T4)

| Model | WER % | CER % | Single-stream RTF | Runaway outputs |
|---|---:|---:|---:|---:|
| whisper-small, zero-shot | 79.19 | 49.44 | 0.173 | 35 |
| whisper-medium, zero-shot (3× larger) | 42.95 | 21.40 | 0.241 | 6 |
| **whisper-small + LoRA r32 (this repo)** | **24.53** | **10.11** | **0.130** | 1 |

Even after re-scoring every model only on the clips where none of them falls into a repetition loop, WER still drops from 55.9 to 23.4. Ablation (one seed each): LoRA rank 8 gives 26.98 WER, and dropping augmentation gives 25.67 WER. Full tables, run IDs, versions and qualitative examples, including the code-switched (Latin-script) subset, are in **[RESULTS.md](RESULTS.md)**. Data, training details and limitations are in **[MODEL_CARD.md](MODEL_CARD.md)**.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[demo,llm]"          # torch, transformers, peft, jiwer, gradio, ...

# the LoRA adapter (about 28 MB) lives in artifacts/adapter/ (gitignored; see "Get the adapter")
hindi-asr question.wav                 # fine-tuned model
hindi-asr --base question.wav          # zero-shot whisper-small, for comparison
python app/demo.py                     # http://127.0.0.1:7860
```

```python
from hindi_asr.inference import transcribe
print(transcribe("question.wav"))                 # whisper-small + Hindi LoRA
print(transcribe("question.wav", adapter=None))   # zero-shot base
```

It runs on CPU (slowly) or on any CUDA GPU in fp16. Audio of any length is split into 30 s windows.

### LLM answers (optional)

The demo and `hindi-asr --ask` pass the transcript to a pluggable client (`src/hindi_asr/llm.py`):

| Env var | Effect |
|---|---|
| `ANTHROPIC_API_KEY` | use Claude (default model `claude-haiku-4-5-20251001`) |
| `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`) | use an OpenAI-compatible endpoint, including a local vLLM or Ollama server |
| `HINDI_ASR_LLM` | force `anthropic`, `openai` or `none` |
| `HINDI_ASR_LLM_MODEL` | override the model name |

Keys are read from the environment only and are never written anywhere.

### Get the adapter

The trained adapter is a Kaggle kernel output, not stored in git. Copy it into `artifacts/adapter/`:

```bash
KAGGLE_API_TOKEN=... kaggle kernels output abhigyan1881/hva-whisper-lora -p kaggle/output/full
mkdir -p artifacts && cp -r kaggle/output/full/adapters/lora-r32-aug artifacts/adapter
```

Alternatively, set `HINDI_ASR_ADAPTER=/path/to/adapter`.

## How it works

| Piece | Where | Notes |
|---|---|---|
| Data | `src/hindi_asr/data.py` | FLEURS `hi_in` from the ungated parquet export (no loading script); 16 kHz mono |
| Augmentation | `data.py: Augment` | 3-way speed perturbation (0.9/1.0/1.1, Kaldi style) plus white noise at 10–30 dB SNR with p=0.3; seeded per (epoch, item) |
| Text normalisation | `src/hindi_asr/text.py` | NFC, punctuation/symbols/danda removed, Latin lowercased. **Whisper's `BasicTextNormalizer` is not used**, because it deletes Devanagari matras and viramas (Unicode combining marks) and so under-counts errors |
| Model | `src/hindi_asr/model.py` | LoRA on `q_proj,k_proj,v_proj,out_proj` of every encoder/decoder attention block (self and cross), alpha = 2·rank, dropout 0.05 |
| Training | `src/hindi_asr/train.py` | plain PyTorch loop, fp16 autocast + GradScaler, AdamW lr 1e-3, 10 % warmup then linear decay, effective batch 16 (8×2 accumulation), 5 epochs, best epoch picked on 200 validation utterances |
| Evaluation | `kaggle/pipeline.py` | greedy decoding, `language=hi, task=transcribe`, full FLEURS test split; corpus WER/CER; single-stream RTF on 60 test clips; batched throughput |
| Kaggle bundling | `kaggle/build_kernel.py` | packs `src/` + the pipeline into one script kernel (`kaggle/kernels/full/`) |

## Reproduce

```bash
python kaggle/build_kernel.py --mode smoke   # 3-minute end-to-end check on a T4
python kaggle/build_kernel.py --mode full    # baselines + 3 LoRA runs + whisper-medium (about 1.8 h on a T4)
KAGGLE_API_TOKEN="$(cat ~/.kaggle-abhigyan1881/access_token)" kaggle kernels push -p kaggle/kernels/full
KAGGLE_API_TOKEN=... kaggle kernels output abhigyan1881/hva-whisper-lora -p kaggle/output/full
python kaggle/report.py kaggle/output/full    # Markdown tables for RESULTS.md
pytest                                         # normaliser / metric / augmentation unit tests
```

## Limitations

See [MODEL_CARD.md](MODEL_CARD.md). In short: FLEURS is read speech of Wikipedia-style sentences, so conversational, noisy, telephone or strongly code-switched audio will be worse. FLEURS writes English loanwords in Devanagari, so the model learns to do the same.

## Possible follow-ups

- A voice front-end for the `guardrailed-sql-analyst` project: ask a data question aloud, then transcribe → SQL agent.
- Add Common Voice / IndicVoices / Shrutilipi data for conversational coverage, then try whisper-medium + LoRA or distil/turbo variants for latency.
- Streaming (chunked) decoding and CTranslate2 / faster-whisper export for CPU serving.

## Licence

Code: MIT. Base model `openai/whisper-small`: Apache-2.0. FLEURS data: CC-BY-4.0. The adapter is a derivative of both (see the model card).
