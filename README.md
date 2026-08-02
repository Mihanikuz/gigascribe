# GigaScribe

GigaScribe now supports one production configuration only:

- ASR: `gigaam-v3-e2e-rnnt` (`v3_e2e_rnnt` checkpoint and tokenizer)
- Diarization: disabled or local `pyannote-community-1`
- Device: NVIDIA CUDA only
- Runtime: offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)

## Model layout

```text
models/
  gigaam/v3_e2e_rnnt/v3_e2e_rnnt.ckpt
  gigaam/v3_e2e_rnnt/v3_e2e_rnnt_tokenizer.model
  gigaam/v3_e2e_rnnt/.gigaam-ready.json
  pyannote/community-1/config.yaml
  pyannote/community-1/.pyannote-ready.json
```

## Prerequisites

- Docker Engine with the Compose plugin (`docker compose`, not the old `docker-compose`).
- An NVIDIA GPU with the driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host, so `--gpus all` works.
- Outbound internet access for the one-time model download step below (the served application itself is fully offline afterwards).
- A Hugging Face access token with access granted to [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1), only if you want speaker diarization (it is a gated model). Not needed if you plan to run with diarization disabled.

## Installation

1. Copy the example environment file and edit it:
   ```bash
   cp .env.example .env
   ```
   At minimum set real values for `GIGASCRIBE_SECRET_KEY` and `GIGASCRIBE_ADMIN_PASSWORD` — the server refuses to start with the placeholder values shipped in `.env.example` (or with `admin`). Set `HF_TOKEN` if you need diarization.

2. Build the image and download the models into `./models` (this step needs real internet access, so it explicitly overrides the container's normal offline mode):
   ```bash
   docker compose build
   docker compose run --rm gigascribe-gpu python scripts/download_models.py
   ```
   Pass `--skip-pyannote` to skip diarization if you don't have an `HF_TOKEN`. Re-run with `--check` at any time to verify what's installed without downloading anything.

   If the build machine can only reach GitHub through a proxy, `export http_proxy=... https_proxy=...` (and `no_proxy` if needed) in the shell before `docker compose build` — Docker forwards those automatically as build args, which both `apt-get` and the `pip install git+https://...` step (for the ASR backend) respect. Don't edit the Dockerfile for this.

3. Start the service and confirm it's healthy:
   ```bash
   docker compose up -d
   curl -f http://127.0.0.1:8000/health/ready
   ```

4. Open `http://<host>:8000`, log in as `admin` with the password from step 1, and create accounts for other users at `/admin`.

CPU, CTC models, pyannote 3.1, compatibility profiles, and compose override profiles have been removed.

## Users and administration

- Log in as `admin` and open `/admin` to create or disable local accounts and manage installed models.
- The transcription history is shared: every logged-in user sees every job. Only the job's owner or an administrator can cancel or retry it.

## Protocol module (AI meeting minutes)

`protocol/` is a self-contained add-on that turns a finished transcript into a structured meeting
protocol (summary, decisions, tasks, open questions, risks) using a local LLM. It is fully
isolated from transcription, diarization, the job queue, and authorization:

- **Own package.** `protocol/` never imports `app.py`, `asr_backend.py`, `diarization_backend.py`,
  or `model_manager.py`. `server.py` is the only integration point, and it talks to the module
  only through `ProtocolService` (`protocol/service.py`) and plain callables/locks it passes in
  (`MODEL_MANAGER.unload_all`, the shared GPU `asyncio.Lock`).
- **Own dependencies.** Heavy LLM dependencies (`llama-cpp-python`) live in
  `requirements-protocol.txt`, not `requirements.txt`/`requirements-base.txt`. The main image does
  not need them.
- **Own storage.** The module adds its own SQLite tables (`protocol_jobs`, `protocol_models`,
  `protocol_prompts`, `protocol_results`, `glossary_*`, `protocol_settings`) to the existing
  `jobs.sqlite3`, linked to `jobs` only via a `job_id` foreign key. Nothing in the existing `jobs`
  table schema changes.
- **Own GPU discipline.** Transcription and protocol generation never run on the GPU at the same
  time: protocol generation waits for the same GPU lock transcription jobs use, unloads GigaAM and
  pyannote (`MODEL_MANAGER.unload_all()`), runs `gc.collect()` + `torch.cuda.empty_cache()`, then
  loads the LLM. The LLM is always unloaded and VRAM freed afterwards, on both success and failure.
  ASR/diarization models simply reload on the next transcription job, as they already do today.

### Enabling / disabling

The module is off by default. Set in `.env`:

```bash
GIGASCRIBE_PROTOCOL_ENABLED=1
```

and restart the container. When unset or `0`:
- the `protocol` package is **never imported** (verified by
  `tests/test_protocol_server_integration.py::test_protocol_package_never_imported_when_disabled`),
- no protocol SQLite tables/migrations run, no LLM is ever loaded,
- the "Создать протокол" button and the admin "Протоколирование" section are hidden,
- `/health/ready` and the rest of the app are completely unaffected.

If the `protocol/` package fails to import (missing dependency, corrupted files) or is deleted
entirely while `GIGASCRIBE_PROTOCOL_ENABLED=1`, `server.py` catches the failure at startup, logs it,
and falls back to running with the module disabled — GigaScribe still starts and transcription
still works.

### Removing the module entirely

1. Delete the `protocol/` directory and `requirements-protocol.txt`.
2. Remove `GIGASCRIBE_PROTOCOL_ENABLED` (and any other `GIGASCRIBE_PROTOCOL_*`) from `.env`.
3. Nothing else needs to change — `server.py` only touches the module behind the
   `GIGASCRIBE_PROTOCOL_ENABLED` gate, and the extra SQLite tables are simply left unused (drop
   them manually with `sqlite3` if you want them gone too; this is optional, they are inert).

### Supported models

Four local GGUF chat models, run via [llama.cpp](https://github.com/ggml-org/llama.cpp)
(`llama-cpp-python`) by default, with an optional [Ollama](https://ollama.com/) engine
(`GIGASCRIBE_OLLAMA_URL`, default `http://127.0.0.1:11434`) selectable per-deployment in
Протоколирование → Настройки:

| Model | Quantization | Context | Approx. VRAM (Q4_K_M) |
|---|---|---|---|
| Qwen3-14B | Q4_K_M | 32768 | ~10-12 GB |
| Qwen3-8B | Q4_K_M | 32768 | ~6-7 GB |
| Gemma 3 12B IT | Q4_K_M | 8192 | ~8-9 GB |
| Ministral 3 8B Instruct | Q4_K_M | 32768 | ~6-7 GB |

VRAM figures are rough Q4_K_M ballpark estimates for planning purposes only — check the actual
GGUF file size for the build you download. GigaAM + pyannote are fully unloaded before the LLM
loads, so the LLM does not need to share VRAM with them.

Model weights are **not** bundled or auto-downloaded. `repo_id`/`filename` are intentionally left
blank in `protocol/models.py`, because the current GGUF quantization filenames on Hugging Face
change over time and shipping a guessed URL would be worse than requiring an explicit one. Install
a model with:

```bash
docker compose run --rm gigascribe-gpu python scripts/download_protocol_model.py \
  --model-id qwen3-8b \
  --repo-id <hugging-face-repo-id-you-verified> \
  --filename <gguf-filename-you-verified>
```

Repeat with `--model-id qwen3-14b`, `--model-id gemma3-12b-it`, or
`--model-id ministral3-8b-instruct` for the other models. `--list` shows current install status;
`--force` re-downloads. This script is the only place in the protocol module that ever reaches the
network, and only for the duration of the download — normal operation stays fully offline, same as
the main app.

After installing, an admin picks the active model, edits its launch parameters (temperature, max
tokens, context, launch args) and its **engine** — Протоколирование → Модели → "Настройка движка
модели" lets an admin choose `llama_cpp` (GGUF, local file, GPU-layer count) or `ollama` (model
tag + local Ollama daemon URL + `keep_alive`) per model, and runs a load test from that same tab.
Setting changes only apply to protocol jobs created afterwards — every job freezes a full
**settings snapshot** (model, engine, context/temperature/token limits, chunking settings, glossary
settings, and the exact prompt versions used) at creation time in `protocol_jobs.settings_snapshot`,
so jobs already queued or in progress are never affected by a later admin change, and a retry of a
failed job replays that same frozen snapshot rather than picking up new settings.

Only an administrator can set an Ollama URL (`/api/protocol/models/{id}/params` requires
`require_admin`); there is no way for a regular user to point the module at an arbitrary URL.

`llama.cpp` models are always called through `create_chat_completion()` with the model's own chat
template (read from the GGUF's `tokenizer.chat_template` metadata) — never a hardcoded raw-completion
format. A GGUF that has no embedded chat template fails to load with a clear error instead of
silently running in an incorrect mode. Qwen3 builds are additionally told not to emit `<think>`
reasoning, and any `<think>...</think>` block is stripped centrally before JSON parsing regardless
of provider, so no internal reasoning can ever leak into a chunk analysis, the merge, the fact-check
result, or the final protocol.

### Prompts and glossary

- Протоколирование → Промпты lets an admin pick "Общий промпт" or one of the four models, then edit
  the chunk-analysis, topic-split, final-merge, fact-check, or HTML-template prompt for that scope,
  preview it, and restore the built-in default. Resolution order when a job runs: the selected
  model's own active prompt → the common active prompt → the built-in default. Prompts are versioned
  (`protocol_prompts` table keeps every saved version with author/timestamp) and never hardcoded in
  `server.py` — defaults live in `protocol/prompts.py`. The exact version of every prompt kind used
  is pinned into the job's settings snapshot at creation and never changes mid-job.
- Протоколирование → Словарь/Предложения manage the term glossary (canonical term + aliases +
  scope: job/project/department/global). Terms only get applied automatically once an admin marks
  them `confirmed`. Suggestions can come from three sources, none ever applied automatically:
  admin-added terms are `confirmed` immediately; everything else is `proposed` until an admin
  confirms it — the chunk-analysis LLM call's own `terms` field (detected/suggested/context/
  confidence, deduplicated and self-replacement-filtered), a document diff, or a user/owner
  submitting `POST /api/jobs/{job_id}/glossary-suggestion` (a small form on the job card does this).
  Import/export supports JSON and CSV.

### How it works

1. User clicks "Создать протокол" on a completed job (button only appears once transcription is
   done, the module is enabled, and at least one model is installed). This creates a
   `protocol_jobs` row with a frozen settings snapshot; repeat clicks reuse the existing active job
   instead of creating a duplicate.
2. The pipeline (`protocol/service.py`) prepares the transcript and splits it into chunks. Topic
   splitting is two-stage so a long meeting is never sent to the model in one piece: rough
   10-15-minute pre-blocks each get a one-line topic hint, then a single call over those hints (not
   the raw transcript) proposes boundaries for the whole meeting; boundaries outside the
   transcript's own range are dropped and near-duplicates are merged. If that's unreliable, or
   disabled, it falls back to fixed time windows with a configurable overlap — a chunk never splits
   a reply mid-way. Each chunk is analyzed into strict JSON (using "не указан" for anything not
   stated in the transcript — the prompts explicitly forbid inventing decisions, deadlines, owners,
   or numbers), then a separate merge call combines all chunks into one document.
3. A **fact-check** stage follows the merge: every decision and task from the merged document is
   sent back to the model alongside the transcript, and the model confirms or flags each one by
   index (never rewriting the item's text). An item that can't be confirmed is marked unverified —
   never removed. If the fact-check call itself fails technically (invalid JSON after every retry),
   the whole protocol job fails rather than publishing an unverified protocol as "ready".
4. Each decision/task's timestamp is matched to the nearest real transcript segment (within a
   tolerance); a match gets a unique, stable HTML anchor (e.g. `timestamp-312-1`) and an appendix
   row (timecode, speaker, source reply, block number, item type). A timestamp that can't be matched
   is shown as plain text, never as a broken link, and the item is noted in `unverified_items`.
5. The result is rendered to `protocol.json` (source of truth — includes the model/engine, prompt
   versions, chunking settings, and app git commit for reproducibility) and `protocol.html`
   (`protocol/renderer.py`, escapes everything, table of contents, print/download buttons) under
   `data/protocols/<job_id>/`, alongside `protocol.log` (full internal detail/tracebacks — never
   shown to the user, who only sees a short status message including a `fact_checking` stage).
6. On any failure, the protocol job is marked `failed` with a short message, the LLM is unloaded,
   and the underlying transcription job is completely untouched — it is not marked failed and its
   results are not deleted. The user can retry from the same button (replaying the original settings
   snapshot, not any settings changed since).
7. On server restart, any protocol job caught mid-run is marked `failed` with a clear "restored
   after restart" message rather than silently resumed, so it can never leave the GPU lock held.

### Where things are stored

```text
data/
  jobs.sqlite3          # existing DB; protocol/* and glossary_* tables added via ALTER/CREATE, no data loss
  protocols/<job_id>/
    protocol.json        # structured result (source of truth)
    protocol.html         # rendered page served at /api/jobs/{id}/protocol
    protocol.log           # full internal log, incl. tracebacks on failure
models/
  protocol/<model_id>/...  # installed GGUF weights
```

### Docker build with (or without) the protocol module

The main Dockerfile stays exactly as before by default: `INSTALL_PROTOCOL` defaults to `0`, so
`requirements-protocol.txt` and `llama-cpp-python` are never installed and the image's size/dependency
set are unaffected.

```bash
# Without the protocol module (default, unchanged) --
docker compose -f compose.yaml build
docker compose -f compose.yaml up -d

# With it -- compose.protocol.yaml is an override, not a replacement --
docker compose -f compose.yaml -f compose.protocol.yaml build
docker compose -f compose.yaml -f compose.protocol.yaml up -d
```

`compose.protocol.yaml` only adds `--build-arg INSTALL_PROTOCOL=1` (which triggers
`CMAKE_ARGS="-DGGML_CUDA=on" pip install -r requirements-protocol.txt` inside the Dockerfile, so
`llama-cpp-python` is built with CUDA GPU offload for CUDA 12.8-capable cards such as the RTX 5060
Ti) and `GIGASCRIBE_PROTOCOL_ENABLED=1`. No proxy settings are added by either file. As with the
base image, building `llama-cpp-python` from source needs a CUDA-capable toolchain available at
build time; if your build host doesn't have one, install `requirements-protocol.txt` from a
prebuilt wheel instead of relying on this Dockerfile step.

### Running the tests

```bash
python -m pytest tests/ -q
```

Protocol-specific tests are the `tests/test_protocol_*.py` files (chunking, providers/JSON
retry-on-invalid-JSON, glossary, SQLite migration, HTML rendering/escaping, the full pipeline
against a fake LLM provider, and `server.py` isolation/access-control checks run in subprocesses).
They require no GPU and no real model weights.
