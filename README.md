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

A plain, proxy-independent checklist — proxies are only relevant on restricted networks and are a
separate, optional note further down ([Restricted networks](#restricted-networks-proxy--no-github-access)), not a step everyone needs:

1. Copy the example environment file and edit it:
   ```bash
   cp .env.example .env
   ```
   At minimum set real values for `GIGASCRIBE_SECRET_KEY` and `GIGASCRIBE_ADMIN_PASSWORD` — the server refuses to start with the placeholder values shipped in `.env.example` (or with `admin`). Set `HF_TOKEN` if you need diarization.

2. Download and verify every model the application needs — GigaAM, pyannote (unless skipped), and,
   if you're using the [protocol module](#protocol-module-ai-meeting-minutes), its LLM — *before*
   building or starting anything:
   ```bash
   ./scripts/download-models.sh
   ```
   This builds the base image (needed to run GigaAM/pyannote's own loaders for verification) and then
   downloads into `./models`. It's safe to re-run — nothing already downloaded and verified is
   re-fetched. Set `GIGASCRIBE_SKIP_PYANNOTE=1` first if you don't have an `HF_TOKEN` and want to skip
   diarization. To also fetch the protocol module's `qwen3-8b` model in this same step, set
   `GIGASCRIBE_PROTOCOL_MODEL_REPO_ID`/`GIGASCRIBE_PROTOCOL_MODEL_FILENAME` first (see
   [Supported models](#supported-models) for why those aren't filled in for you); otherwise install it
   separately later. At any time, re-check what's installed without downloading anything:
   `docker compose run --rm gigascribe-gpu python scripts/download_models.py --check`.

3. If you plan to use the protocol module with the Ollama engine instead of llama.cpp, install and
   start Ollama on the host (or wherever `GIGASCRIBE_OLLAMA_URL` points) now — see
   [Prompts and glossary](#prompts-and-glossary) / [Supported models](#supported-models) for engine
   selection. Nothing else in this checklist depends on it.

4. Build and start:
   ```bash
   docker compose build   # fast: step 2 already built and cached the base image
   docker compose up -d
   curl -f http://127.0.0.1:8000/health/ready
   ```

5. Open `http://<host>:8000`, log in as `admin` with the password from step 1, and create accounts for other users at `/admin`.

CPU, CTC models, pyannote 3.1, compatibility profiles, and compose override profiles have been removed.

## Restricted networks (proxy / no GitHub access)

Two independent problems can show up on a network where GitHub (or Hugging Face, or Ollama's
registry) isn't directly reachable — handle whichever actually applies to you:

**A proxy is required.** `compose.yaml`'s `build.args` already forwards `http_proxy`/`https_proxy`/
`no_proxy` from your shell into the build automatically — just `export` them before `docker compose
build`; you don't need `--build-arg` or to edit anything. Two things commonly still trip people up
even with that in place:
- **The proxy address itself.** If your proxy runs on the Docker host at `127.0.0.1:<port>`, that
  address is *not* reachable from inside the build container — `127.0.0.1` there means the container
  itself, not the host. Use the Docker bridge gateway address instead (typically `172.17.0.1`; confirm
  with `docker network inspect bridge | grep Gateway`), e.g. `export
  http_proxy=http://172.17.0.1:<port>`. Building with `--network=host` is the other common workaround.
- **`no_proxy` scope.** Without it, every build request goes through the proxy, not just the ones
  that need to — set something like `export
  no_proxy=pypi.org,files.pythonhosted.org,download.pytorch.org,huggingface.co,deb.debian.org` (add
  your proxy's own host if it resolves via a name) alongside `http_proxy`/`https_proxy`, before both
  `docker compose build` and `./scripts/download-models.sh`.

**GitHub isn't reachable at all, even via proxy.** `requirements-gigaam.txt` normally installs GigaAM
straight from its GitHub source. To avoid that entirely: clone or extract a copy of
[`salute-developers/GigaAM`](https://github.com/salute-developers/GigaAM) into `vendor/gigaam/`
before running `docker compose build` (that directory is gitignored on purpose — it's a local-only
build input, not something this repo ships). The Dockerfile detects a non-empty `vendor/gigaam/` (a
`setup.py` or `pyproject.toml` present) and installs from it instead of reaching GitHub; with nothing
there, it falls back to the pinned git commit exactly as before.

**Ollama's registry (`registry.ollama.ai`) is unreachable.** Same class of problem as GitHub above,
for a different host, if you've chosen Ollama as the protocol module's engine — it needs outbound
access to pull a model tag (`ollama pull ...`) unless you provide one already present on disk. Route
that pull through the same proxy setup described above, or transfer/import the model to the Ollama
host by another means; this is unrelated to whether GigaScribe's own Docker build can reach GitHub.

## Data storage

Every writable/user-created file lives outside the image, in the two directories `compose.yaml`
bind-mounts from the host: `./data` (→ `GIGASCRIBE_DATA_DIR`, uploaded originals, transcripts,
M4A/FLAC exports, generated protocols, `jobs.sqlite3`, logs) and `./models` (→
`GIGASCRIBE_MODELS_DIR`, GigaAM/pyannote/protocol model weights). Rebuilding, updating, or removing
the container never touches either directory — only `docker compose down -v` (which this project's
compose files don't define named volumes for, so `-v` has nothing to remove here) or manually
deleting the host paths would. To point either at a dedicated external location instead of a
relative `./data`/`./models` next to the repo (e.g. `/opt/gigascribe-data`), set
`GIGASCRIBE_DATA_DIR=/opt/gigascribe-data/app-data` and `GIGASCRIBE_MODELS_DIR=/opt/gigascribe-data/models`
in `.env` and update the two bind-mount paths on the left of the `:` in `compose.yaml`'s `volumes:`
to match; the container-side paths (right of the `:`) don't need to change.

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

#### GPU offload status and the CPU-wheel fallback

`INSTALL_PROTOCOL=1` builds `llama-cpp-python` with CUDA GPU offload (`GGML_CUDA=on`) via a
dedicated multi-stage build step (`protocol-builder-1` in the `Dockerfile`) that temporarily installs
the CUDA devel toolchain (`nvcc`) from NVIDIA's official Debian apt repository, compiles the wheel,
and discards the toolchain — the final image only keeps the compiled wheel and the CUDA runtime `.so`
files it needs. **This has not been verified against real hardware in the environment this was
written in** (no GPU or Docker daemon available there); if it doesn't work for your driver/CUDA
combination, or you'd rather not wait for a CUDA compile at all, the officially-supported fallback is
a plain CPU-built wheel:

```bash
docker compose -f compose.yaml -f compose.protocol.yaml build \
  --build-arg INSTALL_PROTOCOL=1 \
  --build-arg PROTOCOL_CMAKE_ARGS=
```

(An empty `PROTOCOL_CMAKE_ARGS` skips the `-DGGML_CUDA=on` flag inside `protocol-builder-1`, so it
falls through to a plain `pip wheel` there — still no CUDA toolchain needed on the build host, but the
resulting wheel runs LLM inference on CPU. Expect roughly 15-20 minutes for a single protocol job
on a 14B model instead of seconds on GPU — usable for evaluating the module, not for routine
production load.) Please report back whether the GPU path builds and actually offloads to GPU on your
hardware so this note can be updated with a confirmed status.

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
4. Each decision/task's timestamp — a single point or a range like `15:12 - 16:00` — is matched to
   a *genuine* transcript segment (one whose own span actually overlaps the cited time, not merely
   whichever segment happens to start closest) within a tolerance; a match gets a unique, stable
   HTML anchor (e.g. `timestamp-312-1`, never the raw timecode text itself) and an appendix row
   (timecode, speaker, source reply, block number, item type). A timestamp that can't be matched is
   shown as plain text, never as a broken link, and the item is noted in `unverified_items`.
5. The result is rendered from that one validated document into `protocol.json` (source of truth
   — includes the model/engine, prompt versions, chunking settings, and app git commit for
   reproducibility), `protocol.html` (`protocol/renderer.py`, escapes everything, table of
   contents, hides empty sections, print/download buttons for all three formats), and
   `protocol.xml` (same data, validates against `protocol/protocol.xsd`) under
   `data/protocols/<job_id>/`, alongside `protocol.log` (full internal detail/tracebacks — never
   shown to the user, who only sees a short status message including a `fact_checking` stage). Set
   `GIGASCRIBE_PROTOCOL_DEBUG_RAW_RESPONSE=1` to also log every raw LLM response to `protocol.log`
   for debugging — off by default, since a meeting's raw model output can be confidential.
6. On any failure, the protocol job is marked `failed` with a short message, the LLM is unloaded,
   and the underlying transcription job is completely untouched — it is not marked failed and its
   results are not deleted. The user can retry from the same button (replaying the original settings
   snapshot, not any settings changed since).
7. On server restart, any protocol job caught mid-run is marked `failed` with a clear "restored
   after restart" message rather than silently resumed, so it can never leave the GPU lock held.

### Automatic protocol creation

Instead of clicking "Создать протокол" after a transcription finishes, tick "Сразу создать
протокол после завершения" next to the upload button (shown whenever the module is enabled and at
least one model is installed) — the protocol job starts the instant the transcription completes,
with no second manual step. It reuses exactly the same `create_protocol()` call the button makes,
with the same error isolation: if it fails (no model installed, transcript missing, etc.) it's
logged and the transcription job is unaffected either way. The flag is stored per-job
(`jobs.auto_protocol`, set once at upload time, read-only afterwards) and is entirely independent
of the module being enabled at all — if `GIGASCRIBE_PROTOCOL_ENABLED=0`, the checkbox is hidden and
the flag is simply never acted on.

### External API for integrations (`/api/v1/meeting-protocol`)

For machine clients built separately (a Telegram bot, "Пачка" bridge, etc.) that need to submit
audio and poll for a protocol without a browser session:

- `POST /api/v1/meeting-protocol` — multipart file upload, `Authorization: Bearer <token>` header.
  Returns `{"job_id": ...}` immediately; the job always runs with `auto_protocol` on, using the
  admin panel's currently active model/engine/prompts/glossary.
- `GET /api/v1/meeting-protocol/{job_id}` — status and, once ready, the protocol result
  (`protocol_data`, the same document as `protocol.json`). Only returns jobs created by that same
  token; any other job (including another token's) 404s, never 403 — this mirrors the
  ownership-check semantics used elsewhere in the app.

**These two routes accept no configuration of any kind** — there is no field for model, engine,
prompt, or glossary. That's deliberate: the external API can submit audio and read results, and
nothing else. All configuration stays admin-panel-only.

Tokens are bearer strings (`gsp_<id>_<secret>`, shown once at creation and never recoverable — only
the bcrypt hash of the secret half is stored) that authenticate as a specific local username, via a
dependency (`current_user_via_token`) kept fully separate from the cookie-session auth the rest of
the app uses. **Token issuance and revocation are admin-only**, in the admin panel's "API-токены"
section (label + the username the token acts as) — there is deliberately no self-service endpoint,
since letting any authenticated user mint a token for another username would be a privilege hole.

The polling route (`GET .../{job_id}`) has no rate limiting yet — `requirements-base.txt` doesn't
currently pull in a rate-limiting library. Fine for a single internal bridge; worth adding
(e.g. `slowapi`) before exposing this to many external callers.

### Where things are stored

```text
data/
  jobs.sqlite3          # existing DB; protocol/*, glossary_*, and api_tokens tables added via ALTER/CREATE, no data loss
  protocols/<job_id>/
    protocol.json        # structured result (source of truth)
    protocol.html         # rendered page served at /api/jobs/{id}/protocol
    protocol.xml            # same document, validates against protocol/protocol.xsd
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

`compose.protocol.yaml` only adds `--build-arg INSTALL_PROTOCOL=1` and `GIGASCRIBE_PROTOCOL_ENABLED=1`.
That build arg selects a dedicated `protocol-builder-1` stage in the `Dockerfile` that compiles
`llama-cpp-python` with CUDA GPU offload (`GGML_CUDA=on`) for CUDA 12.8-capable cards such as the RTX
5060 Ti, temporarily installing just the CUDA devel toolchain (`nvcc`) it needs to do that and
discarding it afterwards — this needs no CUDA toolchain on the build host itself, only outbound
access to NVIDIA's apt repository during the build. No proxy settings are added by either compose
file (`compose.yaml`'s `build.args` already forwards your shell's own `http_proxy`/`https_proxy`/
`no_proxy` if you have them set — see [Restricted networks](#restricted-networks-proxy--no-github-access)).
See [GPU offload status and the CPU-wheel fallback](#gpu-offload-status-and-the-cpu-wheel-fallback)
for this build step's verification status and the CPU-only fallback if it doesn't work for you.

### Running the tests

```bash
python -m pytest tests/ -q
```

Protocol-specific tests are the `tests/test_protocol_*.py` files (chunking, providers/JSON
retry-on-invalid-JSON, glossary, SQLite migration, HTML rendering/escaping, the full pipeline
against a fake LLM provider, and `server.py` isolation/access-control checks run in subprocesses).
They require no GPU and no real model weights.
