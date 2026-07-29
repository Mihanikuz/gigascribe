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
