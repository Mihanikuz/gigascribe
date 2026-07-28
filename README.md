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

## Run

```bash
docker compose up -d --build
curl -f http://127.0.0.1:8000/health/ready
```

CPU, CTC models, pyannote 3.1, compatibility profiles, and compose override profiles have been removed.

## Users and administration

- Set `GIGASCRIBE_ADMIN_PASSWORD` in `.env` before the first start; the `admin` account is created from it on first run.
- Log in as `admin` and open `/admin` to create or disable local accounts and manage installed models.
- The transcription history is shared: every logged-in user sees every job. Only the job's owner or an administrator can cancel or retry it.
