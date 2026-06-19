# Meetily Speaker Recognition Sidecar

Local Python service for person-level speaker recognition. The Tauri app owns storage and UI; this service only runs ML inference.

## Run Locally

```bash
python3 -m venv backend/diarization_service/.venv
. backend/diarization_service/.venv/bin/activate
pip install -r requirements.txt
HF_TOKEN=your_huggingface_token uvicorn backend.diarization_service.main:app --host 127.0.0.1 --port 8179
```

`HF_TOKEN` must have access to the selected pyannote diarization model. For the
default pipeline, the Hugging Face account must accept both
`pyannote/segmentation-3.0` and `pyannote/speaker-diarization-3.1` conditions.
For desktop-launched Meetily, put local sidecar secrets in either:

- `backend/diarization_service/.env`
- `~/.config/meetily/speaker.env`

Example:

```bash
HF_TOKEN=your_huggingface_token
```

The sidecar auto-selects `cuda` when ROCm/CUDA PyTorch reports a usable GPU. Set
`SPEAKER_DEVICE=cpu` to force CPU inference.

## Endpoints

- `GET /health`
- `POST /diarize`
- `POST /embed-cluster`
- `POST /identify`

Default models:

- Diarization: `pyannote/speaker-diarization-3.1`
- Embeddings: `speechbrain/spkrec-ecapa-voxceleb`

This sidecar intentionally does not write Meetily database state. Rust/Tauri persists profiles, voiceprints, turns, transcript labels, and user confirmations.
