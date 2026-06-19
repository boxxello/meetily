# AGENTS.md

This file gives Codex and other coding agents repo-local instructions. Treat `CLAUDE.md` as the broader architecture guide and keep this file focused on operational rules.

## Fork Workflow

For the `boxxello/meetily` fork, keep `staging/boxxello-fork` as the local integration branch. Create feature branches from that staging branch, keep feature work independently reviewable, and merge or cherry-pick completed work back into staging before release builds or PRs.

Do not stack unrelated product work directly on temporary fix branches once staging exists.

## Speaker Recognition Direction

Speaker labeling for this fork is person-level recognition:

- Use diarization plus speaker profiles and learned voiceprints.
- Do not use microphone/system/stereo/channel topology as the identity model.
- `backend/diarization_service` is the active local sidecar for diarization, embeddings, and voiceprint matching.
- When validating speaker relabeling, inspect speaker metadata columns and counts only unless the user explicitly asks to read transcript text.

## Local Build Notes

- Use `make build-hipblas` for the AMD GPU `.deb` build on this workstation.
- Use `make install-deb` after building when the Ubuntu launcher should run the new local package.
- The sidecar is local and should be restarted after Python dependency or compatibility changes.
