.PHONY: help dev dev-cpu build build-cpu build-hipblas build-install install install-deb run-local run-installed check check-hipblas test-vad-live test-speakers test-speaker-integration audit-transcript-coverage audit-latest-transcript-coverage sidecar-health clean

FRONTEND_DIR := frontend
TAURI_MANIFEST := frontend/src-tauri/Cargo.toml
HIPCC_WRAPPER := /tmp/meetily-hipcc-wrapper
AMDGPU_TARGETS ?= gfx1100
SPEAKER_PYTHON ?= backend/diarization_service/.venv/bin/python
TAURI_BUILD_CONFIG := {"bundle":{"targets":["deb"],"createUpdaterArtifacts":false}}

help:
	@printf '%s\n' \
		'Targets:' \
		'  make dev             Run Tauri dev app with AMD ROCm/HIP' \
		'  make dev-cpu         Run Tauri dev app without GPU features' \
		'  make build           Build AMD ROCm/HIP .deb package' \
		'  make build-hipblas   Build AMD ROCm/HIP package for 7900 GRE by default' \
		'  make build-cpu       Build CPU-only .deb package' \
		'  make build-install   Build HIP .deb and install that exact package' \
		'  make install         Alias for install-deb' \
		'  make install-deb     Install latest generated .deb with dpkg' \
		'  make run-local       Run target/release/meetily directly' \
		'  make run-installed   Run /usr/bin/meetily from installed .deb' \
		'  make check           Run Rust cargo check' \
		'  make check-hipblas   Run Rust cargo check with HIP features' \
		'  make test-vad-live   Run live VAD batching regression test' \
		'  make test-speakers   Run speaker-related Rust tests' \
		'  make test-speaker-integration  Download fixture and run pyannote voiceprint reconciliation test' \
		'  make audit-transcript-coverage MEETING_DIR=/path/to/meeting  Compare audio speech timing vs saved transcripts' \
		'  make audit-latest-transcript-coverage  Audit newest completed recording' \
		'  make sidecar-health  Check local speaker sidecar health endpoint' \
		'  make clean           Clean frontend and Rust build outputs'

dev: dev-hipblas

dev-hipblas:
	cd $(FRONTEND_DIR) && AMDGPU_TARGETS=$(AMDGPU_TARGETS) PATH="$(HIPCC_WRAPPER):$$PATH" pnpm tauri dev --features hipblas

dev-cpu:
	cd $(FRONTEND_DIR) && pnpm tauri dev

build: build-hipblas

build-cpu:
	cd $(FRONTEND_DIR) && pnpm tauri build --config '$(TAURI_BUILD_CONFIG)'

build-hipblas:
	cd $(FRONTEND_DIR) && AMDGPU_TARGETS=$(AMDGPU_TARGETS) PATH="$(HIPCC_WRAPPER):$$PATH" pnpm tauri build --features hipblas --config '$(TAURI_BUILD_CONFIG)'

build-install: build-hipblas install-deb

install: install-deb

install-deb:
	@deb=$$(ls -t target/release/bundle/deb/meetily_*_amd64.deb 2>/dev/null | head -n 1); \
	test -n "$$deb" || (echo "No .deb found under target/release/bundle/deb. Run make build first."; exit 1); \
	echo "Installing $$deb"; \
	sudo -n dpkg -i "$$deb"

run-local:
	@test -x target/release/meetily || (echo "target/release/meetily not found. Run make build first."; exit 1)
	target/release/meetily

run-installed:
	@test -x /usr/bin/meetily || (echo "/usr/bin/meetily not found. Run make install first."; exit 1)
	/usr/bin/meetily

check:
	cargo check --manifest-path $(TAURI_MANIFEST)

check-hipblas:
	AMDGPU_TARGETS=$(AMDGPU_TARGETS) PATH="$(HIPCC_WRAPPER):$$PATH" cargo check --manifest-path $(TAURI_MANIFEST) --features hipblas

test-vad-live:
	cargo test --manifest-path $(TAURI_MANIFEST) test_live_segment_cap_emits_during_continuous_speech --lib

test-speakers:
	cargo test --manifest-path $(TAURI_MANIFEST) speaker --lib

test-speaker-integration:
	MEETILY_RUN_SPEAKER_INTEGRATION=1 $(SPEAKER_PYTHON) -m unittest backend.diarization_service.tests.test_speaker_recognition_integration

audit-transcript-coverage:
	@test -n "$(MEETING_DIR)" || (echo "Set MEETING_DIR=/path/to/meeting"; exit 1)
	python3 scripts/audit-transcript-coverage.py "$(MEETING_DIR)"

audit-latest-transcript-coverage:
	python3 scripts/audit-transcript-coverage.py --latest-completed

sidecar-health:
	curl -fsS http://127.0.0.1:8179/health

clean:
	cd $(FRONTEND_DIR) && pnpm next telemetry disable >/dev/null 2>&1 || true
	rm -rf $(FRONTEND_DIR)/.next
	cargo clean
