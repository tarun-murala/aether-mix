SHELL := /bin/zsh
UV := $(shell command -v uv || echo $$HOME/.local/bin/uv)
PYTHON := engine/.venv/bin/python

.PHONY: setup setup-ml models install-bridge uninstall-bridge install-prereqs check-setup test test-py test-swift spike verify simulate demo soundpack live-mainstage audition-soundpack build-app dmg dev clean eval-list eval-status eval-fetch eval-audition eval-smoke eval-quick eval-provisional eval-full eval-routing eval-closedloop-plan eval-test

setup:
	cd engine && $(UV) sync --extra dev --extra audio

setup-ml:
	cd engine && $(UV) sync --extra dev --extra audio --extra ml

models:
	./scripts/bootstrap_mac.sh

install-bridge:
	mkdir -p $$HOME/Library/LaunchAgents $$HOME/.aethermix
	sed "s|__REPO_ROOT__|$(CURDIR)|g; s|__PYTHON__|$(CURDIR)/engine/.venv/bin/python|g; s|__HOME__|$$HOME|g" \
		scripts/com.aethermix.midibridge.plist > $$HOME/Library/LaunchAgents/com.aethermix.midibridge.plist
	launchctl bootout gui/$$(id -u)/com.aethermix.midibridge 2>/dev/null || true
	launchctl bootstrap gui/$$(id -u) $$HOME/Library/LaunchAgents/com.aethermix.midibridge.plist

uninstall-bridge:
	launchctl bootout gui/$$(id -u)/com.aethermix.midibridge 2>/dev/null || true
	rm -f $$HOME/Library/LaunchAgents/com.aethermix.midibridge.plist

test-py:
	cd engine && $(UV) run --no-sync pytest -v

test-swift:
	cd macos-app && swift test

test: test-py test-swift

spike:
	$(PYTHON) scripts/phase0_spike.py

verify:
	$(PYTHON) scripts/verify_audio_midi.py

simulate:
	$(PYTHON) scripts/simulate_service.py --headless

simulate-live:
	$(PYTHON) scripts/simulate_service.py --blackhole

demo:
	$(PYTHON) scripts/run_demo.py

soundpack:
	$(PYTHON) scripts/build_soundpack.py

live-mainstage:
	$(PYTHON) scripts/play_mainstage_pack.py

audition-soundpack:
	$(PYTHON) scripts/play_mainstage_pack.py --audition --mode jukebox

install-prereqs:
	./scripts/install_prereqs.sh

check-setup:
	python3 scripts/check_setup.py --test-ollama

build-app:
	cd macos-app && swift build -c release

dmg:
	./scripts/build_dmg.sh

dev:
	@echo "Starting standalone MIDI bridge and AetherMix engine (SHADOW_OBSERVE)..."
	$(PYTHON) bridge/midi_bridge_daemon.py & \
	BRIDGE_PID=$$!; \
	trap "kill $$BRIDGE_PID 2>/dev/null || true" EXIT INT TERM; \
	sleep 0.5; \
	$(PYTHON) engine/main.py --service-plan engine/schemas/service_plan.json --mixing-rules engine/schemas/mixing_rules.json

clean:
	rm -rf engine/.pytest_cache macos-app/.build
# --- Multitrack MainStage eval suite (docs/EVAL_SPEC.md) ---
EVAL := $(PYTHON) scripts/run_eval.py

eval-list:
	$(EVAL) list

eval-status:
	$(EVAL) status

# Reads license terms first: make eval-fetch ACCEPT_TERMS=1 [SERVICE=svc1]
eval-fetch:
	$(EVAL) fetch $(if $(ACCEPT_TERMS),--accept-terms,) $(if $(SERVICE),--service $(SERVICE),)

eval-audition:
	$(EVAL) audition

eval-smoke:
	$(EVAL) run all --profile smoke --allow-synthetic --no-fail

eval-quick:
	$(EVAL) run all --profile quick

eval-provisional:
	$(EVAL) run all --profile quick --allow-synthetic --no-fail

eval-full:
	$(EVAL) run all --profile full --write-audio

eval-routing:
	$(EVAL) routing-check

eval-closedloop-plan:
	$(EVAL) closedloop-plan

eval-test:
	cd engine && $(UV) run --no-sync pytest -v tests/test_evals.py
