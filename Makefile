SHELL := /bin/zsh
UV := $(shell command -v uv || echo $$HOME/.local/bin/uv)
PYTHON := engine/.venv/bin/python

.PHONY: setup setup-ml models install-bridge uninstall-bridge test test-py test-swift spike verify simulate demo build-app dev clean

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

build-app:
	cd macos-app && swift build -c release

dev:
	@echo "Starting standalone MIDI bridge and AetherMix engine (SHADOW_OBSERVE)..."
	$(PYTHON) bridge/midi_bridge_daemon.py & \
	BRIDGE_PID=$$!; \
	trap "kill $$BRIDGE_PID 2>/dev/null || true" EXIT INT TERM; \
	sleep 0.5; \
	$(PYTHON) engine/main.py --service-plan engine/schemas/service_plan.json --mixing-rules engine/schemas/mixing_rules.json

clean:
	rm -rf engine/.pytest_cache macos-app/.build
