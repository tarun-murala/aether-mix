import AppKit
import AVFoundation
import Foundation
import Observation

public enum CheckStatus: Sendable, Equatable {
    case checking
    case ok
    case warn
    case fail

    public var isOK: Bool { self == .ok }
}

public enum CheckFix: Sendable, Equatable {
    case installHomebrew        // Official Homebrew installer in Terminal (requires interactive sudo)
    case installBlackHole       // brew install --cask blackhole-16ch + coreaudiod restart in Terminal (sudo)
    case runInstaller           // Full scripts/install_prereqs.sh in Terminal (handles sudo prompts cleanly)
    case syncPythonRuntime      // In-app streaming uv sync into ~/.aethermix/venv + stage silero_vad.onnx
    case installOllama          // In-app streaming brew install ollama
    case startOllama            // Launch Ollama daemon / Ollama.app
    case pullModel              // In-app streaming `ollama pull <model>` with live progress log
    case stageSileroVAD         // In-app copy bundled silero_vad.onnx (or download) into ~/.aethermix/models/
    case launchMainStage
    case requestMicrophone
    case openMicrophoneSettings
    case installBridgeAgent
    case runVerificationDoctor  // In-app streaming scripts/check_setup.py --test-ollama

    public var buttonTitle: String {
        switch self {
        case .installHomebrew: return "Install Homebrew"
        case .installBlackHole: return "Install BlackHole"
        case .runInstaller: return "Run Full Installer"
        case .syncPythonRuntime: return "Install Python Env"
        case .installOllama: return "Install Ollama"
        case .startOllama: return "Start Ollama"
        case .pullModel: return "Pull Model"
        case .stageSileroVAD: return "Install VAD"
        case .launchMainStage: return "Launch MainStage"
        case .requestMicrophone: return "Request Access"
        case .openMicrophoneSettings: return "Open Settings"
        case .installBridgeAgent: return "Install Bridge"
        case .runVerificationDoctor: return "Run Doctor"
        }
    }

    public var requiresTerminal: Bool {
        switch self {
        case .installHomebrew, .installBlackHole, .runInstaller:
            return true
        default:
            return false
        }
    }
}

public struct DependencyCheck: Identifiable, Sendable, Equatable {
    public let id: String
    public let title: String
    public var status: CheckStatus
    public var detail: String
    public var fix: CheckFix?
    /// Critical checks gate the "ready" state; non-critical are advisory.
    public var critical: Bool

    public init(id: String, title: String, status: CheckStatus = .checking, detail: String = "Checking…",
                fix: CheckFix? = nil, critical: Bool = true) {
        self.id = id
        self.title = title
        self.status = status
        self.detail = detail
        self.fix = fix
        self.critical = critical
    }
}

/// Probes every external dependency AetherMix needs on any Apple Silicon Mac
/// (Homebrew, MainStage, BlackHole 16ch, Microphone TCC, Ollama + Qwen, uv, Python runtime
/// with rtmidi/sounddevice/onnxruntime, Silero VAD ONNX, and 5-tune Sound Pack) and offers
/// one-click fixes: in-app streaming execution for non-sudo installs and Terminal scripts for
/// sudo-required installers (Homebrew / BlackHole 16ch).
@MainActor
@Observable
public final class DependencyChecker {
    public static let shared = DependencyChecker()

    nonisolated public static let availableQwenModels: [String] = [
        "qwen3:14b",
        "qwen3:8b",
        "qwen3:32b",
    ]

    public private(set) var checks: [DependencyCheck] = DependencyChecker.placeholders
    public private(set) var isRefreshing = false
    public private(set) var lastRefresh: Date?
    public private(set) var fixMessage: String?
    public private(set) var model: String = "qwen3:14b"

    // Live in-app installer / doctor console state
    public private(set) var isRunningTask = false
    public private(set) var activeTaskTitle: String?
    public private(set) var taskLogLines: [String] = []
    public private(set) var lastTaskSucceeded: Bool?

    private var activeProcess: Process?
    private let maxTaskLogLines = 250

    public var problems: [DependencyCheck] { checks.filter { $0.status == .fail || $0.status == .warn } }
    public var criticalFailures: [DependencyCheck] { checks.filter { $0.critical && $0.status == .fail } }
    public var allCriticalOK: Bool { lastRefresh != nil && criticalFailures.isEmpty }

    public var hasSudoDependencyMissing: Bool {
        checks.contains { ($0.id == "homebrew" || $0.id == "blackhole") && $0.status == .fail }
    }

    private static let placeholders: [DependencyCheck] = [
        DependencyCheck(id: "homebrew", title: "Homebrew package manager"),
        DependencyCheck(id: "mainstage", title: "Apple MainStage", critical: false),
        DependencyCheck(id: "blackhole", title: "BlackHole 16ch audio driver"),
        DependencyCheck(id: "mic", title: "Microphone permission"),
        DependencyCheck(id: "ollama_installed", title: "Ollama installed"),
        DependencyCheck(id: "ollama_running", title: "Ollama server running"),
        DependencyCheck(id: "qwen", title: "Qwen model pulled"),
        DependencyCheck(id: "uv", title: "uv (Python tool manager)"),
        DependencyCheck(id: "python", title: "Python 3.11 runtime (rtmidi, sounddevice, onnxruntime)"),
        DependencyCheck(id: "vad", title: "Silero VAD ONNX model", critical: false),
        DependencyCheck(id: "soundpack", title: "Packaged 5-Tune Sound Pack", critical: false),
    ]

    public init() {}

    // MARK: - Model Selection

    /// Persist a chosen Qwen model (e.g. `qwen3:8b` on 16 GB Macs or `qwen3:14b` on 24 GB+ Macs)
    /// into `~/.aethermix/config/service_plan.json` (or repo plan) and re-check Ollama.
    public func selectModel(_ newModel: String, repoRoot: URL) {
        let trimmed = newModel.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        model = trimmed
        let cfgURL = AetherMixProtocol.defaultAetherMixDirectory.appendingPathComponent("config/service_plan.json")
        let fm = FileManager.default
        if !fm.fileExists(atPath: cfgURL.path) {
            let bundled = repoRoot.appendingPathComponent("engine/schemas/service_plan.json")
            try? fm.createDirectory(at: cfgURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            if fm.fileExists(atPath: bundled.path) {
                try? fm.copyItem(at: bundled, to: cfgURL)
            }
        }
        if let data = try? Data(contentsOf: cfgURL),
           var obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            obj["ollama_model"] = trimmed
            if let updated = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]) {
                try? updated.write(to: cfgURL, options: .atomic)
            }
        }
        fixMessage = "Configured Ollama model -> \(trimmed)"
        Task { await refresh(repoRoot: repoRoot) }
    }

    // MARK: - Refresh

    public func refresh(repoRoot: URL) async {
        guard !isRefreshing else { return }
        isRefreshing = true
        defer { isRefreshing = false; lastRefresh = Date() }

        let configuredModel = Self.configuredModel(repoRoot: repoRoot)
        model = configuredModel
        let mic = AVCaptureDevice.authorizationStatus(for: .audio)
        let mainStageRunning = NSWorkspace.shared.runningApplications
            .contains { $0.bundleIdentifier == "com.apple.mainstage3" }

        let results = await Task.detached(priority: .userInitiated) {
            Self.probe(repoRoot: repoRoot, model: configuredModel, mic: mic, mainStageRunning: mainStageRunning)
        }.value
        checks = results
    }

    // MARK: - Probing (off main actor)

    nonisolated private static func probe(
        repoRoot: URL, model: String, mic: AVAuthorizationStatus, mainStageRunning: Bool
    ) -> [DependencyCheck] {
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser.path
        var out: [DependencyCheck] = []

        // Homebrew
        let brewBin = ["/opt/homebrew/bin/brew", "/usr/local/bin/brew"].first { fm.isExecutableFile(atPath: $0) }
        out.append(brewBin != nil
            ? .init(id: "homebrew", title: "Homebrew package manager", status: .ok, detail: brewBin ?? "")
            : .init(id: "homebrew", title: "Homebrew package manager", status: .fail,
                    detail: "Homebrew is required to install BlackHole 16ch, PortAudio, uv, and Ollama.",
                    fix: .installHomebrew))

        // MainStage
        let msPaths = ["/Applications/MainStage.app", "\(home)/Applications/MainStage.app"]
        let msInstalled = msPaths.contains { fm.fileExists(atPath: $0) }
        if !msInstalled {
            out.append(.init(id: "mainstage", title: "Apple MainStage", status: .warn,
                             detail: "MainStage.app not found in /Applications. Install from the Mac App Store (or use another CoreMIDI DAW).",
                             critical: false))
        } else if mainStageRunning {
            out.append(.init(id: "mainstage", title: "Apple MainStage", status: .ok, detail: "Installed and running.", critical: false))
        } else {
            out.append(.init(id: "mainstage", title: "Apple MainStage", status: .warn,
                             detail: "Installed but not running. Open your concert so AetherMix can drive it.",
                             fix: .launchMainStage, critical: false))
        }

        // BlackHole 16ch
        let bh = fm.fileExists(atPath: "/Library/Audio/Plug-Ins/HAL/BlackHole16ch.driver")
        out.append(bh
            ? .init(id: "blackhole", title: "BlackHole 16ch audio driver", status: .ok,
                    detail: "BlackHole16ch.driver installed.")
            : .init(id: "blackhole", title: "BlackHole 16ch audio driver", status: .fail,
                    detail: "Driver missing. Click Install BlackHole (runs brew cask install + reloads CoreAudio).",
                    fix: brewBin != nil ? .installBlackHole : .installHomebrew))

        // Microphone TCC
        switch mic {
        case .authorized:
            out.append(.init(id: "mic", title: "Microphone permission", status: .ok, detail: "Granted to AetherMix."))
        case .notDetermined:
            out.append(.init(id: "mic", title: "Microphone permission", status: .warn,
                             detail: "Not yet requested. Needed to read BlackHole input.", fix: .requestMicrophone))
        default:
            out.append(.init(id: "mic", title: "Microphone permission", status: .fail,
                             detail: "Denied. Enable AetherMix in System Settings → Privacy & Security → Microphone.",
                             fix: .openMicrophoneSettings))
        }

        // Ollama binary
        let ollamaBin = ["/opt/homebrew/bin/ollama", "/usr/local/bin/ollama", "/Applications/Ollama.app/Contents/Resources/ollama"]
            .first { fm.isExecutableFile(atPath: $0) }
        out.append(ollamaBin != nil
            ? .init(id: "ollama_installed", title: "Ollama installed", status: .ok, detail: ollamaBin ?? "")
            : .init(id: "ollama_installed", title: "Ollama installed", status: .fail,
                    detail: "Ollama not found. Click Install Ollama (`brew install ollama`).",
                    fix: brewBin != nil ? .installOllama : .installHomebrew))

        // Ollama server + Qwen model
        let tags = fetchOllamaTags()
        if let tags {
            out.append(.init(id: "ollama_running", title: "Ollama server running", status: .ok,
                             detail: "Responding on 127.0.0.1:11434 (\(tags.count) models)."))
            let base = model.split(separator: ":").first.map(String.init) ?? model
            let exact = tags.contains { $0 == model || $0 == "\(model):latest" }
            let sameFamily = tags.filter { $0.hasPrefix(base + ":") }
            if exact {
                out.append(.init(id: "qwen", title: "Qwen model pulled", status: .ok, detail: "\(model) is available."))
            } else if !sameFamily.isEmpty {
                out.append(.init(id: "qwen", title: "Qwen model pulled", status: .warn,
                                 detail: "Configured model \(model) not pulled; found \(sameFamily.joined(separator: ", ")). Pull \(model) or switch model above.",
                                 fix: .pullModel))
            } else {
                out.append(.init(id: "qwen", title: "Qwen model pulled", status: .fail,
                                 detail: "\(model) not pulled. Required for the supervisory LLM (otherwise Tier-1 deterministic mix only).",
                                 fix: .pullModel))
            }
        } else {
            out.append(.init(id: "ollama_running", title: "Ollama server running", status: .fail,
                             detail: "Nothing listening on 127.0.0.1:11434.",
                             fix: ollamaBin != nil || fm.fileExists(atPath: "/Applications/Ollama.app") ? .startOllama : .installOllama))
            out.append(.init(id: "qwen", title: "Qwen model pulled", status: .fail,
                             detail: "Cannot verify \(model) until Ollama is running.",
                             fix: ollamaBin != nil ? .startOllama : .installOllama))
        }

        // uv
        let uv = ["\(home)/.local/bin/uv", "/opt/homebrew/bin/uv", "/usr/local/bin/uv"].first { fm.isExecutableFile(atPath: $0) }
        out.append(uv != nil
            ? .init(id: "uv", title: "uv (Python tool manager)", status: .ok, detail: uv ?? "")
            : .init(id: "uv", title: "uv (Python tool manager)", status: .fail,
                    detail: "uv missing. Click Install Python Env or Run Full Installer.",
                    fix: brewBin != nil ? .syncPythonRuntime : .installHomebrew))

        // Python runtime with full audio/MIDI/ONNX extras
        if let py = EngineProcessManager.resolveExistingPython(runtimeRoot: repoRoot) {
            let (code, text) = runCapture(
                py,
                ["-c", "import rtmidi, sounddevice, soundfile, onnxruntime, numpy, scipy, fastapi, uvicorn; print('ok')"]
            )
            if code == 0 {
                out.append(.init(id: "python", title: "Python 3.11 runtime (rtmidi, sounddevice, onnxruntime)", status: .ok, detail: py))
            } else {
                let last = text.split(separator: "\n").last.map(String.init) ?? "import failed"
                out.append(.init(id: "python", title: "Python 3.11 runtime (rtmidi, sounddevice, onnxruntime)", status: .fail,
                                 detail: "\(py) is missing packages: \(last)", fix: .syncPythonRuntime))
            }
        } else {
            out.append(.init(id: "python", title: "Python 3.11 runtime (rtmidi, sounddevice, onnxruntime)", status: .fail,
                             detail: "~/.aethermix/venv not provisioned — click Install Python Env to set it up automatically.",
                             fix: .syncPythonRuntime))
        }

        // Silero VAD ONNX
        let vad = "\(home)/.aethermix/models/silero_vad.onnx"
        let vadSize = (try? fm.attributesOfItem(atPath: vad)[.size] as? Int) ?? 0
        out.append(vadSize > 100_000
            ? .init(id: "vad", title: "Silero VAD ONNX model", status: .ok, detail: vad, critical: false)
            : .init(id: "vad", title: "Silero VAD ONNX model", status: .warn,
                    detail: "Missing at ~/.aethermix/models/silero_vad.onnx — click Install VAD to stage from app bundle.",
                    fix: .stageSileroVAD, critical: false))

        // Packaged 5-Tune Sound Pack
        let packManifest = repoRoot.appendingPathComponent("assets/soundpack/manifest.json")
        out.append(fm.fileExists(atPath: packManifest.path)
            ? .init(id: "soundpack", title: "Packaged 5-Tune Sound Pack", status: .ok,
                    detail: "5 multi-instrument tunes + speech + applause bundled.", critical: false)
            : .init(id: "soundpack", title: "Packaged 5-Tune Sound Pack", status: .warn,
                    detail: "Not found under \(repoRoot.path)/assets/soundpack.", critical: false))

        return out
    }

    nonisolated private static func fetchOllamaTags() -> [String]? {
        guard let url = URL(string: "http://127.0.0.1:11434/api/tags") else { return nil }
        var req = URLRequest(url: url)
        req.timeoutInterval = 1.5
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: [String]?
        URLSession.shared.dataTask(with: req) { data, resp, _ in
            defer { sem.signal() }
            guard let data, (resp as? HTTPURLResponse)?.statusCode == 200,
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let models = obj["models"] as? [[String: Any]] else { return }
            result = models.compactMap { $0["name"] as? String }
        }.resume()
        _ = sem.wait(timeout: .now() + 2.5)
        return result
    }

    nonisolated private static func runCapture(_ exe: String, _ args: [String]) -> (Int32, String) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: exe)
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        do {
            try p.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            return (p.terminationStatus, String(data: data, encoding: .utf8) ?? "")
        } catch {
            return (-1, error.localizedDescription)
        }
    }

    nonisolated static func configuredModel(repoRoot: URL) -> String {
        let candidates = [
            AetherMixProtocol.defaultAetherMixDirectory.appendingPathComponent("config/service_plan.json"),
            repoRoot.appendingPathComponent("engine/schemas/service_plan.json"),
        ]
        for url in candidates {
            if let data = try? Data(contentsOf: url),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let m = obj["ollama_model"] as? String, !m.isEmpty {
                return m
            }
        }
        return "qwen3:14b"
    }

    // MARK: - One-Click Fixes (In-App Streaming & Terminal Sudo Helpers)

    public func cancelRunningTask() {
        if let proc = activeProcess, proc.isRunning {
            proc.terminate()
            appendTaskLog("Cancelled by operator.")
        }
        activeProcess = nil
        isRunningTask = false
        activeTaskTitle = nil
    }

    public func clearTaskLog() {
        taskLogLines.removeAll()
        lastTaskSucceeded = nil
    }

    public func apply(
        _ fix: CheckFix,
        repoRoot: URL,
        processManager: EngineProcessManager,
        onFinished: (@MainActor () -> Void)? = nil
    ) {
        switch fix {
        case .installHomebrew:
            runInTerminal(
                script: "/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"",
                title: "Install Homebrew"
            )
        case .installBlackHole:
            runInTerminal(
                script: """
                export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
                brew install --cask blackhole-16ch
                echo "==> Reloading CoreAudio (sudo password may be required)..."
                sudo killall coreaudiod || true
                """,
                title: "Install BlackHole 16ch"
            )
        case .runInstaller:
            runInTerminal(
                script: "\"\(repoRoot.path)/scripts/install_prereqs.sh\" --model \"\(model)\"",
                title: "AetherMix Full Installer"
            )
        case .installOllama:
            runInAppShell(
                title: "Installing Ollama via Homebrew",
                script: """
                set -euo pipefail
                export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
                brew install ollama
                brew services start ollama || true
                """,
                repoRoot: repoRoot,
                onFinished: onFinished
            )
        case .syncPythonRuntime:
            runInAppShell(
                title: "Provisioning Python 3.11 Runtime & Dependencies",
                script: """
                set -euo pipefail
                export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
                if ! command -v uv >/dev/null 2>&1; then
                  echo "==> Installing uv..."
                  brew install uv
                fi
                if ! brew list --formula portaudio >/dev/null 2>&1; then
                  echo "==> Installing portaudio..."
                  brew install portaudio
                fi
                mkdir -p "$HOME/.aethermix/models" "$HOME/.aethermix/config" "$HOME/.aethermix/storage/rule_backups"
                chmod 700 "$HOME/.aethermix"
                echo "==> Syncing Python 3.11 venv into ~/.aethermix/venv..."
                UV_PROJECT_ENVIRONMENT="$HOME/.aethermix/venv" uv sync --project "\(repoRoot.path)/engine" --extra audio --python 3.11
                if [ -f "\(repoRoot.path)/models/silero_vad.onnx" ] && [ ! -f "$HOME/.aethermix/models/silero_vad.onnx" ]; then
                  cp "\(repoRoot.path)/models/silero_vad.onnx" "$HOME/.aethermix/models/silero_vad.onnx"
                  echo "==> Staged bundled silero_vad.onnx"
                fi
                PYTHONPATH="\(repoRoot.path)" "$HOME/.aethermix/venv/bin/python" -c "from engine.core.protocol import ensure_session_token; ensure_session_token()"
                echo "==> Python runtime ready!"
                """,
                repoRoot: repoRoot,
                onFinished: {
                    processManager.installBridgeLaunchAgent()
                    processManager.ensureEngineRunning()
                    onFinished?()
                }
            )
        case .stageSileroVAD:
            runInAppShell(
                title: "Staging Silero VAD ONNX Model",
                script: """
                set -euo pipefail
                mkdir -p "$HOME/.aethermix/models"
                DEST="$HOME/.aethermix/models/silero_vad.onnx"
                if [ -f "\(repoRoot.path)/models/silero_vad.onnx" ]; then
                  cp "\(repoRoot.path)/models/silero_vad.onnx" "$DEST"
                  echo "Copied bundled silero_vad.onnx -> $DEST"
                else
                  curl -fsSL "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx" -o "${DEST}.tmp"
                  mv "${DEST}.tmp" "$DEST"
                  echo "Downloaded silero_vad.onnx -> $DEST"
                fi
                """,
                repoRoot: repoRoot,
                onFinished: onFinished
            )
        case .pullModel:
            runInAppShell(
                title: "Pulling Ollama Model (\(model))",
                script: """
                set -euo pipefail
                export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
                if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
                  brew services start ollama >/dev/null 2>&1 || true
                  sleep 1
                fi
                ollama pull "\(model)"
                """,
                repoRoot: repoRoot,
                onFinished: onFinished
            )
        case .runVerificationDoctor:
            runInAppShell(
                title: "Running Pre-Flight Verification Doctor",
                script: """
                export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
                python3 "\(repoRoot.path)/scripts/check_setup.py" --model "\(model)" --test-ollama
                """,
                repoRoot: repoRoot,
                onFinished: onFinished
            )
        case .startOllama:
            if FileManager.default.fileExists(atPath: "/Applications/Ollama.app") {
                NSWorkspace.shared.open(URL(fileURLWithPath: "/Applications/Ollama.app"))
            } else {
                let p = Process()
                p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
                var env = ProcessInfo.processInfo.environment
                env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + (env["PATH"] ?? "/usr/bin:/bin")
                p.environment = env
                p.arguments = ["ollama", "serve"]
                p.standardOutput = FileHandle.nullDevice
                p.standardError = FileHandle.nullDevice
                try? p.run()
            }
            fixMessage = "Starting Ollama… re-checking in 3 s."
            scheduleRecheck(repoRoot: repoRoot)
        case .launchMainStage:
            NSWorkspace.shared.open(URL(fileURLWithPath: "/Applications/MainStage.app"))
            fixMessage = "Launching MainStage…"
            scheduleRecheck(repoRoot: repoRoot)
        case .requestMicrophone:
            AVCaptureDevice.requestAccess(for: .audio) { _ in
                Task { @MainActor in await self.refresh(repoRoot: repoRoot) }
            }
        case .openMicrophoneSettings:
            if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                NSWorkspace.shared.open(url)
            }
        case .installBridgeAgent:
            processManager.installBridgeLaunchAgent()
            fixMessage = processManager.lastStatusMessage
            scheduleRecheck(repoRoot: repoRoot)
            onFinished?()
        }
    }

    /// Runs a non-interactive setup or verification command directly inside the AetherMix UI
    /// with live streaming stdout/stderr output and automatic dependency re-check on completion.
    private func runInAppShell(
        title: String,
        script: String,
        repoRoot: URL,
        onFinished: (@MainActor () -> Void)? = nil
    ) {
        guard !isRunningTask else {
            fixMessage = "Another setup task (\(activeTaskTitle ?? "task")) is already running."
            return
        }
        isRunningTask = true
        activeTaskTitle = title
        lastTaskSucceeded = nil
        taskLogLines = ["==> \(title)"]
        fixMessage = "\(title)…"

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/bash")
        proc.arguments = ["-lc", script]
        var env = ProcessInfo.processInfo.environment
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:\(home)/.local/bin:" + (env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin")
        env["TERM"] = "dumb"
        env["NO_COLOR"] = "1"
        proc.environment = env

        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            if let text = String(data: data, encoding: .utf8) {
                let rawLines = text
                    .replacingOccurrences(of: "\r", with: "\n")
                    .split(whereSeparator: \.isNewline)
                    .map(String.init)
                Task { @MainActor [weak self] in
                    for line in rawLines {
                        self?.appendTaskLog(line)
                    }
                }
            }
        }

        proc.terminationHandler = { [weak self] finished in
            let status = finished.terminationStatus
            Task { @MainActor [weak self] in
                guard let self else { return }
                pipe.fileHandleForReading.readabilityHandler = nil
                self.activeProcess = nil
                self.isRunningTask = false
                self.activeTaskTitle = nil
                let ok = (status == 0)
                self.lastTaskSucceeded = ok
                self.appendTaskLog(ok ? "==> Completed successfully." : "==> Exited with status \(status).")
                self.fixMessage = ok ? "\(title): Done!" : "\(title) failed (exit \(status)). See live log below."
                await self.refresh(repoRoot: repoRoot)
                if ok {
                    onFinished?()
                }
            }
        }

        do {
            try proc.run()
            activeProcess = proc
        } catch {
            isRunningTask = false
            activeTaskTitle = nil
            lastTaskSucceeded = false
            appendTaskLog("ERROR: \(error.localizedDescription)")
            fixMessage = "Failed to start \(title): \(error.localizedDescription)"
        }
    }

    private func appendTaskLog(_ line: String) {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        taskLogLines.append(trimmed)
        if taskLogLines.count > maxTaskLogLines {
            taskLogLines.removeFirst(taskLogLines.count - maxTaskLogLines)
        }
    }

    private func scheduleRecheck(repoRoot: URL) {
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(3))
            await refresh(repoRoot: repoRoot)
        }
    }

    private func runInTerminal(script: String, title: String) {
        let dir = AetherMixProtocol.defaultAetherMixDirectory
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let safeName = title.replacingOccurrences(of: " ", with: "-")
        let file = dir.appendingPathComponent("run-\(safeName).command")
        let body = """
        #!/bin/bash
        echo "=== \(title) ==="
        \(script)
        status=$?
        echo
        if [ $status -eq 0 ]; then echo "Done. Return to AetherMix and press Re-check."; else echo "Failed (exit $status)."; fi
        read -r -p "Press Enter to close…"
        """
        do {
            try body.write(to: file, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: file.path)
            NSWorkspace.shared.open(file)
            fixMessage = "\(title) opened in Terminal. Press Re-check when it finishes."
        } catch {
            fixMessage = "Could not write helper script: \(error.localizedDescription)"
        }
    }
}
