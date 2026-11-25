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
    case runInstaller          // install_prereqs.sh in Terminal (brew deps, venv, Ollama+model, LaunchAgent)
    case startOllama
    case pullModel
    case launchMainStage
    case requestMicrophone
    case openMicrophoneSettings
    case installBridgeAgent

    public var buttonTitle: String {
        switch self {
        case .runInstaller: return "Run Installer"
        case .startOllama: return "Start Ollama"
        case .pullModel: return "Pull Model"
        case .launchMainStage: return "Launch MainStage"
        case .requestMicrophone: return "Request Access"
        case .openMicrophoneSettings: return "Open Settings"
        case .installBridgeAgent: return "Install Bridge"
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

/// Probes every external dependency AetherMix needs (MainStage, BlackHole, mic permission, Ollama + Qwen,
/// uv, Python runtime with rtmidi, Silero VAD) and offers one-click fixes.
@MainActor
@Observable
public final class DependencyChecker {
    public static let shared = DependencyChecker()

    public private(set) var checks: [DependencyCheck] = DependencyChecker.placeholders
    public private(set) var isRefreshing = false
    public private(set) var lastRefresh: Date?
    public private(set) var fixMessage: String?
    public private(set) var model: String = "qwen3:14b"

    public var problems: [DependencyCheck] { checks.filter { $0.status == .fail || $0.status == .warn } }
    public var criticalFailures: [DependencyCheck] { checks.filter { $0.critical && $0.status == .fail } }
    public var allCriticalOK: Bool { lastRefresh != nil && criticalFailures.isEmpty }

    private static let placeholders: [DependencyCheck] = [
        DependencyCheck(id: "mainstage", title: "MainStage"),
        DependencyCheck(id: "blackhole", title: "BlackHole 16ch audio driver"),
        DependencyCheck(id: "mic", title: "Microphone permission"),
        DependencyCheck(id: "ollama_installed", title: "Ollama installed"),
        DependencyCheck(id: "ollama_running", title: "Ollama server running"),
        DependencyCheck(id: "qwen", title: "Qwen model pulled"),
        DependencyCheck(id: "uv", title: "uv (Python tool manager)"),
        DependencyCheck(id: "python", title: "Python runtime (rtmidi, sounddevice)"),
        DependencyCheck(id: "vad", title: "Silero VAD model", critical: false),
    ]

    public init() {}

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

        // MainStage
        let msPaths = ["/Applications/MainStage.app", "\(home)/Applications/MainStage.app"]
        let msInstalled = msPaths.contains { fm.fileExists(atPath: $0) }
        if !msInstalled {
            out.append(.init(id: "mainstage", title: "MainStage", status: .fail,
                             detail: "MainStage.app not found in /Applications. Install it from the Mac App Store."))
        } else if mainStageRunning {
            out.append(.init(id: "mainstage", title: "MainStage", status: .ok, detail: "Installed and running."))
        } else {
            out.append(.init(id: "mainstage", title: "MainStage", status: .warn,
                             detail: "Installed but not running. Open your concert so AetherMix can drive it.",
                             fix: .launchMainStage))
        }

        // BlackHole
        let bh = fm.fileExists(atPath: "/Library/Audio/Plug-Ins/HAL/BlackHole16ch.driver")
        out.append(bh
            ? .init(id: "blackhole", title: "BlackHole 16ch audio driver", status: .ok,
                    detail: "BlackHole16ch.driver installed.")
            : .init(id: "blackhole", title: "BlackHole 16ch audio driver", status: .fail,
                    detail: "Driver missing. The installer runs `brew install --cask blackhole-16ch` (reboot or `sudo killall coreaudiod` after).",
                    fix: .runInstaller))

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
                    detail: "Ollama not found. The installer runs `brew install ollama`.", fix: .runInstaller))

        // Ollama server + model
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
                                 detail: "Configured model \(model) not pulled; found \(sameFamily.joined(separator: ", ")). Pull \(model) or change ollama_model.",
                                 fix: .pullModel))
            } else {
                out.append(.init(id: "qwen", title: "Qwen model pulled", status: .fail,
                                 detail: "\(model) not pulled. Required for the supervisory LLM (otherwise Tier-1 deterministic mix only).",
                                 fix: .pullModel))
            }
        } else {
            out.append(.init(id: "ollama_running", title: "Ollama server running", status: .fail,
                             detail: "Nothing listening on 127.0.0.1:11434.",
                             fix: ollamaBin != nil || fm.fileExists(atPath: "/Applications/Ollama.app") ? .startOllama : .runInstaller))
            out.append(.init(id: "qwen", title: "Qwen model pulled", status: .fail,
                             detail: "Cannot verify \(model) until Ollama is running."))
        }

        // uv
        let uv = ["\(home)/.local/bin/uv", "/opt/homebrew/bin/uv", "/usr/local/bin/uv"].first { fm.isExecutableFile(atPath: $0) }
        out.append(uv != nil
            ? .init(id: "uv", title: "uv (Python tool manager)", status: .ok, detail: uv ?? "")
            : .init(id: "uv", title: "uv (Python tool manager)", status: .fail,
                    detail: "uv missing. The installer runs `brew install uv`.", fix: .runInstaller))

        // Python runtime with the audio/MIDI extras
        if let py = EngineProcessManager.resolveExistingPython(runtimeRoot: repoRoot) {
            let (code, text) = runCapture(py, ["-c", "import rtmidi, sounddevice, numpy, fastapi; print('ok')"])
            if code == 0 {
                out.append(.init(id: "python", title: "Python runtime (rtmidi, sounddevice)", status: .ok, detail: py))
            } else {
                let last = text.split(separator: "\n").last.map(String.init) ?? "import failed"
                out.append(.init(id: "python", title: "Python runtime (rtmidi, sounddevice)", status: .fail,
                                 detail: "\(py) is missing packages: \(last)", fix: .runInstaller))
            }
        } else {
            out.append(.init(id: "python", title: "Python runtime (rtmidi, sounddevice)", status: .fail,
                             detail: "~/.aethermix/venv not provisioned — the MIDI bridge cannot start without it.",
                             fix: .runInstaller))
        }

        // Silero VAD
        let vad = "\(home)/.aethermix/models/silero_vad.onnx"
        let vadSize = (try? fm.attributesOfItem(atPath: vad)[.size] as? Int) ?? 0
        out.append(vadSize > 100_000
            ? .init(id: "vad", title: "Silero VAD model", status: .ok, detail: vad, critical: false)
            : .init(id: "vad", title: "Silero VAD model", status: .warn,
                    detail: "Missing — falls back to energy-based VAD.", fix: .runInstaller, critical: false))
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

    // MARK: - Fixes

    public func apply(_ fix: CheckFix, repoRoot: URL, processManager: EngineProcessManager) {
        switch fix {
        case .runInstaller:
            runInTerminal(script: "\"\(repoRoot.path)/scripts/install_prereqs.sh\" --model \"\(model)\"",
                          title: "AetherMix installer")
        case .pullModel:
            runInTerminal(script: "export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH; ollama pull \"\(model)\"",
                          title: "Pull \(model)")
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
        let file = dir.appendingPathComponent("run-\(title.replacingOccurrences(of: " ", with: "-")).command")
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
