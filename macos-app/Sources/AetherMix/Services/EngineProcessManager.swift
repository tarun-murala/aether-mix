import Darwin
import Foundation
import Observation

public enum EngineLaunchMode: String, Sendable, Equatable {
    case stopped = "Stopped"
    case externalMakeDev = "External (make dev)"
    case managedSubprocess = "Managed Subprocess"
}

public enum LaunchContext: String, Sendable, Equatable {
    case terminalDev = "Terminal (make dev / CLI)"
    case standaloneApp = "AetherMix.app Bundle"

    public var tccTargetAppName: String {
        switch self {
        case .terminalDev:
            return ProcessInfo.processInfo.environment["TERM_PROGRAM"] ?? "Terminal.app"
        case .standaloneApp:
            return "AetherMix.app"
        }
    }
}

public enum SafeUpdateCheckResult: Sendable, Equatable {
    case allowed
    case blocked(reason: String)

    public var isAllowed: Bool {
        if case .allowed = self { return true }
        return false
    }
}

public struct DiagnosticLogEntry: Sendable, Identifiable, Equatable {
    public let id: UUID
    public let timestamp: Date
    public let stream: String
    public let line: String

    public init(id: UUID = UUID(), timestamp: Date = Date(), stream: String, line: String) {
        self.id = id
        self.timestamp = timestamp
        self.stream = stream
        self.line = line
    }
}

/// Manages lifecycle of the standalone `com.aethermix.midibridge` LaunchAgent,
/// detects/spawns the local `uv` Python engine if `127.0.0.1:8765` is not already
/// listening, captures stdout/stderr logs for Diagnostics, and enforces the
/// SPEC §8 Safe-Update Lock (`autonomyMode == .active` hard-blocks updates).
@MainActor
@Observable
public final class EngineProcessManager {
    public private(set) var launchAgentInstalled: Bool = false
    public private(set) var launchAgentLoaded: Bool = false
    public private(set) var launchMode: EngineLaunchMode = .stopped
    public private(set) var enginePID: Int32?
    public private(set) var diagnosticLogs: [DiagnosticLogEntry] = []
    public private(set) var lastStatusMessage: String?
    public private(set) var safeUpdateAlert: String?

    public let repoRootURL: URL
    public let launchContext: LaunchContext

    private var engineProcess: Process?
    private let maxLogLines = 500

    public init(repoRootURL: URL? = nil) {
        self.repoRootURL = repoRootURL ?? Self.discoverRepoRoot()
        self.launchContext = Self.detectLaunchContext()
        refreshBridgeLaunchAgentStatus()
    }

    // MARK: - Safe-Update Lock (SPEC §8)

    /// Pure guard enforcing SPEC §8: updates are hard-blocked while `autonomyMode == .active`.
    nonisolated public static func evaluateSafeUpdateLock(autonomyMode: AutonomyMode) -> SafeUpdateCheckResult {
        switch autonomyMode {
        case .active:
            return .blocked(
                reason: "Safe-Update Lock: Updates are hard-blocked while Autonomy Mode is ACTIVE. Switch to SHADOW OBSERVE or EMERGENCY BYPASS first."
            )
        case .shadowObserve, .emergencyBypass:
            return .allowed
        }
    }

    @discardableResult
    public func requestUpdateAction(_ actionName: String, autonomyMode: AutonomyMode) -> SafeUpdateCheckResult {
        let result = Self.evaluateSafeUpdateLock(autonomyMode: autonomyMode)
        switch result {
        case .allowed:
            safeUpdateAlert = nil
            appendLog(stream: "system", line: "Safe-Update Lock cleared for '\(actionName)' (mode: \(autonomyMode.rawValue))")
        case .blocked(let reason):
            safeUpdateAlert = reason
            appendLog(stream: "stderr", line: "BLOCKED '\(actionName)': \(reason)")
        }
        return result
    }

    // MARK: - LaunchAgent Verification & Installation

    public func refreshBridgeLaunchAgentStatus() {
        let plistPath = AetherMixProtocol.defaultLaunchAgentPath.path
        launchAgentInstalled = FileManager.default.fileExists(atPath: plistPath)

        guard launchAgentInstalled else {
            launchAgentLoaded = false
            return
        }

        let uid = getuid()
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        proc.arguments = ["print", "gui/\(uid)/com.aethermix.midibridge"]
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
            proc.waitUntilExit()
            launchAgentLoaded = (proc.terminationStatus == 0)
        } catch {
            launchAgentLoaded = false
        }
    }

    public func installBridgeLaunchAgent() {
        let templateURL = repoRootURL.appendingPathComponent("scripts/com.aethermix.midibridge.plist")
        let targetURL = AetherMixProtocol.defaultLaunchAgentPath
        let homePath = FileManager.default.homeDirectoryForCurrentUser.path
        guard let pythonPath = Self.resolveExistingPython(runtimeRoot: repoRootURL) else {
            lastStatusMessage = "Cannot install bridge: no Python runtime with rtmidi found (~/.aethermix/venv). Run the installer from Setup → System Check first."
            appendLog(stream: "stderr", line: lastStatusMessage ?? "")
            return
        }

        do {
            let template: String
            if FileManager.default.fileExists(atPath: templateURL.path) {
                template = try String(contentsOf: templateURL, encoding: .utf8)
            } else {
                template = Self.fallbackLaunchAgentTemplate
            }

            let rendered = template
                .replacingOccurrences(of: "__REPO_ROOT__", with: repoRootURL.path)
                .replacingOccurrences(of: "__PYTHON__", with: pythonPath)
                .replacingOccurrences(of: "__HOME__", with: homePath)

            try FileManager.default.createDirectory(
                at: targetURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try FileManager.default.createDirectory(
                at: AetherMixProtocol.defaultAetherMixDirectory,
                withIntermediateDirectories: true
            )
            try rendered.write(to: targetURL, atomically: true, encoding: .utf8)

            let uid = getuid()
            runLaunchctl(["bootout", "gui/\(uid)/com.aethermix.midibridge"])
            usleep(300_000)
            runLaunchctl(["enable", "gui/\(uid)/com.aethermix.midibridge"])
            let bootstrapOK = runLaunchctl(["bootstrap", "gui/\(uid)", targetURL.path])

            refreshBridgeLaunchAgentStatus()
            lastStatusMessage = bootstrapOK
                ? "Installed & started com.aethermix.midibridge LaunchAgent"
                : "Wrote plist to \(targetURL.path); launchctl bootstrap returned non-zero"
            appendLog(stream: "system", line: lastStatusMessage ?? "")
        } catch {
            lastStatusMessage = "LaunchAgent install failed: \(error.localizedDescription)"
            appendLog(stream: "stderr", line: lastStatusMessage ?? "")
        }
    }

    // MARK: - Engine Discovery & Managed Subprocess

    public func ensureEngineRunning() {
        if Self.isLocalPortOpen(host: "127.0.0.1", port: 8765) {
            if engineProcess?.isRunning == true {
                launchMode = .managedSubprocess
            } else {
                launchMode = .externalMakeDev
                lastStatusMessage = "Attached to running engine on 127.0.0.1:8765 (make dev)"
                appendLog(stream: "system", line: lastStatusMessage ?? "")
            }
            return
        }
        spawnLocalEngine()
    }

    public func spawnLocalEngine() {
        if let proc = engineProcess, proc.isRunning {
            return
        }

        let mainPy = repoRootURL.appendingPathComponent("engine/main.py")
        guard FileManager.default.fileExists(atPath: mainPy.path) else {
            lastStatusMessage = "engine/main.py not found under \(repoRootURL.path)"
            appendLog(stream: "stderr", line: lastStatusMessage ?? "")
            launchMode = .stopped
            return
        }

        let resolvedPython = Self.resolvePythonPath(runtimeRoot: repoRootURL)
        let (planPath, rulesPath) = Self.resolveConfigPaths(runtimeRoot: repoRootURL, isAppBundle: launchContext == .standaloneApp)

        let proc = Process()
        proc.currentDirectoryURL = repoRootURL

        // Finder-launched .app bundles have minimal PATH; include Homebrew, uv, and Ollama paths.
        var env = ProcessInfo.processInfo.environment
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let extraPaths = ["/opt/homebrew/bin", "/usr/local/bin", "\(home)/.local/bin"]
        let existingPath = env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        env["PATH"] = (extraPaths + [existingPath]).joined(separator: ":")
        let existingPyPath = env["PYTHONPATH"]
        env["PYTHONPATH"] = existingPyPath.map { "\(repoRootURL.path):\($0)" } ?? repoRootURL.path
        // Never let uv write a .venv inside the signed .app bundle.
        env["UV_PROJECT_ENVIRONMENT"] = AetherMixProtocol.defaultAetherMixDirectory.appendingPathComponent("venv").path
        proc.environment = env

        if FileManager.default.isExecutableFile(atPath: resolvedPython) {
            proc.executableURL = URL(fileURLWithPath: resolvedPython)
            proc.arguments = [
                mainPy.path,
                "--service-plan", planPath,
                "--mixing-rules", rulesPath
            ]
        } else {
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            proc.arguments = [
                "uv", "run", "--extra", "audio", "--project", repoRootURL.appendingPathComponent("engine").path,
                "python", mainPy.path,
                "--service-plan", planPath,
                "--mixing-rules", rulesPath
            ]
        }

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        proc.standardOutput = stdoutPipe
        proc.standardError = stderrPipe

        attachPipeReader(stdoutPipe, streamName: "stdout")
        attachPipeReader(stderrPipe, streamName: "stderr")

        proc.terminationHandler = { [weak self] terminated in
            let code = terminated.terminationStatus
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.engineProcess = nil
                self.enginePID = nil
                self.launchMode = .stopped
                let msg = "Managed engine exited with status \(code)"
                self.lastStatusMessage = msg
                self.appendLog(stream: "system", line: msg)
            }
        }

        do {
            try proc.run()
            self.engineProcess = proc
            self.enginePID = proc.processIdentifier
            self.launchMode = .managedSubprocess
            let msg = "Spawned managed AI engine (PID \(proc.processIdentifier))"
            self.lastStatusMessage = msg
            self.appendLog(stream: "system", line: msg)
        } catch {
            self.launchMode = .stopped
            let msg = "Failed to spawn AI engine: \(error.localizedDescription)"
            self.lastStatusMessage = msg
            self.appendLog(stream: "stderr", line: msg)
        }
    }

    public func stopManagedEngine() {
        guard let proc = engineProcess, proc.isRunning else { return }
        proc.terminate()
        engineProcess = nil
        enginePID = nil
        launchMode = .stopped
        appendLog(stream: "system", line: "Sent SIGTERM to managed AI engine")
    }

    public func restartEngine(autonomyMode: AutonomyMode) {
        if autonomyMode == .active {
            // Safe restart only when user explicitly restarts after crash or in non-active mode;
            // if engine is already down, allow recovery immediately.
            if Self.isLocalPortOpen(host: "127.0.0.1", port: 8765) {
                _ = requestUpdateAction("Restart Engine", autonomyMode: autonomyMode)
                return
            }
        }
        stopManagedEngine()
        spawnLocalEngine()
    }

    public func clearLogs() {
        diagnosticLogs.removeAll()
    }

    public func appendLog(stream: String, line: String) {
        let trimmed = line.trimmingCharacters(in: .newlines)
        guard !trimmed.isEmpty else { return }
        diagnosticLogs.append(DiagnosticLogEntry(stream: stream, line: trimmed))
        if diagnosticLogs.count > maxLogLines {
            diagnosticLogs.removeFirst(diagnosticLogs.count - maxLogLines)
        }
    }

    // MARK: - Private Helpers

    private func attachPipeReader(_ pipe: Pipe, streamName: String) {
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            if let text = String(data: data, encoding: .utf8) {
                let lines = text.split(whereSeparator: \.isNewline).map(String.init)
                Task { @MainActor [weak self] in
                    for line in lines {
                        self?.appendLog(stream: streamName, line: line)
                    }
                }
            }
        }
    }

    @discardableResult
    private func runLaunchctl(_ arguments: [String]) -> Bool {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        proc.arguments = arguments
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
            proc.waitUntilExit()
            return proc.terminationStatus == 0
        } catch {
            return false
        }
    }

    public static func isLocalPortOpen(host: String, port: UInt16) -> Bool {
        let sock = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else { return false }
        defer { Darwin.close(sock) }

        var timeout = timeval(tv_sec: 0, tv_usec: 150_000)
        _ = Darwin.setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        _ = Darwin.setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))

        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        _ = host.withCString { Darwin.inet_pton(AF_INET, $0, &addr.sin_addr) }

        let result = withUnsafePointer(to: &addr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.connect(sock, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return result == 0
    }

    private static func detectLaunchContext() -> LaunchContext {
        if Bundle.main.bundlePath.hasSuffix(".app") {
            return .standaloneApp
        }
        return .terminalDev
    }

    private static func resolveConfigPaths(runtimeRoot: URL, isAppBundle: Bool) -> (String, String) {
        let bundledPlan = runtimeRoot.appendingPathComponent("engine/schemas/service_plan.json")
        let bundledRules = runtimeRoot.appendingPathComponent("engine/schemas/mixing_rules.json")
        guard isAppBundle else {
            return (bundledPlan.path, bundledRules.path)
        }
        let cfgDir = AetherMixProtocol.defaultAetherMixDirectory.appendingPathComponent("config", isDirectory: true)
        let userPlan = cfgDir.appendingPathComponent("service_plan.json")
        let userRules = cfgDir.appendingPathComponent("mixing_rules.json")
        let fm = FileManager.default
        try? fm.createDirectory(at: cfgDir, withIntermediateDirectories: true)
        if !fm.fileExists(atPath: userPlan.path), fm.fileExists(atPath: bundledPlan.path) {
            try? fm.copyItem(at: bundledPlan, to: userPlan)
        }
        if !fm.fileExists(atPath: userRules.path), fm.fileExists(atPath: bundledRules.path) {
            try? fm.copyItem(at: bundledRules, to: userRules)
        }
        return (
            fm.fileExists(atPath: userPlan.path) ? userPlan.path : bundledPlan.path,
            fm.fileExists(atPath: userRules.path) ? userRules.path : bundledRules.path
        )
    }

    /// Returns the managed Python interpreter only if it actually exists.
    nonisolated public static func resolveExistingPython(runtimeRoot: URL) -> String? {
        let candidate = resolvePythonPath(runtimeRoot: runtimeRoot)
        return FileManager.default.isExecutableFile(atPath: candidate) ? candidate : nil
    }

    nonisolated private static func resolvePythonPath(runtimeRoot: URL) -> String {
        let candidates = [
            runtimeRoot.appendingPathComponent("engine/.venv/bin/python").path,
            AetherMixProtocol.defaultAetherMixDirectory.appendingPathComponent("venv/bin/python").path,
        ]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            return path
        }
        return candidates[0]
    }

    private static func discoverRepoRoot() -> URL {
        // 1. Inside installed AetherMix.app/Contents/Resources/runtime
        if let resURL = Bundle.main.resourceURL {
            let bundledRuntime = resURL.appendingPathComponent("runtime", isDirectory: true)
            if FileManager.default.fileExists(atPath: bundledRuntime.appendingPathComponent("engine/main.py").path) {
                return bundledRuntime
            }
        }
        // 2. Installed /Applications/AetherMix.app fallback
        let appRuntime = URL(fileURLWithPath: "/Applications/AetherMix.app/Contents/Resources/runtime", isDirectory: true)
        if FileManager.default.fileExists(atPath: appRuntime.appendingPathComponent("engine/main.py").path) {
            return appRuntime
        }
        // 3. Development repo walk-up from current working directory
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        var cursor = cwd
        for _ in 0..<5 {
            let candidate = cursor.appendingPathComponent("engine/main.py")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return cursor
            }
            let parent = cursor.deletingLastPathComponent()
            if parent.path == cursor.path { break }
            cursor = parent
        }
        return cwd
    }

    private static let fallbackLaunchAgentTemplate = """
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
        <key>Label</key>
        <string>com.aethermix.midibridge</string>
        <key>ProgramArguments</key>
        <array>
            <string>__PYTHON__</string>
            <string>__REPO_ROOT__/bridge/midi_bridge_daemon.py</string>
        </array>
        <key>KeepAlive</key>
        <true/>
        <key>ThrottleInterval</key>
        <integer>1</integer>
        <key>ProcessType</key>
        <string>Interactive</string>
        <key>StandardOutPath</key>
        <string>__HOME__/.aethermix/bridge.stdout.log</string>
        <key>StandardErrorPath</key>
        <string>__HOME__/.aethermix/bridge.stderr.log</string>
    </dict>
    </plist>
    """
}
