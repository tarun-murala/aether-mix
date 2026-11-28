import CoreAudio
import CoreMIDI
import SwiftUI

/// Guided Setup & 4-Step Sequential Soundcheck Calibration Wizard (SPEC §5.2 & §8):
/// 1. macOS TCC Microphone Permission (Terminal vs AetherMix.app) & BlackHole 16ch Aggregate check.
/// 2. Standalone Bridge LaunchAgent & MainStage Exclusive Controller check
///    ("Remove direct hardware assignments in MainStage").
/// 3. MIDI Learn Pulse & Mute Polarity Verification (auto-mute forbidden on `ch1_pastor`).
/// 4. 4-Step Sequential Soundcheck Calibration:
///    A1 Room Silence -> A2a Spoken Nominals -> A2b Band Nominals -> B Bleed Floor -> C Baseline Capture.
public struct SetupWizardView: View {
    public let wsClient: EngineWebSocketClient
    public let bridgeClient: BridgeSocketClient
    public let processManager: EngineProcessManager

    @Environment(\.dismiss) private var dismiss

    @State private var currentStep: Int = 0
    @State private var detectedAudioDevices: [String] = []
    @AppStorage("aethermix.mainstageExclusiveConfirmed") private var mainstageExclusiveConfirmed: Bool = false
    @State private var exclusivityReport: String?
    @State private var mutePolarityMap: [String: Bool] = [
        "ch2_lead_vox": true,
        "ch3_bv1": true,
        "ch4_bv2": true
    ]
    @State private var baselinePhase: ServicePhase = .worshipHigh
    @State private var quickDeriveOthers: Bool = true
    @State private var calibrationBusyStep: String?
    @State private var calibrationLog: [String] = []
    @State private var completedCalibrationSteps: Set<String> = []

    @State private var checker = DependencyChecker.shared

    private let stepTitles = [
        "1. System Check",
        "2. TCC & Audio",
        "3. Bridge",
        "4. MIDI & Mute",
        "5. Calibration"
    ]

    public init(
        wsClient: EngineWebSocketClient,
        bridgeClient: BridgeSocketClient,
        processManager: EngineProcessManager
    ) {
        self.wsClient = wsClient
        self.bridgeClient = bridgeClient
        self.processManager = processManager
    }

    private var hasBlackHole16ch: Bool {
        detectedAudioDevices.contains { $0.localizedCaseInsensitiveContains("BlackHole") }
    }

    private var hasAggregateDevice: Bool {
        detectedAudioDevices.contains { $0.localizedCaseInsensitiveContains("Aggregate") }
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("AetherMix Setup & Soundcheck Wizard")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundStyle(MissionTheme.textPrimary)
                    Text("Pre-service verification for CoreAudio TCC, standalone MIDI Bridge, MainStage exclusivity, and 4-step bleed calibration.")
                        .font(.system(size: 11))
                        .foregroundStyle(MissionTheme.textSecondary)
                }
                Spacer()
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }

            // Stepper Pills
            HStack(spacing: 6) {
                ForEach(0..<stepTitles.count, id: \.self) { idx in
                    Button {
                        currentStep = idx
                    } label: {
                        Text(stepTitles[idx])
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(currentStep == idx ? Color.black : MissionTheme.textSecondary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .frame(maxWidth: .infinity)
                            .background(
                                currentStep == idx ? MissionTheme.cyanTarget : MissionTheme.panelBg,
                                in: RoundedRectangle(cornerRadius: 7)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 7)
                                    .stroke(currentStep == idx ? MissionTheme.cyanTarget : MissionTheme.border, lineWidth: 1)
                            )
                    }
                    .buttonStyle(.plain)
                }
            }

            Divider().overlay(MissionTheme.border)

            // Active Step Body
            ScrollView {
                Group {
                    switch currentStep {
                    case 0: systemCheckStep
                    case 1: tccAndAudioStep
                    case 2: bridgeAndMainStageStep
                    case 3: midiLearnAndMuteStep
                    default: soundcheckCalibrationStep
                    }
                }
                .padding(.vertical, 4)
            }
            .frame(minHeight: 360)

            Divider().overlay(MissionTheme.border)

            // Footer Navigation
            HStack {
                Button("Previous") {
                    currentStep = max(0, currentStep - 1)
                }
                .disabled(currentStep == 0)

                Spacer()

                if currentStep < stepTitles.count - 1 {
                    Button("Next Step") {
                        currentStep = min(stepTitles.count - 1, currentStep + 1)
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    Button("Finish Setup") {
                        dismiss()
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
        .padding(18)
        .frame(minWidth: 760, maxWidth: 760, minHeight: 580)
        .background(MissionTheme.canvasBg)
        .preferredColorScheme(.dark)
        .onAppear {
            detectedAudioDevices = CoreAudioDeviceScanner.listInputDeviceNames()
            processManager.refreshBridgeLaunchAgentStatus()
            Task { await checker.refresh(repoRoot: processManager.repoRootURL) }
        }
    }

    // MARK: - Step 0: System Check (all external dependencies + live connections)

    private var liveRows: [DependencyCheck] {
        [
            DependencyCheck(
                id: "engine", title: "AI engine (WebSocket 127.0.0.1:8765)",
                status: wsClient.isConnected ? .ok : .fail,
                detail: wsClient.isConnected
                    ? "Connected — launch mode: \(processManager.launchMode.rawValue)."
                    : (wsClient.lastError ?? processManager.lastStatusMessage ?? "Engine not reachable. Check Python runtime and the Diagnostics log.")
            ),
            DependencyCheck(
                id: "bridge", title: "MIDI bridge (LaunchAgent + socket)",
                status: bridgeClient.isConnected ? .ok : .fail,
                detail: bridgeClient.isConnected
                    ? "Bridge socket connected."
                    : (processManager.launchAgentInstalled
                        ? (processManager.launchAgentLoaded ? "Agent loaded but socket not answering — see ~/.aethermix/bridge.stderr.log."
                                                            : "Plist installed but not loaded in launchd.")
                        : "LaunchAgent not installed."),
                fix: bridgeClient.isConnected ? nil : .installBridgeAgent
            ),
            DependencyCheck(
                id: "controller", title: "MainStage MIDI link (AetherMix virtual port)",
                status: bridgeClient.isConnected ? (bridgeClient.bridgeState.controllerConnected ? .ok : .warn) : .fail,
                detail: bridgeClient.isConnected
                    ? (bridgeClient.bridgeState.controllerConnected
                        ? "Controller/MIDI path active."
                        : "Bridge is up but no hardware controller detected. In MainStage, set the Concert MIDI input to the AetherMix virtual port.")
                    : "Needs the bridge first.",
                critical: false
            ),
        ]
    }

    private var systemCheckStep: some View {
        let rows = checker.checks + liveRows
        let failing = rows.filter { $0.critical && $0.status == .fail }.count
        let anyMissing = checker.checks.contains { $0.status != .ok }
        return VStack(alignment: .leading, spacing: 12) {
            systemCheckHeaderCard(failing: failing, anyMissing: anyMissing)
            if checker.isRunningTask || !checker.taskLogLines.isEmpty {
                liveTaskConsoleCard
            }
            ForEach(rows) { row in
                dependencyRow(row)
            }
        }
    }

    @ViewBuilder
    private func systemCheckHeaderCard(failing: Int, anyMissing: Bool) -> some View {
        let badgeText = checker.isRunningTask
            ? "RUNNING…"
            : (checker.isRefreshing ? "CHECKING…" : (failing == 0 ? "READY" : "\(failing) BLOCKING"))
        let badgeColor = (checker.isRunningTask || checker.isRefreshing)
            ? MissionTheme.cyanTarget
            : (failing == 0 ? MissionTheme.emerald : MissionTheme.crimson)
        wizardCard(title: "Dependency & Connection Status", badge: badgeText, badgeColor: badgeColor) {
            HStack(spacing: 8) {
                Button(checker.isRefreshing ? "Checking…" : "Re-check") {
                    processManager.refreshBridgeLaunchAgentStatus()
                    Task { await checker.refresh(repoRoot: processManager.repoRootURL) }
                }
                .disabled(checker.isRefreshing || checker.isRunningTask)

                Button {
                    checker.apply(.runVerificationDoctor, repoRoot: processManager.repoRootURL, processManager: processManager)
                } label: {
                    Label("Run Doctor (Live Test)", systemImage: "stethoscope")
                }
                .disabled(checker.isRunningTask)
                .help("Runs scripts/check_setup.py --test-ollama in-app to test BlackHole, CoreMIDI, Silero VAD ONNX, and live Qwen JSON inference")

                if anyMissing {
                    Button {
                        checker.apply(.runInstaller, repoRoot: processManager.repoRootURL, processManager: processManager)
                    } label: {
                        Label("Install All Missing", systemImage: "arrow.down.circle.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(checker.isRunningTask)
                    .help("Runs scripts/install_prereqs.sh in Terminal (installs Homebrew deps, BlackHole 16ch, Python venv, Ollama + Qwen, and Bridge LaunchAgent)")
                }

                Spacer()

                qwenModelPicker
            }

            if let msg = checker.fixMessage {
                HStack(spacing: 6) {
                    if checker.isRunningTask {
                        ProgressView().controlSize(.small)
                    }
                    Text(msg)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(checker.lastTaskSucceeded == false ? MissionTheme.crimson : MissionTheme.cyanTarget)
                }
            }
        }
    }

    private var qwenModelPicker: some View {
        HStack(spacing: 4) {
            Text("Qwen:")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(MissionTheme.textSecondary)
            Menu {
                ForEach(DependencyChecker.availableQwenModels, id: \.self) { m in
                    Button {
                        checker.selectModel(m, repoRoot: processManager.repoRootURL)
                    } label: {
                        if m == checker.model {
                            Label(m, systemImage: "checkmark")
                        } else {
                            Text(m)
                        }
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Text(checker.model)
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                    Image(systemName: "chevron.down")
                        .font(.system(size: 8))
                }
                .foregroundStyle(MissionTheme.cyanTarget)
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(MissionTheme.canvasBg, in: RoundedRectangle(cornerRadius: 5))
                .overlay(RoundedRectangle(cornerRadius: 5).stroke(MissionTheme.border, lineWidth: 1))
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .disabled(checker.isRunningTask)
            .help("Select Ollama Qwen model (qwen3:14b for 24 GB+ Macs, qwen3:8b for 16 GB Macs)")
        }
    }

    private var liveTaskConsoleCard: some View {
        let strokeColor: Color = checker.isRunningTask
            ? MissionTheme.cyanTarget.opacity(0.7)
            : (checker.lastTaskSucceeded == true ? MissionTheme.emerald.opacity(0.6) : MissionTheme.border)
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: "terminal.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(MissionTheme.cyanTarget)
                Text(checker.activeTaskTitle ?? "Setup & Doctor Output")
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.textPrimary)
                Spacer()
                if checker.isRunningTask {
                    Button("Cancel") { checker.cancelRunningTask() }
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(MissionTheme.crimson)
                        .buttonStyle(.plain)
                } else {
                    Button("Clear") { checker.clearTaskLog() }
                        .font(.system(size: 10))
                        .foregroundStyle(MissionTheme.textSecondary)
                        .buttonStyle(.plain)
                }
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(checker.taskLogLines.enumerated()), id: \.offset) { idx, line in
                            Text(line)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(logLineColor(line))
                                .textSelection(.enabled)
                                .id(idx)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(height: 120)
                .padding(8)
                .background(MissionTheme.canvasBg, in: RoundedRectangle(cornerRadius: 6))
                .onChange(of: checker.taskLogLines.count) { _, newCount in
                    if newCount > 0 {
                        proxy.scrollTo(newCount - 1, anchor: .bottom)
                    }
                }
            }
        }
        .padding(10)
        .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(strokeColor, lineWidth: 1))
    }

    private func logLineColor(_ line: String) -> Color {
        if line.contains("[FAIL]") || line.contains("ERROR") {
            return MissionTheme.crimson
        }
        if line.contains("[PASS]") || line.contains("Completed successfully") {
            return MissionTheme.emerald
        }
        if line.contains("[WARN]") {
            return MissionTheme.amber
        }
        return MissionTheme.textSecondary
    }

    private func dependencyRow(_ row: DependencyCheck) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: symbol(for: row.status))
                .font(.system(size: 15))
                .foregroundStyle(color(for: row.status))
            VStack(alignment: .leading, spacing: 2) {
                Text(row.title)
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(MissionTheme.textPrimary)
                Text(row.detail)
                    .font(.system(size: 10))
                    .foregroundStyle(MissionTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            if let fix = row.fix, row.status != .ok {
                Button {
                    checker.apply(fix, repoRoot: processManager.repoRootURL, processManager: processManager) {
                        if fix == .installBridgeAgent || fix == .syncPythonRuntime {
                            Task {
                                try? await Task.sleep(for: .seconds(1))
                                bridgeClient.connectOnce()
                                processManager.ensureEngineRunning()
                                wsClient.start()
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        if fix.requiresTerminal {
                            Image(systemName: "terminal").font(.system(size: 9))
                        }
                        Text(fix.buttonTitle)
                    }
                }
                .font(.system(size: 10, weight: .semibold))
                .disabled(checker.isRunningTask && !fix.requiresTerminal)
                .help(
                    fix.requiresTerminal
                        ? "Opens in Terminal (requires macOS admin/sudo prompt)"
                        : "Runs directly inside AetherMix with live progress"
                )
            }
        }
        .padding(10)
        .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(color(for: row.status).opacity(0.5), lineWidth: 1))
    }

    private func symbol(for status: CheckStatus) -> String {
        switch status {
        case .ok: return "checkmark.circle.fill"
        case .warn: return "exclamationmark.triangle.fill"
        case .fail: return "xmark.octagon.fill"
        case .checking: return "hourglass"
        }
    }

    private func color(for status: CheckStatus) -> Color {
        switch status {
        case .ok: return MissionTheme.emerald
        case .warn: return MissionTheme.amber
        case .fail: return MissionTheme.crimson
        case .checking: return MissionTheme.cyanTarget
        }
    }

    // MARK: - Step 1: TCC Microphone & BlackHole 16ch Aggregate Check

    private var tccAndAudioStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            wizardCard(
                title: "macOS TCC Microphone Permission Context",
                badge: processManager.launchContext.rawValue,
                badgeColor: MissionTheme.cyanTarget
            ) {
                Text("Active Launch Context: **\(processManager.launchContext.rawValue)**")
                    .font(.system(size: 12))
                    .foregroundStyle(MissionTheme.textPrimary)

                Text(
                    processManager.launchContext == .terminalDev
                        ? "Under `make dev` (Terminal), macOS grants CoreAudio/BlackHole input permission to **\(processManager.launchContext.tccTargetAppName)** — ensure it is enabled in System Settings → Privacy & Security → Microphone."
                        : "Under standalone **AetherMix.app**, macOS grants CoreAudio/BlackHole input permission directly to **AetherMix.app** via `NSMicrophoneUsageDescription`."
                )
                .font(.system(size: 11))
                .foregroundStyle(MissionTheme.textSecondary)

                if wsClient.snapshot.hardwareWarnings.contains("HARDWARE_SILENCE_WARNING") {
                    Label("HARDWARE_SILENCE_WARNING active: bitwise all-zero input detected (> 2.0 s). Check Microphone TCC permission!", systemImage: "exclamationmark.triangle.fill")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(MissionTheme.crimson)
                }
            }

            wizardCard(
                title: "BlackHole 16ch & 48 kHz Aggregate Device",
                badge: hasBlackHole16ch ? "DETECTED" : "NOT FOUND",
                badgeColor: hasBlackHole16ch ? MissionTheme.emerald : MissionTheme.amber
            ) {
                HStack(spacing: 16) {
                    statusRow(label: "BlackHole 16ch Driver", ok: hasBlackHole16ch)
                    statusRow(label: "macOS Aggregate Device", ok: hasAggregateDevice)
                    Spacer()
                    Button("Rescan CoreAudio") {
                        detectedAudioDevices = CoreAudioDeviceScanner.listInputDeviceNames()
                    }
                }

                Text("MainStage must drive a **48 kHz macOS Aggregate Device** with your physical PA interface as clock master and Drift Correction enabled on BlackHole 16ch. Route Ch 1–12 Pre-Fader, Ch 13–14 Direct Room Mic, Ch 15–16 Post-Fader Master.")
                    .font(.system(size: 11))
                    .foregroundStyle(MissionTheme.textSecondary)

                if !detectedAudioDevices.isEmpty {
                    Text("Detected Input Devices: \(detectedAudioDevices.joined(separator: " • "))")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(MissionTheme.textSecondary)
                }
            }
        }
    }

    // MARK: - Step 2: Standalone Bridge & MainStage Exclusive Controller Check

    private var bridgeAndMainStageStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            wizardCard(
                title: "Standalone MIDI Bridge Daemon (launchd KeepAlive)",
                badge: bridgeClient.isConnected ? "CONNECTED" : "OFFLINE",
                badgeColor: bridgeClient.isConnected ? MissionTheme.emerald : MissionTheme.crimson
            ) {
                HStack(spacing: 16) {
                    statusRow(label: "LaunchAgent Plist Installed", ok: processManager.launchAgentInstalled)
                    statusRow(label: "launchctl Loaded", ok: processManager.launchAgentLoaded)
                    statusRow(label: "Direct Bridge Socket", ok: bridgeClient.isConnected)
                }

                HStack(spacing: 10) {
                    Button("Install / Reload LaunchAgent") {
                        processManager.installBridgeLaunchAgent()
                    }
                    .buttonStyle(.borderedProminent)

                    Button("Reconnect Bridge Socket") {
                        bridgeClient.connectOnce()
                    }
                }

                if let msg = processManager.lastStatusMessage {
                    Text(msg)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(MissionTheme.cyanTarget)
                }
            }

            wizardCard(
                title: "MainStage Exclusive Controller Check (Mandatory)",
                badge: mainstageExclusiveConfirmed ? "VERIFIED" : "ACTION REQUIRED",
                badgeColor: mainstageExclusiveConfirmed ? MissionTheme.emerald : MissionTheme.amber
            ) {
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "exclamationmark.octagon.fill")
                        .font(.system(size: 20))
                        .foregroundStyle(MissionTheme.amber)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Remove direct hardware assignments in MainStage")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundStyle(MissionTheme.amber)
                        Text("Hardware MIDI controllers MUST route exclusively into `midi_bridge_daemon`, which forwards to MainStage via the AetherMix Virtual Port or IAC Bus. Dual-path assignments inside MainStage break soft-takeover pickup and human touch-lock isolation.")
                            .font(.system(size: 11))
                            .foregroundStyle(MissionTheme.textSecondary)
                    }
                }

                Toggle("I have verified that all direct hardware MIDI controller assignments are removed in MainStage (Concert Level).", isOn: $mainstageExclusiveConfirmed)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(MissionTheme.textPrimary)

                HStack(spacing: 10) {
                    Button("Auto-Verify") { runExclusivityScan() }
                        .buttonStyle(.borderedProminent)
                    Button("Open MainStage") {
                        NSWorkspace.shared.open(URL(fileURLWithPath: "/Applications/MainStage.app"))
                    }
                    if mainstageExclusiveConfirmed {
                        Button("Reset") { mainstageExclusiveConfirmed = false; exclusivityReport = nil }
                    }
                }
                if let report = exclusivityReport {
                    Text(report)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(MissionTheme.cyanTarget)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    /// Scans CoreMIDI for physical (non-virtual, non-IAC, non-AetherMix) sources. MainStage's concert file
    /// can't be inspected, so: no hardware sources => dual-path is impossible => verified automatically.
    /// Otherwise the user must confirm the listed devices are unassigned in MainStage.
    private func runExclusivityScan() {
        let hardware = MIDISourceScanner.physicalSourceNames()
        if hardware.isEmpty {
            mainstageExclusiveConfirmed = true
            exclusivityReport = "No physical MIDI controllers attached — no direct MainStage path can exist. Verified."
        } else {
            exclusivityReport = "Physical MIDI sources: \(hardware.joined(separator: ", ")). In MainStage (Layout → Concert Inspector → MIDI Inputs) uncheck these so only the AetherMix port feeds MainStage, then tick the box above."
        }
    }

    // MARK: - Step 3: MIDI Learn Pulse & Mute Polarity Verification

    private var midiLearnAndMuteStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            wizardCard(
                title: "MIDI Learn Pulse (Concert-Level Screen Controls)",
                badge: "DEDUP BYPASS",
                badgeColor: MissionTheme.cyanTarget
            ) {
                Text("Select a channel strip in MainStage Layout/Edit mode, enable MIDI Learn, then click Pulse to emit a distinct CC transition through `midi_bridge_daemon`.")
                    .font(.system(size: 11))
                    .foregroundStyle(MissionTheme.textSecondary)

                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                    ForEach(ChannelCatalog.defaultChannels) { ch in
                        HStack {
                            Text(ch.label)
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(MissionTheme.textPrimary)
                            Spacer()
                            Button("Vol CC \(ch.volCC)") {
                                Task { await wsClient.sendMidiLearnPulse(channelId: ch.id, control: .vol) }
                            }
                            .font(.system(size: 10))
                            if let revCC = ch.reverbCC {
                                Button("Rev CC \(revCC)") {
                                    Task { await wsClient.sendMidiLearnPulse(channelId: ch.id, control: .reverb) }
                                }
                                .font(.system(size: 10))
                            }
                        }
                        .padding(8)
                        .background(MissionTheme.canvasBg, in: RoundedRectangle(cornerRadius: 6))
                    }
                }
            }

            wizardCard(
                title: "Mute Polarity Verification",
                badge: "SAFETY CRITICAL",
                badgeColor: MissionTheme.amber
            ) {
                Text("Verify whether MainStage mutes on `CC 127` (Active-High) or `CC 0` (Active-Low) for mute-capable vocal channels. Note: **Automatic mute is hard-forbidden on `ch1_pastor` (Pastor Mic).**")
                    .font(.system(size: 11))
                    .foregroundStyle(MissionTheme.textSecondary)

                ForEach(ChannelCatalog.defaultChannels.filter(\.hasMute)) { ch in
                    HStack {
                        Text("\(ch.label) (Mute CC \(ch.muteCC ?? 0))")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(MissionTheme.textPrimary)
                        Spacer()
                        Picker("", selection: Binding(
                            get: { mutePolarityMap[ch.id] ?? true },
                            set: { mutePolarityMap[ch.id] = $0 }
                        )) {
                            Text("Active-High (127 = Mute)").tag(true)
                            Text("Active-Low (0 = Mute)").tag(false)
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 280)

                        Button("Verify") {
                            let activeHigh = mutePolarityMap[ch.id] ?? true
                            Task { await wsClient.verifyMutePolarity(channelId: ch.id, activeHigh: activeHigh) }
                        }
                        .disabled(!wsClient.isConnected)
                    }
                    .padding(8)
                    .background(MissionTheme.canvasBg, in: RoundedRectangle(cornerRadius: 6))
                }
            }
        }
    }

    // MARK: - Step 4: 4-Step Sequential Soundcheck Calibration (SPEC §5.2)

    private var soundcheckCalibrationStep: some View {
        VStack(alignment: .leading, spacing: 10) {
            calibrationRow(
                stepKey: "A1",
                title: "Step A1 — Room Silence (5 s)",
                subtitle: "Room Quiet — Nobody Speaks or Plays. Measures ambient voice-band noise floor per mic.",
                actionTitle: "Capture Silence"
            ) {
                let ack = try await wsClient.captureRoomSilence(durationS: 5.0)
                return ack.message ?? "Captured ambient room silence floor"
            }

            calibrationRow(
                stepKey: "A2a",
                title: "Step A2 (Part 1) — Spoken Nominals (10 s)",
                subtitle: "Band Silent — Pastor & Vocalists speak/sing normally to record clean vocal nominals with zero bleed.",
                actionTitle: "Capture Spoken"
            ) {
                let ack = try await wsClient.captureInputNominals(group: "spoken", durationS: 10.0)
                return ack.message ?? "Captured isolated spoken & vocal nominals"
            }

            calibrationRow(
                stepKey: "A2b",
                title: "Step A2 (Part 2) — Band Nominals (15 s)",
                subtitle: "Spoken Mics Silent — Keys, Guitars, Drums/Bass play normally to record instrument nominals.",
                actionTitle: "Capture Band"
            ) {
                let ack = try await wsClient.captureInputNominals(group: "band", durationS: 15.0)
                return ack.message ?? "Captured isolated band nominals"
            }

            calibrationRow(
                stepKey: "B",
                title: "Step B — Acoustic Bleed Floor (10 s)",
                subtitle: "Band Plays, Nobody Speaks. Measures per-mic acoustic bleed transfer ratio so 0 dB SNR = expected bleed only.",
                actionTitle: "Capture Bleed"
            ) {
                let ack = try await wsClient.captureBleedFloor(durationS: 10.0)
                return ack.message ?? "Captured bleed_offset_db transfer matrix"
            }

            // Step C — Baseline Capture
            wizardCard(
                title: "Step C — Baseline Snapshot (from Bridge Authoritative last_sent_cc)",
                badge: completedCalibrationSteps.contains("C") ? "CAPTURED" : "FINAL STEP",
                badgeColor: completedCalibrationSteps.contains("C") ? MissionTheme.emerald : MissionTheme.cyanTarget
            ) {
                HStack(spacing: 12) {
                    Picker("Phase", selection: $baselinePhase) {
                        ForEach(ServicePhase.macroPhases) { phase in
                            Text(phase.displayName).tag(phase)
                        }
                    }
                    .frame(width: 200)

                    Toggle("Quick-derive other phases", isOn: $quickDeriveOthers)
                        .font(.system(size: 11))

                    Spacer()

                    Button(calibrationBusyStep == "C" ? "Capturing…" : "Capture Baseline") {
                        runCalibrationStep("C") {
                            let ack = try await wsClient.captureBaseline(
                                phase: baselinePhase,
                                quickDeriveOthers: quickDeriveOthers
                            )
                            return ack.message ?? "Captured \(baselinePhase.rawValue) baseline from bridge last_sent_cc"
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!wsClient.isConnected || calibrationBusyStep != nil)
                }
            }

            if !calibrationLog.isEmpty {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(calibrationLog, id: \.self) { entry in
                        Text(entry)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(MissionTheme.textSecondary)
                    }
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    private func calibrationRow(
        stepKey: String,
        title: String,
        subtitle: String,
        actionTitle: String,
        action: @escaping () async throws -> String
    ) -> some View {
        let done = completedCalibrationSteps.contains(stepKey)
        let busy = calibrationBusyStep == stepKey
        return HStack(spacing: 12) {
            Image(systemName: done ? "checkmark.circle.fill" : "circle")
                .font(.system(size: 16))
                .foregroundStyle(done ? MissionTheme.emerald : MissionTheme.textSecondary)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(MissionTheme.textPrimary)
                Text(subtitle)
                    .font(.system(size: 10))
                    .foregroundStyle(MissionTheme.textSecondary)
            }

            Spacer()

            Button(busy ? "Running…" : actionTitle) {
                runCalibrationStep(stepKey, action: action)
            }
            .disabled(!wsClient.isConnected || calibrationBusyStep != nil)
        }
        .padding(10)
        .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(done ? MissionTheme.emerald.opacity(0.6) : MissionTheme.border, lineWidth: 1)
        )
    }

    private func runCalibrationStep(_ stepKey: String, action: @escaping () async throws -> String) {
        calibrationBusyStep = stepKey
        Task {
            defer { calibrationBusyStep = nil }
            do {
                let message = try await action()
                completedCalibrationSteps.insert(stepKey)
                calibrationLog.append("[\(stepKey)] OK: \(message)")
            } catch {
                calibrationLog.append("[\(stepKey)] ERROR: \(error.localizedDescription)")
            }
        }
    }

    private func wizardCard<Content: View>(
        title: String,
        badge: String,
        badgeColor: Color,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(MissionTheme.textPrimary)
                Spacer()
                Text(badge)
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundStyle(Color.black)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 2)
                    .background(badgeColor, in: Capsule())
            }
            content()
        }
        .padding(12)
        .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(MissionTheme.border, lineWidth: 1))
    }

    private func statusRow(label: String, ok: Bool) -> some View {
        HStack(spacing: 5) {
            Circle()
                .fill(ok ? MissionTheme.emerald : MissionTheme.amber)
                .frame(width: 8, height: 8)
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MissionTheme.textPrimary)
        }
    }
}

// MARK: - CoreAudio Input Device Scanner

public enum CoreAudioDeviceScanner {
    public static func listInputDeviceNames() -> [String] {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var dataSize: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &dataSize
        ) == noErr, dataSize > 0 else {
            return []
        }

        let count = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
        var deviceIds = [AudioDeviceID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &dataSize,
            &deviceIds
        ) == noErr else {
            return []
        }

        var names: [String] = []
        for id in deviceIds {
            var nameAddress = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyDeviceNameCFString,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            var cfName: Unmanaged<CFString>?
            var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
            let status = withUnsafeMutablePointer(to: &cfName) { ptr in
                AudioObjectGetPropertyData(id, &nameAddress, 0, nil, &nameSize, ptr)
            }
            if status == noErr, let unwrapped = cfName?.takeUnretainedValue() {
                names.append(unwrapped as String)
            }
        }
        return names
    }
}

// MARK: - CoreMIDI Physical Source Scanner

public enum MIDISourceScanner {
    public static func physicalSourceNames() -> [String] {
        var names: [String] = []
        for i in 0..<MIDIGetNumberOfSources() {
            let src = MIDIGetSource(i)
            var entity = MIDIEntityRef()
            guard MIDIEndpointGetEntity(src, &entity) == noErr, entity != 0 else { continue } // virtual endpoint
            var cfName: Unmanaged<CFString>?
            guard MIDIObjectGetStringProperty(src, kMIDIPropertyDisplayName, &cfName) == noErr,
                  let name = cfName?.takeRetainedValue() as String? else { continue }
            let lower = name.lowercased()
            if lower.contains("aethermix") || lower.contains("iac") || lower.contains("network") { continue }
            names.append(name)
        }
        return names
    }
}
