import CoreAudio
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
    @State private var mainstageExclusiveConfirmed: Bool = false
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

    private let stepTitles = [
        "1. TCC & Audio",
        "2. Bridge & MainStage",
        "3. MIDI & Mute Polarity",
        "4. Soundcheck Calibration"
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
                    case 0: tccAndAudioStep
                    case 1: bridgeAndMainStageStep
                    case 2: midiLearnAndMuteStep
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
            }
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
