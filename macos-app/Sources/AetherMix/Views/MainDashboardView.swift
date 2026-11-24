import SwiftUI

/// Primary Live Sound Mission Control Dashboard (min 1280×800, dark mode).
public struct MainDashboardView: View {
    public let wsClient: EngineWebSocketClient
    public let bridgeClient: BridgeSocketClient
    public let processManager: EngineProcessManager

    @State private var showRulesDrawer = false
    @State private var showReviewDrawer = false
    @State private var showSetupWizard = false
    @State private var showDiagnosticsDrawer = false

    private let orderOfService: [ServiceSegment] = ChannelCatalog.defaultOrderOfService

    public init(
        wsClient: EngineWebSocketClient,
        bridgeClient: BridgeSocketClient,
        processManager: EngineProcessManager
    ) {
        self.wsClient = wsClient
        self.bridgeClient = bridgeClient
        self.processManager = processManager
    }

    private var effectiveMode: AutonomyMode {
        if bridgeClient.bridgeState.bypassLatched || wsClient.snapshot.bypassLatched {
            return .emergencyBypass
        }
        return wsClient.snapshot.autonomyMode
    }

    private var isEngineDisconnected: Bool {
        !wsClient.isConnected || (bridgeClient.isConnected && !bridgeClient.bridgeState.engineConnected)
    }

    private var activeWarnings: [String] {
        var warnings = wsClient.snapshot.hardwareWarnings
        if bridgeClient.isConnected && !bridgeClient.bridgeState.controllerConnected
            && !warnings.contains("CONTROLLER_DISCONNECTED") {
            warnings.append("CONTROLLER_DISCONNECTED")
        }
        if bridgeClient.bridgeState.restoredFromDiskAt != nil {
            warnings.append("BRIDGE_RESTORED_FROM_DISK (Verify Unsynced Faders)")
        }
        return warnings
    }

    public var body: some View {
        VStack(spacing: 0) {
            topStatusBar
            orderOfServiceBar

            if !activeWarnings.isEmpty {
                warningsStrip
            }

            if isEngineDisconnected {
                disconnectedManualControlBanner
            }

            // Main Split Workspace: Mixer Deck (~68%) + Agent Brain Stream (~32%)
            HStack(alignment: .top, spacing: 12) {
                MixerTelemetryView(wsClient: wsClient, bridgeClient: bridgeClient)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                AgentBrainStreamView(wsClient: wsClient)
                    .frame(width: 390)
                    .frame(maxHeight: .infinity)
            }
            .padding(12)
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Bottom Feedback & Critique Dock
            FeedbackControlBar(wsClient: wsClient)
        }
        .frame(minWidth: 1280, minHeight: 800)
        .background(MissionTheme.canvasBg)
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showRulesDrawer) {
            RulesManagerView(wsClient: wsClient)
        }
        .sheet(isPresented: $showReviewDrawer) {
            PostServiceReviewView(wsClient: wsClient)
        }
        .sheet(isPresented: $showSetupWizard) {
            SetupWizardView(
                wsClient: wsClient,
                bridgeClient: bridgeClient,
                processManager: processManager
            )
        }
        .sheet(isPresented: $showDiagnosticsDrawer) {
            DiagnosticsDrawerView(
                wsClient: wsClient,
                bridgeClient: bridgeClient,
                processManager: processManager
            )
        }
    }

    // MARK: - Top Status Bar

    private var topStatusBar: some View {
        HStack(spacing: 12) {
            // App Identity + Phase Pill + Confidence + Hysteresis Bar
            HStack(spacing: 10) {
                HStack(spacing: 6) {
                    Image(systemName: "waveform.circle.fill")
                        .font(.system(size: 18))
                        .foregroundStyle(MissionTheme.cyanTarget)
                    Text("AETHERMIX")
                        .font(.system(size: 13, weight: .black, design: .monospaced))
                        .foregroundStyle(MissionTheme.textPrimary)
                }

                Menu {
                    ForEach(ServicePhase.allCases) { phase in
                        Button("Override → \(phase.displayName)") {
                            Task { await wsClient.overridePhase(phase) }
                        }
                    }
                } label: {
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(spacing: 6) {
                            Circle()
                                .fill(MissionTheme.phaseColor(wsClient.snapshot.activePhase))
                                .frame(width: 8, height: 8)
                            Text(wsClient.snapshot.activePhase.rawValue)
                                .font(.system(size: 11, weight: .black, design: .monospaced))
                                .foregroundStyle(MissionTheme.textPrimary)
                            Text(String(format: "%.0f%%", wsClient.snapshot.phaseConfidence * 100.0))
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .foregroundStyle(MissionTheme.cyanTarget)
                        }

                        // Hysteresis Progress Bar (0.0 .. 1.0)
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                Capsule().fill(MissionTheme.canvasBg)
                                Capsule()
                                    .fill(MissionTheme.phaseColor(wsClient.snapshot.activePhase))
                                    .frame(width: geo.size.width * CGFloat(max(0.0, min(1.0, wsClient.snapshot.hysteresisProgress))))
                            }
                        }
                        .frame(width: 170, height: 4)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(MissionTheme.cardBg, in: RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(MissionTheme.phaseColor(wsClient.snapshot.activePhase).opacity(0.6), lineWidth: 1)
                    )
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .help("Active Service Phase, Confidence & Hysteresis Progress (click to manually override)")
            }

            // Master Autonomy Mode Segmented Control + Cmd+. Bypass
            HStack(spacing: 4) {
                ForEach(AutonomyMode.allCases) { mode in
                    let isSelected = effectiveMode == mode
                    let isDirectActiveFromBypass = (effectiveMode == .emergencyBypass && mode == .active)
                    Button {
                        switchAutonomyMode(mode)
                    } label: {
                        Text(mode.shortLabel)
                            .font(.system(size: 10, weight: .black, design: .monospaced))
                            .foregroundStyle(
                                isSelected
                                    ? (mode == .emergencyBypass ? Color.white : Color.black)
                                    : (isDirectActiveFromBypass ? MissionTheme.textSecondary.opacity(0.4) : MissionTheme.textSecondary)
                            )
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(
                                isSelected ? MissionTheme.modeColor(mode) : MissionTheme.canvasBg,
                                in: RoundedRectangle(cornerRadius: 6)
                            )
                    }
                    .buttonStyle(.plain)
                    .disabled(isDirectActiveFromBypass)
                    .help(
                        isDirectActiveFromBypass
                            ? "Switch to SHADOW OBSERVE first to release Emergency Bypass safely before entering ACTIVE"
                            : "Switch Autonomy Mode to \(mode.displayName)"
                    )
                }

                // Dedicated Cmd+. Emergency Bypass Latch Shortcut
                Button {
                    triggerEmergencyBypass()
                } label: {
                    Label("⌘.", systemImage: "hand.raised.fill")
                        .font(.system(size: 10, weight: .black, design: .monospaced))
                        .foregroundStyle(Color.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .background(MissionTheme.crimson, in: RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                .keyboardShortcut(".", modifiers: [.command])
                .help("Emergency Bypass (Cmd+.) — latches directly in standalone MIDI bridge (< 1 ms)")
            }
            .padding(3)
            .background(MissionTheme.cardBg, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(MissionTheme.border, lineWidth: 1))

            // Service Lock (Freeze Learning) Toggle
            Button {
                Task { await wsClient.setFreezeLearning(!wsClient.snapshot.freezeLearning) }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: wsClient.snapshot.freezeLearning ? "lock.fill" : "lock.open")
                        .font(.system(size: 11))
                    Text(wsClient.snapshot.freezeLearning ? "SERVICE LOCK ON" : "Freeze Learning")
                        .font(.system(size: 10, weight: .bold))
                }
                .foregroundStyle(wsClient.snapshot.freezeLearning ? Color.black : MissionTheme.textPrimary)
                .padding(.horizontal, 9)
                .padding(.vertical, 6)
                .background(
                    wsClient.snapshot.freezeLearning ? MissionTheme.amber : MissionTheme.cardBg,
                    in: RoundedRectangle(cornerRadius: 7)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 7)
                        .stroke(wsClient.snapshot.freezeLearning ? MissionTheme.amber : MissionTheme.border, lineWidth: 1)
                )
            }
            .buttonStyle(.plain)
            .disabled(!wsClient.isConnected)
            .help("Freeze Learning queues corrections for post-service review without mutating live Ridge weights")

            // 1-Click Restore Previous Arming
            Button {
                Task { await wsClient.restorePreviousArming() }
            } label: {
                Label("Restore Arming", systemImage: "arrow.uturn.backward.circle.fill")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(MissionTheme.cyanTarget)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 6)
                    .background(MissionTheme.cardBg, in: RoundedRectangle(cornerRadius: 7))
                    .overlay(RoundedRectangle(cornerRadius: 7).stroke(MissionTheme.cyanTarget.opacity(0.45), lineWidth: 1))
            }
            .buttonStyle(.plain)
            .disabled(!wsClient.isConnected)
            .help("Restore previously armed channels from SQLite last_known_state in 1 click")

            Spacer()

            // Heartbeat LED + Diagnostics Button
            Button {
                showDiagnosticsDrawer = true
            } label: {
                HStack(spacing: 6) {
                    Circle()
                        .fill(heartbeatLEDColor)
                        .frame(width: 9, height: 9)
                    VStack(alignment: .leading, spacing: 0) {
                        Text(wsClient.heartbeatHealthy ? "ENGINE OK" : (wsClient.isConnected ? "STALE HB" : "OFFLINE"))
                            .font(.system(size: 9, weight: .black, design: .monospaced))
                            .foregroundStyle(MissionTheme.textPrimary)
                        Text(String(format: "p95 %.1fms • B:%@", wsClient.lastHeartbeat.loopP95Ms, bridgeClient.isConnected ? "OK" : "OFF"))
                            .font(.system(size: 8, design: .monospaced))
                            .foregroundStyle(MissionTheme.textSecondary)
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(MissionTheme.cardBg, in: RoundedRectangle(cornerRadius: 7))
                .overlay(RoundedRectangle(cornerRadius: 7).stroke(MissionTheme.border, lineWidth: 1))
            }
            .buttonStyle(.plain)
            .help("Open Engine & Bridge Diagnostics Drawer")

            // Utility Drawer Buttons
            HStack(spacing: 6) {
                toolbarIconButton(icon: "slider.horizontal.3", title: "Rules", badge: wsClient.snapshot.activeRules.count) {
                    showRulesDrawer = true
                }
                toolbarIconButton(icon: "tray.full", title: "Review", badge: wsClient.snapshot.queuedReviewCount) {
                    showReviewDrawer = true
                }
                toolbarIconButton(icon: "wrench.and.screwdriver", title: "Setup", badge: 0) {
                    showSetupWizard = true
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .background(MissionTheme.panelBg)
        .overlay(alignment: .bottom) {
            Divider().overlay(MissionTheme.border)
        }
    }

    // MARK: - Order-of-Service Stepper Bar

    private var orderOfServiceBar: some View {
        HStack(spacing: 8) {
            Text("ORDER OF SERVICE")
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundStyle(MissionTheme.textSecondary)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(Array(orderOfService.enumerated()), id: \.element.id) { idx, segment in
                        let isCurrent = idx == wsClient.snapshot.serviceStepIdx
                        let isPast = idx < wsClient.snapshot.serviceStepIdx
                        Button {
                            Task { await wsClient.advanceServiceStep(stepIdx: idx) }
                        } label: {
                            HStack(spacing: 5) {
                                Text("\(idx + 1)")
                                    .font(.system(size: 9, weight: .black, design: .monospaced))
                                    .foregroundStyle(isCurrent ? Color.black : MissionTheme.textSecondary)
                                    .frame(width: 16, height: 16)
                                    .background(
                                        isCurrent ? MissionTheme.cyanTarget : (isPast ? MissionTheme.emerald.opacity(0.35) : MissionTheme.canvasBg),
                                        in: Circle()
                                    )
                                Text(segment.title)
                                    .font(.system(size: 10, weight: isCurrent ? .bold : .medium))
                                    .foregroundStyle(isCurrent ? MissionTheme.textPrimary : MissionTheme.textSecondary)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(
                                isCurrent ? MissionTheme.cardBg : Color.clear,
                                in: Capsule()
                            )
                            .overlay(
                                Capsule().stroke(
                                    isCurrent ? MissionTheme.cyanTarget.opacity(0.7) : MissionTheme.border.opacity(0.6),
                                    lineWidth: 1
                                )
                            )
                        }
                        .buttonStyle(.plain)
                        .disabled(!wsClient.isConnected)
                    }
                }
            }

            Button {
                let nextIdx = min(orderOfService.count - 1, wsClient.snapshot.serviceStepIdx + 1)
                Task { await wsClient.advanceServiceStep(stepIdx: nextIdx) }
            } label: {
                Label("Next Segment", systemImage: "forward.fill")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(Color.black)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 4)
                    .background(MissionTheme.cyanTarget, in: Capsule())
            }
            .buttonStyle(.plain)
            .disabled(!wsClient.isConnected)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
        .background(MissionTheme.canvasBg)
        .overlay(alignment: .bottom) {
            Divider().overlay(MissionTheme.border)
        }
    }

    // MARK: - Warnings Strip

    private var warningsStrip: some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(MissionTheme.amber)
            Text(activeWarnings.joined(separator: "  •  "))
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundStyle(MissionTheme.amber)
            Spacer()
            Button("Open Setup Wizard") {
                showSetupWizard = true
            }
            .buttonStyle(.plain)
            .font(.system(size: 10, weight: .bold))
            .foregroundStyle(Color.black)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(MissionTheme.amber, in: Capsule())
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
        .background(MissionTheme.amber.opacity(0.14))
    }

    // MARK: - DISCONNECTED Banner w/ Cached Last Known Faders & Direct Bridge Bypass

    private var disconnectedManualControlBanner: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: "bolt.slash.fill")
                    .font(.system(size: 16))
                    .foregroundStyle(MissionTheme.crimson)

                VStack(alignment: .leading, spacing: 2) {
                    Text(
                        bridgeClient.isConnected
                            ? "ENGINE DISCONNECTED — DIRECT BRIDGE MANUAL CONTROL LIVE (< 2 ms)"
                            : "ENGINE & BRIDGE DISCONNECTED — CHECK LAUNCHAGENT OR RESTART"
                    )
                    .font(.system(size: 12, weight: .black, design: .monospaced))
                    .foregroundStyle(Color.white)

                    Text("Hardware controller & virtual faders below route straight through ~/.aethermix/bridge.sock. MainStage holds last fader values safely (never zeroed).")
                        .font(.system(size: 10))
                        .foregroundStyle(MissionTheme.textPrimary.opacity(0.85))
                }

                Spacer()

                Button {
                    bridgeClient.setBypass(latched: !bridgeClient.bridgeState.bypassLatched)
                } label: {
                    Text(bridgeClient.bridgeState.bypassLatched ? "Release Bridge Bypass" : "Latch Bridge Bypass")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(Color.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(
                            bridgeClient.bridgeState.bypassLatched ? MissionTheme.amber : MissionTheme.crimson,
                            in: RoundedRectangle(cornerRadius: 6)
                        )
                }
                .buttonStyle(.plain)

                Button {
                    processManager.restartEngine(autonomyMode: effectiveMode)
                    wsClient.connectOnce()
                } label: {
                    Label("Restart Engine", systemImage: "arrow.clockwise")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(Color.black)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(MissionTheme.cyanTarget, in: RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
            }

            // Cached Last Known Fader Positions Strip
            HStack(spacing: 8) {
                Text("LAST KNOWN FADERS:")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.textPrimary.opacity(0.8))

                ForEach(ChannelCatalog.defaultChannels) { ch in
                    let cached = wsClient.lastKnownFaders[ch.id]
                    let bridgeCC = bridgeClient.bridgeState.ccKey(channelId: ch.id, control: .vol)
                    let displayCC = bridgeCC ?? cached?.volCC ?? 0
                    let displayDb = cached?.volDb ?? TaperMath.ccToDb(displayCC, taper: TaperMath.defaultVolTaper)

                    HStack(spacing: 4) {
                        Text(ch.label.split(separator: " ").first.map(String.init) ?? ch.id)
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(MissionTheme.textSecondary)
                        Text(TaperMath.formatDb(displayDb))
                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                            .foregroundStyle(Color.white)
                    }
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(MissionTheme.canvasBg.opacity(0.75), in: RoundedRectangle(cornerRadius: 5))
                }
                Spacer()
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .background(MissionTheme.crimson.opacity(0.24))
        .overlay(alignment: .bottom) {
            Divider().overlay(MissionTheme.crimson.opacity(0.7))
        }
    }

    // MARK: - Helpers

    private var heartbeatLEDColor: Color {
        if wsClient.heartbeatHealthy && wsClient.lastHeartbeat.audioOk && wsClient.lastHeartbeat.bridgeOk {
            return MissionTheme.emerald
        }
        if wsClient.isConnected {
            return MissionTheme.amber
        }
        return MissionTheme.crimson
    }

    /// Enforces safe state-machine transition: jumping directly from `EMERGENCY_BYPASS`
    /// to `ACTIVE` is prevented (always steps through `SHADOW_OBSERVE` to clear bypass cleanly).
    nonisolated public static func safeModeTransition(
        from current: AutonomyMode,
        requested: AutonomyMode
    ) -> AutonomyMode {
        if current == .emergencyBypass && requested == .active {
            return .shadowObserve
        }
        return requested
    }

    private func switchAutonomyMode(_ requestedMode: AutonomyMode) {
        let mode = Self.safeModeTransition(from: effectiveMode, requested: requestedMode)
        if mode == .emergencyBypass {
            triggerEmergencyBypass()
        } else {
            bridgeClient.setBypass(latched: false)
            Task { await wsClient.setMode(mode) }
        }
    }

    private func triggerEmergencyBypass() {
        // Latch directly inside standalone MIDI bridge FIRST (< 1 ms, independent of AI engine)
        bridgeClient.setBypass(latched: true)
        Task { await wsClient.setMode(.emergencyBypass) }
    }

    private func toolbarIconButton(icon: String, title: String, badge: Int, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 11))
                Text(title)
                    .font(.system(size: 10, weight: .semibold))
                if badge > 0 {
                    Text("\(badge)")
                        .font(.system(size: 9, weight: .black, design: .monospaced))
                        .foregroundStyle(Color.black)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(MissionTheme.amber, in: Capsule())
                }
            }
            .foregroundStyle(MissionTheme.textPrimary)
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(MissionTheme.cardBg, in: RoundedRectangle(cornerRadius: 7))
            .overlay(RoundedRectangle(cornerRadius: 7).stroke(MissionTheme.border, lineWidth: 1))
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Diagnostics & Safe-Update Drawer

private struct DiagnosticsDrawerView: View {
    let wsClient: EngineWebSocketClient
    let bridgeClient: BridgeSocketClient
    let processManager: EngineProcessManager
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Engine & Bridge Diagnostics")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(MissionTheme.textPrimary)
                Spacer()
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }

            // Subsystem Telemetry Row
            HStack(spacing: 10) {
                subsystemPill("Audio Tap", ok: wsClient.lastHeartbeat.audioOk)
                subsystemPill("MIDI Bridge", ok: bridgeClient.isConnected || wsClient.lastHeartbeat.bridgeOk)
                subsystemPill("CLAP Subproc", ok: wsClient.lastHeartbeat.clapOk)
                subsystemPill("Ollama LLM", ok: wsClient.lastHeartbeat.ollamaOk)
                Spacer()
                Text(String(format: "p95: %.2f ms  •  p99: %.2f ms", wsClient.lastHeartbeat.loopP95Ms, wsClient.lastHeartbeat.loopP99Ms))
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.cyanTarget)
            }
            .padding(10)
            .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 8))

            // Process & Safe-Update Controls
            HStack(spacing: 10) {
                Text("Engine Mode: \(processManager.launchMode.rawValue)")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(MissionTheme.textPrimary)
                if let pid = processManager.enginePID {
                    Text("PID \(pid)")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(MissionTheme.textSecondary)
                }

                Spacer()

                Button("Install Bridge LaunchAgent") {
                    processManager.installBridgeLaunchAgent()
                }

                Button("Check for Updates") {
                    _ = processManager.requestUpdateAction("Check for Updates", autonomyMode: wsClient.snapshot.autonomyMode)
                }
                .disabled(wsClient.snapshot.autonomyMode == .active)
                .help("Blocked while Autonomy Mode is ACTIVE (Safe-Update Lock)")
            }

            if let alert = processManager.safeUpdateAlert {
                Text(alert)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(MissionTheme.amber)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(MissionTheme.amber.opacity(0.14), in: RoundedRectangle(cornerRadius: 6))
            }

            // Live stdout/stderr Log Console
            HStack {
                Text("ENGINE PROCESS LOGS (\(processManager.diagnosticLogs.count))")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.textSecondary)
                Spacer()
                Button("Clear") { processManager.clearLogs() }
                    .font(.system(size: 10))
            }

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 3) {
                    ForEach(processManager.diagnosticLogs) { entry in
                        Text("[\(entry.stream)] \(entry.line)")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(entry.stream == "stderr" ? MissionTheme.crimson : MissionTheme.textSecondary)
                            .textSelection(.enabled)
                    }
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minHeight: 240)
            .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(MissionTheme.border, lineWidth: 1))
        }
        .padding(18)
        .frame(minWidth: 720, maxWidth: 720, minHeight: 500)
        .background(MissionTheme.canvasBg)
        .preferredColorScheme(.dark)
    }

    private func subsystemPill(_ label: String, ok: Bool) -> some View {
        HStack(spacing: 5) {
            Circle().fill(ok ? MissionTheme.emerald : MissionTheme.crimson).frame(width: 7, height: 7)
            Text(label)
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(MissionTheme.textPrimary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(MissionTheme.canvasBg, in: Capsule())
    }
}
