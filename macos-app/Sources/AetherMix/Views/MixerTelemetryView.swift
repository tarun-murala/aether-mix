import SwiftUI

/// Per-channel Live Sound Mixer Strip Deck.
/// Every volume & reverb dual-ring knob drags directly through `BridgeSocketClient.sendManualCC`
/// so manual control stays 100% responsive even if the Python AI engine is offline.
public struct MixerTelemetryView: View {
    public let wsClient: EngineWebSocketClient
    public let bridgeClient: BridgeSocketClient

    public init(wsClient: EngineWebSocketClient, bridgeClient: BridgeSocketClient) {
        self.wsClient = wsClient
        self.bridgeClient = bridgeClient
    }

    private var channelIds: [String] {
        let ids = wsClient.snapshot.channelIds
        return ids.isEmpty ? ChannelCatalog.defaultChannelIds : ids
    }

    public var body: some View {
        VStack(spacing: 10) {
            // Dedicated 30–60 Hz Canvas meter bridge
            MeterCanvasView(
                meterStore: wsClient.meterStore,
                speechOpenByChannel: speechOpenMap,
                instrumentOpenByChannel: instrumentOpenMap
            )

            // Per-channel console strips
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(alignment: .top, spacing: 10) {
                    ForEach(channelIds, id: \.self) { channelId in
                        ChannelStripCard(
                            channelId: channelId,
                            descriptor: ChannelCatalog.descriptor(for: channelId),
                            state: wsClient.snapshot.channels[channelId] ?? ChannelRuntimeState(),
                            bridgeState: bridgeClient.bridgeState,
                            meterStore: wsClient.meterStore,
                            engineConnected: wsClient.isConnected,
                            onToggleArm: { newArmed in
                                Task { await wsClient.setChannelArmed(channelId: channelId, armed: newArmed) }
                            },
                            onConfirmSync: {
                                Task { await wsClient.syncChannel(channelId: channelId) }
                            },
                            onReleaseOffset: { control in
                                Task { await wsClient.releaseHumanOffset(channelId: channelId, control: control) }
                            },
                            onManualCC: { control, cc in
                                // Direct Bridge Unix Socket write — works even when AI engine is dead!
                                bridgeClient.sendManualCC(channelId: channelId, control: control, cc: cc)
                            }
                        )
                    }
                }
                .padding(.vertical, 2)
            }
        }
    }

    private var speechOpenMap: [String: Bool] {
        wsClient.snapshot.channels.mapValues(\.speechOpen)
    }

    private var instrumentOpenMap: [String: Bool] {
        wsClient.snapshot.channels.mapValues(\.instrumentOpen)
    }
}

// MARK: - Single Channel Console Strip

private struct ChannelStripCard: View {
    let channelId: String
    let descriptor: ChannelDescriptor
    let state: ChannelRuntimeState
    let bridgeState: BridgeStatePayload
    let meterStore: MeterStore
    let engineConnected: Bool
    let onToggleArm: (Bool) -> Void
    let onConfirmSync: () -> Void
    let onReleaseOffset: (ControlType) -> Void
    let onManualCC: (ControlType, Int) -> Void

    private var volPickupDir: Int {
        let bridgeDir = bridgeState.pickupDirection(channelId: channelId, control: .vol)
        return bridgeDir != 0 ? bridgeDir : state.volPickupDir
    }

    private var reverbPickupDir: Int {
        let bridgeDir = bridgeState.pickupDirection(channelId: channelId, control: .reverb)
        return bridgeDir != 0 ? bridgeDir : state.reverbPickupDir
    }

    var body: some View {
        VStack(spacing: 8) {
            // Header: Channel Name + Role
            VStack(spacing: 2) {
                Text(descriptor.label)
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(MissionTheme.textPrimary)
                    .lineLimit(1)
                Text(descriptor.role.shortLabel)
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.textSecondary)
            }

            // Arm Toggle + Sync Badge
            HStack(spacing: 5) {
                Button {
                    onToggleArm(!state.armed)
                } label: {
                    Text(state.armed ? "ARMED" : "SAFE")
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundStyle(state.armed ? Color.black : MissionTheme.textSecondary)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(
                            state.armed ? MissionTheme.emerald : MissionTheme.canvasBg,
                            in: Capsule()
                        )
                        .overlay(
                            Capsule().stroke(state.armed ? MissionTheme.emerald : MissionTheme.border, lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)
                .disabled(!engineConnected)

                if state.syncStatus == .synced {
                    Text("SYNCED")
                        .font(.system(size: 8, weight: .bold, design: .monospaced))
                        .foregroundStyle(MissionTheme.emerald)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 3)
                        .background(MissionTheme.emerald.opacity(0.14), in: Capsule())
                } else {
                    Button {
                        onConfirmSync()
                    } label: {
                        Text("SYNC?")
                            .font(.system(size: 8, weight: .bold, design: .monospaced))
                            .foregroundStyle(Color.black)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 3)
                            .background(MissionTheme.amber, in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .help("Confirm & Sync channel with 2.0 dB/s handover glide")
                    .disabled(!engineConnected)
                }
            }

            // Reflex & Feedback Safety Badges
            HStack(spacing: 4) {
                if state.speechOpen {
                    badgePill("SPEECH OPEN", color: MissionTheme.cyanTarget)
                }
                if state.instrumentOpen {
                    badgePill("INST OPEN", color: MissionTheme.emerald)
                }
                if let fbCeil = state.feedbackCeilingDb {
                    badgePill(String(format: "FB %.0fdB", fbCeil), color: MissionTheme.crimson)
                }
            }
            .frame(minHeight: 16)

            // Main Control Area: Inline Canvas Meter + Live 50Hz Dual-Ring Volume Knob
            HStack(alignment: .center, spacing: 10) {
                SingleChannelMeterCanvas(meterStore: meterStore, channelId: channelId)
                    .frame(width: 12, height: 118)

                LiveVolumeKnobSection(
                    channelId: channelId,
                    meterStore: meterStore,
                    fallbackTargetCC: state.volTargetCC,
                    fallbackActualCC: state.volActualCC,
                    bridgeActualCC: bridgeState.ccKey(channelId: channelId, control: .vol),
                    engineConnected: engineConnected,
                    pickupDir: volPickupDir,
                    onManualCC: { cc in onManualCC(.vol, cc) }
                )
            }

            // Volume Touch-Lock / Human Offset Banner
            if state.volTouchLockS > 0.05 || abs(state.volHumanOffsetDb) > 0.05 {
                HumanOffsetPill(
                    label: "VOL",
                    lockSeconds: state.volTouchLockS,
                    offsetDb: state.volHumanOffsetDb,
                    onRelease: { onReleaseOffset(.vol) }
                )
            }

            // Optional Reverb Dual-Ring Send Control (50Hz live via MeterStore)
            if descriptor.hasReverb || state.hasReverb {
                Divider().overlay(MissionTheme.border)

                LiveReverbKnobSection(
                    channelId: channelId,
                    meterStore: meterStore,
                    fallbackTargetCC: state.reverbTargetCC ?? state.reverbActualCC ?? 0,
                    fallbackActualCC: state.reverbActualCC ?? 0,
                    bridgeActualCC: bridgeState.ccKey(channelId: channelId, control: .reverb),
                    engineConnected: engineConnected,
                    pickupDir: reverbPickupDir,
                    onManualCC: { cc in onManualCC(.reverb, cc) }
                )

                if state.reverbTouchLockS > 0.05 || abs(state.reverbHumanOffsetDb) > 0.05 {
                    HumanOffsetPill(
                        label: "REV",
                        lockSeconds: state.reverbTouchLockS,
                        offsetDb: state.reverbHumanOffsetDb,
                        onRelease: { onReleaseOffset(.reverb) }
                    )
                }
            } else {
                Spacer(minLength: 0)
            }
        }
        .padding(10)
        .frame(minWidth: 152, maxWidth: 152, minHeight: 330)
        .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(borderStrokeColor, lineWidth: state.feedbackCeilingDb != nil ? 1.8 : 1.0)
        )
    }

    private var borderStrokeColor: Color {
        if state.feedbackCeilingDb != nil { return MissionTheme.crimson }
        if state.volTouchLockS > 0.05 || state.reverbTouchLockS > 0.05 { return MissionTheme.amber }
        if state.armed { return MissionTheme.emerald.opacity(0.6) }
        return MissionTheme.border
    }

    private func badgePill(_ title: String, color: Color) -> some View {
        Text(title)
            .font(.system(size: 8, weight: .bold, design: .monospaced))
            .foregroundStyle(Color.black)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(color, in: Capsule())
    }
}

// MARK: - Live 50Hz Isolated Volume & Reverb Knob Sections (Observe Only MeterStore)

private struct LiveVolumeKnobSection: View {
    let channelId: String
    let meterStore: MeterStore
    let fallbackTargetCC: Int
    let fallbackActualCC: Int
    let bridgeActualCC: Int?
    let engineConnected: Bool
    let pickupDir: Int
    let onManualCC: (Int) -> Void

    private var targetCC: Int {
        if engineConnected && meterStore.frameCounter > 0 {
            return meterStore.reading(for: channelId).volTargetCC
        }
        return fallbackTargetCC
    }

    private var actualCC: Int {
        if engineConnected && meterStore.frameCounter > 0 {
            return meterStore.reading(for: channelId).volActualCC
        }
        if !engineConnected, let bridgeActualCC {
            return bridgeActualCC
        }
        return fallbackActualCC
    }

    var body: some View {
        VStack(spacing: 6) {
            DualRingKnobControl(
                title: "VOL",
                targetCC: targetCC,
                actualCC: actualCC,
                taper: TaperMath.defaultVolTaper,
                pickupDir: pickupDir,
                size: 74,
                onDragCC: onManualCC
            )

            Text(TaperMath.formatDb(TaperMath.ccToDb(actualCC, taper: TaperMath.defaultVolTaper)))
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundStyle(MissionTheme.textPrimary)

            Text("CC \(actualCC) → \(targetCC)")
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .foregroundStyle(MissionTheme.cyanTarget)
        }
    }
}

private struct LiveReverbKnobSection: View {
    let channelId: String
    let meterStore: MeterStore
    let fallbackTargetCC: Int
    let fallbackActualCC: Int
    let bridgeActualCC: Int?
    let engineConnected: Bool
    let pickupDir: Int
    let onManualCC: (Int) -> Void

    private var targetCC: Int {
        if engineConnected && meterStore.frameCounter > 0,
           let live = meterStore.reading(for: channelId).reverbTargetCC {
            return live
        }
        return fallbackTargetCC
    }

    private var actualCC: Int {
        if engineConnected && meterStore.frameCounter > 0,
           let live = meterStore.reading(for: channelId).reverbActualCC {
            return live
        }
        if !engineConnected, let bridgeActualCC {
            return bridgeActualCC
        }
        return fallbackActualCC
    }

    var body: some View {
        HStack(spacing: 8) {
            DualRingKnobControl(
                title: "REV",
                targetCC: targetCC,
                actualCC: actualCC,
                taper: TaperMath.defaultReverbTaper,
                pickupDir: pickupDir,
                size: 52,
                onDragCC: onManualCC
            )

            VStack(alignment: .leading, spacing: 2) {
                Text("REVERB")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.textSecondary)
                Text(TaperMath.formatDb(TaperMath.ccToDb(actualCC, taper: TaperMath.defaultReverbTaper)))
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.textPrimary)
                Text("CC \(actualCC) → \(targetCC)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(MissionTheme.cyanTarget)
            }
        }
    }
}

// MARK: - Human Touch-Lock Countdown + Release Offset Pill

private struct HumanOffsetPill: View {
    let label: String
    let lockSeconds: Double
    let offsetDb: Double
    let onRelease: () -> Void

    var body: some View {
        VStack(spacing: 3) {
            HStack(spacing: 4) {
                Image(systemName: lockSeconds > 0.05 ? "hand.raised.fill" : "slider.horizontal.3")
                    .font(.system(size: 9))
                if lockSeconds > 0.05 {
                    Text(String(format: "%@ LOCK %.1fs", label, lockSeconds))
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                } else {
                    Text(String(format: "%@ %+.1f dB", label, offsetDb))
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                }
            }
            .foregroundStyle(MissionTheme.amber)

            Button("Release Offset") {
                onRelease()
            }
            .buttonStyle(.plain)
            .font(.system(size: 9, weight: .semibold))
            .foregroundStyle(MissionTheme.textPrimary)
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            .background(MissionTheme.amber.opacity(0.22), in: Capsule())
            .overlay(Capsule().stroke(MissionTheme.amber.opacity(0.6), lineWidth: 0.8))
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 6)
        .frame(maxWidth: .infinity)
        .background(MissionTheme.amber.opacity(0.1), in: RoundedRectangle(cornerRadius: 6))
    }
}

// MARK: - Dual-Ring Volume / Reverb Knob (Cyan Target vs White Actual)

public struct DualRingKnobControl: View {
    public let title: String
    public let targetCC: Int
    public let actualCC: Int
    public let taper: [TaperPoint]
    public let pickupDir: Int
    public let size: CGFloat
    public let onDragCC: (Int) -> Void

    @State private var dragStartCC: Int?

    public init(
        title: String,
        targetCC: Int,
        actualCC: Int,
        taper: [TaperPoint],
        pickupDir: Int = 0,
        size: CGFloat = 72,
        onDragCC: @escaping (Int) -> Void
    ) {
        self.title = title
        self.targetCC = targetCC
        self.actualCC = actualCC
        self.taper = taper
        self.pickupDir = pickupDir
        self.size = size
        self.onDragCC = onDragCC
    }

    private var targetFraction: CGFloat {
        CGFloat(max(0, min(127, targetCC))) / 127.0
    }

    private var actualFraction: CGFloat {
        CGFloat(max(0, min(127, actualCC))) / 127.0
    }

    public var body: some View {
        ZStack {
            // Outer background arc (270 degrees: 135° .. 405°)
            ArcShape(fraction: 1.0)
                .stroke(MissionTheme.border.opacity(0.7), style: StrokeStyle(lineWidth: 5, lineCap: .round))

            // Outer Cyan Target Ring (AI target)
            ArcShape(fraction: targetFraction)
                .stroke(MissionTheme.cyanTarget, style: StrokeStyle(lineWidth: 5, lineCap: .round))

            // Inner background arc
            ArcShape(fraction: 1.0)
                .inset(by: 8)
                .stroke(MissionTheme.canvasBg, style: StrokeStyle(lineWidth: 4.5, lineCap: .round))

            // Inner White Actual Ring (MainStage actual CC)
            ArcShape(fraction: actualFraction)
                .inset(by: 8)
                .stroke(Color.white, style: StrokeStyle(lineWidth: 4.5, lineCap: .round))

            // Center label + pickup direction cue
            VStack(spacing: 1) {
                if pickupDir > 0 {
                    Text("▲ UP")
                        .font(.system(size: 8, weight: .black, design: .monospaced))
                        .foregroundStyle(MissionTheme.amber)
                } else if pickupDir < 0 {
                    Text("▼ DN")
                        .font(.system(size: 8, weight: .black, design: .monospaced))
                        .foregroundStyle(MissionTheme.amber)
                } else {
                    Text(title)
                        .font(.system(size: size >= 64 ? 10 : 8, weight: .bold, design: .monospaced))
                        .foregroundStyle(MissionTheme.textSecondary)
                }
            }
        }
        .frame(width: size, height: size)
        .contentShape(Circle())
        .gesture(
            DragGesture(minimumDistance: 1)
                .onChanged { value in
                    if dragStartCC == nil {
                        dragStartCC = actualCC
                    }
                    let base = dragStartCC ?? actualCC
                    // Dragging up increases CC, dragging down decreases CC (~140pt full travel)
                    let deltaCC = Int((-value.translation.height / 1.1).rounded())
                    let newCC = max(0, min(127, base + deltaCC))
                    onDragCC(newCC)
                }
                .onEnded { _ in
                    dragStartCC = nil
                }
        )
        .help("Drag vertically to send direct MIDI CC via standalone bridge (Cyan = AI Target, White = Actual)")
    }
}

private struct ArcShape: InsettableShape {
    let fraction: CGFloat
    var insetAmount: CGFloat = 0

    func inset(by amount: CGFloat) -> ArcShape {
        var copy = self
        copy.insetAmount += amount
        return copy
    }

    func path(in rect: CGRect) -> Path {
        let insetRect = rect.insetBy(dx: insetAmount + 3, dy: insetAmount + 3)
        let center = CGPoint(x: insetRect.midX, y: insetRect.midY)
        let radius = max(1, min(insetRect.width, insetRect.height) / 2.0)
        let startDeg: Double = 135.0
        let sweepDeg: Double = 270.0 * Double(max(0.0, min(1.0, fraction)))
        var path = Path()
        path.addArc(
            center: center,
            radius: radius,
            startAngle: .degrees(startDeg),
            endAngle: .degrees(startDeg + sweepDeg),
            clockwise: false
        )
        return path
    }
}
