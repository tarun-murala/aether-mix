import SwiftUI

// MARK: - Shared Dark Live-Sound Mission Control Theme

public enum MissionTheme {
    public static let canvasBg = Color(red: 0.043, green: 0.055, blue: 0.078)       // #0B0E14
    public static let panelBg = Color(red: 0.078, green: 0.098, blue: 0.137)        // #141923
    public static let cardBg = Color(red: 0.102, green: 0.129, blue: 0.180)         // #1A212E
    public static let border = Color(red: 0.149, green: 0.192, blue: 0.267)         // #263144
    public static let cyanTarget = Color(red: 0.133, green: 0.827, blue: 0.933)     // #22D3EE
    public static let emerald = Color(red: 0.063, green: 0.725, blue: 0.506)        // #10B981
    public static let amber = Color(red: 0.961, green: 0.620, blue: 0.043)          // #F59E0B
    public static let crimson = Color(red: 0.937, green: 0.267, blue: 0.267)        // #EF4444
    public static let indigo = Color(red: 0.388, green: 0.400, blue: 0.945)         // #6366F1
    public static let textPrimary = Color(red: 0.945, green: 0.961, blue: 0.980)
    public static let textSecondary = Color(red: 0.580, green: 0.639, blue: 0.722)

    public static func phaseColor(_ phase: ServicePhase) -> Color {
        switch phase {
        case .spokenWord: return cyanTarget
        case .prayerUnderscore: return indigo
        case .worshipLow: return emerald
        case .worshipHigh: return amber
        case .transition: return textSecondary
        }
    }

    public static func modeColor(_ mode: AutonomyMode) -> Color {
        switch mode {
        case .shadowObserve: return amber
        case .active: return emerald
        case .emergencyBypass: return crimson
        }
    }

    public static func categoryColor(_ category: BrainCategory) -> Color {
        switch category {
        case .perception: return cyanTarget
        case .action: return emerald
        case .safety: return crimson
        case .llm: return indigo
        case .rlhf: return amber
        }
    }
}

// MARK: - Dedicated Multi-Channel 30–60 Hz Meter Bridge (SwiftUI Canvas)

/// High-frequency meter bridge driven strictly by isolated `MeterStore`.
/// Redraws inside a single GPU-accelerated SwiftUI `Canvas` at 30–60 Hz without
/// invalidating surrounding dashboard views.
public struct MeterCanvasView: View {
    public let meterStore: MeterStore
    public let speechOpenByChannel: [String: Bool]
    public let instrumentOpenByChannel: [String: Bool]

    public init(
        meterStore: MeterStore,
        speechOpenByChannel: [String: Bool] = [:],
        instrumentOpenByChannel: [String: Bool] = [:]
    ) {
        self.meterStore = meterStore
        self.speechOpenByChannel = speechOpenByChannel
        self.instrumentOpenByChannel = instrumentOpenByChannel
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Label("REAL-TIME PRE-FADER METER BRIDGE", systemImage: "waveform.path.ecg")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(MissionTheme.textSecondary)

                Spacer()

                HStack(spacing: 10) {
                    legendDot(color: MissionTheme.cyanTarget, label: "VAD / Speech")
                    legendDot(color: MissionTheme.emerald, label: "Inst Open")
                    legendDot(color: MissionTheme.crimson, label: "Clip")

                    Text(String(format: "MASTER %.1f LUFS", meterStore.masterLufs))
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundStyle(lufsColor(meterStore.masterLufs))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(MissionTheme.canvasBg, in: RoundedRectangle(cornerRadius: 4))
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(MissionTheme.border, lineWidth: 1))

                    Button("Clear Clips") {
                        meterStore.resetClipIndicators()
                    }
                    .buttonStyle(.plain)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(MissionTheme.textSecondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(MissionTheme.cardBg, in: RoundedRectangle(cornerRadius: 4))
                }
            }

            Canvas { context, size in
                let _ = meterStore.frameCounter
                let readings = meterStore.orderedReadings
                guard !readings.isEmpty else { return }

                let count = readings.count
                let leftMargin: CGFloat = 34
                let rightMargin: CGFloat = 10
                let topMargin: CGFloat = 14
                let bottomMargin: CGFloat = 22
                let usableWidth = max(1, size.width - leftMargin - rightMargin)
                let meterHeight = max(1, size.height - topMargin - bottomMargin)

                // Draw dBFS horizontal reference grid lines (-48, -36, -24, -12, -6, 0 dBFS)
                let ticks: [Double] = [0, -6, -12, -24, -36, -48]
                for db in ticks {
                    let y = topMargin + meterHeight * CGFloat(1.0 - normalizedDb(db))
                    var gridPath = Path()
                    gridPath.move(to: CGPoint(x: leftMargin, y: y))
                    gridPath.addLine(to: CGPoint(x: size.width - rightMargin, y: y))
                    context.stroke(
                        gridPath,
                        with: .color(MissionTheme.border.opacity(db == -12 ? 0.85 : 0.45)),
                        lineWidth: db == -12 ? 1.0 : 0.5
                    )

                    let labelText = Text(String(format: "%d", Int(db)))
                        .font(.system(size: 8, weight: .medium, design: .monospaced))
                        .foregroundStyle(MissionTheme.textSecondary)
                    context.draw(labelText, at: CGPoint(x: 16, y: y), anchor: .center)
                }

                let slotWidth = usableWidth / CGFloat(count)
                let barWidth = min(28.0, max(12.0, slotWidth * 0.42))

                for (index, reading) in readings.enumerated() {
                    let centerX = leftMargin + slotWidth * (CGFloat(index) + 0.5)
                    let barX = centerX - barWidth / 2.0

                    // Meter track background
                    let trackRect = CGRect(x: barX, y: topMargin, width: barWidth, height: meterHeight)
                    context.fill(
                        Path(roundedRect: trackRect, cornerRadius: 3),
                        with: .color(MissionTheme.canvasBg)
                    )

                    // RMS gradient bar
                    let rmsNorm = CGFloat(normalizedDb(reading.rmsDb))
                    let fillHeight = meterHeight * rmsNorm
                    if fillHeight > 0.5 {
                        let fillRect = CGRect(
                            x: barX + 1,
                            y: topMargin + meterHeight - fillHeight,
                            width: barWidth - 2,
                            height: fillHeight
                        )
                        let gradient = Gradient(stops: [
                            .init(color: MissionTheme.emerald, location: 0.0),
                            .init(color: MissionTheme.emerald, location: 0.65),
                            .init(color: MissionTheme.amber, location: 0.85),
                            .init(color: MissionTheme.crimson, location: 1.0)
                        ])
                        context.fill(
                            Path(roundedRect: fillRect, cornerRadius: 2),
                            with: .linearGradient(
                                gradient,
                                startPoint: CGPoint(x: centerX, y: topMargin + meterHeight),
                                endPoint: CGPoint(x: centerX, y: topMargin)
                            )
                        )
                    }

                    // Peak hold tick
                    let peakNorm = CGFloat(normalizedDb(reading.peakHoldDb))
                    if peakNorm > 0.01 {
                        let peakY = topMargin + meterHeight * (1.0 - peakNorm)
                        let tickRect = CGRect(x: barX, y: peakY - 1, width: barWidth, height: 2.2)
                        let tickColor = reading.peakHoldDb >= -3.0 ? MissionTheme.crimson : Color.white
                        context.fill(Path(tickRect), with: .color(tickColor))
                    }

                    // Clip LED above track
                    let clipRect = CGRect(x: barX, y: 3, width: barWidth, height: 6)
                    context.fill(
                        Path(roundedRect: clipRect, cornerRadius: 2),
                        with: .color(reading.clip ? MissionTheme.crimson : MissionTheme.border.opacity(0.45))
                    )

                    // VAD probability mini-bar to the right of dBFS bar
                    let vadBarX = barX + barWidth + 3
                    let vadHeight = meterHeight * CGFloat(max(0, min(1, reading.vad)))
                    let vadTrack = CGRect(x: vadBarX, y: topMargin, width: 4, height: meterHeight)
                    context.fill(Path(roundedRect: vadTrack, cornerRadius: 2), with: .color(MissionTheme.canvasBg))
                    if vadHeight > 0.5 {
                        let vadFill = CGRect(
                            x: vadBarX,
                            y: topMargin + meterHeight - vadHeight,
                            width: 4,
                            height: vadHeight
                        )
                        context.fill(
                            Path(roundedRect: vadFill, cornerRadius: 2),
                            with: .color(MissionTheme.cyanTarget)
                        )
                    }

                    // Reflex LEDs (speech_open & instrument_open) + channel short label
                    let speechOpen = speechOpenByChannel[reading.channelId] ?? false
                    let instOpen = instrumentOpenByChannel[reading.channelId] ?? false

                    let ledY = topMargin + meterHeight + 6
                    if speechOpen || instOpen {
                        let dotColor = speechOpen ? MissionTheme.cyanTarget : MissionTheme.emerald
                        let dotRect = CGRect(x: barX - 8, y: ledY - 3, width: 6, height: 6)
                        context.fill(Path(ellipseIn: dotRect), with: .color(dotColor))
                    }

                    let shortName = ChannelCatalog.descriptor(for: reading.channelId).label
                        .replacingOccurrences(of: "Backup Vocal ", with: "BV")
                        .replacingOccurrences(of: "Drums / Bass / Tracks", with: "Drums")
                    let caption = Text(shortName)
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(MissionTheme.textSecondary)
                    context.draw(caption, at: CGPoint(x: centerX, y: topMargin + meterHeight + 11), anchor: .center)
                }
            }
            .frame(height: 112)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(MissionTheme.panelBg, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(MissionTheme.border, lineWidth: 1))
    }

    private func normalizedDb(_ db: Double) -> Double {
        let clamped = max(-60.0, min(0.0, db))
        return (clamped + 60.0) / 60.0
    }

    private func lufsColor(_ lufs: Double) -> Color {
        if lufs > -10.0 { return MissionTheme.crimson }
        if lufs > -15.0 { return MissionTheme.amber }
        return MissionTheme.emerald
    }

    private func legendDot(color: Color, label: String) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(MissionTheme.textSecondary)
        }
    }
}

// MARK: - Compact Per-Channel Inline Meter Canvas (Used Inside Mixer Strips)

public struct SingleChannelMeterCanvas: View {
    public let meterStore: MeterStore
    public let channelId: String

    public init(meterStore: MeterStore, channelId: String) {
        self.meterStore = meterStore
        self.channelId = channelId
    }

    public var body: some View {
        Canvas { context, size in
            let _ = meterStore.frameCounter
            let reading = meterStore.reading(for: channelId)

            // Clip LED at top
            let clipRect = CGRect(x: 0, y: 0, width: size.width, height: 5)
            context.fill(
                Path(roundedRect: clipRect, cornerRadius: 2),
                with: .color(reading.clip ? MissionTheme.crimson : MissionTheme.border.opacity(0.4))
            )

            let topY: CGFloat = 8
            let meterH = max(1, size.height - topY)
            let trackRect = CGRect(x: 0, y: topY, width: size.width, height: meterH)
            context.fill(Path(roundedRect: trackRect, cornerRadius: 3), with: .color(MissionTheme.canvasBg))

            let normRms = CGFloat(max(0.0, min(1.0, (reading.rmsDb + 60.0) / 60.0)))
            let fillH = meterH * normRms
            if fillH > 0.5 {
                let fillRect = CGRect(x: 1, y: topY + meterH - fillH, width: size.width - 2, height: fillH)
                let gradient = Gradient(stops: [
                    .init(color: MissionTheme.emerald, location: 0.0),
                    .init(color: MissionTheme.emerald, location: 0.65),
                    .init(color: MissionTheme.amber, location: 0.85),
                    .init(color: MissionTheme.crimson, location: 1.0)
                ])
                context.fill(
                    Path(roundedRect: fillRect, cornerRadius: 2),
                    with: .linearGradient(
                        gradient,
                        startPoint: CGPoint(x: size.width / 2, y: topY + meterH),
                        endPoint: CGPoint(x: size.width / 2, y: topY)
                    )
                )
            }

            let normPeak = CGFloat(max(0.0, min(1.0, (reading.peakHoldDb + 60.0) / 60.0)))
            if normPeak > 0.01 {
                let py = topY + meterH * (1.0 - normPeak)
                context.fill(
                    Path(CGRect(x: 0, y: py - 1, width: size.width, height: 2)),
                    with: .color(reading.peakHoldDb >= -3.0 ? MissionTheme.crimson : Color.white)
                )
            }
        }
    }
}
