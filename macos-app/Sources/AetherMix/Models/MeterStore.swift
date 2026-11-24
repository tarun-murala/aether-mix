import Foundation
import Observation

/// Per-channel high-frequency meter reading isolated from main dashboard state.
public struct ChannelMeterReading: Sendable, Equatable, Identifiable {
    public let channelId: String
    public var rmsDb: Double
    public var peakDb: Double
    public var peakHoldDb: Double
    public var peakHoldTimestamp: TimeInterval
    public var clip: Bool
    public var clipHoldUntil: TimeInterval
    public var vad: Double
    public var volTargetCC: Int
    public var volActualCC: Int
    public var reverbTargetCC: Int?
    public var reverbActualCC: Int?

    public var id: String { channelId }

    public init(
        channelId: String,
        rmsDb: Double = -60.0,
        peakDb: Double = -60.0,
        peakHoldDb: Double = -60.0,
        peakHoldTimestamp: TimeInterval = 0,
        clip: Bool = false,
        clipHoldUntil: TimeInterval = 0,
        vad: Double = 0.0,
        volTargetCC: Int = 0,
        volActualCC: Int = 0,
        reverbTargetCC: Int? = nil,
        reverbActualCC: Int? = nil
    ) {
        self.channelId = channelId
        self.rmsDb = rmsDb
        self.peakDb = peakDb
        self.peakHoldDb = peakHoldDb
        self.peakHoldTimestamp = peakHoldTimestamp
        self.clip = clip
        self.clipHoldUntil = clipHoldUntil
        self.vad = vad
        self.volTargetCC = volTargetCC
        self.volActualCC = volActualCC
        self.reverbTargetCC = reverbTargetCC
        self.reverbActualCC = reverbActualCC
    }
}

/// Isolated 30–60 Hz meter store keyed by `StateSnapshotPayload.channelIds`.
/// Only `MeterCanvasView` observes this object so high-frequency `meter_frame`
/// traffic never triggers SwiftUI layout diffing of the full dashboard.
@MainActor
@Observable
public final class MeterStore {
    public private(set) var channelIds: [String]
    public private(set) var readingsById: [String: ChannelMeterReading]
    public private(set) var masterLufs: Double = -60.0
    public private(set) var frameCounter: UInt64 = 0
    public private(set) var lastFrameTime: TimeInterval = 0

    /// Peak hold duration before gradual decay (seconds).
    public let peakHoldDuration: TimeInterval = 1.5
    /// Peak hold falloff rate once hold expires (dB/sec).
    public let peakFalloffDbPerSec: Double = 14.0
    /// Clip LED latch duration (seconds).
    public let clipHoldDuration: TimeInterval = 2.0

    public init(channelIds: [String] = ChannelCatalog.defaultChannelIds) {
        self.channelIds = channelIds
        var map: [String: ChannelMeterReading] = [:]
        for id in channelIds {
            map[id] = ChannelMeterReading(channelId: id)
        }
        self.readingsById = map
    }

    public func updateChannelOrder(_ newChannelIds: [String]) {
        guard !newChannelIds.isEmpty, newChannelIds != channelIds else { return }
        self.channelIds = newChannelIds
        var updated: [String: ChannelMeterReading] = [:]
        for id in newChannelIds {
            updated[id] = readingsById[id] ?? ChannelMeterReading(channelId: id)
        }
        self.readingsById = updated
    }

    public func applyMeterFrame(
        _ frame: MeterFramePayload,
        now: TimeInterval = Date().timeIntervalSinceReferenceDate
    ) {
        let dt = lastFrameTime > 0 ? max(0.0, min(0.5, now - lastFrameTime)) : 0.02
        var updated = readingsById

        for (idx, channelId) in channelIds.enumerated() {
            var current = updated[channelId] ?? ChannelMeterReading(channelId: channelId)

            let rms = idx < frame.rmsDb.count ? clampDb(frame.rmsDb[idx]) : -60.0
            let peak = idx < frame.peakDb.count ? clampDb(frame.peakDb[idx]) : -60.0
            let clipFlag = idx < frame.clip.count ? frame.clip[idx] : false
            let vadProb = idx < frame.vad.count ? max(0.0, min(1.0, frame.vad[idx])) : 0.0
            let volTarget = idx < frame.volTargetCC.count ? frame.volTargetCC[idx] : current.volTargetCC
            let volActual = idx < frame.volActualCC.count ? frame.volActualCC[idx] : current.volActualCC
            let revTarget = idx < frame.reverbTargetCC.count ? frame.reverbTargetCC[idx] : current.reverbTargetCC
            let revActual = idx < frame.reverbActualCC.count ? frame.reverbActualCC[idx] : current.reverbActualCC

            current.rmsDb = rms
            current.peakDb = peak

            if peak >= current.peakHoldDb {
                current.peakHoldDb = peak
                current.peakHoldTimestamp = now
            } else if now - current.peakHoldTimestamp > peakHoldDuration {
                current.peakHoldDb = max(peak, current.peakHoldDb - peakFalloffDbPerSec * dt)
            }

            if clipFlag {
                current.clip = true
                current.clipHoldUntil = now + clipHoldDuration
            } else if now >= current.clipHoldUntil {
                current.clip = false
            }

            current.vad = vadProb
            current.volTargetCC = volTarget
            current.volActualCC = volActual
            current.reverbTargetCC = revTarget
            current.reverbActualCC = revActual

            updated[channelId] = current
        }

        self.readingsById = updated
        self.masterLufs = frame.masterLufs
        self.lastFrameTime = now
        self.frameCounter &+= 1
    }

    public func reading(for channelId: String) -> ChannelMeterReading {
        readingsById[channelId] ?? ChannelMeterReading(channelId: channelId)
    }

    public var orderedReadings: [ChannelMeterReading] {
        channelIds.map { reading(for: $0) }
    }

    public func resetClipIndicators() {
        var updated = readingsById
        for id in channelIds {
            if var r = updated[id] {
                r.clip = false
                r.clipHoldUntil = 0
                updated[id] = r
            }
        }
        self.readingsById = updated
    }

    private func clampDb(_ value: Double) -> Double {
        if value.isNaN || value.isInfinite {
            return -60.0
        }
        return max(-60.0, min(6.0, value))
    }
}
