import Foundation

// MARK: - Protocol Constants

public enum AetherMixProtocol {
    public static let protocolVersion = "5.1"
    public static let schemaVersion = 5
    public static let defaultWebSocketURL = URL(string: "ws://127.0.0.1:8765/ws")!

    public static var defaultAetherMixDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".aethermix", isDirectory: true)
    }

    public static var defaultTokenPath: URL {
        defaultAetherMixDirectory.appendingPathComponent("session.token")
    }

    public static var defaultBridgeSocketPath: URL {
        defaultAetherMixDirectory.appendingPathComponent("bridge.sock")
    }

    public static var defaultBridgeStatePath: URL {
        defaultAetherMixDirectory.appendingPathComponent("bridge_state.json")
    }

    public static var defaultLaunchAgentPath: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/LaunchAgents/com.aethermix.midibridge.plist")
    }
}

// MARK: - Core Enums (Mirror of engine/core/protocol.py v5.1)

public enum ServicePhase: String, Codable, Sendable, CaseIterable, Hashable, Identifiable {
    case spokenWord = "SPOKEN_WORD"
    case prayerUnderscore = "PRAYER_UNDERSCORE"
    case worshipLow = "WORSHIP_LOW"
    case worshipHigh = "WORSHIP_HIGH"
    case transition = "TRANSITION"

    public var id: String { rawValue }

    public var isMacroPhase: Bool {
        self != .transition
    }

    public static var macroPhases: [ServicePhase] {
        [.spokenWord, .prayerUnderscore, .worshipLow, .worshipHigh]
    }

    public var displayName: String {
        switch self {
        case .spokenWord: return "Spoken Word"
        case .prayerUnderscore: return "Prayer Underscore"
        case .worshipLow: return "Worship Low"
        case .worshipHigh: return "Worship High"
        case .transition: return "Transition"
        }
    }

    public var shortLabel: String {
        switch self {
        case .spokenWord: return "SPOKEN"
        case .prayerUnderscore: return "PRAYER"
        case .worshipLow: return "WORSHIP LO"
        case .worshipHigh: return "WORSHIP HI"
        case .transition: return "TRANSITION"
        }
    }
}

public enum RulePhase: String, Codable, Sendable, CaseIterable, Hashable, Identifiable {
    case any = "ANY"
    case spokenWord = "SPOKEN_WORD"
    case prayerUnderscore = "PRAYER_UNDERSCORE"
    case worshipLow = "WORSHIP_LOW"
    case worshipHigh = "WORSHIP_HIGH"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .any: return "Any Phase"
        case .spokenWord: return "Spoken Word"
        case .prayerUnderscore: return "Prayer Underscore"
        case .worshipLow: return "Worship Low"
        case .worshipHigh: return "Worship High"
        }
    }
}

public enum AutonomyMode: String, Codable, Sendable, CaseIterable, Hashable, Identifiable {
    case shadowObserve = "SHADOW_OBSERVE"
    case active = "ACTIVE"
    case emergencyBypass = "EMERGENCY_BYPASS"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .shadowObserve: return "SHADOW OBSERVE"
        case .active: return "ACTIVE"
        case .emergencyBypass: return "EMERGENCY BYPASS"
        }
    }

    public var shortLabel: String {
        switch self {
        case .shadowObserve: return "SHADOW"
        case .active: return "ACTIVE"
        case .emergencyBypass: return "BYPASS"
        }
    }
}

public enum SyncStatus: String, Codable, Sendable, CaseIterable, Hashable {
    case unverified = "UNVERIFIED"
    case synced = "SYNCED"
}

public enum ChannelRole: String, Codable, Sendable, CaseIterable, Hashable {
    case spokenPrimary = "spoken_primary"
    case vocalLead = "vocal_lead"
    case vocalBg = "vocal_bg"
    case instrumentPad = "instrument_pad"
    case instrumentGuitar = "instrument_guitar"
    case instrumentBand = "instrument_band"
    case roomAmbience = "room_ambience"
    case masterMeter = "master_meter"

    public var isBandRole: Bool {
        switch self {
        case .instrumentPad, .instrumentGuitar, .instrumentBand:
            return true
        default:
            return false
        }
    }

    public var shortLabel: String {
        switch self {
        case .spokenPrimary: return "SPOKEN"
        case .vocalLead: return "LEAD VOX"
        case .vocalBg: return "BG VOX"
        case .instrumentPad: return "PAD/KEYS"
        case .instrumentGuitar: return "GUITAR"
        case .instrumentBand: return "BAND"
        case .roomAmbience: return "ROOM"
        case .masterMeter: return "MASTER"
        }
    }
}

public enum ControlType: String, Codable, Sendable, CaseIterable, Hashable {
    case vol = "vol"
    case reverb = "reverb"
    case mute = "mute"
}

public enum TakeoverMode: String, Codable, Sendable, CaseIterable, Hashable {
    case pickupCatchup = "pickup_catchup"
    case immediate = "immediate"
    case relativeEncoder = "relative_encoder"
}

public enum MainStageOutMode: String, Codable, Sendable, CaseIterable, Hashable {
    case virtualPort = "virtual_port"
    case iacBus = "iac_bus"
}

public enum RuleParameter: String, Codable, Sendable, CaseIterable, Hashable {
    case volDbOffset = "vol_db_offset"
    case reverbDbOffset = "reverb_db_offset"
    case duckDepthDb = "duck_depth_db"
    case conditionalAttenuate = "conditional_attenuate"
    case conditionalMute = "conditional_mute"

    public var displayName: String {
        switch self {
        case .volDbOffset: return "Vol Offset"
        case .reverbDbOffset: return "Reverb Offset"
        case .duckDepthDb: return "Duck Depth"
        case .conditionalAttenuate: return "Cond. Attenuate"
        case .conditionalMute: return "Cond. Mute"
        }
    }

    public var bounds: ClosedRange<Double> {
        switch self {
        case .volDbOffset: return -12.0...6.0
        case .reverbDbOffset: return -18.0...12.0
        case .duckDepthDb: return -12.0...0.0
        case .conditionalAttenuate: return -36.0...0.0
        case .conditionalMute: return 0.0...1.0
        }
    }
}

public enum RuleScope: String, Codable, Sendable, CaseIterable, Hashable {
    case ephemeralSession = "ephemeral_session"
    case candidatePermanent = "candidate_permanent"
}

public enum RuleExpiry: String, Codable, Sendable, CaseIterable, Hashable {
    case phaseChange = "phase_change"
    case segmentChange = "segment_change"
    case endOfService = "end_of_service"

    public var displayName: String {
        switch self {
        case .phaseChange: return "Phase Change"
        case .segmentChange: return "Segment Change"
        case .endOfService: return "End of Service"
        }
    }
}

public enum ConditionPredicate: String, Codable, Sendable, CaseIterable, Hashable {
    case speechOpen = "speech_open"
    case active = "active"
}

public enum BrainCategory: String, Codable, Sendable, CaseIterable, Hashable, Identifiable {
    case perception = "PERCEPTION"
    case action = "ACTION"
    case safety = "SAFETY"
    case llm = "LLM"
    case rlhf = "RLHF"

    public var id: String { rawValue }
}

public enum PendingRuleAction: String, Codable, Sendable, CaseIterable, Hashable {
    case applySession = "apply_session"
    case promote = "promote"
    case dismiss = "dismiss"
}

public enum FeedbackThumb: String, Codable, Sendable, Hashable {
    case up = "up"
    case down = "down"
}

// MARK: - Canonical Taper Math (Exact Parity with protocol.py)

public struct TaperPoint: Codable, Sendable, Equatable, Hashable {
    public let cc: Int
    public let db: Double?

    public init(cc: Int, db: Double?) {
        self.cc = cc
        self.db = db
    }

    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        self.cc = try container.decode(Int.self)
        self.db = try container.decode(Double?.self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(cc)
        try container.encode(db)
    }
}

public enum TaperMath {
    public static let defaultVolTaper: [TaperPoint] = [
        // Measured against MainStage's native channel-strip fader law (volume mapped Absolute, -inf..+6 dB).
        TaperPoint(cc: 0, db: nil),
        TaperPoint(cc: 1, db: -78.2),
        TaperPoint(cc: 4, db: -54.1),
        TaperPoint(cc: 8, db: -42.1),
        TaperPoint(cc: 16, db: -30.1),
        TaperPoint(cc: 24, db: -23.0),
        TaperPoint(cc: 32, db: -18.0),
        TaperPoint(cc: 48, db: -11.0),
        TaperPoint(cc: 64, db: -6.0),
        TaperPoint(cc: 80, db: -2.1),
        TaperPoint(cc: 96, db: 1.1),
        TaperPoint(cc: 112, db: 3.7),
        TaperPoint(cc: 127, db: 6.0)
    ]

    public static let defaultReverbTaper: [TaperPoint] = [
        TaperPoint(cc: 0, db: nil),
        TaperPoint(cc: 1, db: -60.0),
        TaperPoint(cc: 32, db: -24.0),
        TaperPoint(cc: 64, db: -12.0),
        TaperPoint(cc: 96, db: 0.0),
        TaperPoint(cc: 114, db: 6.0),
        TaperPoint(cc: 127, db: 6.0)
    ]

    public static func ccToDb(_ cc: Int, taper: [TaperPoint] = defaultVolTaper) -> Double {
        let clamped = max(0, min(127, cc))
        if clamped == 0, let first = taper.first, first.db == nil {
            return -.infinity
        }
        let numericPts = taper.compactMap { pt -> (Int, Double)? in
            guard let db = pt.db else { return nil }
            return (pt.cc, db)
        }
        guard let first = numericPts.first, let last = numericPts.last else {
            return -60.0
        }
        if clamped <= first.0 { return first.1 }
        if clamped >= last.0 { return last.1 }
        for idx in 0..<(numericPts.count - 1) {
            let (c0, d0) = numericPts[idx]
            let (c1, d1) = numericPts[idx + 1]
            if c0 <= clamped && clamped <= c1 {
                if c1 == c0 { return d0 }
                let frac = Double(clamped - c0) / Double(c1 - c0)
                return d0 + frac * (d1 - d0)
            }
        }
        return last.1
    }

    public static func dbToCC(_ db: Double, taper: [TaperPoint] = defaultVolTaper) -> Int {
        if db.isInfinite && db < 0 {
            return 0
        }
        let numericPts = taper.compactMap { pt -> (Int, Double)? in
            guard let d = pt.db else { return nil }
            return (pt.cc, d)
        }
        guard let first = numericPts.first, let last = numericPts.last else {
            return 0
        }
        if db < first.1 - 1e-6, taper.first?.db == nil {
            return 0
        }
        if db <= first.1 {
            return first.0
        }
        for idx in 0..<(numericPts.count - 1) {
            let (c0, d0) = numericPts[idx]
            let (c1, d1) = numericPts[idx + 1]
            if db <= d0 + 1e-9 {
                return c0
            }
            if d0 <= db && db <= d1 {
                if abs(d1 - d0) < 1e-9 {
                    return c0
                }
                let frac = (db - d0) / (d1 - d0)
                return Int((Double(c0) + frac * Double(c1 - c0)).rounded())
            }
        }
        let maxDb = last.1
        for (c, d) in numericPts where abs(d - maxDb) < 1e-9 {
            return c
        }
        return last.0
    }

    public static func formatDb(_ db: Double?) -> String {
        guard let db else { return "—" }
        if db.isInfinite && db < 0 || db <= -59.5 {
            return "-∞ dB"
        }
        if abs(db) < 0.05 {
            return "0.0 dB"
        }
        return String(format: "%+.1f dB", db)
    }
}

// MARK: - Strict Sendable Dynamic JSON Value

public enum JSONValue: Codable, Sendable, Hashable {
    case null
    case bool(Bool)
    case int(Int64)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let boolVal = try? container.decode(Bool.self) {
            self = .bool(boolVal)
        } else if let intVal = try? container.decode(Int64.self) {
            self = .int(intVal)
        } else if let doubleVal = try? container.decode(Double.self) {
            self = .double(doubleVal)
        } else if let stringVal = try? container.decode(String.self) {
            self = .string(stringVal)
        } else if let arrayVal = try? container.decode([JSONValue].self) {
            self = .array(arrayVal)
        } else if let objectVal = try? container.decode([String: JSONValue].self) {
            self = .object(objectVal)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported JSONValue"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let v):
            try container.encode(v)
        case .int(let v):
            try container.encode(v)
        case .double(let v):
            try container.encode(v)
        case .string(let v):
            try container.encode(v)
        case .array(let v):
            try container.encode(v)
        case .object(let v):
            try container.encode(v)
        }
    }

    public var stringValue: String? {
        if case .string(let v) = self { return v }
        return nil
    }

    public var boolValue: Bool? {
        if case .bool(let v) = self { return v }
        return nil
    }

    public var intValue: Int? {
        switch self {
        case .int(let v): return Int(v)
        case .double(let v): return Int(v)
        default: return nil
        }
    }

    public var doubleValue: Double? {
        switch self {
        case .double(let v): return v
        case .int(let v): return Double(v)
        default: return nil
        }
    }

    public var objectValue: [String: JSONValue]? {
        if case .object(let v) = self { return v }
        return nil
    }

    public var arrayValue: [JSONValue]? {
        if case .array(let v) = self { return v }
        return nil
    }
}

// MARK: - WebSocket v5.1 Envelope & Payloads

public struct WSEnvelope: Codable, Sendable, Equatable {
    public let type: String
    public let seq: Int
    public let reqId: String?
    public let tsMs: Int64
    public let payload: [String: JSONValue]

    enum CodingKeys: String, CodingKey {
        case type
        case seq
        case reqId = "req_id"
        case tsMs = "ts_ms"
        case payload
    }

    public init(
        type: String,
        seq: Int = 0,
        reqId: String? = nil,
        tsMs: Int64 = Int64(Date().timeIntervalSince1970 * 1000),
        payload: [String: JSONValue] = [:]
    ) {
        self.type = type
        self.seq = seq
        self.reqId = reqId
        self.tsMs = tsMs
        self.payload = payload
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.type = try container.decode(String.self, forKey: .type)
        self.seq = try container.decodeIfPresent(Int.self, forKey: .seq) ?? 0
        self.reqId = try container.decodeIfPresent(String.self, forKey: .reqId)
        self.tsMs = try container.decodeIfPresent(Int64.self, forKey: .tsMs) ?? 0
        self.payload = try container.decodeIfPresent([String: JSONValue].self, forKey: .payload) ?? [:]
    }

    public func decodePayload<T: Decodable>(_ type: T.Type) throws -> T {
        let data = try JSONEncoder().encode(payload)
        return try JSONDecoder().decode(T.self, from: data)
    }
}

public struct WSAckPayload: Codable, Sendable, Equatable {
    public let reqId: String?
    public let ok: Bool
    public let code: String?
    public let message: String?

    enum CodingKeys: String, CodingKey {
        case reqId = "req_id"
        case ok
        case code
        case message
    }

    public init(reqId: String? = nil, ok: Bool, code: String? = nil, message: String? = nil) {
        self.reqId = reqId
        self.ok = ok
        self.code = code
        self.message = message
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.reqId = try container.decodeIfPresent(String.self, forKey: .reqId)
        self.ok = try container.decodeIfPresent(Bool.self, forKey: .ok) ?? false
        self.code = try container.decodeIfPresent(String.self, forKey: .code)
        self.message = try container.decodeIfPresent(String.self, forKey: .message)
    }
}

public struct ChannelRuntimeState: Codable, Sendable, Equatable {
    public var armed: Bool
    public var syncStatus: SyncStatus
    public var volCurrentDb: Double
    public var volTargetDb: Double
    public var volActualCC: Int
    public var volTargetCC: Int
    public var volTouchLockS: Double
    public var volHumanOffsetDb: Double
    public var volPickupDir: Int
    public var reverbCurrentDb: Double?
    public var reverbTargetDb: Double?
    public var reverbActualCC: Int?
    public var reverbTargetCC: Int?
    public var reverbTouchLockS: Double
    public var reverbHumanOffsetDb: Double
    public var reverbPickupDir: Int
    public var feedbackCeilingDb: Double?
    public var speechOpen: Bool
    public var instrumentOpen: Bool

    enum CodingKeys: String, CodingKey {
        case armed
        case syncStatus = "sync_status"
        case volCurrentDb = "vol_current_db"
        case volTargetDb = "vol_target_db"
        case volActualCC = "vol_actual_cc"
        case volTargetCC = "vol_target_cc"
        case volTouchLockS = "vol_touch_lock_s"
        case volHumanOffsetDb = "vol_human_offset_db"
        case volPickupDir = "vol_pickup_dir"
        case reverbCurrentDb = "reverb_current_db"
        case reverbTargetDb = "reverb_target_db"
        case reverbActualCC = "reverb_actual_cc"
        case reverbTargetCC = "reverb_target_cc"
        case reverbTouchLockS = "reverb_touch_lock_s"
        case reverbHumanOffsetDb = "reverb_human_offset_db"
        case reverbPickupDir = "reverb_pickup_dir"
        case feedbackCeilingDb = "feedback_ceiling_db"
        case speechOpen = "speech_open"
        case instrumentOpen = "instrument_open"
    }

    public init(
        armed: Bool = false,
        syncStatus: SyncStatus = .unverified,
        volCurrentDb: Double = -60.0,
        volTargetDb: Double = -60.0,
        volActualCC: Int = 0,
        volTargetCC: Int = 0,
        volTouchLockS: Double = 0.0,
        volHumanOffsetDb: Double = 0.0,
        volPickupDir: Int = 0,
        reverbCurrentDb: Double? = nil,
        reverbTargetDb: Double? = nil,
        reverbActualCC: Int? = nil,
        reverbTargetCC: Int? = nil,
        reverbTouchLockS: Double = 0.0,
        reverbHumanOffsetDb: Double = 0.0,
        reverbPickupDir: Int = 0,
        feedbackCeilingDb: Double? = nil,
        speechOpen: Bool = false,
        instrumentOpen: Bool = false
    ) {
        self.armed = armed
        self.syncStatus = syncStatus
        self.volCurrentDb = volCurrentDb
        self.volTargetDb = volTargetDb
        self.volActualCC = volActualCC
        self.volTargetCC = volTargetCC
        self.volTouchLockS = volTouchLockS
        self.volHumanOffsetDb = volHumanOffsetDb
        self.volPickupDir = volPickupDir
        self.reverbCurrentDb = reverbCurrentDb
        self.reverbTargetDb = reverbTargetDb
        self.reverbActualCC = reverbActualCC
        self.reverbTargetCC = reverbTargetCC
        self.reverbTouchLockS = reverbTouchLockS
        self.reverbHumanOffsetDb = reverbHumanOffsetDb
        self.reverbPickupDir = reverbPickupDir
        self.feedbackCeilingDb = feedbackCeilingDb
        self.speechOpen = speechOpen
        self.instrumentOpen = instrumentOpen
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.armed = try c.decodeIfPresent(Bool.self, forKey: .armed) ?? false
        self.syncStatus = try c.decodeIfPresent(SyncStatus.self, forKey: .syncStatus) ?? .unverified
        self.volCurrentDb = try c.decodeIfPresent(Double.self, forKey: .volCurrentDb) ?? -60.0
        self.volTargetDb = try c.decodeIfPresent(Double.self, forKey: .volTargetDb) ?? -60.0
        self.volActualCC = try c.decodeIfPresent(Int.self, forKey: .volActualCC) ?? 0
        self.volTargetCC = try c.decodeIfPresent(Int.self, forKey: .volTargetCC) ?? 0
        self.volTouchLockS = try c.decodeIfPresent(Double.self, forKey: .volTouchLockS) ?? 0.0
        self.volHumanOffsetDb = try c.decodeIfPresent(Double.self, forKey: .volHumanOffsetDb) ?? 0.0
        self.volPickupDir = try c.decodeIfPresent(Int.self, forKey: .volPickupDir) ?? 0
        self.reverbCurrentDb = try c.decodeIfPresent(Double.self, forKey: .reverbCurrentDb)
        self.reverbTargetDb = try c.decodeIfPresent(Double.self, forKey: .reverbTargetDb)
        self.reverbActualCC = try c.decodeIfPresent(Int.self, forKey: .reverbActualCC)
        self.reverbTargetCC = try c.decodeIfPresent(Int.self, forKey: .reverbTargetCC)
        self.reverbTouchLockS = try c.decodeIfPresent(Double.self, forKey: .reverbTouchLockS) ?? 0.0
        self.reverbHumanOffsetDb = try c.decodeIfPresent(Double.self, forKey: .reverbHumanOffsetDb) ?? 0.0
        self.reverbPickupDir = try c.decodeIfPresent(Int.self, forKey: .reverbPickupDir) ?? 0
        self.feedbackCeilingDb = try c.decodeIfPresent(Double.self, forKey: .feedbackCeilingDb)
        self.speechOpen = try c.decodeIfPresent(Bool.self, forKey: .speechOpen) ?? false
        self.instrumentOpen = try c.decodeIfPresent(Bool.self, forKey: .instrumentOpen) ?? false
    }

    public var hasReverb: Bool {
        reverbActualCC != nil || reverbCurrentDb != nil || reverbTargetCC != nil
    }
}

public struct RuleCondition: Codable, Sendable, Equatable, Hashable {
    public let whenChannel: String
    public let predicate: ConditionPredicate

    enum CodingKeys: String, CodingKey {
        case whenChannel = "when_channel"
        case predicate
    }

    public init(whenChannel: String, predicate: ConditionPredicate) {
        self.whenChannel = whenChannel
        self.predicate = predicate
    }
}

public struct MixingRule: Codable, Sendable, Equatable, Hashable, Identifiable {
    public let id: String
    public let description: String
    public let phase: RulePhase
    public let channelId: String
    public let parameter: RuleParameter
    public let value: Double
    public let condition: RuleCondition?
    public var enabled: Bool
    public let source: String
    public let scope: RuleScope
    public let expiresOn: RuleExpiry?
    public let createdAtMs: Int64

    enum CodingKeys: String, CodingKey {
        case id
        case description
        case phase
        case channelId = "channel_id"
        case parameter
        case value
        case condition
        case enabled
        case source
        case scope
        case expiresOn = "expires_on"
        case createdAtMs = "created_at_ms"
    }

    public init(
        id: String,
        description: String,
        phase: RulePhase,
        channelId: String,
        parameter: RuleParameter,
        value: Double,
        condition: RuleCondition? = nil,
        enabled: Bool = true,
        source: String = "operator_confirmed",
        scope: RuleScope = .candidatePermanent,
        expiresOn: RuleExpiry? = nil,
        createdAtMs: Int64 = Int64(Date().timeIntervalSince1970 * 1000)
    ) {
        self.id = id
        self.description = description
        self.phase = phase
        self.channelId = channelId
        self.parameter = parameter
        self.value = value
        self.condition = condition
        self.enabled = enabled
        self.source = source
        self.scope = scope
        self.expiresOn = expiresOn
        self.createdAtMs = createdAtMs
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(String.self, forKey: .id)
        self.description = try c.decode(String.self, forKey: .description)
        self.phase = try c.decode(RulePhase.self, forKey: .phase)
        self.channelId = try c.decode(String.self, forKey: .channelId)
        self.parameter = try c.decode(RuleParameter.self, forKey: .parameter)
        self.value = try c.decode(Double.self, forKey: .value)
        self.condition = try c.decodeIfPresent(RuleCondition.self, forKey: .condition)
        self.enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
        self.source = try c.decodeIfPresent(String.self, forKey: .source) ?? "operator_confirmed"
        self.scope = try c.decodeIfPresent(RuleScope.self, forKey: .scope) ?? .candidatePermanent
        self.expiresOn = try c.decodeIfPresent(RuleExpiry.self, forKey: .expiresOn)
        self.createdAtMs = try c.decodeIfPresent(Int64.self, forKey: .createdAtMs) ?? 0
    }
}

public struct PendingRuleCard: Codable, Sendable, Equatable, Hashable, Identifiable {
    public let ruleId: String
    public let description: String
    public let phase: RulePhase
    public let channelId: String
    public let parameter: RuleParameter
    public let value: Double
    public let condition: RuleCondition?
    public let scope: RuleScope
    public let expiresOn: RuleExpiry
    public let requiresPreview: Bool
    public let appliedInSession: Bool
    public let createdAtMs: Int64

    public var id: String { ruleId }

    enum CodingKeys: String, CodingKey {
        case ruleId = "rule_id"
        case description
        case phase
        case channelId = "channel_id"
        case parameter
        case value
        case condition
        case scope
        case expiresOn = "expires_on"
        case requiresPreview = "requires_preview"
        case appliedInSession = "applied_in_session"
        case createdAtMs = "created_at_ms"
    }

    public init(
        ruleId: String,
        description: String,
        phase: RulePhase,
        channelId: String,
        parameter: RuleParameter,
        value: Double,
        condition: RuleCondition? = nil,
        scope: RuleScope = .ephemeralSession,
        expiresOn: RuleExpiry = .segmentChange,
        requiresPreview: Bool = false,
        appliedInSession: Bool = false,
        createdAtMs: Int64 = Int64(Date().timeIntervalSince1970 * 1000)
    ) {
        self.ruleId = ruleId
        self.description = description
        self.phase = phase
        self.channelId = channelId
        self.parameter = parameter
        self.value = value
        self.condition = condition
        self.scope = scope
        self.expiresOn = expiresOn
        self.requiresPreview = requiresPreview
        self.appliedInSession = appliedInSession
        self.createdAtMs = createdAtMs
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.ruleId = try c.decode(String.self, forKey: .ruleId)
        self.description = try c.decode(String.self, forKey: .description)
        self.phase = try c.decode(RulePhase.self, forKey: .phase)
        self.channelId = try c.decode(String.self, forKey: .channelId)
        self.parameter = try c.decode(RuleParameter.self, forKey: .parameter)
        self.value = try c.decode(Double.self, forKey: .value)
        self.condition = try c.decodeIfPresent(RuleCondition.self, forKey: .condition)
        self.scope = try c.decodeIfPresent(RuleScope.self, forKey: .scope) ?? .ephemeralSession
        self.expiresOn = try c.decodeIfPresent(RuleExpiry.self, forKey: .expiresOn) ?? .segmentChange
        self.requiresPreview = try c.decodeIfPresent(Bool.self, forKey: .requiresPreview) ?? false
        self.appliedInSession = try c.decodeIfPresent(Bool.self, forKey: .appliedInSession) ?? false
        self.createdAtMs = try c.decodeIfPresent(Int64.self, forKey: .createdAtMs) ?? 0
    }
}

public struct StateSnapshotPayload: Codable, Sendable, Equatable {
    public var channelIds: [String]
    public var autonomyMode: AutonomyMode
    public var bypassLatched: Bool
    public var freezeLearning: Bool
    public var activePhase: ServicePhase
    public var phaseConfidence: Double
    public var hysteresisProgress: Double
    public var serviceStepIdx: Int
    public var channels: [String: ChannelRuntimeState]
    public var activeRules: [MixingRule]
    public var pendingRules: [PendingRuleCard]
    public var queuedReviewCount: Int
    public var hardwareWarnings: [String]

    enum CodingKeys: String, CodingKey {
        case channelIds = "channel_ids"
        case autonomyMode = "autonomy_mode"
        case bypassLatched = "bypass_latched"
        case freezeLearning = "freeze_learning"
        case activePhase = "active_phase"
        case phaseConfidence = "phase_confidence"
        case hysteresisProgress = "hysteresis_progress"
        case serviceStepIdx = "service_step_idx"
        case channels
        case activeRules = "active_rules"
        case pendingRules = "pending_rules"
        case queuedReviewCount = "queued_review_count"
        case hardwareWarnings = "hardware_warnings"
    }

    public init(
        channelIds: [String] = ChannelCatalog.defaultChannelIds,
        autonomyMode: AutonomyMode = .shadowObserve,
        bypassLatched: Bool = false,
        freezeLearning: Bool = false,
        activePhase: ServicePhase = .spokenWord,
        phaseConfidence: Double = 0.0,
        hysteresisProgress: Double = 0.0,
        serviceStepIdx: Int = 0,
        channels: [String: ChannelRuntimeState] = [:],
        activeRules: [MixingRule] = [],
        pendingRules: [PendingRuleCard] = [],
        queuedReviewCount: Int = 0,
        hardwareWarnings: [String] = []
    ) {
        self.channelIds = channelIds
        self.autonomyMode = autonomyMode
        self.bypassLatched = bypassLatched
        self.freezeLearning = freezeLearning
        self.activePhase = activePhase
        self.phaseConfidence = phaseConfidence
        self.hysteresisProgress = hysteresisProgress
        self.serviceStepIdx = serviceStepIdx
        self.channels = channels
        self.activeRules = activeRules
        self.pendingRules = pendingRules
        self.queuedReviewCount = queuedReviewCount
        self.hardwareWarnings = hardwareWarnings
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.channelIds = try c.decode([String].self, forKey: .channelIds)
        self.autonomyMode = try c.decodeIfPresent(AutonomyMode.self, forKey: .autonomyMode) ?? .shadowObserve
        self.bypassLatched = try c.decodeIfPresent(Bool.self, forKey: .bypassLatched) ?? false
        self.freezeLearning = try c.decodeIfPresent(Bool.self, forKey: .freezeLearning) ?? false
        self.activePhase = try c.decodeIfPresent(ServicePhase.self, forKey: .activePhase) ?? .spokenWord
        self.phaseConfidence = try c.decodeIfPresent(Double.self, forKey: .phaseConfidence) ?? 0.0
        self.hysteresisProgress = try c.decodeIfPresent(Double.self, forKey: .hysteresisProgress) ?? 0.0
        self.serviceStepIdx = try c.decodeIfPresent(Int.self, forKey: .serviceStepIdx) ?? 0
        self.channels = try c.decodeIfPresent([String: ChannelRuntimeState].self, forKey: .channels) ?? [:]
        self.activeRules = try c.decodeIfPresent([MixingRule].self, forKey: .activeRules) ?? []
        self.pendingRules = try c.decodeIfPresent([PendingRuleCard].self, forKey: .pendingRules) ?? []
        self.queuedReviewCount = try c.decodeIfPresent(Int.self, forKey: .queuedReviewCount) ?? 0
        self.hardwareWarnings = try c.decodeIfPresent([String].self, forKey: .hardwareWarnings) ?? []
    }
}

public struct MeterFramePayload: Codable, Sendable, Equatable {
    public let rmsDb: [Double]
    public let peakDb: [Double]
    public let clip: [Bool]
    public let vad: [Double]
    public let masterLufs: Double
    public let volTargetCC: [Int]
    public let volActualCC: [Int]
    public let reverbTargetCC: [Int?]
    public let reverbActualCC: [Int?]

    enum CodingKeys: String, CodingKey {
        case rmsDb = "rms_db"
        case peakDb = "peak_db"
        case clip
        case vad
        case masterLufs = "master_lufs"
        case volTargetCC = "vol_target_cc"
        case volActualCC = "vol_actual_cc"
        case reverbTargetCC = "reverb_target_cc"
        case reverbActualCC = "reverb_actual_cc"
    }

    public init(
        rmsDb: [Double],
        peakDb: [Double],
        clip: [Bool],
        vad: [Double],
        masterLufs: Double = -60.0,
        volTargetCC: [Int],
        volActualCC: [Int],
        reverbTargetCC: [Int?],
        reverbActualCC: [Int?]
    ) {
        self.rmsDb = rmsDb
        self.peakDb = peakDb
        self.clip = clip
        self.vad = vad
        self.masterLufs = masterLufs
        self.volTargetCC = volTargetCC
        self.volActualCC = volActualCC
        self.reverbTargetCC = reverbTargetCC
        self.reverbActualCC = reverbActualCC
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.rmsDb = try c.decode([Double].self, forKey: .rmsDb)
        self.peakDb = try c.decode([Double].self, forKey: .peakDb)
        self.clip = try c.decode([Bool].self, forKey: .clip)
        self.vad = try c.decode([Double].self, forKey: .vad)
        self.masterLufs = try c.decodeIfPresent(Double.self, forKey: .masterLufs) ?? -60.0
        self.volTargetCC = try c.decode([Int].self, forKey: .volTargetCC)
        self.volActualCC = try c.decode([Int].self, forKey: .volActualCC)
        self.reverbTargetCC = try c.decode([Int?].self, forKey: .reverbTargetCC)
        self.reverbActualCC = try c.decode([Int?].self, forKey: .reverbActualCC)
    }
}

public struct BrainEventPayload: Codable, Sendable, Equatable, Hashable, Identifiable {
    public let id: String
    public let category: BrainCategory
    public let summary: String
    public let detail: String
    public let confidence: Double
    public let tsMs: Int64

    enum CodingKeys: String, CodingKey {
        case id
        case category
        case summary
        case detail
        case confidence
        case tsMs = "ts_ms"
    }

    public init(
        id: String,
        category: BrainCategory,
        summary: String,
        detail: String = "",
        confidence: Double = 1.0,
        tsMs: Int64 = Int64(Date().timeIntervalSince1970 * 1000)
    ) {
        self.id = id
        self.category = category
        self.summary = summary
        self.detail = detail
        self.confidence = confidence
        self.tsMs = tsMs
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(String.self, forKey: .id)
        self.category = try c.decode(BrainCategory.self, forKey: .category)
        self.summary = try c.decode(String.self, forKey: .summary)
        self.detail = try c.decodeIfPresent(String.self, forKey: .detail) ?? ""
        self.confidence = try c.decodeIfPresent(Double.self, forKey: .confidence) ?? 1.0
        self.tsMs = try c.decodeIfPresent(Int64.self, forKey: .tsMs) ?? 0
    }
}

public struct HeartbeatPayload: Codable, Sendable, Equatable {
    public let uptimeS: Double
    public let loopP95Ms: Double
    public let loopP99Ms: Double
    public let audioOk: Bool
    public let bridgeOk: Bool
    public let clapOk: Bool
    public let ollamaOk: Bool

    enum CodingKeys: String, CodingKey {
        case uptimeS = "uptime_s"
        case loopP95Ms = "loop_p95_ms"
        case loopP99Ms = "loop_p99_ms"
        case audioOk = "audio_ok"
        case bridgeOk = "bridge_ok"
        case clapOk = "clap_ok"
        case ollamaOk = "ollama_ok"
    }

    public init(
        uptimeS: Double = 0,
        loopP95Ms: Double = 0,
        loopP99Ms: Double = 0,
        audioOk: Bool = false,
        bridgeOk: Bool = false,
        clapOk: Bool = false,
        ollamaOk: Bool = false
    ) {
        self.uptimeS = uptimeS
        self.loopP95Ms = loopP95Ms
        self.loopP99Ms = loopP99Ms
        self.audioOk = audioOk
        self.bridgeOk = bridgeOk
        self.clapOk = clapOk
        self.ollamaOk = ollamaOk
    }
}

// MARK: - Standalone Bridge NDJSON Protocol Models (Section 6.5)

public struct BridgeStatePayload: Codable, Sendable, Equatable {
    public var lastSentCC: [String: Int]
    public var humanSeq: [String: Int]
    public var bypassLatched: Bool
    public var controllerConnected: Bool
    public var engineConnected: Bool
    public var pickupPending: [String: Int]
    public var restoredFromDiskAt: Int64?

    enum CodingKeys: String, CodingKey {
        case lastSentCC = "last_sent_cc"
        case humanSeq = "human_seq"
        case bypassLatched = "bypass_latched"
        case controllerConnected = "controller_connected"
        case engineConnected = "engine_connected"
        case pickupPending = "pickup_pending"
        case restoredFromDiskAt = "restored_from_disk_at"
    }

    public init(
        lastSentCC: [String: Int] = [:],
        humanSeq: [String: Int] = [:],
        bypassLatched: Bool = false,
        controllerConnected: Bool = false,
        engineConnected: Bool = false,
        pickupPending: [String: Int] = [:],
        restoredFromDiskAt: Int64? = nil
    ) {
        self.lastSentCC = lastSentCC
        self.humanSeq = humanSeq
        self.bypassLatched = bypassLatched
        self.controllerConnected = controllerConnected
        self.engineConnected = engineConnected
        self.pickupPending = pickupPending
        self.restoredFromDiskAt = restoredFromDiskAt
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.lastSentCC = try c.decodeIfPresent([String: Int].self, forKey: .lastSentCC) ?? [:]
        self.humanSeq = try c.decodeIfPresent([String: Int].self, forKey: .humanSeq) ?? [:]
        self.bypassLatched = try c.decodeIfPresent(Bool.self, forKey: .bypassLatched) ?? false
        self.controllerConnected = try c.decodeIfPresent(Bool.self, forKey: .controllerConnected) ?? false
        self.engineConnected = try c.decodeIfPresent(Bool.self, forKey: .engineConnected) ?? false
        self.pickupPending = try c.decodeIfPresent([String: Int].self, forKey: .pickupPending) ?? [:]
        self.restoredFromDiskAt = try c.decodeIfPresent(Int64.self, forKey: .restoredFromDiskAt)
    }

    public func ccKey(channelId: String, control: ControlType) -> Int? {
        lastSentCC["\(channelId):\(control.rawValue)"]
    }

    public func pickupDirection(channelId: String, control: ControlType) -> Int {
        pickupPending["\(channelId):\(control.rawValue)"] ?? 0
    }
}

public struct BridgeHelloMessage: Codable, Sendable, Equatable {
    public let type: String
    public let role: String
    public let token: String
    public let protocolVersion: String

    enum CodingKeys: String, CodingKey {
        case type
        case role
        case token
        case protocolVersion = "protocol_version"
    }

    public init(token: String, role: String = "ui", protocolVersion: String = AetherMixProtocol.protocolVersion) {
        self.type = "hello"
        self.role = role
        self.token = token
        self.protocolVersion = protocolVersion
    }
}

public struct BridgeManualCCMessage: Codable, Sendable, Equatable {
    public let type: String
    public let channelId: String
    public let control: ControlType
    public let cc: Int

    enum CodingKeys: String, CodingKey {
        case type
        case channelId = "channel_id"
        case control
        case cc
    }

    public init(channelId: String, control: ControlType, cc: Int) {
        self.type = "ui_manual_cc"
        self.channelId = channelId
        self.control = control
        self.cc = max(0, min(127, cc))
    }
}

public struct BridgeSetBypassMessage: Codable, Sendable, Equatable {
    public let type: String
    public let latched: Bool

    public init(latched: Bool) {
        self.type = "set_bypass"
        self.latched = latched
    }
}

public struct BridgeGetStateMessage: Codable, Sendable, Equatable {
    public let type: String

    public init() {
        self.type = "get_bridge_state"
    }
}

public struct BridgeHumanTouchEvent: Codable, Sendable, Equatable {
    public let channelId: String
    public let control: ControlType
    public let humanSeq: Int
    public let pickedUp: Bool
    public let source: String
    public let tsMs: Int64

    enum CodingKeys: String, CodingKey {
        case channelId = "channel_id"
        case control
        case humanSeq = "human_seq"
        case pickedUp = "picked_up"
        case source
        case tsMs = "ts_ms"
    }
}

public struct BridgeHumanCCEvent: Codable, Sendable, Equatable {
    public let channelId: String
    public let control: ControlType
    public let cc: Int
    public let humanSeq: Int
    public let source: String
    public let tsMs: Int64

    enum CodingKeys: String, CodingKey {
        case channelId = "channel_id"
        case control
        case cc
        case humanSeq = "human_seq"
        case source
        case tsMs = "ts_ms"
    }
}

public struct BridgeActionButtonEvent: Codable, Sendable, Equatable {
    public let action: String
    public let tsMs: Int64

    enum CodingKeys: String, CodingKey {
        case action
        case tsMs = "ts_ms"
    }
}

public enum BridgeInboundMessage: Sendable, Equatable {
    case bridgeState(BridgeStatePayload)
    case humanTouch(BridgeHumanTouchEvent)
    case humanCC(BridgeHumanCCEvent)
    case actionButton(BridgeActionButtonEvent)
    case helloAck(ok: Bool, message: String?)
    case error(message: String)
    case unknown(type: String)
}

// MARK: - Service Plan UI Metadata & Cached State

public struct ServiceSegment: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let title: String
    public let expectedPhase: ServicePhase

    enum CodingKeys: String, CodingKey {
        case id
        case title
        case expectedPhase = "expected_phase"
    }

    public init(id: String, title: String, expectedPhase: ServicePhase) {
        self.id = id
        self.title = title
        self.expectedPhase = expectedPhase
    }
}

public struct ChannelDescriptor: Sendable, Equatable, Identifiable {
    public let id: String
    public let label: String
    public let role: ChannelRole
    public let hasReverb: Bool
    public let hasMute: Bool
    public let volCC: Int
    public let reverbCC: Int?
    public let muteCC: Int?

    public init(
        id: String,
        label: String,
        role: ChannelRole,
        hasReverb: Bool,
        hasMute: Bool,
        volCC: Int,
        reverbCC: Int? = nil,
        muteCC: Int? = nil
    ) {
        self.id = id
        self.label = label
        self.role = role
        self.hasReverb = hasReverb
        self.hasMute = hasMute
        self.volCC = volCC
        self.reverbCC = reverbCC
        self.muteCC = muteCC
    }
}

public enum ChannelCatalog {
    public static let defaultChannels: [ChannelDescriptor] = [
        ChannelDescriptor(id: "ch1_pastor", label: "Pastor Mic", role: .spokenPrimary, hasReverb: true, hasMute: false, volCC: 20, reverbCC: 30, muteCC: nil),
        ChannelDescriptor(id: "ch2_lead_vox", label: "Lead Vocal", role: .vocalLead, hasReverb: true, hasMute: true, volCC: 21, reverbCC: 31, muteCC: 41),
        ChannelDescriptor(id: "ch3_bv1", label: "Backup Vocal 1", role: .vocalBg, hasReverb: false, hasMute: true, volCC: 22, reverbCC: nil, muteCC: 42),
        ChannelDescriptor(id: "ch4_bv2", label: "Backup Vocal 2", role: .vocalBg, hasReverb: false, hasMute: true, volCC: 23, reverbCC: nil, muteCC: 43),
        ChannelDescriptor(id: "ch5_keys", label: "Keys / Pad", role: .instrumentPad, hasReverb: false, hasMute: false, volCC: 24, reverbCC: nil, muteCC: nil),
        ChannelDescriptor(id: "ch7_guitars", label: "Guitars", role: .instrumentGuitar, hasReverb: false, hasMute: false, volCC: 25, reverbCC: nil, muteCC: nil),
        ChannelDescriptor(id: "ch9_drums", label: "Drums / Bass / Tracks", role: .instrumentBand, hasReverb: false, hasMute: false, volCC: 26, reverbCC: nil, muteCC: nil)
    ]

    public static let defaultChannelIds: [String] = defaultChannels.map(\.id)

    public static let defaultOrderOfService: [ServiceSegment] = [
        ServiceSegment(id: "seg_1", title: "Opening Worship (Songs 1–2)", expectedPhase: .worshipHigh),
        ServiceSegment(id: "seg_2", title: "Song 3 (Acoustic Refrain)", expectedPhase: .worshipLow),
        ServiceSegment(id: "seg_3", title: "Pastoral Prayer", expectedPhase: .prayerUnderscore),
        ServiceSegment(id: "seg_4", title: "Sermon", expectedPhase: .spokenWord),
        ServiceSegment(id: "seg_5", title: "Closing Song & Benediction", expectedPhase: .worshipHigh)
    ]

    public static func descriptor(for channelId: String) -> ChannelDescriptor {
        if let found = defaultChannels.first(where: { $0.id == channelId }) {
            return found
        }
        return ChannelDescriptor(
            id: channelId,
            label: channelId.replacingOccurrences(of: "_", with: " ").capitalized,
            role: .instrumentBand,
            hasReverb: false,
            hasMute: false,
            volCC: 20
        )
    }
}

public struct CachedFaderState: Sendable, Equatable, Identifiable {
    public let channelId: String
    public let volDb: Double
    public let volCC: Int
    public let reverbDb: Double?
    public let reverbCC: Int?
    public let updatedAt: Date

    public var id: String { channelId }

    public init(
        channelId: String,
        volDb: Double,
        volCC: Int,
        reverbDb: Double?,
        reverbCC: Int?,
        updatedAt: Date = Date()
    ) {
        self.channelId = channelId
        self.volDb = volDb
        self.volCC = volCC
        self.reverbDb = reverbDb
        self.reverbCC = reverbCC
        self.updatedAt = updatedAt
    }
}

public struct QueuedReviewSample: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let phase: ServicePhase
    public let channelId: String
    public let control: ControlType
    public let targetOffsetDb: Double
    public let source: String
    public let tsMs: Int64

    enum CodingKeys: String, CodingKey {
        case id
        case phase
        case channelId = "channel_id"
        case control
        case targetOffsetDb = "target_offset_db"
        case source
        case tsMs = "ts_ms"
    }

    public init(
        id: String = UUID().uuidString,
        phase: ServicePhase,
        channelId: String,
        control: ControlType,
        targetOffsetDb: Double,
        source: String,
        tsMs: Int64 = Int64(Date().timeIntervalSince1970 * 1000)
    ) {
        self.id = id
        self.phase = phase
        self.channelId = channelId
        self.control = control
        self.targetOffsetDb = targetOffsetDb
        self.source = source
        self.tsMs = tsMs
    }
}

// MARK: - Session Token Helper

public enum SessionTokenLoader {
    public static func loadToken(from url: URL = AetherMixProtocol.defaultTokenPath) -> String? {
        guard let raw = try? String(contentsOf: url, encoding: .utf8) else {
            return nil
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
