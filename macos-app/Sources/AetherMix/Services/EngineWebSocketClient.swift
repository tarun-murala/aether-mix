import Foundation
import Observation

public enum EngineRPCError: Error, LocalizedError, Sendable {
    case notConnected
    case encodingFailed
    case timeout(String)
    case rejected(code: String, message: String)

    public var errorDescription: String? {
        switch self {
        case .notConnected:
            return "AI Engine WebSocket is not connected"
        case .encodingFailed:
            return "Failed to encode WebSocket RPC envelope"
        case .timeout(let type):
            return "Timed out waiting for ack to '\(type)'"
        case .rejected(let code, let message):
            return "[\(code)] \(message)"
        }
    }
}

/// Authenticated WebSocket client (`ws://127.0.0.1:8765/ws`) with v5.1 hello handshake,
/// `req_id` RPC correlation, 1 Hz heartbeat watchdog, cached Last Known Fader Positions
/// for the DISCONNECTED banner, and zero-diff routing of 30–50 Hz `meter_frame` updates
/// exclusively into `MeterStore`.
@MainActor
@Observable
public final class EngineWebSocketClient {
    public private(set) var isConnected: Bool = false
    public private(set) var isHandshakeComplete: Bool = false
    public private(set) var connectionStatusText: String = "Disconnected"
    public private(set) var lastError: String?

    public private(set) var snapshot: StateSnapshotPayload = StateSnapshotPayload()
    public private(set) var brainEvents: [BrainEventPayload] = []
    public private(set) var lastHeartbeat: HeartbeatPayload = HeartbeatPayload()
    public private(set) var lastHeartbeatAt: Date?
    public private(set) var heartbeatHealthy: Bool = false

    /// Cached Last Known Fader Positions preserved across engine crashes/disconnects.
    public private(set) var lastKnownFaders: [String: CachedFaderState] = [:]
    public private(set) var queuedReviewSamples: [QueuedReviewSample] = []

    /// Isolated meter store: mutated directly on `meter_frame` without mutating this client's observed fields.
    public let meterStore: MeterStore

    public let endpointURL: URL
    private let tokenProvider: @Sendable () -> String?
    private let urlSession: URLSession

    private var webSocketTask: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var heartbeatMonitorTask: Task<Void, Never>?
    private var reconnectTask: Task<Void, Never>?

    private var shouldStayConnected: Bool = false
    private var outboundSeq: Int = 0
    private var reconnectAttempt: Int = 0
    private var pendingRPCs: [String: CheckedContinuation<WSAckPayload, Error>] = [:]

    private let maxBrainEvents = 500

    public init(
        endpointURL: URL = AetherMixProtocol.defaultWebSocketURL,
        tokenPath: URL = AetherMixProtocol.defaultTokenPath,
        tokenOverride: String? = nil,
        meterStore: MeterStore = MeterStore()
    ) {
        self.endpointURL = endpointURL
        self.meterStore = meterStore
        if let tokenOverride {
            self.tokenProvider = { tokenOverride }
        } else {
            self.tokenProvider = { SessionTokenLoader.loadToken(from: tokenPath) }
        }
        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = false
        self.urlSession = URLSession(configuration: config)
    }

    public func start() {
        guard !shouldStayConnected else { return }
        shouldStayConnected = true
        startHeartbeatMonitor()
        connectOnce()
    }

    public func stop() {
        shouldStayConnected = false
        reconnectTask?.cancel()
        reconnectTask = nil
        heartbeatMonitorTask?.cancel()
        heartbeatMonitorTask = nil
        teardownSocket(reason: "Stopped by client")
    }

    public func connectOnce() {
        teardownSocket(reason: nil)

        var request = URLRequest(url: endpointURL)
        request.timeoutInterval = 10
        if let token = tokenProvider(), !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        connectionStatusText = "Connecting…"
        let task = urlSession.webSocketTask(with: request)
        self.webSocketTask = task
        task.resume()

        self.isConnected = true
        self.connectionStatusText = "Handshaking…"
        self.reconnectAttempt = 0
        self.lastError = nil

        receiveTask = Task { @MainActor [weak self] in
            await self?.receiveLoop(task: task)
        }

        Task { @MainActor [weak self] in
            await self?.sendHelloHandshake()
        }
    }

    private func sendHelloHandshake() async {
        do {
            let ack = try await sendRPC(
                type: "hello",
                payload: [
                    "protocol_version": .string(AetherMixProtocol.protocolVersion),
                    "client": .string("macos-app")
                ],
                timeoutSeconds: 5.0
            )
            if ack.ok {
                self.isHandshakeComplete = true
                self.connectionStatusText = "Connected (v\(AetherMixProtocol.protocolVersion))"
            } else {
                self.lastError = ack.message ?? "Handshake failed"
                self.connectionStatusText = "Handshake Rejected"
            }
        } catch {
            // Engine may immediately push state_snapshot without separate hello ack in some dev setups.
            self.lastError = error.localizedDescription
        }
    }

    private func receiveLoop(task: URLSessionWebSocketTask) async {
        while !Task.isCancelled && webSocketTask === task {
            do {
                let message = try await task.receive()
                switch message {
                case .string(let text):
                    handleInboundText(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        handleInboundText(text)
                    }
                @unknown default:
                    break
                }
            } catch {
                if webSocketTask === task {
                    teardownSocket(reason: error.localizedDescription)
                    scheduleReconnectIfNeeded()
                }
                return
            }
        }
    }

    /// Processes a single inbound WebSocket JSON frame.
    /// Exposed `public` for deterministic unit testing without a live server.
    public func handleInboundText(_ text: String) {
        guard let data = text.data(using: .utf8),
              let envelope = try? JSONDecoder().decode(WSEnvelope.self, from: data) else {
            return
        }

        switch envelope.type {
        case "meter_frame":
            // Route EXCLUSIVELY to isolated MeterStore (zero observation invalidation on EngineWebSocketClient!).
            if let frame = try? envelope.decodePayload(MeterFramePayload.self) {
                meterStore.applyMeterFrame(frame)
            }

        case "state_snapshot":
            if let snap = try? envelope.decodePayload(StateSnapshotPayload.self) {
                self.snapshot = snap
                self.isHandshakeComplete = true
                if self.isConnected {
                    self.connectionStatusText = "Connected (v\(AetherMixProtocol.protocolVersion))"
                }
                meterStore.updateChannelOrder(snap.channelIds)
                updateCachedFaders(from: snap)
            }

        case "brain_event":
            if let event = try? envelope.decodePayload(BrainEventPayload.self) {
                brainEvents.insert(event, at: 0)
                if brainEvents.count > maxBrainEvents {
                    brainEvents.removeLast(brainEvents.count - maxBrainEvents)
                }
            }

        case "heartbeat":
            if let hb = try? envelope.decodePayload(HeartbeatPayload.self) {
                self.lastHeartbeat = hb
                self.lastHeartbeatAt = Date()
                self.heartbeatHealthy = true
            }

        case "ack", "error":
            let ack = (try? envelope.decodePayload(WSAckPayload.self))
                ?? WSAckPayload(
                    reqId: envelope.reqId,
                    ok: envelope.type == "ack",
                    code: envelope.payload["code"]?.stringValue,
                    message: envelope.payload["message"]?.stringValue
                )
            let resolvedReqId = ack.reqId ?? envelope.reqId
            if let reqId = resolvedReqId, let continuation = pendingRPCs.removeValue(forKey: reqId) {
                if ack.ok && envelope.type != "error" {
                    continuation.resume(returning: ack)
                } else {
                    let err = EngineRPCError.rejected(
                        code: ack.code ?? "RPC_ERROR",
                        message: ack.message ?? "Engine rejected request"
                    )
                    self.lastError = err.localizedDescription
                    continuation.resume(throwing: err)
                }
            } else if !ack.ok || envelope.type == "error" {
                self.lastError = ack.message ?? "Engine error"
            }

        case "queued_reviews":
            if let arr = envelope.payload["samples"]?.arrayValue,
               let encoded = try? JSONEncoder().encode(arr),
               let samples = try? JSONDecoder().decode([QueuedReviewSample].self, from: encoded) {
                self.queuedReviewSamples = samples
            }

        default:
            break
        }
    }

    private func updateCachedFaders(from snap: StateSnapshotPayload) {
        let now = Date()
        var cached = lastKnownFaders
        for id in snap.channelIds {
            if let ch = snap.channels[id] {
                cached[id] = CachedFaderState(
                    channelId: id,
                    volDb: ch.volCurrentDb,
                    volCC: ch.volActualCC,
                    reverbDb: ch.reverbCurrentDb,
                    reverbCC: ch.reverbActualCC,
                    updatedAt: now
                )
            }
        }
        self.lastKnownFaders = cached
    }

    private func startHeartbeatMonitor() {
        heartbeatMonitorTask?.cancel()
        heartbeatMonitorTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                guard let self else { return }
                if let last = self.lastHeartbeatAt {
                    self.heartbeatHealthy = self.isConnected && Date().timeIntervalSince(last) <= 3.5
                } else {
                    self.heartbeatHealthy = false
                }
            }
        }
    }

    private func teardownSocket(reason: String?) {
        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        isConnected = false
        isHandshakeComplete = false
        heartbeatHealthy = false
        connectionStatusText = "Disconnected"
        if let reason {
            lastError = reason
        }
        let orphaned = pendingRPCs
        pendingRPCs.removeAll()
        for (_, cont) in orphaned {
            cont.resume(throwing: EngineRPCError.notConnected)
        }
    }

    private func scheduleReconnectIfNeeded() {
        guard shouldStayConnected else { return }
        reconnectTask?.cancel()
        reconnectAttempt += 1
        let backoff: Double = min(5.0, Double(reconnectAttempt))
        reconnectTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(backoff))
            guard let self, self.shouldStayConnected, !self.isConnected else { return }
            self.connectOnce()
        }
    }

    // MARK: - Core Correlated RPC Transport

    @discardableResult
    public func sendRPC(
        type: String,
        payload: [String: JSONValue] = [:],
        timeoutSeconds: Double = 5.0
    ) async throws -> WSAckPayload {
        guard isConnected, let task = webSocketTask else {
            throw EngineRPCError.notConnected
        }

        outboundSeq += 1
        let reqId = UUID().uuidString
        let envelope = WSEnvelope(
            type: type,
            seq: outboundSeq,
            reqId: reqId,
            tsMs: Int64(Date().timeIntervalSince1970 * 1000),
            payload: payload
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(envelope),
              let text = String(data: data, encoding: .utf8) else {
            throw EngineRPCError.encodingFailed
        }

        return try await withCheckedThrowingContinuation { continuation in
            pendingRPCs[reqId] = continuation

            Task { @MainActor [weak self] in
                do {
                    try await task.send(.string(text))
                } catch {
                    if let cont = self?.pendingRPCs.removeValue(forKey: reqId) {
                        cont.resume(throwing: error)
                    }
                }
            }

            Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(timeoutSeconds))
                if let cont = self?.pendingRPCs.removeValue(forKey: reqId) {
                    cont.resume(throwing: EngineRPCError.timeout(type))
                }
            }
        }
    }

    // MARK: - v5.1 Typed Client -> Server RPCs (SPEC §6.4)

    public func setMode(_ mode: AutonomyMode) async {
        _ = try? await sendRPC(type: "set_mode", payload: ["mode": .string(mode.rawValue)])
    }

    public func setChannelArmed(channelId: String, armed: Bool) async {
        _ = try? await sendRPC(
            type: "set_channel_armed",
            payload: ["channel_id": .string(channelId), "armed": .bool(armed)]
        )
    }

    public func restorePreviousArming() async {
        _ = try? await sendRPC(type: "restore_previous_arming")
    }

    public func syncChannel(channelId: String, control: ControlType? = nil) async {
        var payload: [String: JSONValue] = ["channel_id": .string(channelId)]
        if let control {
            payload["control"] = .string(control.rawValue)
        }
        _ = try? await sendRPC(type: "sync_channel", payload: payload)
    }

    public func releaseHumanOffset(channelId: String, control: ControlType) async {
        _ = try? await sendRPC(
            type: "release_human_offset",
            payload: [
                "channel_id": .string(channelId),
                "control": .string(control.rawValue)
            ]
        )
    }

    public func setFreezeLearning(_ enabled: Bool) async {
        _ = try? await sendRPC(type: "set_freeze_learning", payload: ["enabled": .bool(enabled)])
    }

    public func overridePhase(_ phase: ServicePhase) async {
        _ = try? await sendRPC(type: "override_phase", payload: ["phase": .string(phase.rawValue)])
    }

    public func advanceServiceStep(stepIdx: Int? = nil) async {
        var payload: [String: JSONValue] = [:]
        if let stepIdx {
            payload["step_idx"] = .int(Int64(stepIdx))
        }
        _ = try? await sendRPC(type: "advance_service_step", payload: payload)
    }

    public func setUseServicePlan(_ enabled: Bool) async {
        _ = try? await sendRPC(
            type: "set_use_service_plan",
            payload: ["enabled": .bool(enabled)]
        )
    }

    public func updateOrderOfService(_ segments: [ServiceSegment], useServicePlan: Bool? = nil) async {
        let encodedSegments: [JSONValue] = segments.map { seg in
            .object([
                "id": .string(seg.id),
                "title": .string(seg.title),
                "expected_phase": .string(seg.expectedPhase.rawValue)
            ])
        }
        var payload: [String: JSONValue] = ["segments": .array(encodedSegments)]
        if let useServicePlan {
            payload["use_service_plan"] = .bool(useServicePlan)
        }
        _ = try? await sendRPC(type: "update_order_of_service", payload: payload)
    }

    public func sendFeedbackThumb(_ thumb: FeedbackThumb) async {
        _ = try? await sendRPC(type: "feedback_thumb", payload: ["thumb": .string(thumb.rawValue)])
    }

    public func submitCritique(_ text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        _ = try? await sendRPC(
            type: "submit_critique",
            payload: ["text": .string(trimmed)],
            timeoutSeconds: 10.0
        )
    }

    public func resolvePendingRule(ruleId: String, action: PendingRuleAction) async {
        _ = try? await sendRPC(
            type: "resolve_pending_rule",
            payload: [
                "rule_id": .string(ruleId),
                "action": .string(action.rawValue)
            ]
        )
    }

    public func toggleRule(ruleId: String, enabled: Bool) async {
        _ = try? await sendRPC(
            type: "toggle_rule",
            payload: ["rule_id": .string(ruleId), "enabled": .bool(enabled)]
        )
    }

    public func deleteRule(ruleId: String) async {
        _ = try? await sendRPC(type: "delete_rule", payload: ["rule_id": .string(ruleId)])
    }

    public func resetLearning(phase: ServicePhase? = nil, channelId: String? = nil) async {
        var payload: [String: JSONValue] = [:]
        if let phase {
            payload["phase"] = .string(phase.rawValue)
        }
        if let channelId {
            payload["channel_id"] = .string(channelId)
        }
        _ = try? await sendRPC(type: "reset_learning", payload: payload)
    }

    public func commitQueuedReviews(commit: Bool, sampleIds: [String]? = nil) async {
        var payload: [String: JSONValue] = ["action": .string(commit ? "commit" : "discard")]
        if let sampleIds {
            payload["sample_ids"] = .array(sampleIds.map { .string($0) })
        }
        _ = try? await sendRPC(type: "commit_queued_reviews", payload: payload)
        if sampleIds == nil {
            queuedReviewSamples.removeAll()
        } else if let ids = sampleIds {
            let idSet = Set(ids)
            queuedReviewSamples.removeAll { idSet.contains($0.id) }
        }
    }

    public func captureRoomSilence(durationS: Double = 5.0) async throws -> WSAckPayload {
        try await sendRPC(
            type: "capture_room_silence",
            payload: ["duration_s": .double(durationS)],
            timeoutSeconds: durationS + 10.0
        )
    }

    public func captureInputNominals(group: String, durationS: Double = 10.0) async throws -> WSAckPayload {
        try await sendRPC(
            type: "capture_input_nominals",
            payload: ["group": .string(group), "duration_s": .double(durationS)],
            timeoutSeconds: durationS + 10.0
        )
    }

    public func captureBleedFloor(durationS: Double = 10.0) async throws -> WSAckPayload {
        try await sendRPC(
            type: "capture_bleed_floor",
            payload: ["duration_s": .double(durationS)],
            timeoutSeconds: durationS + 10.0
        )
    }

    public func captureBaseline(phase: ServicePhase, quickDeriveOthers: Bool = false) async throws -> WSAckPayload {
        try await sendRPC(
            type: "capture_baseline",
            payload: [
                "phase": .string(phase.rawValue),
                "quick_derive_others": .bool(quickDeriveOthers)
            ],
            timeoutSeconds: 10.0
        )
    }

    public func sendMidiLearnPulse(channelId: String, control: ControlType) async {
        _ = try? await sendRPC(
            type: "midi_learn_pulse",
            payload: [
                "channel_id": .string(channelId),
                "control": .string(control.rawValue)
            ]
        )
    }

    public static func makeVerifyMutePolarityPayload(
        channelId: String,
        activeHigh: Bool,
        testMute: Bool = true
    ) -> [String: JSONValue] {
        [
            "channel_id": .string(channelId),
            "mute_active_high": .bool(activeHigh),
            "active_high": .bool(activeHigh),
            "test_mute": .bool(testMute)
        ]
    }

    public func verifyMutePolarity(channelId: String, activeHigh: Bool, testMute: Bool = true) async {
        _ = try? await sendRPC(
            type: "verify_mute_polarity",
            payload: Self.makeVerifyMutePolarityPayload(
                channelId: channelId,
                activeHigh: activeHigh,
                testMute: testMute
            )
        )
    }
}
