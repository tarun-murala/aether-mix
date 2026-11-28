import Darwin
import Foundation
import Testing
#if canImport(AetherMixCore)
@testable import AetherMixCore
#else
@testable import AetherMix
#endif

@Suite("AetherMix v5.1 Protocol, MeterStore, Bridge NDJSON & Safe-Update Tests")
struct ProtocolTests {

    // MARK: - 1. v5.1 WebSocket Envelope & StateSnapshotPayload Fixture

    @Test("Decodes v5.1 StateSnapshotPayload matching engine/core/protocol.py")
    func decodeStateSnapshotEnvelopeV51() throws {
        let json = """
        {
          "type": "state_snapshot",
          "seq": 42,
          "req_id": null,
          "ts_ms": 1732380000123,
          "payload": {
            "channel_ids": ["ch1_pastor", "ch2_lead_vox", "ch3_bv1", "ch5_keys"],
            "autonomy_mode": "SHADOW_OBSERVE",
            "bypass_latched": false,
            "freeze_learning": true,
            "active_phase": "PRAYER_UNDERSCORE",
            "phase_confidence": 0.94,
            "hysteresis_progress": 1.0,
            "service_step_idx": 2,
            "channels": {
              "ch1_pastor": {
                "armed": true,
                "sync_status": "SYNCED",
                "vol_current_db": 0.0,
                "vol_target_db": 0.0,
                "vol_actual_cc": 96,
                "vol_target_cc": 96,
                "vol_touch_lock_s": 12.5,
                "vol_human_offset_db": 1.5,
                "vol_pickup_dir": 0,
                "reverb_current_db": -20.0,
                "reverb_target_db": -18.0,
                "reverb_actual_cc": 43,
                "reverb_target_cc": 48,
                "reverb_touch_lock_s": 0.0,
                "reverb_human_offset_db": -1.0,
                "reverb_pickup_dir": -1,
                "feedback_ceiling_db": -4.0,
                "speech_open": true,
                "instrument_open": false
              }
            },
            "active_rules": [
              {
                "id": "rule_002",
                "description": "Attenuate backup vocals whenever pastor speaks",
                "phase": "ANY",
                "channel_id": "ch3_bv1",
                "parameter": "conditional_attenuate",
                "value": -24.0,
                "condition": {"when_channel": "ch1_pastor", "predicate": "speech_open"},
                "enabled": true,
                "source": "operator_confirmed",
                "scope": "candidate_permanent",
                "expires_on": null,
                "created_at_ms": 1732379900000
              }
            ],
            "pending_rules": [
              {
                "rule_id": "pend_001",
                "description": "Cut keys by -5 dB during prayer",
                "phase": "PRAYER_UNDERSCORE",
                "channel_id": "ch5_keys",
                "parameter": "vol_db_offset",
                "value": -5.0,
                "condition": null,
                "scope": "ephemeral_session",
                "expires_on": "segment_change",
                "requires_preview": true,
                "applied_in_session": false,
                "created_at_ms": 1732380000000
              }
            ],
            "queued_review_count": 3,
            "hardware_warnings": ["CONTROLLER_DISCONNECTED"]
          }
        }
        """

        let envelope = try JSONDecoder().decode(WSEnvelope.self, from: Data(json.utf8))
        #expect(envelope.type == "state_snapshot")
        #expect(envelope.seq == 42)
        #expect(envelope.reqId == nil)

        let snap = try envelope.decodePayload(StateSnapshotPayload.self)
        #expect(snap.channelIds == ["ch1_pastor", "ch2_lead_vox", "ch3_bv1", "ch5_keys"])
        #expect(snap.autonomyMode == .shadowObserve)
        #expect(!snap.bypassLatched)
        #expect(snap.freezeLearning)
        #expect(snap.activePhase == .prayerUnderscore)
        #expect(abs(snap.phaseConfidence - 0.94) < 1e-6)
        #expect(snap.serviceStepIdx == 2)
        #expect(snap.queuedReviewCount == 3)
        #expect(snap.hardwareWarnings == ["CONTROLLER_DISCONNECTED"])

        let pastor = try #require(snap.channels["ch1_pastor"])
        #expect(pastor.armed)
        #expect(pastor.syncStatus == .synced)
        #expect(pastor.volActualCC == 96)
        #expect(abs(pastor.volTouchLockS - 12.5) < 1e-6)
        #expect(abs((pastor.reverbCurrentDb ?? -99) - (-20.0)) < 1e-6)
        #expect(pastor.reverbActualCC == 43)
        #expect(pastor.reverbTargetCC == 48)
        #expect(pastor.reverbPickupDir == -1)
        #expect(abs((pastor.feedbackCeilingDb ?? 0) - (-4.0)) < 1e-6)
        #expect(pastor.speechOpen)
        #expect(!pastor.instrumentOpen)

        #expect(snap.activeRules.count == 1)
        #expect(snap.activeRules[0].parameter == .conditionalAttenuate)
        #expect(snap.activeRules[0].condition?.whenChannel == "ch1_pastor")
        #expect(snap.activeRules[0].condition?.predicate == .speechOpen)

        #expect(snap.pendingRules.count == 1)
        #expect(snap.pendingRules[0].ruleId == "pend_001")
        #expect(snap.pendingRules[0].requiresPreview)
        #expect(!snap.pendingRules[0].appliedInSession)
        #expect(snap.pendingRules[0].expiresOn == .segmentChange)
    }

    // MARK: - 2. MeterFramePayload, BrainEventPayload & HeartbeatPayload Fixtures

    @Test("Decodes MeterFramePayload, BrainEventPayload & HeartbeatPayload fixtures")
    func decodeMeterBrainAndHeartbeatFixtures() throws {
        let meterJSON = """
        {
          "rms_db": [-18.5, -24.0, -48.0],
          "peak_db": [-12.0, -19.2, -40.0],
          "clip": [false, true, false],
          "vad": [0.92, 0.15, 0.0],
          "master_lufs": -21.4,
          "vol_target_cc": [96, 80, 64],
          "vol_actual_cc": [94, 80, 62],
          "reverb_target_cc": [48, 60, null],
          "reverb_actual_cc": [43, 58, null]
        }
        """
        let meter = try JSONDecoder().decode(MeterFramePayload.self, from: Data(meterJSON.utf8))
        #expect(meter.rmsDb == [-18.5, -24.0, -48.0])
        #expect(meter.clip == [false, true, false])
        #expect(abs(meter.masterLufs - (-21.4)) < 1e-6)
        #expect(meter.volTargetCC == [96, 80, 64])
        #expect(meter.volActualCC == [94, 80, 62])
        #expect(meter.reverbTargetCC == [48, 60, nil])
        #expect(meter.reverbActualCC == [43, 58, nil])

        let brainJSON = """
        {
          "id": "evt_99",
          "category": "SAFETY",
          "summary": "Anti-feedback cut -4.0 dB on Pastor Mic (2.4 kHz)",
          "detail": "Peak +17.2 dB above local median floor for 280 ms",
          "confidence": 0.99,
          "ts_ms": 1732380005000
        }
        """
        let brain = try JSONDecoder().decode(BrainEventPayload.self, from: Data(brainJSON.utf8))
        #expect(brain.category == .safety)
        #expect(abs(brain.confidence - 0.99) < 1e-6)

        let hbJSON = """
        {
          "uptime_s": 1240.5,
          "loop_p95_ms": 1.8,
          "loop_p99_ms": 6.2,
          "audio_ok": true,
          "bridge_ok": true,
          "clap_ok": true,
          "ollama_ok": false
        }
        """
        let hb = try JSONDecoder().decode(HeartbeatPayload.self, from: Data(hbJSON.utf8))
        #expect(abs(hb.uptimeS - 1240.5) < 1e-6)
        #expect(abs(hb.loopP95Ms - 1.8) < 1e-6)
        #expect(hb.audioOk)
        #expect(hb.bridgeOk)
        #expect(!hb.ollamaOk)
    }

    // MARK: - 3. Canonical Taper Parity with protocol.py

    @Test("Canonical dB <-> CC taper math matches protocol.py")
    func canonicalTaperMathMatchesProtocolPy() {
        #expect(TaperMath.ccToDb(0).isInfinite)
        #expect(TaperMath.ccToDb(0) < 0)
        #expect(abs(TaperMath.ccToDb(1) - (-78.2)) < 1e-6)
        #expect(abs(TaperMath.ccToDb(64) - (-6.0)) < 1e-6)
        #expect(abs(TaperMath.ccToDb(96) - 1.1) < 1e-6)
        #expect(abs(TaperMath.ccToDb(112) - 3.7) < 1e-6)
        #expect(abs(TaperMath.ccToDb(127) - 6.0) < 1e-6)

        #expect(TaperMath.dbToCC(-.infinity) == 0)
        #expect(TaperMath.dbToCC(-90.0) == 0)
        #expect(TaperMath.dbToCC(-78.2) == 1)
        #expect(TaperMath.dbToCC(-6.0) == 64)
        #expect(TaperMath.dbToCC(1.1) == 96)
        // Top of range: +6.0 dB is only reached at CC 127 on the MainStage-measured taper
        #expect(TaperMath.dbToCC(6.0) == 127)
        #expect(TaperMath.dbToCC(9.0) == 127)
    }

    // MARK: - 4. MeterStore Isolated High-Frequency Updates

    @MainActor
    @Test("MeterStore tracks channel_ids[], peak hold decay, and clip LED latch")
    func meterStoreUpdatesPeakHoldAndClipLatch() {
        let store = MeterStore(channelIds: ["ch1_pastor", "ch5_keys"])
        let frame1 = MeterFramePayload(
            rmsDb: [-14.0, -28.0],
            peakDb: [-6.0, -22.0],
            clip: [true, false],
            vad: [0.95, 0.0],
            masterLufs: -19.5,
            volTargetCC: [96, 64],
            volActualCC: [95, 64],
            reverbTargetCC: [48, nil],
            reverbActualCC: [45, nil]
        )

        store.applyMeterFrame(frame1, now: 100.0)
        #expect(store.frameCounter == 1)
        #expect(abs(store.masterLufs - (-19.5)) < 1e-6)
        #expect(abs(store.reading(for: "ch1_pastor").peakHoldDb - (-6.0)) < 1e-6)
        #expect(store.reading(for: "ch1_pastor").clip)
        #expect(store.reading(for: "ch1_pastor").reverbTargetCC == 48)
        #expect(store.reading(for: "ch5_keys").reverbActualCC == nil)

        // 0.5s later: signal drops, peak hold & clip LED remain latched
        let frame2 = MeterFramePayload(
            rmsDb: [-40.0, -45.0],
            peakDb: [-36.0, -40.0],
            clip: [false, false],
            vad: [0.1, 0.0],
            masterLufs: -32.0,
            volTargetCC: [96, 64],
            volActualCC: [96, 64],
            reverbTargetCC: [48, nil],
            reverbActualCC: [48, nil]
        )
        store.applyMeterFrame(frame2, now: 100.5)
        #expect(abs(store.reading(for: "ch1_pastor").peakHoldDb - (-6.0)) < 1e-6)
        #expect(store.reading(for: "ch1_pastor").clip)

        // 2.5s later: clip latch expires and peak hold starts decaying
        store.applyMeterFrame(frame2, now: 102.5)
        #expect(!store.reading(for: "ch1_pastor").clip)
        #expect(store.reading(for: "ch1_pastor").peakHoldDb < -6.0)
    }

    @MainActor
    @Test("EngineWebSocketClient routes state_snapshot & meter_frame and caches Last Known Faders")
    func engineWebSocketClientRoutesSnapshotAndMeterFrames() {
        let store = MeterStore()
        let ws = EngineWebSocketClient(tokenOverride: "test-token", meterStore: store)

        let snapFrame = """
        {
          "type": "state_snapshot",
          "seq": 1,
          "ts_ms": 1000,
          "payload": {
            "channel_ids": ["ch1_pastor", "ch5_keys"],
            "autonomy_mode": "ACTIVE",
            "active_phase": "WORSHIP_HIGH",
            "channels": {
              "ch1_pastor": {"armed": true, "sync_status": "SYNCED", "vol_current_db": -2.0, "vol_actual_cc": 90},
              "ch5_keys": {"armed": true, "sync_status": "SYNCED", "vol_current_db": -4.0, "vol_actual_cc": 85}
            }
          }
        }
        """
        ws.handleInboundText(snapFrame)
        #expect(ws.snapshot.autonomyMode == .active)
        #expect(ws.lastKnownFaders["ch1_pastor"]?.volCC == 90)
        #expect(abs((ws.lastKnownFaders["ch5_keys"]?.volDb ?? 0) - (-4.0)) < 1e-6)
        #expect(store.channelIds == ["ch1_pastor", "ch5_keys"])

        let meterEnvelope = """
        {
          "type": "meter_frame",
          "seq": 2,
          "ts_ms": 1020,
          "payload": {
            "rms_db": [-20.0, -15.0],
            "peak_db": [-14.0, -10.0],
            "clip": [false, false],
            "vad": [0.85, 0.0],
            "master_lufs": -14.2,
            "vol_target_cc": [90, 88],
            "vol_actual_cc": [90, 85],
            "reverb_target_cc": [40, null],
            "reverb_actual_cc": [40, null]
          }
        }
        """
        ws.handleInboundText(meterEnvelope)
        #expect(store.frameCounter == 1)
        #expect(abs(store.masterLufs - (-14.2)) < 1e-6)
        #expect(store.reading(for: "ch5_keys").volTargetCC == 88)
    }

    // MARK: - 5. Bridge NDJSON Wire Codec & Live AF_UNIX Loopback Test

    @Test("Bridge NDJSON codec encodes hello/ui_manual_cc/set_bypass and decodes bridge_state/human_touch")
    func bridgeNDJSONEncodingAndDecoding() throws {
        let helloLine = try #require(BridgeSocketClient.encodeHello(token: "sec_123"))
        #expect(helloLine.hasSuffix("\n"))
        let helloObj = try JSONDecoder().decode([String: JSONValue].self, from: Data(helloLine.utf8))
        #expect(helloObj["type"]?.stringValue == "hello")
        #expect(helloObj["role"]?.stringValue == "ui")
        #expect(helloObj["token"]?.stringValue == "sec_123")
        #expect(helloObj["protocol_version"]?.stringValue == "5.1")

        let ccLine = try #require(BridgeSocketClient.encodeManualCC(channelId: "ch5_keys", control: .vol, cc: 150))
        let ccObj = try JSONDecoder().decode([String: JSONValue].self, from: Data(ccLine.utf8))
        #expect(ccObj["type"]?.stringValue == "ui_manual_cc")
        #expect(ccObj["channel_id"]?.stringValue == "ch5_keys")
        #expect(ccObj["control"]?.stringValue == "vol")
        #expect(ccObj["cc"]?.intValue == 127) // Clamped to 7-bit MIDI max

        let bypassLine = try #require(BridgeSocketClient.encodeSetBypass(latched: true))
        let bypassObj = try JSONDecoder().decode([String: JSONValue].self, from: Data(bypassLine.utf8))
        #expect(bypassObj["type"]?.stringValue == "set_bypass")
        #expect(bypassObj["latched"]?.boolValue == true)

        let stateLine = """
        {"type":"bridge_state","last_sent_cc":{"ch1_pastor:vol":96,"ch5_keys:vol":74},"human_seq":{"ch5_keys:vol":3},"bypass_latched":true,"controller_connected":true,"engine_connected":false,"pickup_pending":{"ch5_keys:vol":-1},"restored_from_disk_at":1732380000}
        """
        let decoded = BridgeSocketClient.decodeInboundLine(stateLine)
        guard case .bridgeState(let payload) = decoded else {
            Issue.record("Expected .bridgeState, got \(decoded)")
            return
        }
        #expect(payload.ccKey(channelId: "ch1_pastor", control: .vol) == 96)
        #expect(payload.pickupDirection(channelId: "ch5_keys", control: .vol) == -1)
        #expect(payload.bypassLatched)
        #expect(!payload.engineConnected)
        #expect(payload.restoredFromDiskAt == 1732380000)

        let touchLine = """
        {"type":"human_touch_event","channel_id":"ch5_keys","control":"vol","human_seq":4,"picked_up":false,"source":"hardware","ts_ms":1732380001}
        """
        guard case .humanTouch(let touch) = BridgeSocketClient.decodeInboundLine(touchLine) else {
            Issue.record("Expected .humanTouch")
            return
        }
        #expect(touch.channelId == "ch5_keys")
        #expect(touch.humanSeq == 4)
        #expect(!touch.pickedUp)
    }

    @MainActor
    @Test("Live POSIX AF_UNIX BridgeSocketClient sends hello/ui_manual_cc/set_bypass and receives bridge_state with engine offline")
    func liveAFUnixBridgeSocketRoundtrip() async throws {
        // Real kernel POSIX AF_UNIX SOCK_STREAM pair (exercises full non-blocking DispatchSource
        // read/write NDJSON pipeline natively both inside and outside the macOS sandbox).
        var pair: [Int32] = [-1, -1]
        #expect(Darwin.socketpair(AF_UNIX, SOCK_STREAM, 0, &pair) == 0)
        let clientFD = pair[0]
        let serverFD = pair[1]

        let receivedByServer = LockedLines()
        let serverTask = Task.detached {
            defer { Darwin.close(serverFD) }

            let reply = """
            {"type":"bridge_state","last_sent_cc":{"ch5_keys:vol":82},"human_seq":{"ch5_keys:vol":1},"bypass_latched":false,"controller_connected":true,"engine_connected":false,"pickup_pending":{},"restored_from_disk_at":null}\n
            """
            let replyBytes = Array(reply.utf8)
            _ = replyBytes.withUnsafeBytes { Darwin.write(serverFD, $0.baseAddress, replyBytes.count) }

            var buffer = Data()
            var tmp = [UInt8](repeating: 0, count: 1024)
            let deadline = Date().addingTimeInterval(2.0)
            while Date() < deadline && receivedByServer.count() < 4 {
                let n = Darwin.read(serverFD, &tmp, tmp.count)
                if n > 0 {
                    buffer.append(contentsOf: tmp[0..<n])
                    while let nl = buffer.firstRange(of: Data([0x0A])) {
                        let lineData = buffer.subdata(in: buffer.startIndex..<nl.lowerBound)
                        buffer.removeSubrange(buffer.startIndex..<nl.upperBound)
                        if let line = String(data: lineData, encoding: .utf8), !line.isEmpty {
                            receivedByServer.append(line)
                        }
                    }
                } else {
                    break
                }
            }
        }

        let client = BridgeSocketClient(socketPath: "/tmp/unused_pair.sock", tokenOverride: "tok_live_test")
        client.attachConnectedDescriptor(clientFD)

        for _ in 0..<40 {
            if client.isConnected && client.bridgeState.ccKey(channelId: "ch5_keys", control: .vol) == 82 {
                break
            }
            try await Task.sleep(for: .milliseconds(25))
        }

        #expect(client.isConnected)
        #expect(client.bridgeState.ccKey(channelId: "ch5_keys", control: .vol) == 82)
        #expect(!client.bridgeState.engineConnected)

        client.sendManualCC(channelId: "ch5_keys", control: .vol, cc: 70)
        client.setBypass(latched: true)

        for _ in 0..<40 {
            if receivedByServer.count() >= 4 { break }
            try await Task.sleep(for: .milliseconds(25))
        }

        client.stop()
        await serverTask.value

        let lines = receivedByServer.snapshot()
        #expect(lines.contains { $0.contains("\"type\":\"hello\"") && $0.contains("\"role\":\"ui\"") && $0.contains("\"protocol_version\":\"5.1\"") })
        #expect(lines.contains { $0.contains("\"type\":\"get_bridge_state\"") })
        #expect(lines.contains { $0.contains("\"type\":\"ui_manual_cc\"") && $0.contains("\"cc\":70") })
        #expect(lines.contains { $0.contains("\"type\":\"set_bypass\"") && $0.contains("\"latched\":true") })
    }

    // MARK: - 6. Safe-Update Lock Enforcement (SPEC §8)

    @MainActor
    @Test("Safe-Update Lock hard-blocks updates in ACTIVE mode and allows SHADOW/BYPASS")
    func safeUpdateLockBlocksOnlyDuringActiveMode() {
        let activeCheck = EngineProcessManager.evaluateSafeUpdateLock(autonomyMode: .active)
        #expect(!activeCheck.isAllowed)
        if case .blocked(let reason) = activeCheck {
            #expect(reason.contains("ACTIVE"))
        } else {
            Issue.record("Expected blocked result during ACTIVE mode")
        }

        #expect(EngineProcessManager.evaluateSafeUpdateLock(autonomyMode: .shadowObserve).isAllowed)
        #expect(EngineProcessManager.evaluateSafeUpdateLock(autonomyMode: .emergencyBypass).isAllowed)

        let manager = EngineProcessManager()
        let blocked = manager.requestUpdateAction("Model Update", autonomyMode: .active)
        #expect(!blocked.isAllowed)
        #expect(manager.safeUpdateAlert != nil)

        let allowed = manager.requestUpdateAction("Model Update", autonomyMode: .shadowObserve)
        #expect(allowed.isAllowed)
        #expect(manager.safeUpdateAlert == nil)
    }

    // MARK: - 7. Cross-System QA Regression Guards (#3, #11)

    @MainActor
    @Test("verify_mute_polarity payload aligns with engine/main.py (mute_active_high, active_high, test_mute)")
    func verifyMutePolarityPayloadMatchesEngine() {
        let payload = EngineWebSocketClient.makeVerifyMutePolarityPayload(
            channelId: "ch2_lead_vox",
            activeHigh: false,
            testMute: true
        )
        #expect(payload["channel_id"]?.stringValue == "ch2_lead_vox")
        #expect(payload["mute_active_high"]?.boolValue == false)
        #expect(payload["active_high"]?.boolValue == false)
        #expect(payload["test_mute"]?.boolValue == true)
    }

    @Test("Autonomy mode transition blocks direct EMERGENCY_BYPASS -> ACTIVE jump")
    func safeModeTransitionPreventsDirectBypassToActive() {
        #expect(
            MainDashboardView.safeModeTransition(from: .emergencyBypass, requested: .active)
                == .shadowObserve
        )
        #expect(
            MainDashboardView.safeModeTransition(from: .emergencyBypass, requested: .shadowObserve)
                == .shadowObserve
        )
        #expect(
            MainDashboardView.safeModeTransition(from: .shadowObserve, requested: .active)
                == .active
        )
        #expect(
            MainDashboardView.safeModeTransition(from: .active, requested: .emergencyBypass)
                == .emergencyBypass
        )
    }

    @Test("StateSnapshotPayload decodes use_service_plan and custom order_of_service segments")
    func decodesIgnoreOrderOfServiceAndCustomSegments() throws {
        let json = """
        {
          "channel_ids": ["ch1_pastor", "ch5_keys"],
          "autonomy_mode": "SHADOW_OBSERVE",
          "bypass_latched": false,
          "freeze_learning": false,
          "use_service_plan": false,
          "order_of_service": [
            {"id": "s1", "title": "Spontaneous Praise", "expected_phase": "WORSHIP_HIGH"},
            {"id": "s2", "title": "Prayer Time", "expected_phase": "PRAYER_UNDERSCORE"}
          ],
          "active_phase": "WORSHIP_HIGH",
          "phase_confidence": 0.91,
          "hysteresis_progress": 1.0,
          "service_step_idx": 1,
          "channels": {},
          "active_rules": [],
          "pending_rules": [],
          "queued_review_count": 0,
          "hardware_warnings": []
        }
        """
        let snap = try JSONDecoder().decode(StateSnapshotPayload.self, from: Data(json.utf8))
        #expect(snap.useServicePlan == false)
        #expect(snap.orderOfService.count == 2)
        #expect(snap.orderOfService[0].title == "Spontaneous Praise")
        #expect(snap.orderOfService[1].expectedPhase == .prayerUnderscore)
        #expect(snap.serviceStepIdx == 1)
    }
}

private final class LockedLines: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []

    func append(_ line: String) {
        lock.lock()
        lines.append(line)
        lock.unlock()
    }

    func count() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return lines.count
    }

    func snapshot() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return lines
    }
}
